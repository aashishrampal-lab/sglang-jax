import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.managers.schedule_batch import (
    _SPEC_CACHE_LOC_FIT,
    spec_cache_loc_precompile_sizes,
)
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

logger = logging.getLogger(__name__)


class EAGLEWorker(BaseSpecWorker):
    """Standard EAGLE speculative decode orchestrator.

    Composes a ``target_worker`` (full model) with an ``EagleDraftWorker``
    (draft model).  Implements the ``BaseSpecWorker`` contract so the
    scheduler interface is unchanged.
    """

    def __init__(self, server_args, target_worker: ModelWorker, draft_worker=None):
        super().__init__(
            server_args,
            target_worker,
            draft_worker or EagleDraftWorker(server_args, target_worker),
        )

    # -- BaseSpecWorker provides target_worker/draft_worker/verify/
    #    forward_target_extend/forward_batch_speculative_generation --

    # -- Precompilation --

    def run_spec_decode_precompile(self):
        disagg_mode = getattr(self.server_args, "disaggregation_mode", "null")
        use_overlap = not self.server_args.disable_overlap_schedule
        if disagg_mode in ("prefill", "decode"):
            use_overlap = use_overlap and bool(
                getattr(self.server_args, "disaggregation_enable_overlap_schedule", False)
            )
        if use_overlap:
            self.init_spec_relay_buffers()
        self.precompile_spec_extend()
        if disagg_mode != "prefill":
            self.precompile_spec_decode()
        # FIXME precompile some kernel

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        dp_size = self.server_args.dp_size
        bs, _ = self.draft_worker.get_max_padded_size()
        token_paddings = list(self.precompile_token_paddings)
        user_tok_paddings = getattr(self.server_args, "precompile_token_paddings", None)
        if (
            getattr(self.server_args, "disaggregation_mode", "null") == "decode"
            and user_tok_paddings
        ):
            max_user_tok = max(user_tok_paddings)
            capped = [t for t in token_paddings if t <= max_user_tok]
            if capped:
                token_paddings = capped
        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s dp_size=%d",
            self.precompile_bs_paddings[-1:],
            token_paddings,
            dp_size,
        )

        # extend-shaped spec batches pad bs to the largest bucket; with the
        # fitted cache_loc padding on, their page table may land on any rung
        # of that bucket, so compile each rung.
        cache_loc_sizes = (
            spec_cache_loc_precompile_sizes(
                len(self.precompile_bs_paddings) - 1,
                self.precompile_bs_paddings,
                self.precompile_cache_loc_paddings,
                self.page_size,
                fit_enabled=True,
            )
            if _SPEC_CACHE_LOC_FIT
            else [self.precompile_cache_loc_paddings[-1]]
        )
        pairs = list(itertools.product([bs], token_paddings, cache_loc_sizes))

        with tqdm(pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens, cache_loc_size = pair[0], pair[1], pair[2]
                pbar.set_postfix(
                    bs=bs, tokens=num_tokens, cache_loc=cache_loc_size, dp_size=dp_size
                )
                if bs > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs, num_tokens)
                    continue
                if bs % dp_size != 0:
                    logger.warning(
                        "[SPEC_EXTEND] skip bs=%d (not divisible by dp_size=%d)", bs, dp_size
                    )
                    continue
                per_dp_bs = bs // dp_size
                model_worker_batch = self.draft_worker.compilation_manager._make_dummy_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    cache_loc_size,
                    speculative_algorithm=self.speculative_algorithm,
                    dp_size=dp_size,
                    per_dp_bs_size=per_dp_bs,
                )
                if not self._can_use_fused_spec_prefill(model_worker_batch):
                    logger.warning(
                        "[SPEC_EXTEND] skip fused precompile because fused spec prefill is disabled"
                    )
                    continue
                if self.spec_relay_buffers is not None:
                    self.forward_batch_speculative_prefill_overlap(model_worker_batch)
                    jax.block_until_ready(self.spec_relay_buffers)
                else:
                    self.forward_batch_speculative_generation(model_worker_batch)
                chunk_size = int(getattr(self.server_args, "chunked_prefill_size", 0) or 0)
                if (
                    getattr(self.server_args, "disaggregation_mode", "null") in ("null", "prefill")
                    and chunk_size > 0
                    and num_tokens >= chunk_size
                ):
                    model_worker_batch.extend_next_token_ids = np.zeros(
                        (model_worker_batch.req_pool_indices.shape[0],), dtype=np.int32
                    )
                    model_worker_batch.forward_batch = None
                    if self.spec_relay_buffers is not None:
                        self.forward_batch_speculative_prefill_overlap(model_worker_batch)
                        jax.block_until_ready(self.spec_relay_buffers)
                    else:
                        self.forward_batch_speculative_generation(model_worker_batch)
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        dp_size = self.server_args.dp_size
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s dp_size=%d",
            self.precompile_bs_paddings,
            dp_size,
        )

        # One (bs, cache_loc) shape per bs bucket, or every rung of the bucket
        # when the fitted spec cache_loc padding is on (the runtime fit may
        # only land on those rungs, so this precompile covers all of them).
        shapes = []
        for i, bs in enumerate(self.precompile_bs_paddings):
            legacy = (
                (bs * self.draft_worker.max_req_len + self.page_size - 1)
                // self.page_size
                * self.page_size
            )
            sizes = (
                spec_cache_loc_precompile_sizes(
                    i,
                    self.precompile_bs_paddings,
                    self.precompile_cache_loc_paddings,
                    self.page_size,
                    fit_enabled=True,
                )
                if _SPEC_CACHE_LOC_FIT
                else [legacy]
            )
            shapes.extend((bs, size) for size in sizes)
        logger.info(
            "[SPEC_DECODE] cache_loc_fit=%s precompile shapes (bs, cache_loc)=%s",
            _SPEC_CACHE_LOC_FIT,
            shapes,
        )
        if self.spec_relay_buffers is not None:
            from sgl_jax.srt.speculative.relay_buffer import update_spec_relay_buffers_admit

            max_running = int(getattr(self.server_args, "max_running_requests", 256) or 256)
            per_dp_cap = max(1, max_running // max(1, dp_size))
            total_slots = dp_size * per_dp_cap
            relay_cap = int(self.spec_relay_buffers.verified_id.shape[1])
            relay_steps = int(self.spec_relay_buffers.topk_index.shape[-1])
            fut_idx_np = np.arange(total_slots, dtype=np.int32) % max(1, relay_cap)
            val_mask_np = np.ones((total_slots,), dtype=np.bool_)
            topk_idx_np = np.ones((total_slots, relay_steps), dtype=np.int32)
            ver_id_np = np.ones((total_slots,), dtype=np.int32)
            new_seq_np = np.full((total_slots,), 16, dtype=np.int32)
            sh_1d = NamedSharding(self.mesh, P("data"))
            sh_2d = NamedSharding(self.mesh, P("data", None))
            with jax.set_mesh(self.mesh):
                self.spec_relay_buffers = update_spec_relay_buffers_admit(
                    self.spec_relay_buffers,
                    jax.device_put(fut_idx_np, sh_1d),
                    jax.device_put(val_mask_np, sh_1d),
                    jax.device_put(topk_idx_np, sh_2d),
                    jax.device_put(ver_id_np, sh_1d),
                    jax.device_put(new_seq_np, sh_1d),
                    dp_size=dp_size,
                )
            jax.block_until_ready(self.spec_relay_buffers)

        with tqdm(shapes, desc="[SPEC_DECODE] PRECOMPILE", leave=False) as pbar:
            for bs, aligned_cache_loc_size in pbar:
                pbar.set_postfix(bs=bs, cache_loc=aligned_cache_loc_size, dp_size=dp_size)
                if bs % dp_size != 0:
                    logger.warning(
                        "[SPEC_DECODE] skip bs=%d (not divisible by dp_size=%d)", bs, dp_size
                    )
                    continue
                per_dp_bs = bs // dp_size

                def _make_decode_batch(
                    *,
                    bs=bs,
                    per_dp_bs=per_dp_bs,
                    aligned_cache_loc_size=aligned_cache_loc_size,
                ):
                    batch = self.draft_worker.compilation_manager._make_dummy_batch(
                        bs,
                        bs,
                        ForwardMode.DECODE,
                        aligned_cache_loc_size,
                        speculative_algorithm=self.speculative_algorithm,
                        dp_size=dp_size,
                        per_dp_bs_size=per_dp_bs,
                    )
                    if _SPEC_CACHE_LOC_FIT:
                        batch.spec_cache_loc_size = aligned_cache_loc_size
                    # Pad out_cache_loc to the conservative decode allocation
                    # bucket used by runtime _get_spec_decode_mwb_dp.
                    ocl_target = bs * self.speculative_num_draft_tokens * 2
                    if batch.out_cache_loc.shape[0] < ocl_target:
                        pad_len = ocl_target - batch.out_cache_loc.shape[0]
                        batch.out_cache_loc = np.concatenate(
                            [
                                np.asarray(batch.out_cache_loc, dtype=np.int32),
                                np.full(pad_len, -1, dtype=np.int32),
                            ]
                        )
                    return batch

                model_worker_batch = _make_decode_batch()
                assert not model_worker_batch.return_logprob
                assert not model_worker_batch.return_output_logprob_only
                assert model_worker_batch.sampling_info.is_all_greedy
                num_steps = self.speculative_num_steps
                from sgl_jax.srt.speculative.multi_layer_draft_worker import (
                    MultiLayerDraftWorker,
                )

                is_multi_layer = (
                    isinstance(self.draft_worker, MultiLayerDraftWorker)
                    or self.server_args.speculative_algorithm == "NEXTN"
                )
                if is_multi_layer and self.topk == 1:
                    topk_shape = (bs, num_steps)
                else:
                    topk_shape = (bs, num_steps, self.topk) if is_multi_layer else (bs, self.topk)
                data_sharding = NamedSharding(self.mesh, P("data"))
                def _make_non_relay_spec_info(batch):
                    return EagleDraftInput(
                        topk_p=jax.device_put(np.ones(topk_shape, dtype=np.float32), data_sharding),
                        topk_index=jax.device_put(np.ones(topk_shape, dtype=np.int32), data_sharding),
                        hidden_states=np.ones(
                            (bs, self.draft_worker.model_config.hidden_size),
                            dtype=(
                                jnp.bfloat16 if self.server_args.dtype == "bfloat16" else np.float32
                            ),
                        ),
                        verified_id=jax.device_put(np.ones((bs,), dtype=np.int32), data_sharding),
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        num_tokens_per_batch=np.asarray(1, dtype=np.int32),
                        num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
                        allocate_lens=batch.seq_lens + max(3 * EagleDraftInput.ALLOC_LEN_PER_DECODE, 2 * num_steps),
                    )

                def _make_relay_spec_info(batch):
                    return EagleDraftInput(
                        future_indices=np.asarray(batch.req_pool_indices, dtype=np.int32),
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        num_tokens_per_batch=np.asarray(1, dtype=np.int32),
                        num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
                        allocate_lens=batch.seq_lens + max(3 * EagleDraftInput.ALLOC_LEN_PER_DECODE, 2 * num_steps),
                    )

                greedy_modes = (True, False)
                for is_greedy in greedy_modes:
                    top_p_modes = (False,) if is_greedy else (True, False)
                    for use_top_p in top_p_modes:
                        model_worker_batch = _make_decode_batch()
                        model_worker_batch.sampling_info.is_all_greedy = is_greedy
                        if not is_greedy:
                            model_worker_batch.sampling_info.temperatures = np.full(
                                (bs, 1), 1.0, dtype=np.float32
                            )
                            model_worker_batch.sampling_info.top_ks = np.full(
                                (bs,), -1, dtype=np.int32
                            )
                            if use_top_p:
                                model_worker_batch.sampling_info.top_ps = np.full(
                                    (bs,), 0.95, dtype=np.float32
                                )
                        spec_info = _make_non_relay_spec_info(model_worker_batch)
                        if self.spec_relay_buffers is not None:
                            if bs == shapes[0][0]:
                                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                                model_worker_batch.spec_info_padded = spec_info
                                model_worker_batch.speculative_eagle_topk = self.topk
                                model_worker_batch.speculative_num_draft_tokens = (
                                    self.speculative_num_draft_tokens
                                )
                                model_worker_batch.speculative_num_steps = (
                                    self.speculative_num_steps
                                )
                                self.forward_batch_speculative_decode_overlap(model_worker_batch)
                                jax.block_until_ready(self.spec_relay_buffers)
                                model_worker_batch = _make_decode_batch()
                                model_worker_batch.sampling_info.is_all_greedy = is_greedy
                                if not is_greedy:
                                    model_worker_batch.sampling_info.temperatures = np.full(
                                        (bs, 1), 1.0, dtype=np.float32
                                    )
                                    model_worker_batch.sampling_info.top_ks = np.full(
                                        (bs,), -1, dtype=np.int32
                                    )
                                    if use_top_p:
                                        model_worker_batch.sampling_info.top_ps = np.full(
                                            (bs,), 0.95, dtype=np.float32
                                        )
                            spec_info = _make_relay_spec_info(model_worker_batch)
                        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                        model_worker_batch.spec_info_padded = spec_info
                        model_worker_batch.speculative_eagle_topk = self.topk
                        model_worker_batch.speculative_num_draft_tokens = (
                            self.speculative_num_draft_tokens
                        )
                        model_worker_batch.speculative_num_steps = self.speculative_num_steps
                        if self.spec_relay_buffers is not None:
                            self.forward_batch_speculative_decode_overlap(model_worker_batch)
                            jax.block_until_ready(self.spec_relay_buffers)
                        else:
                            self.forward_batch_speculative_generation(model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)
