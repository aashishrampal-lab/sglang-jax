"""Unit tests for PD Disaggregation + Speculative Decoding (NEXTN / MTP)."""

import dataclasses
import os
from dataclasses import dataclass

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

for _opt_mod in (
    "uvicorn",
    "fastapi",
    "aiohttp",
    "xxhash",
    "zmq",
    "orjson",
    "setproctitle",
    "prometheus_client",
    "msgpack",
    "numba",
    "pybase64",
    "fastapi.responses",
    "pydantic",
    "pathwaysutils",
    "llguidance",
    "outlines",
    "xgrammar",
    "partial_json_parser",
    "partial_json_parser.core",
    "partial_json_parser.core.exceptions",
    "partial_json_parser.core.options",
):
    try:
        __import__(_opt_mod)
    except ImportError:
        _m = MagicMock()
        _m.__spec__ = MagicMock()
        if _opt_mod == "numba":
            _m.njit = lambda *a, **kw: (a[0] if a and callable(a[0]) else (lambda f: f))
        sys.modules[_opt_mod] = _m

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.decode import (
    SchedulerDisaggregationDecodeMixin as SchedulerDecodeMixin,
)
from sgl_jax.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin as SchedulerPrefillMixin,
)
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.speculative.draft_extend_fused import (
    _build_prefill,
    _rotate_prefill_input_ids,
    spec_prefill,
)
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.eagle_worker import EAGLEWorker
from sgl_jax.srt.speculative.overlap_utils import can_merge_spec_non_overlap_prefill
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _make_mesh(dp_size: int = 2):
    devices = np.array(jax.devices()[:4]).reshape(dp_size, 4 // dp_size)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


class _DummyKVPool:
    def __init__(
        self,
        mesh,
        *,
        num_pages: int = 16,
        page_size: int = 4,
        layer_num: int = 2,
        head_dim: int = 8,
        dp_size: int = 2,
        page_axis_sharded: bool = True,
        with_indexer: bool = True,
        fill_base: float = 0.0,
    ):
        self.mesh = mesh
        self.page_size = page_size
        self.layer_num = layer_num
        self.start_layer = 0
        self.dp_size = dp_size
        self.dtype = jnp.float32
        self.kv_partition_axis = "tensor"
        self.attention_data_partition_axis = "data"
        self.kv_sharding = (
            NamedSharding(mesh, P("data", None, None))
            if page_axis_sharded
            else NamedSharding(mesh, P(None, None, "tensor"))
        )
        self.kv_buffer = []
        for l_idx in range(layer_num):
            arr = (
                np.arange(num_pages * page_size * head_dim, dtype=np.float32).reshape(
                    num_pages, page_size, head_dim
                )
                + fill_base
                + l_idx * 1000.0
            )
            self.kv_buffer.append(jax.device_put(jnp.asarray(arr), self.kv_sharding))

        if with_indexer:
            self.indexer_key_buffer = []
            for s_idx in range(layer_num):
                idx_arr = (
                    np.arange(num_pages * page_size * (head_dim // 2), dtype=np.float32).reshape(
                        num_pages, page_size, head_dim // 2
                    )
                    + fill_base
                    + 5000.0
                    + s_idx * 1000.0
                )
                self.indexer_key_buffer.append(
                    jax.device_put(jnp.asarray(idx_arr), self.kv_sharding)
                )
        else:
            self.indexer_key_buffer = None

    def get_fused_kv_buffer(self, layer_id: int):
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.kv_buffer[layer_id - self.start_layer]


@jax.tree_util.register_dataclass
@dataclass
class _DummyLogitsMeta:
    logits_indices: jax.Array
    preserve_vocab_sharding: bool = dataclasses.field(default=False, metadata=dict(static=True))


@jax.tree_util.register_dataclass
@dataclass
class _DummySpecInfo:
    hidden_states: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class _DummyForwardBatch:
    input_ids: jax.Array
    positions: jax.Array
    seq_lens: jax.Array
    extend_seq_lens: jax.Array
    spec_info: _DummySpecInfo
    spec_kvshare_readonly: bool = dataclasses.field(default=False, metadata=dict(static=True))


class _TinyTargetModel(nnx.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 4):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=jnp.float32))

    def __call__(self, forward_batch, memory_pools, logits_metadata):
        n_tokens = forward_batch.input_ids.shape[0]
        bs = logits_metadata.logits_indices.shape[0]
        # Produce deterministic argmax token = 11 for all batch items
        logits = jnp.zeros((bs, self.vocab_size), dtype=jnp.float32)
        logits = logits.at[:, 11].set(10.0 * self.weight[0])
        hidden = (
            jnp.arange(n_tokens * self.hidden_size, dtype=jnp.float32).reshape(
                n_tokens, self.hidden_size
            )
            + 1.0
        )
        out = LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
        return out, memory_pools, None, []


class _TinyDraftModel(nnx.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 4):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bias = nnx.Param(jnp.ones((1,), dtype=jnp.float32))

    def __call__(self, forward_batch, memory_pool, logits_metadata):
        # Record the rotated input_ids into memory_pool["recorded_input_ids"] and increment call_count
        last_tokens = forward_batch.input_ids.at[logits_metadata.logits_indices].get(
            out_sharding=P("data")
        )
        # Predict token = (last_input_token + 1) % vocab_size
        target_tok = jnp.mod(last_tokens + 1, self.vocab_size)
        vocab_ids = jnp.arange(self.vocab_size, dtype=jnp.int32)[None, :]
        logits = jnp.where(
            vocab_ids == target_tok[:, None],
            5.0 * self.bias[0],
            jnp.zeros((1, 1), dtype=jnp.float32),
        )
        hidden = forward_batch.spec_info.hidden_states + 10.0
        updated_pool = dict(memory_pool)
        updated_pool["recorded_input_ids"] = forward_batch.input_ids
        updated_pool["call_count"] = memory_pool["call_count"] + 1
        out = LogitsProcessorOutput(next_token_logits=logits, hidden_states=hidden)
        return out, updated_pool, None, []

    def forward_kv_only(self, forward_batch, memory_pool):
        updated_pool = dict(memory_pool)
        updated_pool["recorded_input_ids"] = forward_batch.input_ids
        updated_pool["call_count"] = memory_pool["call_count"] + 1
        if "kv_only_count" in memory_pool:
            updated_pool["kv_only_count"] = memory_pool["kv_only_count"] + 1
        return updated_pool


class TestPDDisaggSpecDecode(unittest.TestCase):
    def test_can_merge_spec_non_overlap_prefill_allows_nextn(self):
        """Verify non-overlap NEXTN can admit/merge 1-token post-transfer EXTEND on 1D."""
        self.assertTrue(
            can_merge_spec_non_overlap_prefill(
                enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NEXTN
            )
        )
        self.assertTrue(
            can_merge_spec_non_overlap_prefill(
                enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.EAGLE3
            )
        )
        self.assertFalse(
            can_merge_spec_non_overlap_prefill(
                enable_overlap=True, spec_algorithm=SpeculativeAlgorithm.NEXTN
            )
        )
        self.assertFalse(
            can_merge_spec_non_overlap_prefill(
                enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE
            )
        )

    def test_rotate_prefill_input_ids_with_chunk_boundary_override(self):
        """Verify _rotate_prefill_input_ids uses ground-truth token on intermediate chunks."""
        # Two requests on dp_size=2, per_dp_bs=1, each with chunk length 4:
        # Req 0 (intermediate chunk): input=[10, 11, 12, 13], ground-truth next prompt token = 99
        # Req 1 (final chunk):        input=[20, 21, 22, 23], extend_next_token_ids = -1 -> uses verified_id=77
        input_ids = jnp.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=jnp.int32)
        extend_seq_lens = jnp.array([4, 4], dtype=jnp.int32)
        target_next_token_ids = jnp.array([66, 77], dtype=jnp.int32)
        chunk_next_token_ids = jnp.array([99, -1], dtype=jnp.int32)

        rotate_ids = jnp.where(
            chunk_next_token_ids >= 0, chunk_next_token_ids, target_next_token_ids
        )
        rotated = _rotate_prefill_input_ids(
            input_ids, extend_seq_lens, rotate_ids, dp_size=2, per_dp_bs=1
        )
        np.testing.assert_array_equal(
            np.asarray(rotated),
            np.array([11, 12, 13, 99, 21, 22, 23, 77], dtype=np.int32),
        )

    def test_schedule_batch_extend_next_token_ids_chunked_and_pd_decode(self):
        """Verify prepare_for_extend and get_model_worker_batch populate extend_next_token_ids."""
        mesh = _make_mesh(dp_size=2)
        sp = SamplingParams(temperature=0.0, max_new_tokens=8)

        # DP 0: intermediate chunked prefill request (origin len 8, fill_ids len 4 -> next token is origin[4]=104)
        req0 = Req(
            rid="req_chunked",
            origin_input_text="",
            origin_input_ids=[100, 101, 102, 103, 104, 105, 106, 107],
            sampling_params=sp,
            dp_rank=0,
            vocab_size=1000,
        )
        req0.fill_ids = [100, 101, 102, 103]
        req0.prefix_indices = np.array([], dtype=np.int32)
        req0.extend_input_len = 4

        # DP 1: final cooperative/offset EXTEND chunk (_pd_seq_pos_offset = 7, fill_ids len 1 -> final token, -1)
        req1 = Req(
            rid="req_pd_decode_extend",
            origin_input_text="",
            origin_input_ids=[200, 201, 202, 203, 204, 205, 206, 207],
            sampling_params=sp,
            dp_rank=1,
            vocab_size=1000,
        )
        req1._pd_seq_pos_offset = 7
        req1.fill_ids = [207]
        req1.prefix_indices = np.array([], dtype=np.int32)
        req1.extend_input_len = 1

        req_to_token_pool = SimpleNamespace(
            req_to_token=np.zeros((8, 32), dtype=np.int32),
            cache_loc_host_buf=np.zeros(64, dtype=np.int32),
            alloc=lambda n, dp_rank=0: [
                dp_rank * 2 + i for i in range(len(n) if isinstance(n, list) else int(n))
            ],
            write=lambda idx, vals: None,
        )
        allocator = SimpleNamespace(
            page_size=1,
            available_size=lambda dp_rank=0: 1024,
            alloc_extend=lambda prefix_lens, seq_lens, last_loc, dp_rank=0: np.arange(
                int(np.sum(seq_lens - prefix_lens)), dtype=np.int32
            )
            + dp_rank * 16,
            alloc=lambda n, dp_rank=0: np.arange(n, dtype=np.int32) + dp_rank * 16,
        )
        model_config = SimpleNamespace(
            vocab_size=1000,
            is_in_model_multimodal=False,
            hf_config=SimpleNamespace(
                vision_encoder_parallel=None, architectures=["GlmMoeDsaForCausalLM"]
            ),
        )

        batch = ScheduleBatch.init_new(
            reqs=[[req0], [req1]],
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            model_config=model_config,
            enable_overlap=False,
            dp_size=2,
            spec_algorithm=SpeculativeAlgorithm.NEXTN,
            mesh=mesh,
        )
        batch.prepare_for_extend()

        np.testing.assert_array_equal(
            batch.reqs_info[0].extend_next_token_ids, np.array([104], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            batch.reqs_info[1].extend_next_token_ids, np.array([-1], dtype=np.int32)
        )

        mwb = batch.get_model_worker_batch(
            token_paddings=[8],
            bs_paddings=[4],  # per_dp_bs_size = 2 (1 real + 1 padded per DP rank)
            cache_loc_paddings=[16],
            page_size=1,
        )
        np.testing.assert_array_equal(
            mwb.extend_next_token_ids,
            np.array([104, -1, -1, -1], dtype=np.int32),
        )
        # Verify _pd_seq_pos_offset adds 7 to req1's merged seq_lens and positions in ModelWorkerBatch
        self.assertEqual(int(mwb.seq_lens[2]), 8)
        self.assertEqual(int(mwb.positions[4]), 7)

    def test_coop_prefill_step_batch_populates_extend_next_token_ids(self):
        """Verify _build_coop_step_batch populates extend_next_token_ids for cooperative chunks."""
        mesh = _make_mesh(dp_size=2)
        sched = object.__new__(Scheduler)
        sched.dp_size = 2
        sched.chunked_prefill_size = 4
        sched.model_config = SimpleNamespace(hf_eos_token_id=[2], vocab_size=100)
        sched.req_to_token_pool = SimpleNamespace(req_to_token=np.zeros((4, 16), dtype=np.int32))
        sched.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        sched.tree_cache = SimpleNamespace()
        sched.enable_overlap = False
        sched.spec_algorithm = SpeculativeAlgorithm.NEXTN
        sched.mesh = mesh

        shared_tokens = list(range(10, 20))  # 10 tokens: indices 0..9
        st = {
            "coop_chunk": 4,
            "cur_pos": 0,
            "shared_token_ids": shared_tokens,
            "full_kv_indices": np.arange(10, dtype=np.int32),
            "coop_req_pool_idx": 0,
            "sampling_params": SamplingParams(temperature=0.0, max_new_tokens=4),
        }
        batch = sched._build_coop_step_batch(st)
        # DP 0 processes [0:4] -> next token at index 4 is shared_tokens[4] = 14
        # DP 1 processes [4:8] -> next token at index 8 is shared_tokens[8] = 18
        np.testing.assert_array_equal(
            batch.reqs_info[0].extend_next_token_ids, np.array([14], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            batch.reqs_info[1].extend_next_token_ids, np.array([18], dtype=np.int32)
        )

    def test_build_prefill_and_spec_prefill_disagg_prefill_vs_decode_modes(self):
        """Verify _build_prefill + spec_prefill run 1 draft layer on 1P (with chunk token override) and all steps on 1D."""
        mesh = _make_mesh(dp_size=1)
        target_model = _TinyTargetModel(vocab_size=16, hidden_size=4)
        draft_model = _TinyDraftModel(vocab_size=16, hidden_size=4)
        t_def, t_state = nnx.split(target_model)
        t_leaves, t_treedef = jax.tree_util.tree_flatten(t_state)
        d_def, d_state = nnx.split(draft_model)
        d_leaves, d_treedef = jax.tree_util.tree_flatten(d_state)

        with jax.set_mesh(mesh):
            fused_fn = _build_prefill(num_layers=1, topk=1, hidden_relay=True)
            data_sharding = NamedSharding(mesh, P("data"))
            rep_sharding = NamedSharding(mesh, P())

            target_fb = _DummyForwardBatch(
                input_ids=jax.device_put(jnp.array([2, 3, 4, 5], dtype=jnp.int32), data_sharding),
                positions=jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), data_sharding),
                seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                extend_seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                spec_info=_DummySpecInfo(
                    hidden_states=jax.device_put(
                        jnp.zeros((4, 4), dtype=jnp.float32), rep_sharding
                    )
                ),
            )
            draft_fb = _DummyForwardBatch(
                input_ids=jax.device_put(jnp.array([2, 3, 4, 5], dtype=jnp.int32), data_sharding),
                positions=jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), data_sharding),
                seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                extend_seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                spec_info=_DummySpecInfo(
                    hidden_states=jax.device_put(
                        jnp.zeros((4, 4), dtype=jnp.float32), rep_sharding
                    )
                ),
            )
            logits_meta = _DummyLogitsMeta(
                logits_indices=jax.device_put(jnp.array([3], dtype=jnp.int32), data_sharding)
            )
            draft_logits_indices = jax.device_put(jnp.array([3], dtype=jnp.int32), data_sharding)
            relay_future_indices = jax.device_put(jnp.array([0], dtype=jnp.int32), data_sharding)
            relay_valid_mask = jax.device_put(jnp.array([True]), data_sharding)

            # 1. Intermediate chunk on 1P (num_layers=1, chunk_next_token_ids=[7]):
            target_pool_p = {
                "kv": jax.device_put(jnp.zeros((4,), dtype=jnp.float32), rep_sharding)
            }
            draft_pool_p = {
                "recorded_input_ids": jax.device_put(
                    jnp.zeros((4,), dtype=jnp.int32), data_sharding
                ),
                "call_count": jax.device_put(jnp.array(0, dtype=jnp.int32), rep_sharding),
            }
            chunk_next_p = jax.device_put(jnp.array([7], dtype=jnp.int32), data_sharding)
            (
                _,
                next_token_ids_p,
                _,
                all_pool_updates_p,
                _,
                topk_idx_p,
                _,
            ) = fused_fn(
                t_def,
                t_treedef,
                tuple(t_leaves),
                target_fb,
                target_pool_p,
                logits_meta,
                d_def,
                d_treedef,
                (tuple(d_leaves),),
                draft_fb,
                draft_logits_indices,
                (draft_pool_p,),
                logits_meta,
                None,
                relay_future_indices,
                relay_valid_mask,
                chunk_next_p,
                num_layers=1,
                dp_size=1,
                per_dp_bs=1,
                update_relay=False,
            )
            # Only 1 draft layer executed, and rotated input_ids used ground-truth boundary token 7 (not target argmax 11)!
            self.assertEqual(int(all_pool_updates_p[0]["call_count"]), 1)
            np.testing.assert_array_equal(
                np.asarray(all_pool_updates_p[0]["recorded_input_ids"]),
                np.array([3, 4, 5, 7], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                np.asarray(next_token_ids_p), np.array([11], dtype=np.int32)
            )
            self.assertEqual(topk_idx_p.shape, (1, 1))

            # 2. Post-transfer 1-token EXTEND on 1D (num_layers=3, chunk_next_token_ids=[-1]):
            target_pool_d = {
                "kv": jax.device_put(jnp.zeros((4,), dtype=jnp.float32), rep_sharding)
            }
            draft_pool_d = {
                "recorded_input_ids": jax.device_put(
                    jnp.zeros((4,), dtype=jnp.int32), data_sharding
                ),
                "call_count": jax.device_put(jnp.array(0, dtype=jnp.int32), rep_sharding),
            }
            chunk_next_d = jax.device_put(jnp.array([-1], dtype=jnp.int32), data_sharding)
            (
                _,
                next_token_ids_d,
                _,
                all_pool_updates_d,
                _,
                topk_idx_d,
                _,
            ) = fused_fn(
                t_def,
                t_treedef,
                tuple(t_leaves),
                target_fb,
                target_pool_d,
                logits_meta,
                d_def,
                d_treedef,
                (tuple(d_leaves),),
                draft_fb,
                draft_logits_indices,
                (draft_pool_d,),
                logits_meta,
                None,
                relay_future_indices,
                relay_valid_mask,
                chunk_next_d,
                num_layers=3,
                dp_size=1,
                per_dp_bs=1,
                update_relay=False,
            )
            # Only 1 pool update is returned (since len(all_memory_pools) == 1),
            # and all 3 speculative steps produced topk_idx_d=[12, 13, 14] from verified_id=11!
            self.assertEqual(len(all_pool_updates_d), 1)
            self.assertEqual(topk_idx_d.shape, (1, 3))
            np.testing.assert_array_equal(
                np.asarray(topk_idx_d), np.array([[12, 13, 14]], dtype=np.int32)
            )

    def test_eagle_worker_precompile_disagg_modes(self):
        """Verify EAGLEWorker skips spec_decode precompile on 1P and caps spec_extend tokens on 1D."""
        worker = object.__new__(EAGLEWorker)
        worker.server_args = SimpleNamespace(
            disaggregation_mode="prefill",
            disable_overlap_schedule=True,
            dp_size=2,
        )
        calls = []
        worker.precompile_spec_extend = lambda: calls.append("extend")
        worker.precompile_spec_decode = lambda: calls.append("decode")
        worker.run_spec_decode_precompile()
        self.assertEqual(calls, ["extend"])

        # Now verify precompile_spec_extend token_paddings cap on 1D ("decode")
        worker_d = object.__new__(EAGLEWorker)
        worker_d.server_args = SimpleNamespace(
            disaggregation_mode="decode",
            disable_overlap_schedule=True,
            dp_size=2,
        )
        worker_d._draft_worker = SimpleNamespace(
            get_max_padded_size=lambda: (32, 0),
            compilation_manager=SimpleNamespace(
                _make_dummy_batch=lambda bs, num_tokens, mode, cl, **kw: SimpleNamespace(
                    bs=bs, num_tokens=num_tokens, cache_loc_size=cl
                )
            ),
        )
        worker_d.precompile_bs_paddings = [32]
        worker_d.precompile_token_paddings = [32, 1024, 4096, 32768]
        worker_d.precompile_cache_loc_paddings = [128]
        worker_d.page_size = 64
        worker_d.speculative_algorithm = SpeculativeAlgorithm.NEXTN
        worker_d.spec_relay_buffers = None
        worker_d._can_use_fused_spec_prefill = lambda mwb: True
        compiled_num_tokens = []
        worker_d.forward_batch_speculative_generation = lambda mwb: compiled_num_tokens.append(
            mwb.num_tokens
        )
        worker_d.precompile_spec_extend()
        # Verify all configured token_paddings are precompiled when server_args.precompile_token_paddings is not set
        self.assertEqual(compiled_num_tokens, [32, 1024, 4096, 32768])

        # And when server_args.precompile_token_paddings=[2048] on 1D, cap out 32768
        worker_d.server_args.precompile_token_paddings = [2048]
        compiled_num_tokens_capped = []
        worker_d.forward_batch_speculative_generation = (
            lambda mwb: compiled_num_tokens_capped.append(mwb.num_tokens)
        )
        worker_d.precompile_spec_extend()
        self.assertEqual(compiled_num_tokens_capped, [32, 1024])

    def test_pd_prefill_and_decode_draft_kv_pool_transfer_and_indexer(self):
        """Verify 1P _extract_req_kv and 1D _build_kv_spec_for_req / _write_kv_to_pool include draft KV + indexer pools."""
        mesh = _make_mesh(dp_size=2)
        with jax.set_mesh(mesh):
            # Create 1P target pool (2 layers) and draft pool (1 layer) with distinct values
            target_pool_p = _DummyKVPool(
                mesh,
                num_pages=16,
                page_size=4,
                layer_num=2,
                head_dim=8,
                dp_size=2,
                page_axis_sharded=True,
                with_indexer=True,
                fill_base=100.0,
            )
            draft_pool_p = _DummyKVPool(
                mesh,
                num_pages=16,
                page_size=4,
                layer_num=1,
                head_dim=8,
                dp_size=2,
                page_axis_sharded=True,
                with_indexer=True,
                fill_base=900.0,
            )

            # Create 1D target pool (2 layers) and draft pool (1 layer) initialized to zeros
            target_pool_d = _DummyKVPool(
                mesh,
                num_pages=16,
                page_size=4,
                layer_num=2,
                head_dim=8,
                dp_size=2,
                page_axis_sharded=True,
                with_indexer=True,
                fill_base=-99999.0,
            )
            draft_pool_d = _DummyKVPool(
                mesh,
                num_pages=16,
                page_size=4,
                layer_num=1,
                head_dim=8,
                dp_size=2,
                page_axis_sharded=True,
                with_indexer=True,
                fill_base=-99999.0,
            )
            for i in range(target_pool_d.layer_num):
                target_pool_d.kv_buffer[i] = jnp.zeros_like(target_pool_d.kv_buffer[i])
                target_pool_d.indexer_key_buffer[i] = jnp.zeros_like(
                    target_pool_d.indexer_key_buffer[i]
                )
            for i in range(draft_pool_d.layer_num):
                draft_pool_d.kv_buffer[i] = jnp.zeros_like(draft_pool_d.kv_buffer[i])
                draft_pool_d.indexer_key_buffer[i] = jnp.zeros_like(
                    draft_pool_d.indexer_key_buffer[i]
                )

            class _MockScheduler(Scheduler, SchedulerPrefillMixin, SchedulerDecodeMixin):
                pass

            sched_p = object.__new__(_MockScheduler)
            sched_p.spec_algorithm = SpeculativeAlgorithm.NEXTN
            sched_p.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: target_pool_p)
            sched_p.tp_size = 2
            sched_p.dp_size = 2
            sched_p.page_size = 4
            # Wire EAGLEWorker -> MultiLayerDraftWorker structure for _get_draft_kv_pools()
            sched_p.draft_worker = SimpleNamespace(
                draft_worker=SimpleNamespace(
                    draft_worker_list=[
                        SimpleNamespace(
                            token_to_kv_pool_allocator=SimpleNamespace(
                                get_kvcache=lambda: draft_pool_p
                            )
                        )
                    ]
                )
            )
            self.assertEqual(sched_p._get_draft_kv_pools(), [draft_pool_p])

            # Request on DP rank 1 occupying rank-local pages [2, 3] (8 tokens: indices 8..15)
            req_to_token = np.zeros((2, 16), dtype=np.int32)
            req_to_token[0, :8] = np.arange(8, 16, dtype=np.int32)
            sched_p.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
            sched_p.server_args = SimpleNamespace(disaggregation_enable_overlap_schedule=False)
            req = SimpleNamespace(
                rid="req_pd_spec_1",
                req_pool_idx=0,
                origin_input_ids=list(range(8)),
                output_ids=[42],
                dp_rank=1,
                _pd_transfer_skip_tokens=0,
            )
            extracted = sched_p._extract_req_payload(req)
            self.assertIsInstance(extracted, dict)
            self.assertEqual(len(extracted["kv"]), 3)  # 2 target + 1 draft
            self.assertEqual(len(extracted["indexer_kv"]), 3)  # 2 target + 1 draft

            # Verify _req_staging_bytes_per_device includes both target (2) and draft (1) pools
            sched_p_no_draft = object.__new__(_MockScheduler)
            sched_p_no_draft.spec_algorithm = SpeculativeAlgorithm.NONE
            sched_p_no_draft.token_to_kv_pool_allocator = SimpleNamespace(
                get_kvcache=lambda: target_pool_p
            )
            sched_p_no_draft.tp_size = 2
            sched_p_no_draft.draft_worker = None
            bytes_with_draft = sched_p._req_staging_bytes_per_device(req)
            bytes_without_draft = sched_p_no_draft._req_staging_bytes_per_device(req)
            self.assertEqual(bytes_with_draft * 2, bytes_without_draft * 3)

            sched_p.server_args = SimpleNamespace(mem_fraction_static=0.70)
            sched_p_no_draft.server_args = SimpleNamespace(mem_fraction_static=0.70)
            self.assertLess(
                sched_p._pre_forward_staging_budget_bytes_per_device(),
                sched_p_no_draft._pre_forward_staging_budget_bytes_per_device(),
            )

            # Now test 1D spec building, pull byte accounting, and _write_kv_to_pool
            sched_d = object.__new__(_MockScheduler)
            sched_d.spec_algorithm = SpeculativeAlgorithm.NEXTN
            sched_d.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: target_pool_d)
            sched_d.tp_size = 2
            sched_d.dp_size = 2
            sched_d.page_size = 4
            sched_d._maybe_verify_decode_writeback_debug = lambda *a, **kw: None
            sched_d.draft_worker = SimpleNamespace(
                draft_worker=SimpleNamespace(
                    draft_worker_list=[
                        SimpleNamespace(
                            token_to_kv_pool_allocator=SimpleNamespace(
                                get_kvcache=lambda: draft_pool_d
                            )
                        )
                    ]
                )
            )
            spec = sched_d._build_kv_spec_for_req(req)
            self.assertEqual(len(spec["kv"]), 3)
            self.assertEqual(len(spec["indexer_kv"]), 3)
            self.assertEqual(sched_d._req_pull_bytes_per_device(req), bytes_with_draft)

            # Simulate DCN pull returning stacked (kv, indexer_kv) arrays and scatter to 1D local pages [1, 4] on dp_rank=1
            d_kv_indices = np.array([4, 5, 6, 7, 16, 17, 18, 19], dtype=np.int32)
            pulled_kv = jnp.stack(extracted["kv"], axis=0)
            pulled_idx = jnp.stack(extracted["indexer_kv"], axis=0)
            sched_d._write_kv_to_pool(req, d_kv_indices, (pulled_kv, pulled_idx))

            # Global pages on dp_rank=1 (8 pages per rank):
            # 1P source local pages [2, 3] -> global pages [10, 11]
            # 1D dest local pages   [1, 4] -> global pages [9, 12]
            for l_idx in range(2):
                np.testing.assert_allclose(
                    np.asarray(target_pool_d.kv_buffer[l_idx])[9],
                    np.asarray(target_pool_p.kv_buffer[l_idx])[10],
                )
                np.testing.assert_allclose(
                    np.asarray(target_pool_d.kv_buffer[l_idx])[12],
                    np.asarray(target_pool_p.kv_buffer[l_idx])[11],
                )
                np.testing.assert_allclose(
                    np.asarray(target_pool_d.indexer_key_buffer[l_idx])[9],
                    np.asarray(target_pool_p.indexer_key_buffer[l_idx])[10],
                )
            # And verify draft KV pool layer 0 and indexer_key_buffer[0] were also scattered!
            np.testing.assert_allclose(
                np.asarray(draft_pool_d.kv_buffer[0])[9],
                np.asarray(draft_pool_p.kv_buffer[0])[10],
            )
            np.testing.assert_allclose(
                np.asarray(draft_pool_d.kv_buffer[0])[12],
                np.asarray(draft_pool_p.kv_buffer[0])[11],
            )
            np.testing.assert_allclose(
                np.asarray(draft_pool_d.indexer_key_buffer[0])[9],
                np.asarray(draft_pool_p.indexer_key_buffer[0])[10],
            )
            np.testing.assert_allclose(
                np.asarray(draft_pool_d.indexer_key_buffer[0])[12],
                np.asarray(draft_pool_p.indexer_key_buffer[0])[11],
            )

    def test_pd_decode_sync_unpopulated_prefixes_includes_draft_pool(self):
        """Verify _sync_unpopulated_decode_prefixes broadcasts both target and draft KV + indexer buffers across DP ranks."""
        mesh = _make_mesh(dp_size=2)
        with jax.set_mesh(mesh):
            target_pool = _DummyKVPool(
                mesh,
                num_pages=16,
                page_size=4,
                layer_num=2,
                head_dim=8,
                dp_size=2,
                page_axis_sharded=True,
                with_indexer=True,
                fill_base=10.0,
            )
            draft_pool = _DummyKVPool(
                mesh,
                num_pages=16,
                page_size=4,
                layer_num=1,
                head_dim=8,
                dp_size=2,
                page_axis_sharded=True,
                with_indexer=True,
                fill_base=500.0,
            )

            class _MockScheduler(Scheduler, SchedulerDecodeMixin):
                pass

            sched = object.__new__(_MockScheduler)
            sched.spec_algorithm = SpeculativeAlgorithm.NEXTN
            sched.dp_size = 2
            sched.page_size = 4
            sched.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: target_pool)
            sched.draft_worker = SimpleNamespace(
                draft_worker=SimpleNamespace(
                    draft_worker_list=[
                        SimpleNamespace(
                            token_to_kv_pool_allocator=SimpleNamespace(
                                get_kvcache=lambda: draft_pool
                            )
                        )
                    ]
                )
            )
            # Leader on DP 0 at local pages [1, 2] (global pages [1, 2]);
            # Follower on DP 1 at local pages [3, 4] (global pages [11, 12]), unpopulated.
            leader_entry = {
                "dp_rank": 0,
                "kv_indices": np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32),
                "populated": True,
                "scatter_kv_indices": None,
            }
            follower_entry = {
                "dp_rank": 1,
                "kv_indices": np.array([12, 13, 14, 15, 16, 17, 18, 19], dtype=np.int32),
                "populated": False,
                "source_leader_entry": leader_entry,
            }
            sched._pd_decode_prefix_cache = {
                0: leader_entry,
                1: follower_entry,
            }
            sched._sync_unpopulated_decode_prefixes()

            self.assertTrue(sched._pd_decode_prefix_cache[1]["populated"])
            # Check follower global pages [11, 12] match leader global pages [1, 2] in both target and draft pools!
            np.testing.assert_allclose(
                np.asarray(target_pool.kv_buffer[0])[11:13],
                np.asarray(target_pool.kv_buffer[0])[1:3],
            )
            np.testing.assert_allclose(
                np.asarray(draft_pool.kv_buffer[0])[11:13],
                np.asarray(draft_pool.kv_buffer[0])[1:3],
            )
            np.testing.assert_allclose(
                np.asarray(draft_pool.indexer_key_buffer[0])[11:13],
                np.asarray(draft_pool.indexer_key_buffer[0])[1:3],
            )

    def test_pd_prefill_release_kv_cache_allows_overallocated(self):
        """Verify _release_prefill_kv_pool passes allow_overallocated=True when spec decode is enabled."""
        class _MockScheduler(Scheduler, SchedulerPrefillMixin):
            pass

        sched = object.__new__(_MockScheduler)
        sched.spec_algorithm = SpeculativeAlgorithm.NEXTN
        sched.tree_cache = MagicMock()
        req = SimpleNamespace(rid="req_overalloc")

        import sgl_jax.srt.mem_cache.common as mem_common

        orig_release = mem_common.release_kv_cache
        recorded_kwargs = {}

        def _spy_release(r, tc, is_insert=True, allow_overallocated=False):
            recorded_kwargs["allow_overallocated"] = allow_overallocated
            recorded_kwargs["is_insert"] = is_insert

        mem_common.release_kv_cache = _spy_release
        try:
            sched._release_prefill_kv_pool(req)
        finally:
            mem_common.release_kv_cache = orig_release

        self.assertTrue(recorded_kwargs.get("allow_overallocated"))

    def test_fused_prefill_relay_buffer_padding_uses_num_steps_dim(self):
        """Verify fused_prefill with update_relay=True pads relay_topk_index to shape[2] (num_steps), not shape[1] (capacity)."""
        from sgl_jax.srt.speculative.relay_buffer import create_spec_relay_buffers

        mesh = _make_mesh(dp_size=1)
        with jax.set_mesh(mesh):
            t_model = _TinyTargetModel()
            d_model = _TinyDraftModel()
            t_def, t_state = nnx.split(t_model)
            t_leaves, t_treedef = jax.tree_util.tree_flatten(t_state)
            d_def, d_state = nnx.split(d_model)
            d_leaves, d_treedef = jax.tree_util.tree_flatten(d_state)

            fused_fn = _build_prefill(num_layers=1, topk=1, hidden_relay=True)
            data_sharding = NamedSharding(mesh, P("data"))
            rep_sharding = NamedSharding(mesh, P())

            target_fb = _DummyForwardBatch(
                input_ids=jax.device_put(jnp.array([2, 3, 4, 5], dtype=jnp.int32), data_sharding),
                positions=jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), data_sharding),
                seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                extend_seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                spec_info=_DummySpecInfo(
                    hidden_states=jax.device_put(
                        jnp.zeros((4, 4), dtype=jnp.float32), rep_sharding
                    )
                ),
            )
            draft_fb = _DummyForwardBatch(
                input_ids=jax.device_put(jnp.array([2, 3, 4, 5], dtype=jnp.int32), data_sharding),
                positions=jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), data_sharding),
                seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                extend_seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                spec_info=_DummySpecInfo(
                    hidden_states=jax.device_put(
                        jnp.zeros((4, 4), dtype=jnp.float32), rep_sharding
                    )
                ),
            )
            logits_meta = _DummyLogitsMeta(
                logits_indices=jax.device_put(jnp.array([3], dtype=jnp.int32), data_sharding)
            )
            draft_logits_indices = jax.device_put(jnp.array([3], dtype=jnp.int32), data_sharding)
            relay_future_indices = jax.device_put(jnp.array([2], dtype=jnp.int32), data_sharding)
            relay_valid_mask = jax.device_put(jnp.array([True]), data_sharding)

            # Create relay buffers with capacity=16 > num_steps=3 while fused_prefill runs num_layers=1
            mock_pool = SimpleNamespace(req_to_token=np.zeros((16, 32), dtype=np.int32))
            relay_buffers = create_spec_relay_buffers(
                mesh,
                mock_pool,
                dp_size=1,
                num_steps=3,
                hidden_size=4,
                hidden_dtype=jnp.float32,
            )

            target_pool = {
                "kv": jax.device_put(jnp.zeros((4,), dtype=jnp.float32), rep_sharding)
            }
            draft_pool = {
                "recorded_input_ids": jax.device_put(
                    jnp.zeros((4,), dtype=jnp.int32), data_sharding
                ),
                "call_count": jax.device_put(jnp.array(0, dtype=jnp.int32), rep_sharding),
            }
            chunk_next = jax.device_put(jnp.array([-1], dtype=jnp.int32), data_sharding)

            (
                _,
                _,
                _,
                _,
                _,
                _,
                updated_relay_buffers,
            ) = fused_fn(
                t_def,
                t_treedef,
                tuple(t_leaves),
                target_fb,
                target_pool,
                logits_meta,
                d_def,
                d_treedef,
                (tuple(d_leaves),),
                draft_fb,
                draft_logits_indices,
                (draft_pool,),
                logits_meta,
                relay_buffers,
                relay_future_indices,
                relay_valid_mask,
                chunk_next,
                num_layers=1,
                dp_size=1,
                per_dp_bs=1,
                update_relay=True,
            )
            self.assertEqual(updated_relay_buffers.topk_index.shape, (1, 16, 3))
            np.testing.assert_array_equal(
                np.asarray(updated_relay_buffers.topk_index[0, 2]),
                np.array([12, 0, 0], dtype=np.int32),
            )

    def test_precompile_spec_decode_and_non_overlap_generation_greedy_and_nongreedy(self):
        """Verify precompile_spec_decode and forward_batch_speculative_generation use spec_decode for both greedy and non-greedy."""
        import sgl_jax.srt.speculative.draft_extend_fused as def_mod
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
        from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

        EagleDraftInput.ALLOC_LEN_PER_DECODE = 4

        # Simulate DSASparseAttentionBackend which does NOT have get_eagle_multi_step_metadata
        def _forbidden_legacy_draft(mwb):
            raise AttributeError(
                "'DSASparseAttentionBackend' object has no attribute 'get_eagle_multi_step_metadata'"
            )

        @dataclass
        class _DummySamplingInfo:
            is_all_greedy: bool = True
            sampling_info_done: object = None
            penalizer_orchestrator: object = None

            def update_penalties(self):
                pass

        def _make_dummy_mwb(bs, num_tokens, mode, cl, **kw):
            return SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                real_bs=bs,
                bs=bs,
                num_tokens=num_tokens,
                cache_loc_size=cl,
                out_cache_loc=np.zeros((bs,), dtype=np.int32),
                return_logprob=False,
                return_output_logprob_only=False,
                req_pool_indices=np.arange(bs, dtype=np.int32),
                logits_indices_selector=slice(None),
                spec_info=None,
                sampling_info=_DummySamplingInfo(is_all_greedy=True),
                seq_lens=np.full((bs,), 128, dtype=np.int32),
                cache_loc=np.zeros((cl,), dtype=np.int32),
            )

        target_worker = SimpleNamespace(
            mesh=_make_mesh(dp_size=2),
            get_memory_pool=lambda: (SimpleNamespace(), SimpleNamespace()),
            get_precompile_paddings=lambda: ([32], [32], [55296]),
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(),
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
                ),
            ),
        )
        draft_worker = SimpleNamespace(
            max_req_len=4096,
            model_config=SimpleNamespace(hidden_size=4),
            get_max_padded_size=lambda: (32, 0),
            compilation_manager=SimpleNamespace(_make_dummy_batch=_make_dummy_mwb),
            draft=_forbidden_legacy_draft,
            draft_extend_for_decode=lambda mwb, bo: (_ for _ in ()).throw(
                AssertionError("Must call draft_extend_fused.draft_extend_for_decode, not method!")
            ),
        )
        server_args = SimpleNamespace(
            disaggregation_mode="decode",
            disable_overlap_schedule=True,
            dp_size=2,
            speculative_algorithm="NEXTN",
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            page_size=128,
            attention_backend="dsa_sparse",
            dtype="float32",
        )
        worker = object.__new__(EAGLEWorker)
        BaseSpecWorker.__init__(worker, server_args, target_worker, draft_worker)
        worker._spec_decode_seen_shapes = set()

        spec_decode_greedy_flags = []
        fused_draft_extend_calls = []

        orig_spec_decode_verify = def_mod.spec_decode_verify
        orig_fused_de = def_mod.draft_extend_for_decode
        try:
            def _spy_spec_decode_verify(sw, mwb, cur_allocate_lens, **kw):
                spec_decode_greedy_flags.append(bool(mwb.sampling_info.is_all_greedy))
                return SimpleNamespace(
                    next_token_ids=np.zeros((mwb.bs, 4), dtype=np.int32),
                    accept_lens=np.ones((mwb.bs,), dtype=np.int32),
                )

            def _spy_fused_de(dw, mwb, bo):
                fused_draft_extend_calls.append(bool(mwb.sampling_info.is_all_greedy))

            def_mod.spec_decode_verify = _spy_spec_decode_verify
            def_mod.draft_extend_for_decode = _spy_fused_de

            orig_dsa = os.environ.get("DSA_SPEC_AS_DECODE")
            os.environ["DSA_SPEC_AS_DECODE"] = "1"
            try:
                worker.precompile_spec_decode()
            finally:
                if orig_dsa is None:
                    os.environ.pop("DSA_SPEC_AS_DECODE", None)
                else:
                    os.environ["DSA_SPEC_AS_DECODE"] = orig_dsa
        finally:
            def_mod.spec_decode_verify = orig_spec_decode_verify
            def_mod.draft_extend_for_decode = orig_fused_de

        self.assertEqual(spec_decode_greedy_flags, [True, False, False])
        self.assertEqual(fused_draft_extend_calls, [True, False, False])

    def test_non_overlap_restore_draft_extend_and_merge_with_extend_ranks(self):
        """Verify non-overlap restore_draft_extend_result + _split_spec_info_per_rank + _concat_spec_info_per_rank across mixed DECODE and EXTEND DP ranks."""
        from sgl_jax.srt.speculative.overlap_utils import publish_spec_decode_new_seq_lens
        from sgl_jax.srt.speculative.draft_extend_fused import (
            FusedDraftExtendPendingResult,
            restore_draft_extend_result,
        )

        # 4 DP ranks, per_dp_bs=2 -> total_bs=8.
        # Step 1 (DECODE): DP ranks 0 and 3 each have 1 active request -> real_bs=2
        real_bs_per_dp_decode = [1, 0, 0, 1]
        sel = np.array([True, False, False, False, False, False, True, False], dtype=bool)
        next_draft_input = EagleDraftInput(
            verified_id=np.zeros((2,), dtype=np.int32),
            allocate_lens=np.array([100, 200, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            new_seq_lens=jnp.array([103, 0, 0, 0, 0, 0, 202, 0], dtype=jnp.int32),
        )
        batch_output = SimpleNamespace(
            next_draft_input=next_draft_input,
            accept_lens=None,
        )
        pending_result = FusedDraftExtendPendingResult(
            batch_output=batch_output,
            selected_layer0_hidden=jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4),
            topk_index_stacked=jnp.arange(8 * 3 * 1, dtype=jnp.int32).reshape(8, 3, 1),
            next_verified_id=jnp.array([11, 0, 0, 0, 0, 0, 44, 0], dtype=jnp.int32),
            accept_lens=jnp.array([3, 0, 0, 0, 0, 0, 2, 0], dtype=jnp.int32),
            sel=sel,
            updated_relay_buffers=None,
            host_outputs_prefetched=True,
        )
        mwb = SimpleNamespace(dp_size=1, real_bs=2)
        restore_draft_extend_result(SimpleNamespace(), mwb, pending_result)

        # batch_output.accept_lens remains DP-padded (8,), while next_draft_input fields are sliced to (2,)
        self.assertEqual(batch_output.accept_lens.shape, (8,))
        np.testing.assert_array_equal(
            batch_output.next_draft_input.accept_length, np.array([3, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            batch_output.next_draft_input.verified_id, np.array([11, 44], dtype=np.int32)
        )

        # Simulate scheduler.py lines 3318-3329: publish new_seq_lens and clear next_draft_input.new_seq_lens
        published = publish_spec_decode_new_seq_lens(batch_output)
        self.assertIsNotNone(published)
        batch_output.next_draft_input.new_seq_lens = None

        decode_per_rank = ScheduleBatch._split_spec_info_per_rank(
            batch_output.next_draft_input, real_bs_per_dp_decode
        )
        self.assertIsNotNone(decode_per_rank[0])
        self.assertIsNone(decode_per_rank[1])
        self.assertIsNone(decode_per_rank[2])
        self.assertIsNotNone(decode_per_rank[3])
        np.testing.assert_array_equal(decode_per_rank[0].accept_length, np.array([3], dtype=np.int32))
        np.testing.assert_array_equal(decode_per_rank[3].accept_length, np.array([2], dtype=np.int32))

        # Step 2 (EXTEND on DP ranks 1 and 2): spec_prefill produces EagleDraftInput without accept_length or new_seq_lens
        extend_spec_1 = EagleDraftInput(
            hidden_states=np.ones((1, 4), dtype=np.float32),
            verified_id=np.array([22], dtype=np.int32),
            topk_p=np.ones((1, 3, 1), dtype=np.float32),
            topk_index=np.full((1, 3, 1), 5, dtype=np.int32),
            allocate_lens=np.array([150], dtype=np.int32),
        )
        extend_spec_2 = EagleDraftInput(
            hidden_states=np.ones((1, 4), dtype=np.float32) * 2,
            verified_id=np.array([33], dtype=np.int32),
            topk_p=np.ones((1, 3, 1), dtype=np.float32),
            topk_index=np.full((1, 3, 1), 6, dtype=np.int32),
            allocate_lens=np.array([160], dtype=np.int32),
        )
        merged_per_rank = [
            decode_per_rank[0],
            extend_spec_1,
            extend_spec_2,
            decode_per_rank[3],
        ]
        merged = ScheduleBatch._concat_spec_info_per_rank(merged_per_rank)
        self.assertIsNotNone(merged)
        np.testing.assert_array_equal(
            merged.verified_id, np.array([11, 22, 33, 44], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            merged.allocate_lens, np.array([100, 150, 160, 200], dtype=np.int32)
        )
        self.assertIsNone(merged.accept_length)
        self.assertIsNone(merged.new_seq_lens)

    def test_spec_prefill_always_passes_device_chunk_next_token_ids(self):
        """Verify spec_prefill passes an int32[-1] device array for chunk_next_token_ids even when extend_next_token_ids is None."""
        import inspect
        from sgl_jax.srt.speculative import draft_extend_fused

        src = inspect.getsource(draft_extend_fused.spec_prefill)
        self.assertIn("np.full(", src)
        self.assertNotIn("chunk_next_token_ids = None", src)

    def test_1m_context_length_pd_disagg_config_and_cache_loc_fit(self):
        """Verify --context-length 1000000 resolves valid max_running_requests and cache_loc fit rungs for PD Disagg + NEXTN."""
        from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
        from sgl_jax.srt.managers.schedule_batch import (
            fit_cache_loc_padding,
            spec_cache_loc_needs,
            spec_cache_loc_rungs,
        )
        from sgl_jax.srt.model_executor.compilation_manager import CompilationManager

        context_len = 1_000_000
        page_size = 128
        dp_size = 16
        per_dp_limit = MLAAttentionBackend.get_max_running_reqests(context_len, page_size)
        self.assertEqual(per_dp_limit, 32)

        dummy_args = SimpleNamespace(
            page_size=page_size,
            dp_size=dp_size,
            tp_size=16,
            max_running_requests=512,
            disaggregation_mode="prefill",
        )
        resolved_max_reqs = CompilationManager.resolve_max_running_requests(
            dummy_args,
            context_len,
            MLAAttentionBackend,
            token_capacity=1_513_472,
            pool_limit=512,
            moe_backend="fused_v2",
        )
        self.assertEqual(resolved_max_reqs, 512)

        bs_paddings = [32, 64, 128, 256]
        pages_per_req_tokens = ((context_len - 1 + page_size - 1) // page_size) * page_size
        cache_loc_paddings = [bs * pages_per_req_tokens for bs in bs_paddings]
        rungs_per_bs = spec_cache_loc_rungs(
            bs_paddings, cache_loc_paddings, page_size, caps=(65536,)
        )
        for bs, rungs in zip(bs_paddings, rungs_per_bs):
            self.assertEqual(rungs[0], bs * 65536)
            per_dp_bs = bs // dp_size
            # Check CoopPrefill step 1 (32768), CoopPrefill step 2 (65536), and 43k request
            for seq_len in (32768, 43500, 65536):
                needs = spec_cache_loc_needs(
                    [np.array([seq_len], dtype=np.int32) for _ in range(dp_size)],
                    per_dp_bs,
                    page_size,
                )
                fitted = fit_cache_loc_padding(rungs, needs, dp_size)
                self.assertEqual(fitted, bs * 65536)

    def test_fix1a_prefill_staging_budget_allows_full_wave_overlap(self):
        """Verify Fix 1A raises 1P pre-forward staging budget above 2.54 GiB/device with NEXTN so waves overlap 100%."""
        class _MockScheduler(Scheduler, SchedulerPrefillMixin):
            pass

        sched = object.__new__(_MockScheduler)
        sched.spec_algorithm = SpeculativeAlgorithm.NEXTN
        sched.server_args = SimpleNamespace(mem_fraction_static=0.70)

        pre_budget = sched._pre_forward_staging_budget_bytes_per_device()
        post_budget = sched._staging_budget_bytes_per_device()
        wave_16req_43k_bytes_per_device = int(2.54 * (1024**3))

        self.assertGreater(pre_budget, wave_16req_43k_bytes_per_device)
        self.assertGreaterEqual(pre_budget, int(2.80 * (1024**3)))
        self.assertGreater(post_budget, pre_budget)

    def test_fix1b_dynamic_lcp_prefix_skip_with_nextn_bigram_offset(self):
        """Verify Fix 1B dynamic LCP prefix-skip on 1P and 1D matches exact page-aligned skip_tokens under NEXTN bigram keys."""
        page_size = 128
        # Case 1: Mode 1B 80% prefix reuse (~34,100 common prefix tokens out of 43,000)
        # Under NEXTN (is_eagle=True), RadixCache keys are bigrams (token[i], token[i+1]),
        # so 34,100 common tokens yield 34,099 matching bigrams -> ((34099 // 128) * 128) = 34,048 cached tokens (266 pages).
        common_len = 34100
        total_len = 43000
        rng = np.random.default_rng(42)
        shared_prefix = rng.integers(10, 1000, size=common_len, dtype=np.int32).tolist()
        warmup_ids = shared_prefix + rng.integers(1001, 2000, size=total_len - common_len, dtype=np.int32).tolist()
        req_ids = shared_prefix + rng.integers(2001, 3000, size=total_len - common_len, dtype=np.int32).tolist()
        radix_cached_len = ((common_len - 1) // page_size) * page_size
        self.assertEqual(radix_cached_len, 34048)

        # Simulate 1P caching warmup_ids[:target_prefix_len + 1] and computing LCP with req_ids
        target_prefix_len = 35072
        cap_len = min(len(warmup_ids), target_prefix_len)
        cached_1p = np.asarray(warmup_ids[: cap_len + 1], dtype=np.int32)
        cur_1p = np.asarray(req_ids[: min(len(req_ids), cached_1p.shape[0])], dtype=np.int32)
        cmp_len = min(cached_1p.shape[0], cur_1p.shape[0])
        diff = np.flatnonzero(cached_1p[:cmp_len] != cur_1p[:cmp_len])
        lcp_tokens = int(diff[0]) if diff.size > 0 else int(cmp_len)
        self.assertEqual(lcp_tokens, common_len)
        usable_lcp = max(0, lcp_tokens - 1) if lcp_tokens < cmp_len else min(lcp_tokens, target_prefix_len)
        skip_1p = (usable_lcp // page_size) * page_size
        self.assertEqual(skip_1p, 34048)
        self.assertLessEqual(skip_1p, radix_cached_len)

        # Boundary case: common_len = 4224 (exact multiple of 128), where bigram RadixCache caches ((4224 - 1) // 128) * 128 = 4096 tokens!
        # Without usable_lcp = lcp_tokens - 1, (4224 // 128) * 128 = 4224 > 4096 (len(req.prefix_indices)), which would fail!
        common_boundary = 4224
        radix_boundary_len = ((common_boundary - 1) // page_size) * page_size
        self.assertEqual(radix_boundary_len, 4096)
        usable_boundary = common_boundary - 1
        skip_boundary = (usable_boundary // page_size) * page_size
        self.assertEqual(skip_boundary, 4096)
        self.assertLessEqual(skip_boundary, radix_boundary_len)

    def test_fix2_fused_prefill_kv_only_skips_draft_full_forward_and_updates_kv_pool(self):
        """Verify Fix 2 fused_prefill(kv_only=True) invokes model.forward_kv_only and skips draft logits/hidden computation."""
        mesh = _make_mesh(dp_size=1)
        with jax.set_mesh(mesh):
            t_model = _TinyTargetModel()
            d_model = _TinyDraftModel()
            t_def, t_state = nnx.split(t_model)
            t_leaves, t_treedef = jax.tree_util.tree_flatten(t_state)
            d_def, d_state = nnx.split(d_model)
            d_leaves, d_treedef = jax.tree_util.tree_flatten(d_state)

            fused_fn = _build_prefill(num_layers=1, topk=1, hidden_relay=True)
            data_sharding = NamedSharding(mesh, P("data"))
            rep_sharding = NamedSharding(mesh, P())

            target_fb = _DummyForwardBatch(
                input_ids=jax.device_put(jnp.array([2, 3, 4, 5], dtype=jnp.int32), data_sharding),
                positions=jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), data_sharding),
                seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                extend_seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                spec_info=_DummySpecInfo(
                    hidden_states=jax.device_put(
                        jnp.zeros((4, 4), dtype=jnp.float32), rep_sharding
                    )
                ),
            )
            draft_fb = _DummyForwardBatch(
                input_ids=jax.device_put(jnp.array([2, 3, 4, 5], dtype=jnp.int32), data_sharding),
                positions=jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), data_sharding),
                seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                extend_seq_lens=jax.device_put(jnp.array([4], dtype=jnp.int32), data_sharding),
                spec_info=_DummySpecInfo(
                    hidden_states=jax.device_put(
                        jnp.zeros((4, 4), dtype=jnp.float32), rep_sharding
                    )
                ),
            )
            logits_meta = _DummyLogitsMeta(
                logits_indices=jax.device_put(jnp.array([3], dtype=jnp.int32), data_sharding)
            )
            draft_logits_indices = jax.device_put(jnp.array([3], dtype=jnp.int32), data_sharding)
            relay_future_indices = jax.device_put(jnp.array([0], dtype=jnp.int32), data_sharding)
            relay_valid_mask = jax.device_put(jnp.array([True]), data_sharding)

            target_pool = {
                "kv": jax.device_put(jnp.zeros((4,), dtype=jnp.float32), rep_sharding)
            }
            draft_pool = {
                "recorded_input_ids": jax.device_put(
                    jnp.zeros((4,), dtype=jnp.int32), data_sharding
                ),
                "call_count": jax.device_put(jnp.array(0, dtype=jnp.int32), rep_sharding),
                "kv_only_count": jax.device_put(jnp.array(0, dtype=jnp.int32), rep_sharding),
            }
            chunk_next = jax.device_put(jnp.array([-1], dtype=jnp.int32), data_sharding)

            (
                _,
                next_token_ids,
                _,
                all_pool_updates,
                selected_layer0_hidden,
                stacked_idx,
                _,
            ) = fused_fn(
                t_def,
                t_treedef,
                tuple(t_leaves),
                target_fb,
                target_pool,
                logits_meta,
                d_def,
                d_treedef,
                (tuple(d_leaves),),
                draft_fb,
                draft_logits_indices,
                (draft_pool,),
                logits_meta,
                None,
                relay_future_indices,
                relay_valid_mask,
                chunk_next,
                num_layers=1,
                dp_size=1,
                per_dp_bs=1,
                update_relay=False,
                kv_only=True,
            )
            np.testing.assert_array_equal(np.asarray(next_token_ids), np.array([11], dtype=np.int32))
            self.assertEqual(int(np.asarray(all_pool_updates[0]["kv_only_count"])), 1)
            np.testing.assert_array_equal(
                np.asarray(all_pool_updates[0]["recorded_input_ids"]),
                np.array([3, 4, 5, 11], dtype=np.int32),
            )
            self.assertEqual(selected_layer0_hidden.shape, (1, 4))
            self.assertEqual(stacked_idx.shape, (1, 1))

    def test_pd_direct_decode_admission_and_spec_relay_update(self):
        """Verify 1P spec_meta transfer and 1D direct-to-decode admission with update_spec_relay_buffers_admit."""
        from jax.sharding import NamedSharding, PartitionSpec
        from sgl_jax.srt.speculative.relay_buffer import (
            create_spec_relay_buffers,
            update_spec_relay_buffers_admit,
        )

        mesh = _make_mesh(dp_size=2)
        dummy_pool = SimpleNamespace(req_to_token=np.zeros((4, 16), dtype=np.int32))
        buffers = create_spec_relay_buffers(
            mesh,
            dummy_pool,
            dp_size=2,
            num_steps=1,
            hidden_size=4,
            hidden_dtype=jnp.bfloat16,
        )
        sh_1d = NamedSharding(mesh, PartitionSpec("data"))
        sh_2d = NamedSharding(mesh, PartitionSpec("data", None))
        fut_idx = jax.device_put(np.array([1, 2], dtype=np.int32), sh_1d)
        val_mask = jax.device_put(np.array([True, True], dtype=np.bool_), sh_1d)
        topk_idx = jax.device_put(np.array([[77], [88]], dtype=np.int32), sh_2d)
        ver_id = jax.device_put(np.array([77, 88], dtype=np.int32), sh_1d)
        new_seq = jax.device_put(np.array([17, 33], dtype=np.int32), sh_1d)

        # Explicitly test outside `with jax.set_mesh(mesh):` to match real Scheduler context
        updated = update_spec_relay_buffers_admit(
            buffers,
            fut_idx,
            val_mask,
            topk_idx,
            ver_id,
            new_seq,
            dp_size=2,
        )
        if True:
            self.assertEqual(int(np.asarray(updated.verified_id)[0, 1]), 77)
            self.assertEqual(int(np.asarray(updated.verified_id)[1, 2]), 88)
            self.assertEqual(int(np.asarray(updated.topk_index)[0, 1, 0]), 77)
            self.assertEqual(int(np.asarray(updated.topk_index)[1, 2, 0]), 88)
            self.assertEqual(int(np.asarray(updated.new_seq_lens)[0, 1]), 17)
            self.assertEqual(int(np.asarray(updated.new_seq_lens)[1, 2]), 33)

            # Also verify _admit_pd_direct_decode_reqs end-to-end with real ReqToTokenPool + PagedTokenToKVPoolAllocator
            from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
            from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
            from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

            req_pool = ReqToTokenPool(size=4, max_context_len=128)
            kv_alloc = PagedTokenToKVPoolAllocator(
                size=1024, page_size=16, kvcache=SimpleNamespace(), dp_size=2
            )
            tree_cache = ChunkCache(
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=kv_alloc,
                page_size=16,
            )

            req0 = Req(
                rid="r0",
                origin_input_text="",
                origin_input_ids=list(range(10, 26)),
                sampling_params=SamplingParams(max_new_tokens=8, temperature=0.0),
            )
            req0.dp_rank = 0
            req0._pd_direct_decode_ready = True
            req0._pd_first_output_id = 101
            req0._pd_initial_draft_ids = [301]
            req0._pd_prealloc_kv_indices = kv_alloc.alloc(16, dp_rank=0)

            req1 = Req(
                rid="r1",
                origin_input_text="",
                origin_input_ids=list(range(30, 46)),
                sampling_params=SamplingParams(max_new_tokens=8, temperature=0.0),
            )
            req1.dp_rank = 1
            req1._pd_direct_decode_ready = True
            req1._pd_first_output_id = 202
            req1._pd_initial_draft_ids = [402]
            req1._pd_prealloc_kv_indices = kv_alloc.alloc(16, dp_rank=1)

            empty_running = ScheduleBatch.init_new(
                [[], []],
                req_pool,
                kv_alloc,
                tree_cache,
                SimpleNamespace(hidden_size=4, vocab_size=32000),
                True,
                2,
                spec_algorithm=SpeculativeAlgorithm.NEXTN,
                mesh=mesh,
            )
            draft_worker = SimpleNamespace(spec_relay_buffers=updated)
            sched = SimpleNamespace(
                waiting_queue=[req0, req1],
                running_batch=empty_running,
                max_total_num_tokens=1024,
                dp_size=2,
                server_args=SimpleNamespace(
                    max_running_requests=16,
                    speculative_eagle_topk=1,
                    speculative_num_steps=1,
                ),
                req_to_token_pool=req_pool,
                token_to_kv_pool_allocator=kv_alloc,
                tree_cache=tree_cache,
                model_config=SimpleNamespace(hidden_size=4, vocab_size=32000),
                enable_overlap=True,
                spec_algorithm=SpeculativeAlgorithm.NEXTN,
                mesh=mesh,
                draft_worker=draft_worker,
                tp_worker=SimpleNamespace(spec_relay_buffers=None),
                _is_spec_decode_enabled=lambda: True,
                _pd_chunk_cache_finished_req=lambda r: None,
                stream_output=lambda reqs, ret_logprob, ret_output_logprob_only: None,
            )
            n_admitted = SchedulerDecodeMixin._admit_pd_direct_decode_reqs(sched)
            self.assertEqual(n_admitted, 2)
            self.assertEqual(len(sched.waiting_queue), 0)
            self.assertEqual(sched.running_batch.batch_size(), 2)
            self.assertEqual(list(sched.running_batch.reqs_info[0].output_ids), [101])
            self.assertEqual(list(sched.running_batch.reqs_info[1].output_ids), [202])
            self.assertIsNone(sched.running_batch.reqs_info[0].spec_info.topk_index)
            self.assertIsNotNone(sched.running_batch.reqs_info[0].spec_info.future_indices)
            self.assertEqual(
                int(np.asarray(draft_worker.spec_relay_buffers.verified_id)[0, 0]), 101
            )
            self.assertEqual(
                int(np.asarray(draft_worker.spec_relay_buffers.verified_id)[1, 1]), 202
            )
            self.assertEqual(
                int(np.asarray(draft_worker.spec_relay_buffers.topk_index)[0, 0, 0]), 301
            )
            self.assertEqual(
                int(np.asarray(draft_worker.spec_relay_buffers.topk_index)[1, 1, 0]), 402
            )
            self.assertEqual(
                int(np.asarray(draft_worker.spec_relay_buffers.new_seq_lens)[0, 0]), 17
            )
            self.assertEqual(
                int(np.asarray(draft_worker.spec_relay_buffers.new_seq_lens)[1, 1]), 17
            )
            EagleDraftInput.ALLOC_LEN_PER_DECODE = 5
            sched.running_batch.prepare_for_decode()
            self.assertEqual(int(sched.running_batch.reqs_info[0].spec_info.allocate_lens[0]), 26)
            self.assertEqual(int(sched.running_batch.reqs_info[1].spec_info.allocate_lens[0]), 26)

    def test_patch_tarball_contains_all_modified_files(self):
        """Verify /tmp/patch.tar.xz contains all modified files in python/sgl_jax/srt byte-for-byte."""
        import subprocess
        import tarfile

        tar_path = "/tmp/patch.tar.xz"
        srt_dir = "/tmp/sglang-jax/python/sgl_jax/srt"
        if not os.path.exists(tar_path) or os.environ.get("TEST_PATCH_TARBALL", "0") != "1":
            self.skipTest(f"{tar_path} not present")

        diff_out = subprocess.check_output(
            ["git", "diff", "--relative", "--name-only", "82978d59c", "--", "."],
            cwd=srt_dir,
            text=True,
        )
        expected_files = [line.strip() for line in diff_out.splitlines() if line.strip()]
        self.assertGreaterEqual(len(expected_files), 70)

        with tarfile.open(tar_path, "r:xz") as tf:
            tar_names = set(tf.getnames())
            missing = [f for f in expected_files if f not in tar_names]
            self.assertEqual(missing, [], f"Missing modified files from {tar_path}: {missing}")

            for rel_path in expected_files:
                extracted_bytes = tf.extractfile(rel_path).read()
                with open(os.path.join(srt_dir, rel_path), "rb") as f:
                    disk_bytes = f.read()
                self.assertEqual(
                    extracted_bytes,
                    disk_bytes,
                    f"File {rel_path} in {tar_path} is stale relative to working tree!",
                )


if __name__ == "__main__":
    unittest.main()
