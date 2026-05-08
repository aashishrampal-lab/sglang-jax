"""Scheduler score-from-cache request execution."""

import logging
import time

import jax
import numpy as np
from jax import numpy as jnp

from sgl_jax.srt.managers.io_struct import ScoreFromCacheReqInput, ScoreFromCacheReqOutput
from sgl_jax.srt.utils.common_utils import get_bool_env_var

logger = logging.getLogger(__name__)

SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE = get_bool_env_var(
    "SGLANG_SCORE_FROM_CACHE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE"
)


class SchedulerScoringExecuteMixin:
    def score_from_cache_v2(self, recv_req: ScoreFromCacheReqInput) -> ScoreFromCacheReqOutput:
        self.score_from_cache_v2_attempted += 1
        dispatch_count = 0
        queue_wait_s = 0.0
        device_compute_s = 0.0
        host_orchestration_s = 0.0
        score_start = time.perf_counter()

        try:
            if self.enable_overlap:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="unsupported_scheduler_mode",
                    error_msg="score-from-cache v2 does not support overlap schedule.",
                )

            is_valid, fallback_reason, error_msg = self._score_from_cache_v2_validate_items(
                recv_req
            )
            if not is_valid:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason=fallback_reason,
                    error_msg=error_msg,
                )

            self._evict_expired_scoring_cache_nodes()
            entry = self.scoring_cache_nodes.get(recv_req.cache_handle)
            if entry is None:
                self._record_scoring_cache_lookup(
                    path="score_from_cache_v2",
                    hit=False,
                    lane_name="default",
                )
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="missing_cache_handle",
                    error_msg=(
                        f"Missing scoring cache handle '{recv_req.cache_handle}'. "
                        "The cached prefix may have expired or been released."
                    ),
                )
            cached_last_node, _, prefix_ids, prefix_indices, cached_extra_key, _ = (
                self._unpack_scoring_cache_entry(entry)
            )
            lane_name = self._score_scheduler_lane_from_prefix_len(self, len(prefix_indices))
            self._record_scoring_cache_lookup(
                path="score_from_cache_v2",
                hit=True,
                lane_name=lane_name,
            )
            if cached_last_node is None:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="missing_cache_handle",
                    error_msg=f"Scoring cache handle '{recv_req.cache_handle}' has no radix node.",
                )

            label_only_logprob = bool(
                getattr(self.server_args, "multi_item_score_label_only_logprob", False)
            )
            use_direct_label_only = self._score_from_cache_v2_use_direct_label_only(
                label_only_logprob=label_only_logprob
            )
            if label_only_logprob:
                backend = str(getattr(self.server_args, "device", "")).lower()
                if backend not in {"tpu", "gpu", "cuda", "cpu"}:
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="unsupported_backend",
                        error_msg=(
                            "Label-only logprob fastpath requires TPU/GPU/CPU backend, "
                            f"got device={backend!r}."
                        ),
                    )

            items_per_step = int(recv_req.items_per_step or 0)
            default_items_per_step = int(
                getattr(self.server_args, "multi_item_score_from_cache_v2_items_per_step", 64)
            )
            if default_items_per_step <= 0:
                default_items_per_step = 1
            if use_direct_label_only:
                default_items_per_step = max(default_items_per_step, len(recv_req.items_2d))
            if items_per_step <= 0:
                items_per_step = default_items_per_step
            requested_items_per_step = max(1, items_per_step)
            if use_direct_label_only:
                requested_items_per_step = max(requested_items_per_step, len(recv_req.items_2d))
            requested_token_budget = max(
                0,
                int(recv_req.token_budget or 0),
                int(
                    getattr(self.server_args, "multi_item_score_from_cache_v2_token_budget", 0) or 0
                ),
            )
            # Keep chunk size within request-slot capacity so large configured values
            # (e.g., 64 with max_running_requests=24) do not trigger alloc_req_slots failures.
            capacity_caps: list[int] = []
            max_running_requests = int(getattr(self.server_args, "max_running_requests", 0) or 0)
            if (
                not use_direct_label_only
                and max_running_requests > 0
                and not SCORE_V2_ALLOW_REQPOOL_OVERSUBSCRIBE
            ):
                capacity_caps.append(max_running_requests)
            req_to_token_pool = getattr(self, "req_to_token_pool", None)
            if (
                not use_direct_label_only
                and req_to_token_pool is not None
                and hasattr(req_to_token_pool, "available_size")
            ):
                try:
                    req_pool_available = int(req_to_token_pool.available_size())
                except Exception:
                    req_pool_available = 0
                if req_pool_available > 0:
                    capacity_caps.append(req_pool_available)
            effective_capacity = min(capacity_caps) if capacity_caps else requested_items_per_step
            if effective_capacity <= 0:
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="req_slot_exhausted",
                    error_msg=(
                        "Fastpath v2 requires at least one free request slot "
                        f"(requested_items_per_step={requested_items_per_step})."
                    ),
                    dispatch_count=dispatch_count,
                    queue_wait_s=queue_wait_s,
                    device_compute_s=device_compute_s,
                    host_orchestration_s=host_orchestration_s,
                )
            items_per_step = self._resolve_score_from_cache_v2_items_per_step(
                requested_items_per_step=requested_items_per_step,
                default_items_per_step=default_items_per_step,
                effective_capacity=effective_capacity,
                total_items=len(recv_req.items_2d),
                lane_name=lane_name,
            )
            total_items = len(recv_req.items_2d)
            (
                items_per_step,
                dispatch_token_budget,
                replica_lane_count,
                topology_name,
            ) = self._score_from_cache_v2_topology_dispatch_policy(
                lane_name=lane_name,
                prefix_len=len(prefix_ids),
                requested_items_per_step=requested_items_per_step,
                effective_items_per_step=items_per_step,
                effective_capacity=effective_capacity,
                total_items=total_items,
                requested_token_budget=requested_token_budget,
                max_total_tokens=max(
                    int(recv_req.max_total_tokens or 0),
                    max((len(item) for item in recv_req.items_2d), default=0),
                ),
            )
            if total_items == 0:
                self.score_from_cache_v2_succeeded += 1
                self._record_score_from_cache_v2_timing(
                    queue_wait_s=0.0,
                    device_compute_s=0.0,
                    host_orchestration_s=0.0,
                )
                return ScoreFromCacheReqOutput(
                    rid=recv_req.rid,
                    success=True,
                    scores=[],
                    fallback_reason=None,
                    error_msg="",
                    dispatch_count=0,
                    lifecycle_requests_sent=0,
                    lifecycle_results_received=0,
                    queue_wait_s=0.0,
                    device_compute_s=0.0,
                    host_orchestration_s=0.0,
                    effective_items_per_step=items_per_step,
                    dispatch_token_budget=dispatch_token_budget,
                    replica_lane_count=replica_lane_count,
                    topology_name=topology_name,
                )

            label_token_ids_arr = None
            if label_only_logprob:
                label_token_ids_arr = jnp.asarray(recv_req.label_token_ids, dtype=jnp.int32)
            chunk_plan = self._build_score_from_cache_v2_chunk_plan(
                recv_req.items_2d,
                items_per_step,
                prefix_len=len(prefix_ids),
                token_budget=dispatch_token_budget,
            )

            for _, chunk_items in chunk_plan:
                if not chunk_items:
                    continue

                int32_max = np.iinfo(np.int32).max
                max_seq_len = max((len(prefix_ids) + len(item) for item in chunk_items), default=0)
                estimated_words = self._estimate_score_from_cache_v2_words(
                    prefix_len=len(prefix_ids),
                    items=chunk_items,
                )
                if max_seq_len >= int32_max or estimated_words >= int(int32_max * 0.9):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="size_guard",
                        error_msg=(
                            "Fastpath v2 size guard triggered. "
                            f"max_seq_len={max_seq_len}, estimated_words={estimated_words}"
                        ),
                        dispatch_count=dispatch_count,
                        queue_wait_s=queue_wait_s,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )

            self._touch_scoring_cache_entry(recv_req.cache_handle)

            all_scores: list[list[float] | None] = [None] * total_items
            deferred_direct_chunks: list[tuple[list[int], jax.Array]] = []
            first_dispatch_started = False
            for chunk_indices, chunk_items in chunk_plan:
                if not chunk_items:
                    continue
                if not first_dispatch_started:
                    queue_wait_s = max(0.0, time.perf_counter() - score_start)
                    first_dispatch_started = True
                chunk_host_start = time.perf_counter()
                if label_only_logprob:
                    if use_direct_label_only:
                        chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                            self._run_score_from_cache_v2_direct_chunk_label_only(
                                cache_handle=recv_req.cache_handle,
                                chunk_items=chunk_items,
                                label_token_ids=recv_req.label_token_ids,
                                label_token_ids_arr=label_token_ids_arr,
                                apply_softmax=recv_req.apply_softmax,
                                cached_last_node=cached_last_node,
                                cached_prefix_indices=prefix_indices,
                                prefix_ids=prefix_ids,
                                cached_extra_key=cached_extra_key,
                            )
                        )
                    else:
                        chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                            self._run_score_from_cache_v2_chunk_label_only(
                                cache_handle=recv_req.cache_handle,
                                chunk_items=chunk_items,
                                label_token_ids=recv_req.label_token_ids,
                                label_token_ids_arr=label_token_ids_arr,
                                apply_softmax=recv_req.apply_softmax,
                                cached_last_node=cached_last_node,
                                cached_prefix_indices=prefix_indices,
                                prefix_ids=prefix_ids,
                                cached_extra_key=cached_extra_key,
                            )
                        )
                else:
                    chunk_scores, chunk_device_compute_s, chunk_host_overhead_s = (
                        self._run_score_from_cache_v2_chunk(
                            cache_handle=recv_req.cache_handle,
                            chunk_items=chunk_items,
                            label_token_ids=recv_req.label_token_ids,
                            apply_softmax=recv_req.apply_softmax,
                            cached_last_node=cached_last_node,
                            cached_prefix_indices=prefix_indices,
                            prefix_ids=prefix_ids,
                            cached_extra_key=cached_extra_key,
                        )
                    )
                chunk_score_count = (
                    int(chunk_scores.shape[0])
                    if isinstance(chunk_scores, jax.Array)
                    else len(chunk_scores)
                )
                if chunk_score_count != len(chunk_indices):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="runtime_exception",
                        error_msg=(
                            "score-from-cache v2 chunk output count mismatch: "
                            f"scores={chunk_score_count}, chunk_items={len(chunk_indices)}."
                        ),
                        dispatch_count=dispatch_count,
                        queue_wait_s=queue_wait_s,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )
                if isinstance(chunk_scores, jax.Array):
                    deferred_direct_chunks.append((list(chunk_indices), chunk_scores))
                else:
                    for original_idx, score_row in zip(chunk_indices, chunk_scores, strict=True):
                        all_scores[original_idx] = score_row
                dispatch_count += 1
                device_compute_s += max(0.0, chunk_device_compute_s)
                # host_orchestration_s excludes device time by design.
                chunk_total = max(0.0, time.perf_counter() - chunk_host_start)
                host_orchestration_s += max(
                    0.0,
                    max(chunk_host_overhead_s, chunk_total - chunk_device_compute_s),
                )

            if deferred_direct_chunks:
                materialize_start = time.perf_counter()
                merged_chunk_scores = jnp.concatenate(
                    [chunk_scores for _, chunk_scores in deferred_direct_chunks],
                    axis=0,
                )
                merged_scores_np = np.asarray(jax.device_get(merged_chunk_scores))
                merged_indices = [
                    original_idx
                    for chunk_indices, _ in deferred_direct_chunks
                    for original_idx in chunk_indices
                ]
                if merged_scores_np.shape[0] != len(merged_indices):
                    return self._score_from_cache_v2_fallback_output(
                        recv_req,
                        reason="runtime_exception",
                        error_msg=(
                            "score-from-cache v2 merged direct output count mismatch: "
                            f"scores={merged_scores_np.shape[0]}, chunk_items={len(merged_indices)}."
                        ),
                        dispatch_count=dispatch_count,
                        queue_wait_s=queue_wait_s,
                        device_compute_s=device_compute_s,
                        host_orchestration_s=host_orchestration_s,
                    )
                for original_idx, score_row in zip(
                    merged_indices,
                    merged_scores_np,
                    strict=True,
                ):
                    all_scores[original_idx] = score_row.tolist()
                host_orchestration_s += max(0.0, time.perf_counter() - materialize_start)

            if any(score_row is None for score_row in all_scores):
                return self._score_from_cache_v2_fallback_output(
                    recv_req,
                    reason="runtime_exception",
                    error_msg=(
                        "score-from-cache v2 failed to assemble scores in original order "
                        f"({sum(score_row is not None for score_row in all_scores)} / {total_items})."
                    ),
                    dispatch_count=dispatch_count,
                    queue_wait_s=queue_wait_s,
                    device_compute_s=device_compute_s,
                    host_orchestration_s=host_orchestration_s,
                )
            ordered_scores = [score_row for score_row in all_scores if score_row is not None]

            self.score_from_cache_v2_succeeded += 1
            self._record_score_from_cache_v2_timing(
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
            )
            return ScoreFromCacheReqOutput(
                rid=recv_req.rid,
                success=True,
                scores=ordered_scores,
                fallback_reason=None,
                error_msg="",
                dispatch_count=dispatch_count,
                lifecycle_requests_sent=0,
                lifecycle_results_received=0,
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
                effective_items_per_step=items_per_step,
                dispatch_token_budget=dispatch_token_budget,
                replica_lane_count=replica_lane_count,
                topology_name=topology_name,
            )
        except Exception as e:
            logger.exception("score-from-cache v2 failed; falling back to baseline path.")
            return self._score_from_cache_v2_fallback_output(
                recv_req,
                reason="runtime_exception",
                error_msg=str(e),
                dispatch_count=dispatch_count,
                queue_wait_s=queue_wait_s,
                device_compute_s=device_compute_s,
                host_orchestration_s=host_orchestration_s,
            )
