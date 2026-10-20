"""Decode-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll, KVReceiver
from sgl_jax.srt.disaggregation.base.transfer import (
    AdmissionState,
    DecodeMetadataContext,
    DecodeTransferContext,
    TransferBackend,
)
from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient, PrefillInfoCache
from sgl_jax.srt.disaggregation.common.capacity import per_rank_inflight_limit
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, get_disagg_transport_id
from sgl_jax.srt.mem_cache.memory_pool import write_kv_layer
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

_SHARDED_PREFIX_PAGES = int(os.environ.get("DSA_SHARDED_PREFIX_PAGES", "0"))


@partial(jax.jit, static_argnames=("out_sharding",), donate_argnums=(0,))
def _jit_scatter_one_layer(buf, page_indices, layer_kv, out_sharding):
    """Scatter ``layer_kv`` into ``buf`` at ``page_indices`` in-place per layer."""
    return buf.at[page_indices].set(layer_kv, out_sharding=out_sharding)


@partial(jax.jit, static_argnames=("out_sharding",), donate_argnums=(0,))
def _jit_scatter_all_layers(bufs, page_indices, stacked_kv, out_sharding):
    """Scatter ``stacked_kv`` (shape ``[num_layers, num_pages, ...]``) into ``bufs`` in one fused XLA program."""
    return [
        buf.at[page_indices].set(stacked_kv[i], out_sharding=out_sharding)
        for i, buf in enumerate(bufs)
    ]


@partial(jax.jit, static_argnames=("out_sharding", "dp_size"), donate_argnums=(0,))
def _jit_broadcast_prefix_one_layer(
    buf, src_page_indices, dst_page_indices, out_sharding, dp_size: int
):
    """Broadcast ``src_page_indices`` from the leader DP rank to ``dst_page_indices`` across all DP ranks."""
    src_pages = buf.at[src_page_indices].get(
        out_sharding=NamedSharding(out_sharding.mesh, PartitionSpec(None, *out_sharding.spec[1:]))
    )
    tiled_pages = jnp.tile(src_pages, (dp_size,) + (1,) * (src_pages.ndim - 1))
    return buf.at[dst_page_indices].set(tiled_pages, out_sharding=out_sharding)


@partial(jax.jit, static_argnames=("out_sharding",), donate_argnums=(0,))
def _jit_scatter_prefix_shards_one_layer(
    buf, src_page_indices, dst_page_indices, out_sharding
):
    """Scatter distinct prefix page shards from the leader DP rank across DP ranks."""
    src_pages = buf.at[src_page_indices].get(
        out_sharding=NamedSharding(out_sharding.mesh, PartitionSpec(None, *out_sharding.spec[1:]))
    )
    return buf.at[dst_page_indices].set(src_pages, out_sharding=out_sharding)


def _batch_req_count(batch) -> int:
    if batch is None:
        return 0
    reqs_info = getattr(batch, "reqs_info", None)
    if reqs_info is None:
        return len(getattr(batch, "reqs", ()) or ())
    return sum(len(info.reqs) if info.reqs else 0 for info in reqs_info)


def _batch_req_count_for_dp(batch, dp_rank: int) -> int:
    if batch is None:
        return 0
    reqs_info = getattr(batch, "reqs_info", None)
    if reqs_info is not None:
        info = reqs_info[dp_rank]
        return len(info.reqs) if info.reqs else 0
    return sum(
        1
        for req in (getattr(batch, "reqs", ()) or ())
        if (getattr(req, "dp_rank", None) if getattr(req, "dp_rank", None) is not None else 0)
        == dp_rank
    )


def _request_dp_rank(req, dp_size: int, *, prefill: bool = False) -> int:
    field = "disagg_prefill_dp_rank" if prefill else "dp_rank"
    value = getattr(req, field, None)
    if value is None:
        if dp_size == 1:
            return 0
        raise ValueError(f"PD request {req.rid!r} is missing {field} for dp_size={dp_size}")
    value = int(value)
    if not 0 <= value < dp_size:
        raise ValueError(f"PD request {req.rid!r} has {field}={value} outside [0, {dp_size})")
    return value


@dataclass
class DecodeBookkeeping:
    """Per-request decode-side state."""

    req_id: str
    req: Req
    receiver: KVReceiver | None = None
    # Indices into the paged pool reserved for this request.
    kv_indices: object | None = None
    # Whether the receiver has been initialized + poll started.
    started: bool = False
    # Set by _drain_transfer_queue_synced on multi-host so downstream
    # does not re-poll (a poll() that raised would re-raise and desync).
    synced_state: KVPoll | None = None
    # Prefill-side info from bootstrap, stashed at intake so KV alloc +
    # receiver setup can be deferred to the capacity-gated admission step.
    p_info: dict | None = None
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)


class DecodePreallocQueue:
    """PD reqs awaiting capacity-gated KV alloc. FIFO, thread-safe.

    Entries enter at intake holding only ``p_info`` (no KV indices, no
    receiver). The decode loop's admission gate pops them in FIFO order
    once the paged pool has room; reqs that don't fit stay queued and are
    retried next tick (deferral, never abort).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(f"DecodePreallocQueue already tracks req_id={entry.req_id!r}")
            self._entries[entry.req_id] = entry

    def items_fifo(self) -> list[DecodeBookkeeping]:
        """FIFO snapshot for the admission gate (does not remove)."""

        with self._lock:
            return list(self._entries.values())

    def remove(self, req_id: str) -> None:
        """Drop an admitted (or failed) entry by id."""

        with self._lock:
            self._entries.pop(req_id, None)

    def abort_matching(self, rid_prefix: str, abort_all: bool) -> list[DecodeBookkeeping]:
        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid in list(self._entries):
                if abort_all or rid.startswith(rid_prefix):
                    out.append(self._entries.pop(rid))
        return out


class DecodeTransferQueue:
    """Receivers in TRANSFERRING; polled each tick."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(f"DecodeTransferQueue already tracks req_id={entry.req_id!r}")
            self._entries[entry.req_id] = entry

    def count_by_rank(self, dp_size: int) -> list[int]:
        counts = [0] * dp_size
        with self._lock:
            for entry in self._entries.values():
                counts[_request_dp_rank(entry.req, dp_size)] += 1
        return counts

    def drain_terminal(self) -> list[DecodeBookkeeping]:
        """Return entries whose receiver reached SUCCESS or FAILED."""

        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid, entry in list(self._entries.items()):
                assert entry.receiver is not None
                state = entry.receiver.poll()
                if state in (KVPoll.SUCCESS, KVPoll.FAILED):
                    out.append(entry)
                    del self._entries[rid]
        return out

    def abort_matching(self, rid_prefix: str, abort_all: bool) -> list[DecodeBookkeeping]:
        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid in list(self._entries):
                if abort_all or rid.startswith(rid_prefix):
                    out.append(self._entries.pop(rid))
        return out

    def cancel_matching(self, rid_prefix: str, abort_all: bool) -> list[DecodeBookkeeping]:
        """Mark matching transfers cancelled while retaining their KV pages."""

        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid, entry in self._entries.items():
                if abort_all or rid.startswith(rid_prefix):
                    entry.cancelled = True
                    out.append(entry)
        return out


class SchedulerDisaggregationDecodeMixin:
    """Mixin for PD decode mode on Scheduler."""

    disagg_kv_manager: TransferBackend
    disagg_bootstrap_client: BootstrapClient
    disagg_prefill_info_cache: PrefillInfoCache
    disagg_prealloc_queue: DecodePreallocQueue
    disagg_transfer_queue: DecodeTransferQueue

    def event_loop_overlap_disagg_decode(self: Scheduler) -> None:
        """Overlap D scheduling with forward, retaining the PD admission gates.

        Exactly one unconsumed result crosses an iteration boundary. The first
        batch (including the post-transfer EXTEND) uses the normal overlap
        worker's dummy sampling handshake. Later batches consume their
        predecessor only after dispatch, so future token IDs and request slots
        follow the same ownership protocol as the non-PD overlap loop.
        """
        self.result_queue = deque()
        wd = self.disagg_decode_watchdog
        wd.start()
        while True:
            wd.beat("recv_requests")
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)
            wd.beat("process_input_requests")
            self.process_input_requests_disagg_decode(recv_reqs)

            # pause_generation normally consumes the outstanding result. Also
            # handle a paused flag set without that handler, without admitting
            # new transfers or leaving a worker result stranded indefinitely.
            if self._engine_paused:
                self._drain_disagg_decode_overlap_results()
                self._wait_donation_safe()
                self._drain_decode_transfer_terminals()
                continue

            # Raiden resolves the current pool arrays outside the worker. Wait
            # for replace_all, not device completion: received pages are owned
            # by the transfer queue and cannot appear in the running batch.
            wd.beat("wait_donation_safe")
            self._wait_donation_safe()
            wd.beat("process_decode_queue")
            self.process_decode_queue()
            wd.beat("get_next_batch")
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            had_pending_result = bool(self.result_queue)

            if batch is not None:
                batch.launch_done = threading.Event()
                wd.beat("run_batch")
                result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), result))
                if not had_pending_result:
                    dummy = ScheduleBatch.init_new(
                        reqs=[[] for _ in range(self.dp_size)],
                        req_to_token_pool=self.req_to_token_pool,
                        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                        tree_cache=self.tree_cache,
                        model_config=self.model_config,
                        enable_overlap=self.enable_overlap,
                        dp_size=self.dp_size,
                        spec_algorithm=self.spec_algorithm,
                        mesh=self.mesh,
                    )
                    dummy.forward_mode = ForwardMode.DUMMY_FIRST
                    dummy.next_batch_sampling_info = (
                        self._current_sampling_info_owner().cur_sampling_info
                    )
                    self.process_batch_result(dummy, None, batch.launch_done)

            if had_pending_result:
                wd.beat("process_batch_result")
                previous, result = self.result_queue.popleft()
                previous.next_batch_sampling_info = (
                    self._current_sampling_info_owner().cur_sampling_info
                    if batch is not None
                    else None
                )
                self.process_batch_result(
                    previous, result, batch.launch_done if batch is not None else None
                )
            elif batch is None:
                wd.beat("idle")
                # Generic on_idle checks omit the reservations in PD queues.
                self.new_token_ratio = self.init_new_token_ratio
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)
            self.last_batch = batch

    def _drain_disagg_decode_overlap_results(self: Scheduler) -> None:
        """Quiesce pending compute before pause/retraction releases its pages."""
        if self.result_queue:
            self._wait_donation_safe()
            jax.block_until_ready(self.token_to_kv_pool_allocator.get_kvcache().kv_buffer)
            for dpool in self._get_draft_kv_pools():
                jax.block_until_ready(dpool.kv_buffer)
        while self.result_queue:
            batch, result = self.result_queue.popleft()
            batch.next_batch_sampling_info = None
            self.process_batch_result(batch, result)
        self.last_batch = None
        self.cur_batch = None

    def _wait_decode_admission_safe(self: Scheduler) -> None:
        """Fence previous device writes before Raiden reuses freed KV pages.

        Result N-1 can retire a request while the already launched step N writes
        its pages. JAX orders subsequent forwards, but native Raiden writes do
        not participate in that dependency chain. Until page quarantine is
        implemented, new native receive admission requires a whole-pool device
        fence. Steady decode and polling existing receives need no such fence.
        """
        with jax.profiler.TraceAnnotation("pd_decode_admission_fence"):
            self._wait_donation_safe()
            jax.block_until_ready(self.token_to_kv_pool_allocator.get_kvcache().kv_buffer)
            for dpool in self._get_draft_kv_pools():
                jax.block_until_ready(dpool.kv_buffer)

    def event_loop_normal_disagg_decode(self: Scheduler) -> None:
        """Decode event loop."""

        wd = self.disagg_decode_watchdog
        wd.start()

        while True:
            self.cur_batch = None
            wd.beat("recv_requests")
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)
            wd.beat("process_input_requests")
            self.process_input_requests_disagg_decode(recv_reqs)

            if self._engine_paused:
                self._drain_decode_transfer_terminals()
                continue

            wd.beat("process_decode_queue")
            self.process_decode_queue()

            if len(self.disagg_transfer_queue) == 0:
                self._sync_unpopulated_decode_prefixes()

            wd.beat("get_next_batch")
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                self._sync_unpopulated_decode_prefixes()
                wd.beat("run_batch")
                result = self.run_batch(batch)
                self.cur_batch = None
                wd.beat("process_batch_result")
                self.process_batch_result(batch, result)
            else:
                wd.beat("idle")
                # Skip check_memory / check_tree_cache for PD decode.
                self.new_token_ratio = self.init_new_token_ratio
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.last_batch = batch

    def _decode_backlog_snapshot(self: Scheduler) -> str:
        """One-line backlog snapshot for the watchdog stall report.

        Cheap reads only; never raises (the watchdog suppresses, but a
        clean string is more useful in the log than a swallowed error).
        """

        prealloc = len(self.disagg_prealloc_queue or ())
        transfer = len(self.disagg_transfer_queue or ())
        try:
            ns, nr = self.disagg_kv_manager.inflight_count()
        except Exception:
            ns, nr = (-1, -1)
        try:
            kv_avail = [
                self.token_to_kv_pool_allocator.available_size(dp_rank)
                for dp_rank in range(self.dp_size)
            ]
        except Exception:
            kv_avail = -1
        running = _batch_req_count(self.running_batch)
        return (
            f"prealloc_q={prealloc} transfer_q={transfer} "
            f"inflight_send={ns} inflight_recv={nr} "
            f"kv_avail={kv_avail} running_reqs={running}"
        )

    def process_input_requests_disagg_decode(self: Scheduler, recv_reqs) -> None:
        """Decode-mode request intake. PD reqs are extracted from
        waiting_queue and routed to the prealloc queue.
        """

        self.process_input_requests(recv_reqs)

        recv_pd_rids = {
            getattr(r, "rid", None)
            for r in recv_reqs
            if getattr(r, "bootstrap_room", None) is not None
        }
        new_pd_reqs = self._extract_pd_reqs_from_waiting_queue(recv_pd_rids) if recv_pd_rids else []

        # Retry reqs deferred on a previous tick because no prefill was
        # registered yet (bootstrap cache miss). They go ahead of new reqs so
        # FIFO ordering is preserved across deferrals.
        pending = self._pd_pending_bootstrap
        self._pd_pending_bootstrap = []
        pd_reqs = pending + new_pd_reqs
        if not pd_reqs:
            return

        for req in pd_reqs:
            try:
                from sgl_jax.srt.disaggregation.common.metrics import time_phase

                prefill_dp_rank = _request_dp_rank(req, self.dp_size, prefill=True)
                self._pd_mark_time(req, "bootstrap_start")
                with time_phase("bootstrap", "decode"):
                    if jax.process_count() > 1:
                        # Multi-host caches the matched peer after the first
                        # lookup, so this does no per-request network I/O.
                        p_info = self._pick_prefill_peer_for_this_host(prefill_dp_rank)
                    else:
                        # Local cache resolution (sglang-style): a warm cache
                        # does zero network I/O, so this no longer blocks the
                        # event loop.
                        p_info = self.disagg_prefill_info_cache.pick_for_room(
                            req.bootstrap_room, prefill_dp_rank
                        )
                self._pd_mark_time(req, "bootstrap_done")
            except Exception:
                logger.exception(
                    "bootstrap lookup failed for req_id=%s bootstrap_room=%s; releasing resources",
                    req.rid,
                    req.bootstrap_room,
                )
                self._record_decode_transfer_failure("bootstrap_lookup")
                self._abort_decode_request(req, "bootstrap_lookup")
                continue

            if p_info is None:
                # No prefill registered yet (or the rate-limited refresh was
                # skipped this tick). Defer and retry next tick — never abort.
                self._pd_pending_bootstrap.append(req)
                continue

            req.disagg_peer_process_index = int(p_info.get("jax_process_index", 0))

            try:
                from sgl_jax.srt.disaggregation.bootstrap import (
                    check_prefill_compat,
                    resolve_kv_dtype_name,
                )

                local_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                manager = getattr(self, "disagg_kv_manager", None)
                check_prefill_compat(
                    p_info,
                    local_page_size=self.server_args.page_size,
                    local_kv_dtype=resolve_kv_dtype_name(local_kv_pool.dtype),
                    expected_transfer_engine=(None if manager is None else manager.engine_name),
                    expected_dp_rank=prefill_dp_rank,
                    expected_dp_size=self.dp_size,
                    expected_chunk_prefill_transfer=bool(
                        getattr(
                            self.server_args,
                            "disaggregation_enable_chunk_prefill_transfer",
                            False,
                        )
                    ),
                )
            except ValueError as exc:
                logger.error(
                    "prefill/decode KV layout mismatch for req_id=%s: %s",
                    req.rid,
                    exc,
                )
                self._record_decode_transfer_failure("config_mismatch")
                self._abort_decode_request(req, "config_mismatch")
                continue

            # KV alloc + receiver setup are deferred to the capacity-gated
            # admission step (process_decode_queue). At intake the entry holds
            # only p_info and consumes no paged-pool slots, so a backlog of
            # waiting reqs cannot exhaust decode KV cache.
            entry = DecodeBookkeeping(
                req_id=req.rid,
                req=req,
                p_info=p_info,
            )
            self._pd_mark_time(req, "prealloc_entry")
            self.disagg_prealloc_queue.add(entry)

    def _extract_pd_reqs_from_waiting_queue(self: Scheduler, rids: set) -> list[Req]:
        """Extract PD reqs from waiting_queue by rid set."""

        out: list[Req] = []
        queue = self.waiting_queue
        survivors = []
        for req in queue:
            if req.rid in rids and req.bootstrap_room is not None:
                out.append(req)
            else:
                survivors.append(req)
        queue.clear()
        queue.extend(survivors)
        return out

    def process_decode_queue(self: Scheduler) -> None:
        """Drive prealloc -> transfer -> ready transitions."""

        self.cur_batch = None
        self._admit_decode_prealloc()
        self._drain_decode_transfer_terminals()
        running_empty = self.running_batch is None or self.running_batch.is_empty()
        has_pskip_waiting = any(
            "__pskip_" in str(getattr(r, "disagg_transfer_id", "") or "")
            for r in self.waiting_queue
        )
        if (
            len(self.waiting_queue) > 0
            and len(self.disagg_transfer_queue) > 0
            and (running_empty or has_pskip_waiting)
        ):
            import time as _time

            wd = getattr(self, "disagg_decode_watchdog", None)
            for _ in range(200):
                if len(self.disagg_transfer_queue) == 0:
                    break
                self._drain_decode_transfer_terminals()
                if len(self.disagg_transfer_queue) > 0:
                    if wd is not None:
                        wd.beat("drain_wave_transfers")
                    _time.sleep(0.002)
        if len(self.disagg_transfer_queue) == 0:
            self._sync_unpopulated_decode_prefixes()

    def _sync_unpopulated_decode_prefixes(self: Scheduler) -> None:
        prefix_cache = getattr(self, "_pd_decode_prefix_cache", None)
        if not prefix_cache:
            return
        by_leader: dict[int, tuple[dict, list[tuple[int, dict]]]] = {}
        for dp_rank in sorted(prefix_cache.keys()):
            val = prefix_cache[dp_rank]
            entries = val if isinstance(val, list) else ([val] if val is not None else [])
            for entry in entries:
                if (
                    entry is not None
                    and not entry.get("populated", True)
                    and entry.get("source_leader_entry") is not None
                    and entry.get("kv_indices") is not None
                ):
                    leader_entry = entry["source_leader_entry"]
                    if (
                        leader_entry.get("populated", False)
                        and leader_entry.get("kv_indices") is not None
                    ):
                        lid = id(leader_entry)
                        if lid not in by_leader:
                            by_leader[lid] = (leader_entry, [])
                        by_leader[lid][1].append((dp_rank, entry))
        if not by_leader:
            return

        from sgl_jax.srt.disaggregation.prefill import _globalize_rank_local_page_ids

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = int(kv_pool.page_size)
        dp_size = int(getattr(kv_pool, "dp_size", self.dp_size))
        total_pages = int(kv_pool.kv_buffer[0].shape[0])
        pages_per_rank = total_pages // max(1, dp_size)

        for leader_entry, followers in by_leader.values():
            leader_dp = int(leader_entry.get("dp_rank", 0))
            scatter_kv = leader_entry.get("scatter_kv_indices")
            if _SHARDED_PREFIX_PAGES > 0 and scatter_kv is not None:
                scatter_kv_indices = np.asarray(scatter_kv)
                shard_pages = int(_SHARDED_PREFIX_PAGES)
                total_scatter_pages = dp_size * shard_pages
                if len(scatter_kv_indices) < total_scatter_pages * page_size:
                    continue
                src_local_pages = (
                    scatter_kv_indices[: total_scatter_pages * page_size : page_size] // page_size
                )
                src_global_pages = _globalize_rank_local_page_ids(
                    src_local_pages,
                    dp_rank=leader_dp,
                    dp_size=dp_size,
                    total_pages=total_pages,
                )
                follower_map = {r: e for r, e in followers}
                dst_chunks = []
                for r in range(dp_size):
                    if r in follower_map:
                        f_kv_indices = np.asarray(follower_map[r]["kv_indices"])
                        f_local_pages = (
                            f_kv_indices[: shard_pages * page_size : page_size] // page_size
                        )
                        f_global_pages = _globalize_rank_local_page_ids(
                            f_local_pages,
                            dp_rank=r,
                            dp_size=dp_size,
                            total_pages=total_pages,
                        )
                        dst_chunks.append(f_global_pages)
                    else:
                        dst_chunks.append(
                            np.full(shard_pages, r * pages_per_rank, dtype=src_global_pages.dtype)
                        )
                dst_global_pages = np.concatenate(dst_chunks)

                idx_sharding = NamedSharding(kv_pool.mesh, PartitionSpec(None))
                src_idx_jax = jax.device_put(src_global_pages, idx_sharding)
                dst_idx_jax = jax.device_put(dst_global_pages, idx_sharding)

                all_pools = [kv_pool] + list(self._get_draft_kv_pools())
                for pool in all_pools:
                    for layer_idx in range(pool.layer_num):
                        pool.kv_buffer[layer_idx] = _jit_scatter_prefix_shards_one_layer(
                            pool.kv_buffer[layer_idx],
                            src_idx_jax,
                            dst_idx_jax,
                            pool.kv_sharding,
                        )
                    indexer_buffers = getattr(pool, "indexer_key_buffer", None)
                    if indexer_buffers:
                        for slot_idx in range(len(indexer_buffers)):
                            pool.indexer_key_buffer[slot_idx] = _jit_scatter_prefix_shards_one_layer(
                                pool.indexer_key_buffer[slot_idx],
                                src_idx_jax,
                                dst_idx_jax,
                                pool.kv_sharding,
                            )
                last_pool = all_pools[-1]
                last_indexer = getattr(last_pool, "indexer_key_buffer", None)
                if last_indexer:
                    jax.block_until_ready(last_indexer[-1])
                else:
                    jax.block_until_ready(last_pool.kv_buffer[-1])

                with suppress(Exception):
                    src_idx_jax.delete()
                with suppress(Exception):
                    dst_idx_jax.delete()
            else:
                leader_kv_indices = np.asarray(leader_entry["kv_indices"])
                num_pages = len(leader_kv_indices) // page_size
                if num_pages <= 0:
                    continue
                src_local_pages = leader_kv_indices[: num_pages * page_size : page_size] // page_size
                src_global_pages = _globalize_rank_local_page_ids(
                    src_local_pages,
                    dp_rank=leader_dp,
                    dp_size=dp_size,
                    total_pages=total_pages,
                )

                follower_map = {r: e for r, e in followers}
                dst_chunks = []
                for r in range(dp_size):
                    if r in follower_map:
                        f_kv_indices = np.asarray(follower_map[r]["kv_indices"])
                        f_local_pages = (
                            f_kv_indices[: num_pages * page_size : page_size] // page_size
                        )
                        f_global_pages = _globalize_rank_local_page_ids(
                            f_local_pages,
                            dp_rank=r,
                            dp_size=dp_size,
                            total_pages=total_pages,
                        )
                        dst_chunks.append(f_global_pages)
                    else:
                        # Page 0 of each DP rank's local slice is the reserved sentinel page.
                        dst_chunks.append(
                            np.full(num_pages, r * pages_per_rank, dtype=src_global_pages.dtype)
                        )
                dst_global_pages = np.concatenate(dst_chunks)

                idx_sharding = NamedSharding(kv_pool.mesh, PartitionSpec(None))
                src_idx_jax = jax.device_put(src_global_pages, idx_sharding)
                dst_idx_jax = jax.device_put(dst_global_pages, idx_sharding)

                all_pools = [kv_pool] + list(self._get_draft_kv_pools())
                for pool in all_pools:
                    for layer_idx in range(pool.layer_num):
                        pool.kv_buffer[layer_idx] = _jit_broadcast_prefix_one_layer(
                            pool.kv_buffer[layer_idx],
                            src_idx_jax,
                            dst_idx_jax,
                            pool.kv_sharding,
                            dp_size,
                        )
                    indexer_buffers = getattr(pool, "indexer_key_buffer", None)
                    if indexer_buffers:
                        for slot_idx in range(len(indexer_buffers)):
                            pool.indexer_key_buffer[slot_idx] = _jit_broadcast_prefix_one_layer(
                                pool.indexer_key_buffer[slot_idx],
                                src_idx_jax,
                                dst_idx_jax,
                                pool.kv_sharding,
                                dp_size,
                            )
                last_pool = all_pools[-1]
                last_indexer = getattr(last_pool, "indexer_key_buffer", None)
                if last_indexer:
                    jax.block_until_ready(last_indexer[-1])
                else:
                    jax.block_until_ready(last_pool.kv_buffer[-1])

                with suppress(Exception):
                    src_idx_jax.delete()
                with suppress(Exception):
                    dst_idx_jax.delete()

            for r, f_entry in followers:
                f_entry["populated"] = True
                f_entry["source_leader_entry"] = None
                leader_entry["ref_count"] = max(0, int(leader_entry.get("ref_count", 1)) - 1)
            if (
                _SHARDED_PREFIX_PAGES > 0
                and leader_entry.get("scatter_kv_extra") is not None
                and all(
                    r in prefix_cache
                    and prefix_cache[r] is not None
                    and (
                        any(e.get("populated", False) for e in prefix_cache[r])
                        if isinstance(prefix_cache[r], list)
                        else prefix_cache[r].get("populated", False)
                    )
                    for r in range(dp_size)
                )
            ):
                self._release_decode_kv_indices(leader_entry["scatter_kv_extra"], leader_dp)
                leader_entry["scatter_kv_extra"] = None
                leader_entry["scatter_kv_indices"] = None
            if leader_entry.get("ref_count", 0) == 0 and leader_entry.get("evict_on_zero", False):
                if leader_entry.get("scatter_kv_extra") is not None:
                    self._release_decode_kv_indices(leader_entry["scatter_kv_extra"], leader_dp)
                    leader_entry["scatter_kv_extra"] = None
                    leader_entry["scatter_kv_indices"] = None
                l_kv = leader_entry.get("kv_indices")
                if l_kv is not None:
                    self._release_decode_kv_indices(l_kv, leader_dp)
                    leader_entry["kv_indices"] = None

    def _release_failed_transfer_entry(self: Scheduler, entry: DecodeBookkeeping) -> None:
        prefix_entry = getattr(entry.req, "_pd_shared_prefix_entry", None)
        skip_tokens = int(getattr(entry.req, "_pd_transfer_skip_tokens", 0) or 0)
        if prefix_entry is not None:
            entry.req._pd_shared_prefix_entry = None
            src_leader = prefix_entry.get("source_leader_entry")
            if src_leader is not None:
                prefix_entry["source_leader_entry"] = None
                src_leader["ref_count"] = max(0, int(src_leader.get("ref_count", 1)) - 1)
                if src_leader["ref_count"] == 0 and src_leader.get("evict_on_zero", False):
                    if src_leader.get("scatter_kv_extra") is not None:
                        self._release_decode_kv_indices(
                            src_leader["scatter_kv_extra"], src_leader.get("dp_rank", 0)
                        )
                        src_leader["scatter_kv_extra"] = None
                        src_leader["scatter_kv_indices"] = None
                    l_kv = src_leader.get("kv_indices")
                    if l_kv is not None:
                        self._release_decode_kv_indices(l_kv, src_leader.get("dp_rank", 0))
                    src_leader["kv_indices"] = None
            if skip_tokens == 0:
                prefix_entry["kv_indices"] = None
                prefix_entry["scatter_kv_extra"] = None
                prefix_entry["scatter_kv_indices"] = None
                prefix_entry["ref_count"] = 0
                prefix_entry["populated"] = False
            else:
                prefix_entry["ref_count"] = max(0, int(prefix_entry.get("ref_count", 1)) - 1)
                if prefix_entry["ref_count"] == 0 and (
                    prefix_entry.get("evict_on_zero", False)
                    or not prefix_entry.get("populated", True)
                ):
                    if prefix_entry.get("scatter_kv_extra") is not None:
                        self._release_decode_kv_indices(
                            prefix_entry["scatter_kv_extra"], entry.req.dp_rank
                        )
                        prefix_entry["scatter_kv_extra"] = None
                        prefix_entry["scatter_kv_indices"] = None
                    shared_kv = prefix_entry.get("kv_indices")
                    if shared_kv is not None:
                        self._release_decode_kv_indices(shared_kv, entry.req.dp_rank)
                    prefix_entry["kv_indices"] = None
        if entry.kv_indices is not None:
            self._release_decode_kv_indices(entry.kv_indices, entry.req.dp_rank)
            entry.kv_indices = None

    def _drain_decode_transfer_terminals(self: Scheduler) -> None:
        """Finalize terminal receivers without admitting new work."""

        for entry in self._drain_transfer_queue_synced():
            assert entry.receiver is not None
            state = entry.synced_state
            if state is None:
                try:
                    state = entry.receiver.poll()
                except Exception:
                    logger.exception("receiver.poll() raised for req_id=%s", entry.req_id)
                    state = KVPoll.FAILED
            if entry.cancelled:
                self._release_failed_transfer_entry(entry)
                continue
            if state == KVPoll.SUCCESS:
                try:
                    entry.receiver.commit(
                        lambda kv, req=entry.req, indices=entry.kv_indices: (
                            self._install_received_kv(req, indices, kv)
                        )
                    )
                    skip_tokens = int(getattr(entry.req, "_pd_transfer_skip_tokens", 0) or 0)
                    prefix_entry = getattr(entry.req, "_pd_shared_prefix_entry", None)
                    page_size = self.token_to_kv_pool_allocator.page_size
                    target_prefix_len = (
                        int(
                            os.environ.get(
                                "SGL_PD_SHARED_PREFIX_LEN",
                                "262144" if _SHARDED_PREFIX_PAGES == 0 else "92160",
                            )
                        )
                        // page_size
                    ) * page_size
                    if skip_tokens == 0 and prefix_entry is not None:
                        prefix_entry["populated"] = True
                    if (
                        skip_tokens > 0
                        and prefix_entry is not None
                        and prefix_entry.get("kv_indices") is not None
                    ):
                        reuse_kv = (
                            prefix_entry["kv_indices"][:skip_tokens]
                            if _SHARDED_PREFIX_PAGES == 0
                            else prefix_entry["kv_indices"]
                        )
                        full_kv_indices = np.concatenate(
                            [reuse_kv, np.asarray(entry.kv_indices)]
                        )
                    elif (
                        skip_tokens == 0
                        and _SHARDED_PREFIX_PAGES > 0
                        and prefix_entry is not None
                        and prefix_entry.get("kv_indices") is not None
                        and len(entry.kv_indices) >= target_prefix_len
                    ):
                        full_kv_indices = np.concatenate(
                            [
                                prefix_entry["kv_indices"],
                                np.asarray(entry.kv_indices[target_prefix_len:]),
                            ]
                        )
                    else:
                        full_kv_indices = np.asarray(entry.kv_indices)
                    first_output_id = int(getattr(entry.receiver, "first_output_id", -1))
                    entry.req._pd_first_output_id = first_output_id
                    entry.req._pd_initial_draft_ids = getattr(
                        entry.receiver, "initial_draft_ids", None
                    )
                    use_direct_decode = (
                        os.environ.get("SGL_PD_DIRECT_DECODE", "1") == "1"
                        and first_output_id >= 0
                        and (
                            self._is_spec_decode_enabled()
                            or not getattr(self, "enable_overlap", False)
                        )
                        and not getattr(entry.req, "return_logprob", False)
                        and not getattr(entry.req, "return_output_logprob_only", False)
                        and getattr(entry.req, "grammar", None) is None
                    )
                    # D recomputes the last prompt token only when falling back to EXTEND.
                    # With direct-to-decode admission, P already wrote KV for all
                    # len(origin_input_ids) prompt tokens and transferred t_0.
                    pos_off = int(getattr(entry.req, "_pd_seq_pos_offset", 0) or 0)
                    prefix_len = len(entry.req.origin_input_ids) - pos_off - 1
                    if not use_direct_decode and prefix_len >= 0 and prefix_len % page_size == 0:
                        unused_tail = full_kv_indices[prefix_len:]
                        full_kv_indices = full_kv_indices[:prefix_len]
                        self._release_decode_kv_indices(unused_tail, entry.req.dp_rank)
                    entry.kv_indices = full_kv_indices
                    entry.req._pd_direct_decode_ready = bool(use_direct_decode)
                    self._set_decode_bookkeeping(entry.req, entry.kv_indices)
                    self._enqueue_for_decode(entry.req)
                    self._pd_mark_time(entry.req, "first_token")
                    from sgl_jax.srt.disaggregation.req_time_stats import (
                        maybe_log_time_stats,
                    )

                    maybe_log_time_stats(
                        entry.req.pd_time_stats,
                        req_id=entry.req_id,
                        enabled=getattr(
                            self.server_args,
                            "enable_request_time_stats_logging",
                            False,
                        ),
                    )
                except Exception:
                    logger.exception(
                        "failed to install KV / enqueue decode for req_id=%s; releasing resources",
                        entry.req_id,
                    )
                    self._release_failed_transfer_entry(entry)
                    self._abort_decode_request(
                        entry.req,
                        "kv_writeback",
                        cleanup_transfer=False,
                    )
            else:
                logger.warning(
                    "KVReceiver for req_id=%s reached %s; releasing resources and aborting request",
                    entry.req_id,
                    state.value,
                )
                self._record_decode_transfer_failure("receiver_terminal_failed")
                self._release_failed_transfer_entry(entry)
                self._abort_decode_request(
                    entry.req,
                    "receiver_terminal_failed",
                    cleanup_transfer=False,
                )

    def _pick_prefill_peer_for_this_host(self: Scheduler, dp_rank: int) -> dict[str, object]:
        """Multi-host: find the P host whose jax_process_index matches ours.
        That host's local KV shard is exactly the slice this D host needs.
        Requires P/D to have the same nproc (same-TP constraint).
        """
        peers = getattr(self, "_disagg_prefill_peers", None)
        if peers is None:
            peers = self._disagg_prefill_peers = {}
        if dp_rank in peers:
            return peers[dp_rank]
        my_pidx = jax.process_index()
        my_nproc = jax.process_count()
        all_p = self.disagg_bootstrap_client.list_prefills()
        for p in all_p:
            if (
                int(p.get("jax_process_index", -1)) == my_pidx
                and int(p.get("system_dp_rank", 0)) == dp_rank
            ):
                if int(p.get("jax_process_count", 0)) != my_nproc:
                    raise RuntimeError(
                        f"P/D process_count mismatch: P={p.get('jax_process_count')} "
                        f"D={my_nproc}. Per-host shard transfer requires same nproc."
                    )
                peers[dp_rank] = p
                return p
        raise RuntimeError(
            f"no prefill host with jax_process_index={my_pidx}, dp_rank={dp_rank} "
            "registered (got "
            f"{[(p.get('host'), p.get('jax_process_index'), p.get('system_dp_rank')) for p in all_p]})"
        )

    def _drain_transfer_queue_synced(self: Scheduler) -> list[DecodeBookkeeping]:
        """On multi-host, only drain entries whose receiver has reached a
        terminal state on every NP — _write_kv_to_pool issues a cross-host
        jit and all NPs must enter it for the same set of reqs.
        """
        if jax.process_count() <= 1:
            return self.disagg_transfer_queue.drain_terminal()
        from sgl_jax.srt.disaggregation.common.multihost_sync import (
            synced_terminal_rooms,
        )

        with self.disagg_transfer_queue._lock:
            entries = list(self.disagg_transfer_queue._entries.values())
        success, failed = synced_terminal_rooms(
            entries,
            poll_fn=lambda e: e.receiver.poll(),
            room_fn=lambda e: getattr(e.req, "bootstrap_room", None),
        )
        if not success and not failed:
            return []
        out: list[DecodeBookkeeping] = []
        with self.disagg_transfer_queue._lock:
            for rid, e in list(self.disagg_transfer_queue._entries.items()):
                room = getattr(e.req, "bootstrap_room", None)
                if room in failed:
                    self.disagg_transfer_queue._entries.pop(rid, None)
                    with suppress(Exception):
                        e.receiver.fail(reason="peer_np_failed")
                    e.synced_state = KVPoll.FAILED
                    out.append(e)
                elif room in success:
                    self.disagg_transfer_queue._entries.pop(rid, None)
                    e.synced_state = KVPoll.SUCCESS
                    out.append(e)
        return out

    def _req_pull_bytes_per_device(self: Scheduler, req: Req | None) -> int:
        if req is None:
            return 0
        seqlen = len(req.origin_input_ids)
        skip_tokens = int(getattr(req, "_pd_transfer_skip_tokens", 0) or 0)
        transfer_tokens = max(0, seqlen - skip_tokens)
        if transfer_tokens <= 0:
            return 0
        from sgl_jax.srt.disaggregation.prefill import (
            _pad_to_page_bucket,
            _page_transfer_axis_and_multiple,
        )

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = max(1, int(kv_pool.page_size))
        num_pages = (transfer_tokens + page_size - 1) // page_size
        _, _, min_multiple = _page_transfer_axis_and_multiple(kv_pool)
        padded_pages = _pad_to_page_bucket(num_pages, min_multiple=min_multiple)
        bytes_per_page_global = 0
        for pool in [kv_pool] + list(self._get_draft_kv_pools()):
            bytes_per_page_global += (
                int(pool.layer_num)
                * int(np.prod(pool.kv_buffer[0].shape[1:]))
                * int(np.dtype(pool.dtype).itemsize)
            )
            indexer_buffers = getattr(pool, "indexer_key_buffer", None)
            if indexer_buffers:
                bytes_per_page_global += (
                    int(len(indexer_buffers))
                    * int(np.prod(indexer_buffers[0].shape[1:]))
                    * int(np.dtype(indexer_buffers[0].dtype).itemsize)
                )
        return (padded_pages * bytes_per_page_global) // max(1, int(self.tp_size))

    def _inflight_pull_bytes_per_device(self: Scheduler) -> int:
        with self.disagg_transfer_queue._lock:
            reqs = [e.req for e in self.disagg_transfer_queue._entries.values()]
        return sum(self._req_pull_bytes_per_device(r) for r in reqs)

    def _pull_budget_bytes_per_device(self: Scheduler) -> int:
        # Since event_loop_normal_disagg_decode drains all in-flight transfers
        # in disagg_transfer_queue (which scatters and deletes the temporary
        # pull buffers) BEFORE launching run_batch(), pull buffers and the
        # post-transfer EXTEND scratch never coexist in HBM. With cross-DP
        # prefix deduplication, a 16-request wave pulls only ~0.77 GiB/device.
        peak_extend_scratch_bytes = int(2.70 * (1024**3))
        mem_frac = float(getattr(self.server_args, "mem_fraction_static", 0.90) or 0.90)
        total_hbm_bytes = int(95.73 * (1024**3))
        free_hbm_bytes = int(max(0.05, 1.0 - mem_frac) * total_hbm_bytes)
        return min(
            int(3.20 * (1024**3)),
            max(int(2.00 * (1024**3)), free_hbm_bytes - peak_extend_scratch_bytes),
        )

    def _admit_decode_prealloc(self: Scheduler) -> None:
        """Admit transfers without consuming non-retractable Decode headroom."""

        allocator = self.token_to_kv_pool_allocator
        if allocator is None:
            return

        page_size = allocator.page_size
        reserved_per = self.server_args.disaggregation_num_reserved_decode_tokens
        # Prefill schedules at most 1 request per DP rank per wave, so cap
        # Decode transfer admission to 1 per DP rank (dp_size total) per wave.
        max_inflight = min(
            int(self.server_args.disaggregation_max_inflight_transfers or self.dp_size),
            int(self.dp_size),
        )
        per_rank_inflight = per_rank_inflight_limit(max_inflight, self.dp_size)
        n_transfer = len(self.disagg_transfer_queue)
        admitted = 0
        inflight_pull_bytes = getattr(self, "_inflight_pull_bytes_per_device", lambda: 0)()
        pull_budget_bytes = getattr(self, "_pull_budget_bytes_per_device", lambda: 1 << 60)()
        transfer_per_dp = self.disagg_transfer_queue.count_by_rank(self.dp_size)
        waiting_per_dp = [0] * self.dp_size
        for wreq in getattr(self, "waiting_queue", ()):
            w_dp = _request_dp_rank(wreq, self.dp_size)
            if 0 <= w_dp < self.dp_size:
                waiting_per_dp[w_dp] += 1
        admitted_per_dp = [0] * self.dp_size
        capacity_blocked_ranks: set[int] = set()
        admission_fenced = False

        candidates = self.disagg_prealloc_queue.items_fifo()
        extract_dpseq = getattr(self, "_extract_pd_dpseq", None)
        if callable(extract_dpseq) and len(candidates) > 1:
            candidates.sort(
                key=lambda e: (
                    extract_dpseq(e.req) if extract_dpseq(e.req) is not None else 10**9,
                    0 if "__pfull_" in str(getattr(e.req, "disagg_transfer_id", "") or "") else 1,
                    _request_dp_rank(e.req, self.dp_size),
                )
            )

        active_wave_dpseq = None
        if callable(extract_dpseq) and n_transfer > 0:
            with self.disagg_transfer_queue._lock:
                for te in self.disagg_transfer_queue._entries.values():
                    ttid = str(getattr(te.req, "disagg_transfer_id", "") or "")
                    if "__pskip_" in ttid or "__pfull_" in ttid:
                        s = extract_dpseq(te.req)
                        if s is not None:
                            active_wave_dpseq = (
                                s if active_wave_dpseq is None else min(active_wave_dpseq, s)
                            )
        if callable(extract_dpseq) and active_wave_dpseq is None and len(self.waiting_queue) > 0:
            for wreq in self.waiting_queue:
                if getattr(wreq, "bootstrap_room", None) is not None:
                    wtid = str(getattr(wreq, "disagg_transfer_id", "") or "")
                    if "__pskip_" in wtid or "__pfull_" in wtid:
                        s = extract_dpseq(wreq)
                        if s is not None:
                            active_wave_dpseq = (
                                s if active_wave_dpseq is None else min(active_wave_dpseq, s)
                            )

        prefix_cache = getattr(self, "_pd_decode_prefix_cache", None)
        if prefix_cache is None:
            prefix_cache = self._pd_decode_prefix_cache = {}
        target_prefix_len = (
            int(
                os.environ.get(
                    "SGL_PD_SHARED_PREFIX_LEN",
                    "262144" if _SHARDED_PREFIX_PAGES == 0 else "92160",
                )
            )
            // page_size
        ) * page_size
        shard_prefix_len = (
            int(_SHARDED_PREFIX_PAGES) * page_size if _SHARDED_PREFIX_PAGES > 0 else 0
        )
        eff_prefix_len = shard_prefix_len if shard_prefix_len > 0 else target_prefix_len

        for entry in candidates:
            decode_dp_rank = _request_dp_rank(entry.req, self.dp_size)
            prefill_dp_rank = _request_dp_rank(entry.req, self.dp_size, prefill=True)
            # Pulled KV buffers remain outside paged-pool accounting until scatter.
            if max_inflight > 0 and (n_transfer + admitted) >= max_inflight:
                break
            # Preserve FIFO within a blocked rank without stalling other ranks.
            if decode_dp_rank in capacity_blocked_ranks:
                continue
            if (
                per_rank_inflight > 0
                and transfer_per_dp[decode_dp_rank] + admitted_per_dp[decode_dp_rank]
                >= per_rank_inflight
            ):
                continue
            n_running = _batch_req_count_for_dp(self.running_batch, decode_dp_rank)
            n_active_dp = (
                n_running
                + waiting_per_dp[decode_dp_rank]
                + transfer_per_dp[decode_dp_rank]
                + admitted_per_dp[decode_dp_rank]
            )
            if n_active_dp >= getattr(self, "per_dp_max_running_requests", 1 << 30):
                capacity_blocked_ranks.add(decode_dp_rank)
                continue

            entry_tid = str(getattr(entry.req, "disagg_transfer_id", "") or "")
            use_cross_dp_sync = "__pskip_" in entry_tid or "__pfull_" in entry_tid
            dpseq = (
                extract_dpseq(entry.req)
                if (callable(extract_dpseq) and use_cross_dp_sync)
                else None
            )
            if dpseq is not None:
                if active_wave_dpseq is not None and dpseq != active_wave_dpseq:
                    capacity_blocked_ranks.add(decode_dp_rank)
                    continue
                next_seq_map = getattr(self, "_pd_next_expected_dpseq", None)
                if next_seq_map is None:
                    next_seq_map = self._pd_next_expected_dpseq = {}
                wait_map = getattr(self, "_pd_dpseq_wait_ticks", None)
                if wait_map is None:
                    wait_map = self._pd_dpseq_wait_ticks = {}
                if dpseq == 0 and next_seq_map.get(decode_dp_rank, 0) > 0:
                    next_seq_map[decode_dp_rank] = 0
                expected_seq = next_seq_map.get(decode_dp_rank, 0)
                if dpseq > expected_seq:
                    wkey = (decode_dp_rank, dpseq)
                    wait_map[wkey] = wait_map.get(wkey, 0) + 1
                    if wait_map[wkey] < 200:
                        capacity_blocked_ranks.add(decode_dp_rank)
                        continue

            seqlen = len(entry.req.origin_input_ids)
            tid = str(getattr(entry.req, "disagg_transfer_id", "") or "")
            force_pfull = "__pfull_" in tid
            force_pskip = "__pskip_" in tid and seqlen >= target_prefix_len + page_size
            is_spec = (
                getattr(self, "spec_algorithm", None) is not None
                and not self.spec_algorithm.is_none()
            )

            max_prefix_slots = (
                1
                if _SHARDED_PREFIX_PAGES > 0
                else int(os.environ.get("SGL_PD_MAX_PREFIX_SLOTS", "8"))
            )
            raw_slots = prefix_cache.get(decode_dp_rank)
            if raw_slots is None:
                slot_list: list[dict] = []
            elif isinstance(raw_slots, list):
                slot_list = [
                    e
                    for e in raw_slots
                    if isinstance(e, dict) and e.get("kv_indices") is not None
                ]
            elif isinstance(raw_slots, dict):
                slot_list = [raw_slots] if raw_slots.get("kv_indices") is not None else []
            else:
                slot_list = []
            prefix_cache[decode_dp_rank] = slot_list

            def _evict_prefix_entry(ev_entry: dict) -> None:
                if ev_entry.get("kv_indices") is None:
                    return
                if int(ev_entry.get("ref_count", 0)) <= 0:
                    if ev_entry.get("scatter_kv_extra") is not None:
                        self._release_decode_kv_indices(
                            ev_entry["scatter_kv_extra"], decode_dp_rank
                        )
                        ev_entry["scatter_kv_extra"] = None
                        ev_entry["scatter_kv_indices"] = None
                    self._release_decode_kv_indices(ev_entry["kv_indices"], decode_dp_rank)
                    ev_entry["kv_indices"] = None
                else:
                    ev_entry["evict_on_zero"] = True

            cached_entry = None
            reuse_skip_tokens = 0
            reuse_lcp_tokens = 0
            reuse_cmp_len = 0
            if not force_pfull and slot_list and seqlen >= page_size * 2:
                for cand_entry in reversed(slot_list):
                    if (
                        cand_entry.get("kv_indices") is None
                        or cand_entry.get("evict_on_zero", False)
                    ):
                        continue
                    c_ids = cand_entry.get("prefix_ids", ())
                    cap_cmp = min(
                        target_prefix_len,
                        len(cand_entry["kv_indices"]),
                        ((seqlen - page_size) // page_size) * page_size,
                    )
                    cmp_len = min(len(c_ids), cap_cmp + 1)
                    if cmp_len >= 4096:
                        if entry.req.origin_input_ids[:cmp_len] == c_ids[:cmp_len]:
                            lcp_tokens = cmp_len
                        else:
                            arr_cur = np.asarray(
                                entry.req.origin_input_ids[:cmp_len], dtype=np.int32
                            )
                            arr_prev = np.asarray(c_ids[:cmp_len], dtype=np.int32)
                            diff = np.nonzero(arr_cur != arr_prev)[0]
                            lcp_tokens = int(diff[0]) if len(diff) > 0 else cmp_len
                        usable_lcp = (
                            max(0, lcp_tokens - 1)
                            if (is_spec and lcp_tokens < cmp_len)
                            else min(lcp_tokens, cap_cmp)
                        )
                        cand_skip = (usable_lcp // page_size) * page_size
                        if (
                            cand_skip >= 4096
                            and seqlen >= cand_skip + page_size
                            and (_SHARDED_PREFIX_PAGES == 0 or cand_skip == target_prefix_len)
                            and cand_skip > reuse_skip_tokens
                        ):
                            cached_entry = cand_entry
                            reuse_skip_tokens = cand_skip
                            reuse_lcp_tokens = lcp_tokens
                            reuse_cmp_len = cmp_len
            can_reuse_prefix = reuse_skip_tokens > 0 and cached_entry is not None
            if not can_reuse_prefix and (force_pfull or max_prefix_slots <= 1):
                for ev_e in list(slot_list):
                    if int(ev_e.get("ref_count", 0)) <= 0:
                        _evict_prefix_entry(ev_e)
                        slot_list.remove(ev_e)

            can_bcast_from_leader = False
            leader_entry = None
            if force_pskip and not can_reuse_prefix:
                leader_entry = getattr(self, "_pd_latest_leader_prefix", None)
                if (
                    leader_entry is not None
                    and leader_entry.get("kv_indices") is not None
                    and not leader_entry.get("evict_on_zero", False)
                    and seqlen >= target_prefix_len + page_size
                    and len(leader_entry.get("prefix_ids", ())) >= target_prefix_len
                    and entry.req.origin_input_ids[:target_prefix_len]
                    == leader_entry["prefix_ids"][:target_prefix_len]
                ):
                    can_bcast_from_leader = True
                else:
                    capacity_blocked_ranks.add(decode_dp_rank)
                    continue

            if can_reuse_prefix:
                skip_tokens = reuse_skip_tokens
            elif can_bcast_from_leader:
                skip_tokens = target_prefix_len
            else:
                skip_tokens = 0
            entry.req._pd_transfer_skip_tokens = skip_tokens
            req_pull_bytes = getattr(self, "_req_pull_bytes_per_device", lambda r: 0)(entry.req)
            if (n_transfer + admitted) > 0 and (inflight_pull_bytes + req_pull_bytes) > pull_budget_bytes:
                # Keep requests that would exceed Decode HBM pull budget waiting in
                # disagg_prealloc_queue (CPU) until in-flight transfers scatter & free.
                break

            transfer_tokens = max(1, seqlen - skip_tokens)
            page_aligned = ((transfer_tokens + page_size - 1) // page_size) * page_size
            alloc_tokens = (
                eff_prefix_len + page_aligned if can_bcast_from_leader else page_aligned
            )
            req_sp = getattr(entry.req, "sampling_params", None)
            req_max_new = int(getattr(req_sp, "max_new_tokens", 0) or 0)
            if not getattr(req_sp, "ignore_eos", False):
                req_max_new = min(req_max_new, reserved_per)
            eff_reserved_per = max(
                reserved_per, ((req_max_new + page_size - 1) // page_size) * page_size
            )
            running_rem_tokens = 0
            if (
                self.running_batch is not None
                and not getattr(self.running_batch, "is_empty", lambda: False)()
                and hasattr(self.running_batch, "reqs_info")
                and decode_dp_rank < len(self.running_batch.reqs_info)
                and self.running_batch.reqs_info[decode_dp_rank].reqs
            ):
                for r_req in self.running_batch.reqs_info[decode_dp_rank].reqs:
                    r_sp = getattr(r_req, "sampling_params", None)
                    r_max = int(getattr(r_sp, "max_new_tokens", 0) or reserved_per)
                    if not getattr(r_sp, "ignore_eos", False):
                        r_max = min(r_max, reserved_per)
                    r_aligned = ((r_max + page_size - 1) // page_size) * page_size
                    r_gen = len(getattr(r_req, "output_ids", ()) or ())
                    running_rem_tokens += max(reserved_per, r_aligned - r_gen)
            non_running_active = (
                waiting_per_dp[decode_dp_rank]
                + transfer_per_dp[decode_dp_rank]
                + admitted_per_dp[decode_dp_rank]
            )
            suffix_tokens = max(
                1,
                seqlen - (target_prefix_len if seqlen >= target_prefix_len + page_size else 0),
            )
            suffix_page_aligned = ((suffix_tokens + page_size - 1) // page_size) * page_size
            reserved = running_rem_tokens + eff_reserved_per * (non_running_active + 1) + 2048
            extra_wave_reserve = (
                (suffix_page_aligned + eff_reserved_per)
                if (not can_reuse_prefix and seqlen >= target_prefix_len + page_size)
                else 0
            )
            if (
                not can_reuse_prefix
                and slot_list
                and alloc_tokens + reserved + extra_wave_reserve
                > allocator.available_size(decode_dp_rank)
            ):
                for ev_e in list(slot_list):
                    if (
                        alloc_tokens + reserved + extra_wave_reserve
                        <= allocator.available_size(decode_dp_rank)
                    ):
                        break
                    if int(ev_e.get("ref_count", 0)) <= 0:
                        _evict_prefix_entry(ev_e)
                        slot_list.remove(ev_e)
            if alloc_tokens + reserved + extra_wave_reserve > allocator.available_size(decode_dp_rank):
                capacity_blocked_ranks.add(decode_dp_rank)
                continue

            metadata_ready = self.disagg_kv_manager.poll_decode_metadata(
                DecodeMetadataContext(
                    req_id=entry.req.rid,
                    transfer_id=get_disagg_transport_id(entry.req),
                    bootstrap_room=entry.req.bootstrap_room,
                    prefill_dp_rank=prefill_dp_rank,
                    peer_info=entry.p_info or {},
                )
            )
            if not metadata_ready:
                self._expire_decode_prealloc(entry)
                continue

            # Fence only candidates that passed admission gates. A full transfer
            # window or pending metadata must not serialize otherwise independent
            # decode compute. No forwards launch inside this sweep, so one fence
            # protects every native receive admitted here.
            if getattr(self, "enable_overlap", False) and not admission_fenced:
                self._wait_decode_admission_safe()
                admission_fenced = True

            full_alloc = allocator.alloc(alloc_tokens, dp_rank=decode_dp_rank)
            if full_alloc is None:
                capacity_blocked_ranks.add(decode_dp_rank)
                continue

            if can_bcast_from_leader:
                prefix_kv_indices = np.asarray(full_alloc[:eff_prefix_len]).copy()
                kv_indices = full_alloc[eff_prefix_len:]
            else:
                prefix_kv_indices = None
                kv_indices = full_alloc

            try:
                admission = self.disagg_kv_manager.try_start_decode(
                    DecodeTransferContext(
                        req_id=entry.req.rid,
                        transfer_id=get_disagg_transport_id(entry.req),
                        bootstrap_room=entry.req.bootstrap_room,
                        decode_dp_rank=decode_dp_rank,
                        prefill_dp_rank=prefill_dp_rank,
                        peer_info=entry.p_info or {},
                        kv_indices=kv_indices,
                        page_size=page_size,
                        prompt_tokens=transfer_tokens,
                        spec_factory=lambda req=entry.req: self._build_kv_spec_for_req(req),
                        direct_commit=lambda expected, req=entry.req, indices=kv_indices: (
                            self._commit_direct_received_kv(req, indices, expected)
                        ),
                    )
                )
            except Exception:
                logger.exception(
                    "failed to set up KVReceiver for req_id=%s",
                    entry.req.rid,
                )
                self._record_decode_transfer_failure("receiver_init")
                self._release_decode_kv_indices(full_alloc, decode_dp_rank)
                self.disagg_prealloc_queue.remove(entry.req_id)
                self._abort_decode_request(entry.req, "receiver_init")
                continue

            if admission.state == AdmissionState.DEFERRED:
                self._release_decode_kv_indices(full_alloc, decode_dp_rank)
                self._expire_decode_prealloc(entry)
                continue

            receiver = admission.receiver
            if receiver is None:
                raise RuntimeError("admitted transfer is missing a receiver")

            if dpseq is not None:
                if active_wave_dpseq is None:
                    active_wave_dpseq = dpseq
                self._pd_next_expected_dpseq[decode_dp_rank] = dpseq + 1
            if can_reuse_prefix and cached_entry is not None:
                if cached_entry in slot_list:
                    slot_list.remove(cached_entry)
                slot_list.append(cached_entry)
                cached_entry["ref_count"] = int(cached_entry.get("ref_count", 0)) + 1
                entry.req._pd_shared_prefix_entry = cached_entry
                prefix_len = skip_tokens if shard_prefix_len == 0 else eff_prefix_len
                if (
                    shard_prefix_len == 0
                    and skip_tokens > 0
                    and os.environ.get("SGL_PD_MULTI_TURN_PREFIX_EXTEND", "1") == "1"
                ):
                    cap_len = min(
                        target_prefix_len,
                        ((len(entry.req.origin_input_ids) - page_size) // page_size) * page_size,
                    )
                    if reuse_lcp_tokens == reuse_cmp_len and cap_len > skip_tokens:
                        extra_tokens = cap_len - skip_tokens
                        cached_entry["kv_indices"] = np.concatenate(
                            [
                                cached_entry["kv_indices"],
                                np.asarray(kv_indices, dtype=np.int32)[:extra_tokens],
                            ]
                        )
                        cached_entry["prefix_ids"] = list(
                            entry.req.origin_input_ids[: cap_len + 1]
                        )
                        prefix_len = cap_len
                entry.req._pd_shared_prefix_len = prefix_len
                entry.req._pd_seq_pos_offset = (
                    0 if shard_prefix_len == 0 else (target_prefix_len - eff_prefix_len)
                )
            elif can_bcast_from_leader and leader_entry is not None:
                while len(slot_list) >= max_prefix_slots and slot_list:
                    ev_victim = next(
                        (e for e in slot_list if int(e.get("ref_count", 0)) <= 0),
                        slot_list[0],
                    )
                    slot_list.remove(ev_victim)
                    _evict_prefix_entry(ev_victim)
                new_cache_entry = {
                    "prefix_ids": list(entry.req.origin_input_ids[: target_prefix_len + 1]),
                    "kv_indices": prefix_kv_indices,
                    "scatter_kv_indices": None,
                    "scatter_kv_extra": None,
                    "ref_count": 1,
                    "evict_on_zero": False,
                    "populated": False,
                    "source_leader_entry": leader_entry,
                    "dp_rank": decode_dp_rank,
                }
                leader_entry["ref_count"] = int(leader_entry.get("ref_count", 0)) + 1
                slot_list.append(new_cache_entry)
                entry.req._pd_shared_prefix_entry = new_cache_entry
                entry.req._pd_shared_prefix_len = eff_prefix_len
                entry.req._pd_seq_pos_offset = target_prefix_len - eff_prefix_len
            else:
                cap_len = min(
                    target_prefix_len,
                    ((seqlen - page_size) // page_size) * page_size,
                )
                if cap_len >= 4096 and (shard_prefix_len == 0 or seqlen >= target_prefix_len + page_size):
                    while len(slot_list) >= max_prefix_slots and slot_list:
                        ev_victim = next(
                            (e for e in slot_list if int(e.get("ref_count", 0)) <= 0),
                            slot_list[0],
                        )
                        slot_list.remove(ev_victim)
                        _evict_prefix_entry(ev_victim)
                    kv_indices_np = np.asarray(kv_indices)
                    store_kv_len = eff_prefix_len if shard_prefix_len > 0 else cap_len
                    new_cache_entry = {
                        "prefix_ids": list(entry.req.origin_input_ids[: cap_len + 1]),
                        "kv_indices": kv_indices_np[:store_kv_len].copy(),
                        "scatter_kv_indices": (
                            kv_indices_np[:target_prefix_len].copy()
                            if shard_prefix_len > 0
                            else None
                        ),
                        "scatter_kv_extra": (
                            kv_indices_np[eff_prefix_len:target_prefix_len].copy()
                            if shard_prefix_len > 0
                            else None
                        ),
                        "ref_count": 1,
                        "evict_on_zero": False,
                        "populated": False,
                        "source_leader_entry": None,
                        "dp_rank": decode_dp_rank,
                    }
                    slot_list.append(new_cache_entry)
                    self._pd_latest_leader_prefix = new_cache_entry
                    entry.req._pd_shared_prefix_entry = new_cache_entry
                    entry.req._pd_shared_prefix_len = store_kv_len
                    entry.req._pd_seq_pos_offset = (
                        target_prefix_len - eff_prefix_len if shard_prefix_len > 0 else 0
                    )
                else:
                    if max_prefix_slots <= 1:
                        for ev_e in list(slot_list):
                            _evict_prefix_entry(ev_e)
                        slot_list.clear()
                    entry.req._pd_shared_prefix_entry = None
                    entry.req._pd_shared_prefix_len = 0
                    entry.req._pd_seq_pos_offset = 0

            entry.kv_indices = kv_indices
            entry.receiver = receiver
            entry.started = True
            self._pd_mark_time(entry.req, "transfer_entry")
            self.disagg_prealloc_queue.remove(entry.req_id)
            self.disagg_transfer_queue.add(entry)
            admitted += 1
            admitted_per_dp[decode_dp_rank] += 1
            inflight_pull_bytes += req_pull_bytes

    def _expire_decode_prealloc(self: Scheduler, entry: DecodeBookkeeping) -> None:
        timeout_s = max(3600.0, float(self.server_args.disaggregation_pull_timeout_seconds))
        if timeout_s <= 0 or time.monotonic() - entry.created_at < timeout_s:
            return
        self.disagg_prealloc_queue.remove(entry.req_id)
        self._record_decode_transfer_failure("metadata_timeout")
        self._abort_decode_request(entry.req, "metadata_timeout")

    # ------------------------------------------------------------------
    # Overridable / test-friendly hooks
    # ------------------------------------------------------------------

    def _pd_mark_time(self: Scheduler, req: Req, name: str) -> None:
        """Record a PD lifecycle mark on ``req`` (no-op unless enabled)."""

        if not getattr(self.server_args, "enable_request_time_stats_logging", False):
            return
        from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

        ts = req.pd_time_stats
        if ts is None:
            role = getattr(self.server_args, "disaggregation_mode", "decode")
            ts = TimeStats(role)
            req.pd_time_stats = ts
        ts.mark(name)

    def _release_decode_kv_indices(self: Scheduler, kv_indices, dp_rank: int | None) -> None:
        """Release KV indices back to the allocator."""

        if kv_indices is None:
            return
        allocator = self.token_to_kv_pool_allocator
        if allocator is not None:
            try:
                rank = 0 if dp_rank is None and self.dp_size == 1 else int(dp_rank)
                allocator.free(kv_indices, dp_rank=rank)
            except Exception:
                logger.exception("failed to free kv_indices=%r", kv_indices)

    def _use_pd_spec_meta_transfer(self: Scheduler) -> bool:
        if os.environ.get("SGL_PD_DIRECT_DECODE", "1") != "1":
            return False
        if getattr(self, "disagg_use_d2h_staging", False):
            return False
        mgr = getattr(self, "disagg_kv_manager", None)
        return getattr(mgr, "engine_name", None) == "jax" and not getattr(
            mgr, "requires_host_staging", False
        )

    def _admit_pd_direct_decode_reqs(self: Scheduler) -> int:
        """Directly admit PD requests that already received their first output token into running_batch."""
        if not self.waiting_queue:
            return 0
        from jax.sharding import NamedSharding, PartitionSpec
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
        from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
        from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
        from sgl_jax.srt.speculative.relay_buffer import update_spec_relay_buffers_admit

        running_per_dp = [
            len(info.reqs) if info.reqs else 0 for info in self.running_batch.reqs_info
        ]
        max_bs_per_dp = max(1, int(self.max_total_num_tokens) // max(1, int(self.dp_size)))
        if getattr(self.server_args, "max_running_requests", None):
            max_bs_per_dp = min(
                max_bs_per_dp,
                max(1, int(self.server_args.max_running_requests) // max(1, int(self.dp_size))),
            )

        ready_reqs: list[Req] = []
        remaining_waiting: list[Req] = []
        avail_req_slots = self.req_to_token_pool.available_size()
        for req in self.waiting_queue:
            if not getattr(req, "_pd_direct_decode_ready", False):
                remaining_waiting.append(req)
                continue
            prefix_entry = getattr(req, "_pd_shared_prefix_entry", None)
            if prefix_entry is not None and not prefix_entry.get("populated", True):
                remaining_waiting.append(req)
                continue
            dp_rank = _request_dp_rank(req, self.dp_size)
            if (
                running_per_dp[dp_rank] >= max_bs_per_dp
                or avail_req_slots < 1
            ):
                remaining_waiting.append(req)
                continue
            running_per_dp[dp_rank] += 1
            avail_req_slots -= 1
            ready_reqs.append(req)

        if not ready_reqs:
            return 0

        self.waiting_queue = remaining_waiting
        is_spec = self._is_spec_decode_enabled()
        topk = int(getattr(self.server_args, "speculative_eagle_topk", 1) or 1)
        num_steps = int(getattr(self.server_args, "speculative_num_steps", 1) or 1)
        from sgl_jax.srt.mem_cache.common import release_kv_cache

        admitted_reqs: list[Req] = []
        for req in ready_reqs:
            req.req_pool_idx = None
            req_pool_indices = self.req_to_token_pool.alloc([req])
            if req_pool_indices is None:
                self.waiting_queue.insert(0, req)
                continue
            req.req_pool_idx = req_pool_indices[0]
            pos_off = int(getattr(req, "_pd_seq_pos_offset", 0) or 0)
            seqlen = len(req.origin_input_ids) - pos_off
            raw_prealloc = getattr(req, "_pd_prealloc_kv_indices", None)
            if raw_prealloc is None:
                raw_prealloc = getattr(req, "_decode_prealloc_tokens", None)
            prealloc = np.asarray(raw_prealloc if raw_prealloc is not None else [], dtype=np.int32)
            write_len = min(seqlen, len(prealloc))
            if write_len > 0:
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(0, write_len)), prealloc[:write_len]
                )
            req.kv_committed_len = seqlen
            req.kv_allocated_len = seqlen
            req.fill_ids = list(req.origin_input_ids)
            req.extend_input_len = 0
            req.prefix_indices = np.empty((0,), dtype=np.int32)
            first_tok = int(getattr(req, "_pd_first_output_id", 0))
            req.output_ids.append(first_tok)
            req.check_finished()
            if req.finished():
                release_kv_cache(req, self.tree_cache, allow_overallocated=is_spec)
            req._pd_direct_decode_ready = False
            admitted_reqs.append(req)

        if not admitted_reqs:
            return 0

        self.stream_output(admitted_reqs, False, False)
        active_reqs = [r for r in admitted_reqs if not r.finished()]
        if not active_reqs:
            return len(admitted_reqs)

        reqs_per_dp: list[list[Req]] = [[] for _ in range(self.dp_size)]
        for r in active_reqs:
            reqs_per_dp[_request_dp_rank(r, self.dp_size)].append(r)

        admit_batch = ScheduleBatch.init_new(
            reqs_per_dp,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.dp_size,
            spec_algorithm=self.spec_algorithm,
            mesh=self.mesh,
            embedding_pool=getattr(self, "embedding_pool", None),
        )
        relay_owner = (
            getattr(self, "draft_worker", None)
            if getattr(getattr(self, "draft_worker", None), "spec_relay_buffers", None) is not None
            else getattr(self, "tp_worker", None)
        )
        relay_buffers = (
            getattr(relay_owner, "spec_relay_buffers", None) if relay_owner is not None else None
        )
        hidden_size = int(getattr(self.model_config, "hidden_size", 1) or 1)
        topk_width = num_steps if num_steps > 1 and topk == 1 else topk

        for dp_rank, info in enumerate(admit_batch.reqs_info):
            if not info.reqs:
                continue
            info.req_pool_indices = np.asarray([r.req_pool_idx for r in info.reqs], dtype=np.int32)
            if is_spec:
                seq_lens_list = [
                    len(r.origin_input_ids)
                    - int(getattr(r, "_pd_seq_pos_offset", 0) or 0)
                    + len(r.output_ids)
                    for r in info.reqs
                ]
                info.orig_seq_lens = [
                    len(r.origin_input_ids) + len(r.output_ids) for r in info.reqs
                ]
            else:
                seq_lens_list = [
                    len(r.origin_input_ids)
                    - int(getattr(r, "_pd_seq_pos_offset", 0) or 0)
                    + max(0, len(r.output_ids) - 1)
                    for r in info.reqs
                ]
                info.orig_seq_lens = [
                    len(r.origin_input_ids) + max(0, len(r.output_ids) - 1)
                    for r in info.reqs
                ]
            info.seq_lens = np.asarray(seq_lens_list, dtype=np.int32)
            info.seq_lens_sum = int(info.seq_lens.sum())
            info.extend_lens = [1] * len(info.reqs)
            info.prefix_lens = [
                len(r.origin_input_ids) - int(getattr(r, "_pd_seq_pos_offset", 0) or 0)
                for r in info.reqs
            ]
            info.positions = list(seq_lens_list)
            info.output_ids = np.asarray([int(r.output_ids[-1]) for r in info.reqs], dtype=np.int32)
            if self.model_config is not None and hasattr(self.model_config, "vocab_size"):
                info.sampling_info = SamplingBatchInfo.from_schedule_batch(
                    info,
                    self.model_config.vocab_size,
                    batch=admit_batch,
                )
            if is_spec:
                n_dp = len(info.reqs)
                first_toks_dp = info.output_ids
                if relay_buffers is not None:
                    info.spec_info = EagleDraftInput(
                        future_indices=info.req_pool_indices,
                        allocate_lens=info.seq_lens.copy(),
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        num_tokens_per_batch=np.asarray(1, dtype=np.int32),
                        num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
                    )
                else:
                    topk_idx_dp = np.repeat(first_toks_dp[:, None], topk_width, axis=1)
                    for req_i, r in enumerate(info.reqs):
                        draft_ids = getattr(r, "_pd_initial_draft_ids", None)
                        if draft_ids:
                            take_k = min(topk_width, len(draft_ids))
                            topk_idx_dp[req_i, :take_k] = np.asarray(
                                draft_ids[:take_k], dtype=np.int32
                            )
                    info.spec_info = EagleDraftInput(
                        topk_p=np.ones((n_dp, topk_width), dtype=np.float32),
                        topk_index=topk_idx_dp,
                        hidden_states=np.zeros((n_dp, hidden_size), dtype=np.float32),
                        verified_id=first_toks_dp.copy(),
                        accept_length=np.zeros((n_dp,), dtype=np.int32),
                        allocate_lens=info.seq_lens.copy(),
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        num_tokens_per_batch=np.asarray(1, dtype=np.int32),
                        num_tokens_for_logprob_per_batch=np.asarray(1, dtype=np.int32),
                    )

        if is_spec and relay_buffers is not None:
            dp_size = int(self.dp_size)
            max_running = int(getattr(self.server_args, "max_running_requests", 256) or 256)
            per_dp_cap = max(
                1,
                max_running // max(1, dp_size),
                max(len(r_list) for r_list in reqs_per_dp),
            )
            total_slots = dp_size * per_dp_cap
            fut_idx_np = np.zeros((total_slots,), dtype=np.int32)
            val_mask_np = np.zeros((total_slots,), dtype=np.bool_)
            relay_num_steps = int(relay_buffers.topk_index.shape[-1])
            topk_idx_np = np.zeros((total_slots, relay_num_steps), dtype=np.int32)
            ver_id_np = np.zeros((total_slots,), dtype=np.int32)
            new_seq_np = np.zeros((total_slots,), dtype=np.int32)
            dp_cursors = [0] * dp_size
            for r in active_reqs:
                d = _request_dp_rank(r, dp_size)
                slot = d * per_dp_cap + dp_cursors[d]
                dp_cursors[d] += 1
                fut_idx_np[slot] = int(r.req_pool_idx)
                val_mask_np[slot] = True
                t0 = int(r.output_ids[-1])
                topk_idx_np[slot, :] = t0
                draft_ids = getattr(r, "_pd_initial_draft_ids", None)
                if draft_ids:
                    take_k = min(relay_num_steps, len(draft_ids))
                    topk_idx_np[slot, :take_k] = np.asarray(
                        draft_ids[:take_k], dtype=np.int32
                    )
                ver_id_np[slot] = t0
                new_seq_np[slot] = (
                    len(r.origin_input_ids)
                    - int(getattr(r, "_pd_seq_pos_offset", 0) or 0)
                    + len(r.output_ids)
                )
            sh_1d = NamedSharding(self.mesh, PartitionSpec("data"))
            sh_2d = NamedSharding(self.mesh, PartitionSpec("data", None))
            with jax.set_mesh(self.mesh):
                updated_relay = update_spec_relay_buffers_admit(
                    relay_buffers,
                    jax.device_put(fut_idx_np, sh_1d),
                    jax.device_put(val_mask_np, sh_1d),
                    jax.device_put(topk_idx_np, sh_2d),
                    jax.device_put(ver_id_np, sh_1d),
                    jax.device_put(new_seq_np, sh_1d),
                    dp_size=dp_size,
                )
            relay_owner.spec_relay_buffers = updated_relay

        if self.running_batch.is_empty():
            self.running_batch = admit_batch
        else:
            self.running_batch.merge_batch(admit_batch)
        return len(admitted_reqs)

    def _build_kv_spec_for_req(self: Scheduler, req: Req):
        """Build per-layer ShapeDtypeStructs matching P's KV layout (and indexer_kv if present)."""

        from sgl_jax.srt.disaggregation.prefill import (
            _pad_to_page_bucket,
            _page_transfer_axis_and_multiple,
            local_kv_spec_for_pool,
            local_spec_meta_sharding,
        )

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        all_pools = [kv_pool] + list(self._get_draft_kv_pools())
        total_layers = sum(int(p.layer_num) for p in all_pools)
        total_indexer_layers = sum(
            len(getattr(p, "indexer_key_buffer", None) or ()) for p in all_pools
        )
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        skip_tokens = int(getattr(req, "_pd_transfer_skip_tokens", 0) or 0)
        transfer_tokens = max(1, seqlen - skip_tokens)
        num_pages = (transfer_tokens + page_size - 1) // page_size
        _, _, min_multiple = _page_transfer_axis_and_multiple(kv_pool)
        padded_pages = _pad_to_page_bucket(num_pages, min_multiple=min_multiple)
        indexer_buffers = getattr(kv_pool, "indexer_key_buffer", None)
        use_spec_meta = self._use_pd_spec_meta_transfer()
        if use_spec_meta:
            meta_shape, meta_sharding = local_spec_meta_sharding()
            spec_meta_spec = jax.ShapeDtypeStruct(
                meta_shape,
                jnp.int32,
                sharding=meta_sharding,
            )
        else:
            spec_meta_spec = None
        if jax.process_count() > 1:
            kv_spec = local_kv_spec_for_pool(kv_pool, total_layers, padded_pages)
            if indexer_buffers and total_indexer_layers > 0:
                idx_tail = indexer_buffers[0].shape[1:]
                idx_spec = local_kv_spec_for_pool(
                    kv_pool,
                    total_indexer_layers,
                    padded_pages,
                    per_layer_tail=idx_tail,
                )
                out = {"kv": kv_spec, "indexer_kv": idx_spec}
                if spec_meta_spec is not None:
                    out["spec_meta"] = spec_meta_spec
                return out
            if spec_meta_spec is not None:
                return {"kv": kv_spec, "spec_meta": spec_meta_spec}
            return kv_spec
        per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
        shape = (padded_pages, *per_layer_tail)
        sharding = kv_pool.kv_sharding
        kv_spec = [
            jax.ShapeDtypeStruct(shape, kv_pool.dtype, sharding=sharding)
            for _ in range(total_layers)
        ]
        if indexer_buffers and total_indexer_layers > 0:
            idx_tail = indexer_buffers[0].shape[1:]
            idx_shape = (padded_pages, *idx_tail)
            idx_spec = [
                jax.ShapeDtypeStruct(idx_shape, kv_pool.dtype, sharding=sharding)
                for _ in range(total_indexer_layers)
            ]
            out = {"kv": kv_spec, "indexer_kv": idx_spec}
            if spec_meta_spec is not None:
                out["spec_meta"] = spec_meta_spec
            return out
        if spec_meta_spec is not None:
            return {"kv": kv_spec, "spec_meta": spec_meta_spec}
        return kv_spec

    def _install_received_kv(self: Scheduler, req: Req, kv_indices, kv) -> None:
        self._maybe_log_decode_pull_debug(req, kv)
        self._write_kv_to_pool(req, kv_indices, kv)
        self._record_decode_transfer_bytes(kv)

    def _commit_direct_received_kv(
        self: Scheduler,
        req: Req,
        kv_indices,
        expected_debug: dict[str, object] | None,
    ) -> None:
        """Record a backend that wrote the destination KV pages directly."""

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        all_pools = [kv_pool] + list(self._get_draft_kv_pools())
        num_pages = (len(req.origin_input_ids) + kv_pool.page_size - 1) // kv_pool.page_size
        total = 0
        for pool in all_pools:
            for layer in pool.kv_buffer:
                total += num_pages * (int(layer.nbytes) // int(layer.shape[0]))
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            PD_TRANSFER_BYTES_TOTAL.labels(direction="h2d", role="decode").inc(total)
        self._maybe_verify_direct_receive_debug(req, kv_indices, expected_debug)

    def _write_kv_to_pool(self: Scheduler, req: Req, kv_indices, kv: jax.Array) -> None:
        """Write pulled KV into the local paged pool (in place)."""

        if kv_indices is None:
            raise RuntimeError(
                f"_write_kv_to_pool: kv_indices is None for req "
                f"{req.rid!r}; allocator may have OOM'd"
            )

        indexer_kv = None
        if isinstance(kv, tuple):
            kv, indexer_kv = kv

        from sgl_jax.srt.disaggregation.prefill import _page_transfer_axis_and_multiple

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        all_pools = [kv_pool] + list(self._get_draft_kv_pools())
        pool_pspec = kv_pool.kv_sharding.spec
        is_page_axis_sharded, page_axis, _ = _page_transfer_axis_and_multiple(kv_pool)

        if jax.process_count() > 1 and getattr(kv, "is_fully_addressable", False):
            # Pulled KV is this host's local shard on a 1-D local mesh.
            # Assemble it into the global pool sharding (zero-copy: each NP
            # contributes its own addressable_shards).
            if is_page_axis_sharded:
                stacked_spec = PartitionSpec(None, page_axis, *pool_pspec[1:])
                gsh = NamedSharding(kv_pool.mesh, stacked_spec)
                per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
                gshape = (kv.shape[0], kv.shape[1] * jax.process_count()) + per_layer_tail
            else:
                stacked_spec = PartitionSpec(None, None, *pool_pspec[1:])
                gsh = NamedSharding(kv_pool.mesh, stacked_spec)
                per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
                gshape = (kv.shape[0], kv.shape[1]) + per_layer_tail
            kv = jax.make_array_from_single_device_arrays(
                gshape, gsh, [s.data for s in kv.addressable_shards]
            )
            if indexer_kv is not None and getattr(indexer_kv, "is_fully_addressable", False):
                idx_tail = kv_pool.indexer_key_buffer[0].shape[1:]
                if is_page_axis_sharded:
                    idx_gshape = (
                        indexer_kv.shape[0],
                        indexer_kv.shape[1] * jax.process_count(),
                    ) + idx_tail
                else:
                    idx_gshape = (indexer_kv.shape[0], indexer_kv.shape[1]) + idx_tail
                indexer_kv = jax.make_array_from_single_device_arrays(
                    idx_gshape, gsh, [s.data for s in indexer_kv.addressable_shards]
                )

        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        skip_tokens = int(getattr(req, "_pd_transfer_skip_tokens", 0) or 0)
        transfer_tokens = max(1, seqlen - skip_tokens)
        num_pages = (transfer_tokens + page_size - 1) // page_size
        kv_indices_np = (
            np.asarray(kv_indices) if not isinstance(kv_indices, np.ndarray) else kv_indices
        )
        padded_pages = kv[0].shape[0]
        # page_ids_padded is only consumed by the debug verifier below, which is
        # a no-op unless SGL_JAX_PD_DEBUG_KV is set. The write itself is
        # token-level via ``loc`` (or page-level scatter for page-axis-sharded
        # pools), so skip this numpy work on the production path unless needed.
        from sgl_jax.srt.disaggregation.debug_utils import kv_debug_enabled

        if kv_debug_enabled(req.rid):
            page_ids_np = kv_indices_np[::page_size] // page_size
            page_ids_np = page_ids_np[:num_pages]
            if num_pages < padded_pages:
                pad = np.full(padded_pages - num_pages, page_ids_np[-1], dtype=page_ids_np.dtype)
                page_ids_padded = np.concatenate([page_ids_np, pad])
            else:
                page_ids_padded = page_ids_np
        else:
            page_ids_padded = None

        if is_page_axis_sharded:
            from sgl_jax.srt.disaggregation.prefill import _globalize_rank_local_page_ids

            local_page_ids = kv_indices_np[: num_pages * page_size : page_size] // page_size
            if padded_pages > num_pages:
                # Page 0 of each DP rank's local slice is the reserved sentinel
                # padding page (allocated and never assigned to active requests).
                local_page_ids = np.concatenate(
                    [
                        local_page_ids,
                        np.zeros(padded_pages - num_pages, dtype=local_page_ids.dtype),
                    ]
                )
            global_page_ids = _globalize_rank_local_page_ids(
                local_page_ids,
                dp_rank=int(getattr(req, "dp_rank", 0) or 0),
                dp_size=int(getattr(kv_pool, "dp_size", 1)),
                total_pages=int(kv_pool.kv_buffer[0].shape[0]),
            )
            page_indices = jax.device_put(
                global_page_ids,
                NamedSharding(kv_pool.mesh, PartitionSpec(None)),
            )
            t_scatter0 = time.perf_counter()
            kv_cursor = 0
            idx_cursor = 0
            use_fused_scatter = (
                os.environ.get("SGL_FUSED_KV_SCATTER", "1") == "1" and padded_pages <= 256
            )
            for pool in all_pools:
                n_layers = pool.layer_num
                if use_fused_scatter and n_layers > 1:
                    try:
                        chunk_sz = 13 if n_layers % 13 == 0 else n_layers
                        for s in range(0, n_layers, chunk_sz):
                            e = min(s + chunk_sz, n_layers)
                            kv_vals = [kv[kv_cursor + j] for j in range(s, e)]
                            pool.kv_buffer[s:e] = _jit_scatter_all_layers(
                                pool.kv_buffer[s:e],
                                page_indices,
                                kv_vals,
                                pool.kv_sharding,
                            )
                    except Exception:
                        for layer_idx in range(n_layers):
                            pool.kv_buffer[layer_idx] = _jit_scatter_one_layer(
                                pool.kv_buffer[layer_idx],
                                page_indices,
                                kv[kv_cursor + layer_idx],
                                pool.kv_sharding,
                            )
                else:
                    for layer_idx in range(n_layers):
                        pool.kv_buffer[layer_idx] = _jit_scatter_one_layer(
                            pool.kv_buffer[layer_idx],
                            page_indices,
                            kv[kv_cursor + layer_idx],
                            pool.kv_sharding,
                        )
                kv_cursor += n_layers
                if indexer_kv is not None and getattr(pool, "indexer_key_buffer", None):
                    n_idx = len(pool.indexer_key_buffer)
                    if use_fused_scatter and n_idx > 1:
                        try:
                            idx_chunk = 7 if n_idx % 7 == 0 else n_idx
                            for s in range(0, n_idx, idx_chunk):
                                e = min(s + idx_chunk, n_idx)
                                idx_vals = [indexer_kv[idx_cursor + j] for j in range(s, e)]
                                pool.indexer_key_buffer[s:e] = _jit_scatter_all_layers(
                                    pool.indexer_key_buffer[s:e],
                                    page_indices,
                                    idx_vals,
                                    pool.kv_sharding,
                                )
                        except Exception:
                            for slot_idx in range(n_idx):
                                pool.indexer_key_buffer[slot_idx] = _jit_scatter_one_layer(
                                    pool.indexer_key_buffer[slot_idx],
                                    page_indices,
                                    indexer_kv[idx_cursor + slot_idx],
                                    pool.kv_sharding,
                                )
                    else:
                        for slot_idx in range(n_idx):
                            pool.indexer_key_buffer[slot_idx] = _jit_scatter_one_layer(
                                pool.indexer_key_buffer[slot_idx],
                                page_indices,
                                indexer_kv[idx_cursor + slot_idx],
                                pool.kv_sharding,
                            )
                    idx_cursor += n_idx
            if os.environ.get("SGL_DEBUG_STEP_TIMING", "0") == "1":
                scatter_Count = getattr(self, "_pd_scatter_log_count", 0) + 1
                self._pd_scatter_log_count = scatter_Count
                if scatter_Count <= 10 or scatter_Count % 25 == 0:
                    logger.info(
                        "[PD-SCATTER-TIMING] req_id=%s pages=%d/%d fused=%s dispatch_ms=%.2f",
                        req.rid,
                        num_pages,
                        padded_pages,
                        use_fused_scatter,
                        (time.perf_counter() - t_scatter0) * 1000.0,
                    )
            last_pool = all_pools[-1]
            if kv_debug_enabled(req.rid):
                if indexer_kv is not None and getattr(last_pool, "indexer_key_buffer", None):
                    jax.block_until_ready(last_pool.indexer_key_buffer[-1])
                else:
                    jax.block_until_ready(last_pool.kv_buffer[-1])
                self._maybe_verify_decode_writeback_debug(req, kv_pool, page_ids_padded, kv)
            return

        # Write via the in-place Pallas kernel (``update_fused_kv_cache_vectorized``
        # with ``input_output_aliases``), so the footprint scales with the tokens
        # written. ``loc`` is per-token absolute pool slots; -1 marks padding
        # tokens that are skipped.
        total_tokens = padded_pages * page_size
        loc_np = np.full(total_tokens, -1, dtype=np.int32)
        loc_np[:seqlen] = kv_indices_np[:seqlen]
        loc = jax.device_put(
            jnp.asarray(loc_np),
            NamedSharding(kv_pool.mesh, PartitionSpec(kv_pool.attention_data_partition_axis)),
        )

        kv_cursor = 0
        idx_cursor = 0
        for pool in all_pools:
            for layer_idx in range(pool.layer_num):
                pool.kv_buffer[layer_idx] = write_kv_layer(
                    kv[kv_cursor],
                    loc,
                    pool.kv_buffer[layer_idx],
                    page_size,
                    pool.kv_partition_axis,
                    pool.attention_data_partition_axis,
                    pool.mesh,
                )
                kv_cursor += 1
            if indexer_kv is not None and getattr(pool, "indexer_key_buffer", None):
                for slot_idx in range(len(pool.indexer_key_buffer)):
                    pool.indexer_key_buffer[slot_idx] = write_kv_layer(
                        indexer_kv[idx_cursor],
                        loc,
                        pool.indexer_key_buffer[slot_idx],
                        page_size,
                        pool.kv_partition_axis,
                        pool.attention_data_partition_axis,
                        pool.mesh,
                    )
                    idx_cursor += 1
        self._maybe_verify_decode_writeback_debug(req, kv_pool, page_ids_padded, kv)

    def _set_decode_bookkeeping(self: Scheduler, req: Req, kv_indices) -> None:
        if kv_indices is None:
            raise RuntimeError(f"missing KV indices for req_id={req.rid!r}")
        kv_indices_np = (
            np.asarray(kv_indices) if not isinstance(kv_indices, np.ndarray) else kv_indices
        )
        pos_off = int(getattr(req, "_pd_seq_pos_offset", 0) or 0)
        seqlen = len(req.origin_input_ids) - pos_off
        # The unused received tail page may already have been reclaimed.
        # Slice by prompt length rather than dropping another retained slot.
        req.prefix_indices = kv_indices_np[: max(seqlen - 1, 0)]
        req.last_matched_prefix_len = len(req.prefix_indices)
        req._pd_skip_prefix_match = True
        req._pd_prealloc_kv_indices = kv_indices_np
        req.fill_ids = list(req.origin_input_ids[pos_off:]) + list(req.output_ids)

    def _enqueue_for_decode(self: Scheduler, req: Req) -> None:
        """Put ``req`` into the scheduler's decode-ready queue."""

        if req not in self.waiting_queue:
            self.waiting_queue.append(req)

    def _release_decode_req_resources(self: Scheduler, req: Req) -> None:
        """Best-effort release of req_to_token_pool slot. Does NOT
        call cache_finished_req (req never went through prefill).
        """

        if req.req_pool_idx is None:
            return
        try:
            self.req_to_token_pool.free(req)
        except Exception:
            logger.exception(
                "failed to free req_to_token_pool slot %d for req_id=%s",
                req.req_pool_idx,
                req.rid,
            )

    def _abort_decode_request(
        self: Scheduler,
        req: Req,
        reason: str,
        *,
        cleanup_transfer: bool = True,
    ) -> None:
        """Release resources AND send AbortReq back to tokenizer."""

        manager = getattr(self, "disagg_kv_manager", None)
        room = getattr(req, "bootstrap_room", None)
        if manager is not None and cleanup_transfer:
            try:
                prefill_dp_rank = _request_dp_rank(req, self.dp_size, prefill=True)
            except Exception:
                logger.warning(
                    "cannot resolve Prefill DP rank while cleaning up aborted "
                    "req_id=%s bootstrap_room=%s reason=%s; transfer metadata "
                    "will expire by TTL",
                    req.rid,
                    room,
                    reason,
                    exc_info=True,
                )
            else:
                try:
                    manager.cleanup_transfer(
                        room,
                        jax_process_index=getattr(req, "disagg_peer_process_index", None),
                        prefill_dp_rank=prefill_dp_rank,
                        expected_transfer_id=get_disagg_transport_id(req),
                    )
                except Exception:
                    logger.warning(
                        "failed to clean up transfer metadata for req_id=%s "
                        "bootstrap_room=%s prefill_dp_rank=%s reason=%s",
                        req.rid,
                        room,
                        prefill_dp_rank,
                        reason,
                        exc_info=True,
                    )
        self._release_decode_req_resources(req)
        try:
            from sgl_jax.srt.managers.io_struct import AbortReq

            abort_out = AbortReq(rid=req.rid)
            if self._comm_backend is not None:
                self._comm_backend.send_pyobj(abort_out)
            else:
                self.send_to_tokenizer.send_pyobj(abort_out)
        except Exception:
            logger.exception(
                "failed to send AbortReq for req_id=%s (reason=%s)",
                req.rid,
                reason,
            )

    def _record_decode_transfer_failure(self, reason: str) -> None:
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_FAILURES_TOTAL,
            )

            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()

    def _record_decode_transfer_bytes(self, kv) -> None:
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            if kv is None:
                return
            leaves = jax.tree.leaves(kv)
            total = int(sum(int(x.nbytes) for x in leaves))
            PD_TRANSFER_BYTES_TOTAL.labels(direction="h2d", role="decode").inc(total)

    def _maybe_log_decode_pull_debug(self, req: Req, kv) -> None:
        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            kv_debug_enabled,
        )

        if not kv_debug_enabled(req.rid):
            return

        if isinstance(kv, tuple):
            kv = kv[0]
        snapshot = build_kv_debug_snapshot(kv)
        logger.warning(
            "PD-KV-DEBUG decode_pull req_id=%s shape=%s dtype=%s sharding=%s digest=%s sample=%s",
            req.rid,
            snapshot.shape,
            snapshot.dtype,
            snapshot.sharding,
            snapshot.global_digest,
            snapshot.sample_page_digests(),
        )

    def _maybe_verify_decode_writeback_debug(self, req: Req, kv_pool, page_ids_padded, kv) -> None:
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            count_kv_debug_mismatches,
            find_first_kv_debug_mismatch,
            kv_debug_enabled,
        )
        from sgl_jax.srt.disaggregation.prefill import _jit_gather_all_layers

        if not kv_debug_enabled(req.rid):
            return

        page_ids_jax = jax.device_put(
            page_ids_padded,
            NamedSharding(kv_pool.mesh, PartitionSpec(None)),
        )
        gather_out_sharding = NamedSharding(
            kv_pool.mesh,
            PartitionSpec(None, *kv_pool.kv_sharding.spec[1:]),
        )
        get_buf = getattr(kv_pool, "get_fused_kv_buffer", kv_pool.get_kv_buffer)
        layer_buffers = [
            get_buf(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        readback = jnp.stack(
            _jit_gather_all_layers(
                layer_buffers,
                page_ids_jax,
                gather_out_sharding,
            ),
            axis=0,
        )

        expected = build_kv_debug_snapshot(kv)
        actual = build_kv_debug_snapshot(readback)
        mismatch_count = count_kv_debug_mismatches(expected, actual)
        first_mismatch = find_first_kv_debug_mismatch(expected, actual)

        logger.warning(
            "PD-KV-DEBUG decode_writeback req_id=%s expected_digest=%s "
            "readback_digest=%s mismatch_count=%d first_mismatch=%s "
            "expected_sample=%s readback_sample=%s page_ids=%s",
            req.rid,
            expected.global_digest,
            actual.global_digest,
            mismatch_count,
            first_mismatch,
            expected.sample_page_digests(),
            actual.sample_page_digests(),
            page_ids_padded.tolist(),
        )

    def _maybe_verify_direct_receive_debug(
        self: Scheduler,
        req: Req,
        kv_indices,
        expected_metadata: dict[str, object] | None,
    ) -> None:
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.disaggregation.base.transfer import slots_to_page_ids
        from sgl_jax.srt.disaggregation.debug_utils import (
            KVDebugSnapshot,
            build_kv_debug_snapshot,
            count_kv_debug_mismatches,
            find_first_kv_debug_mismatch,
            kv_debug_enabled,
        )
        from sgl_jax.srt.disaggregation.prefill import _jit_gather_all_layers

        if not kv_debug_enabled(req.rid):
            return

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_ids = np.asarray(
            slots_to_page_ids(kv_indices, kv_pool.page_size, len(req.origin_input_ids)),
            dtype=np.int32,
        )
        page_ids_jax = jax.device_put(
            page_ids,
            NamedSharding(kv_pool.mesh, PartitionSpec(None)),
        )
        gather_out_sharding = NamedSharding(
            kv_pool.mesh,
            PartitionSpec(None, *kv_pool.kv_sharding.spec[1:]),
        )
        layer_buffers = [
            kv_pool.get_kv_buffer(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        readback = jnp.stack(
            _jit_gather_all_layers(layer_buffers, page_ids_jax, gather_out_sharding),
            axis=0,
        )
        actual = build_kv_debug_snapshot(readback)
        if expected_metadata is None:
            logger.warning(
                "PD-KV-DEBUG raiden_direct req_id=%s digest=%s sample=%s "
                "expected=missing page_ids=%s",
                req.rid,
                actual.global_digest,
                actual.sample_page_digests(),
                page_ids.tolist(),
            )
            return

        expected = KVDebugSnapshot(
            shape=tuple(int(value) for value in expected_metadata.get("shape", ())),
            dtype=str(expected_metadata.get("dtype", "")),
            sharding="prefill",
            global_digest=str(expected_metadata.get("global_digest", "")),
            page_digests=tuple(
                tuple(str(digest) for digest in row)
                for row in expected_metadata.get("page_digests", ())
            ),
        )
        mismatch_count = count_kv_debug_mismatches(expected, actual)
        first_mismatch = find_first_kv_debug_mismatch(expected, actual)
        logger.warning(
            "PD-KV-DEBUG raiden_direct req_id=%s expected_digest=%s "
            "readback_digest=%s mismatch_count=%d first_mismatch=%s "
            "expected_sample=%s readback_sample=%s page_ids=%s",
            req.rid,
            expected.global_digest,
            actual.global_digest,
            mismatch_count,
            first_mismatch,
            expected.sample_page_digests(),
            actual.sample_page_digests(),
            page_ids.tolist(),
        )
