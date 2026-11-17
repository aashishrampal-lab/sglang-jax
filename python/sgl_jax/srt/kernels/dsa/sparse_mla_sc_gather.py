"""Token-level SparseCore KV Gather + Double-Buffered TensorCore MLA Attention.

Supports both:
1. Native 576 B/tok SparseCore ``uint32`` MLA KV cache ``[Pn, page_size + page_size // 8, 128]``
   (``DSA_NATIVE_SC_KV=1``):
   - Zero ``%sparse-core-data-format-call`` relayouts (``use_tc_tiling_on_sc=False``).
   - 1.0x HBM read efficiency (exact 512B NOPE + 64B ROPE per selected token vs 2560B/tok).
   - Overlapped NOPE async HBM DMA + 6-op SWAR ``transpose_quad`` ROPE unpack on SparseCore.
   - Zero-cycle ``pltpu.bitcast`` unpack inside double-buffered split-Q TensorCore MLA.
2. Legacy 4D TensorCore-tiled MLA KV cache ``[Pn, page_size // pk, pk, 640]`` fallback.
"""

from __future__ import annotations

import functools
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import api as jax_api
from jax._src import core as jax_core
from jax._src.pallas import core as pl_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import paged_write_back
from sgl_jax.srt.kernels.sparse_core import core_map_helper
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

DEFAULT_SC_GATHER_CHUNK_SIZE = int(os.environ.get("DSA_SC_GATHER_CHUNK_SIZE", "64"))
DEFAULT_ATTN_KERNEL_BATCH_SIZE = int(os.environ.get("DSA_SC_ATTN_BATCH_SIZE", "16"))
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

_GATHER_PT_CHUNK = 128
_SC_NUM_CORES = 2
_SC_NUM_SUBCORES = 16
_SC_NUM_LANES = 16
_ROPE_PACKING = 4
_QUADS_PER_ATOM = 16
_OCCURRENCES_PER_ATOM = _QUADS_PER_ATOM * _ROPE_PACKING  # 64
_WORD_BYTES = 4
_SKIP_ROW = -1


def cdiv(a, b):
    return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def _largest_divisor(x: int, cap: int) -> int:
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def _floor_divmod_const(x: jax.Array, d: int) -> tuple[jax.Array, jax.Array]:
    """Exact integer ``(x // d, x % d)`` without hardware vector ``idiv``."""
    if d <= 0:
        raise ValueError(f"divisor must be positive, got {d}")
    if d & (d - 1) == 0:
        shift = d.bit_length() - 1
        return lax.shift_right_arithmetic(x, jnp.int32(shift)), x & jnp.int32(d - 1)

    inv = np.float32(1.0 / d)
    q = jnp.floor(x.astype(jnp.float32) * inv).astype(jnp.int32)
    r = x - q * jnp.int32(d)
    q2 = jnp.floor(r.astype(jnp.float32) * inv).astype(jnp.int32)
    q = q + q2
    r = r - q2 * jnp.int32(d)
    lo = r < 0
    q = jnp.where(lo, q - 1, q)
    r = jnp.where(lo, r + jnp.int32(d), r)
    hi = r >= d
    q = jnp.where(hi, q + 1, q)
    r = jnp.where(hi, r - jnp.int32(d), r)
    return q, r


def _gather_pt_1d_kernel(pt_ref, idx_ref, out_ref, *, num_chunks: int):
    """Resolve ``page_indices[pt_idx]`` in TensorCore VMEM via 128-lane permutes."""
    idx = idx_ref[...]
    out = jnp.zeros_like(idx)
    for c in range(num_chunks):
        chunk = pt_ref[:, c * _GATHER_PT_CHUNK : (c + 1) * _GATHER_PT_CHUNK]
        local = idx - c * _GATHER_PT_CHUNK
        gathered = jnp.take_along_axis(
            chunk, jnp.clip(local, 0, _GATHER_PT_CHUNK - 1), axis=1
        )
        out = jnp.where((local >= 0) & (local < _GATHER_PT_CHUNK), gathered, out)
    out_ref[...] = out


def gather_pt_1d_pallas(
    page_indices: jax.Array,  # i32[PT_len]
    pt_idx: jax.Array,  # i32[T_q, topk]
    *,
    block_tokens: int = 8,
) -> jax.Array:
    """Fast VMEM replacement for scalar HBM ``page_indices[pt_idx]``."""
    t_q, topk = pt_idx.shape
    pt_len = page_indices.shape[0]
    if (
        not is_tpu_runtime()
        or topk % _GATHER_PT_CHUNK != 0
        or pt_len <= 0
        or pt_len > 16384
        or os.environ.get("DSA_PALLAS_PT_GATHER", "1") != "1"
    ):
        return page_indices[pt_idx]

    num_chunks = cdiv(pt_len, _GATHER_PT_CHUNK)
    padded_pt_len = num_chunks * _GATHER_PT_CHUNK
    pt_1d = (
        jnp.pad(page_indices, (0, padded_pt_len - pt_len))
        if padded_pt_len != pt_len
        else page_indices
    )
    pt_2d = jnp.broadcast_to(pt_1d[None, :], (block_tokens, padded_pt_len))

    padded_t = align_to(t_q, block_tokens)
    idx_in = (
        jnp.pad(pt_idx, ((0, padded_t - t_q), (0, 0))) if padded_t != t_q else pt_idx
    )

    out = pl.pallas_call(
        functools.partial(_gather_pt_1d_kernel, num_chunks=num_chunks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, padded_pt_len), lambda t: (0, 0)),
                pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            ],
            out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            grid=(padded_t // block_tokens,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_t, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name="gather_pt_1d",
    )(pt_2d, idx_in)
    return out[:t_q]


def prepare_sc_gather_metadata(
    topk_indices: jax.Array,  # i32[T_q, topk]
    seq_lens: jax.Array,  # i32[S]
    page_indices: jax.Array,  # i32[PT_len] (packed via cu_kv_lens // page_size)
    cu_q_lens: jax.Array,  # i32[S + 1]
    cu_kv_lens: jax.Array,  # i32[S + 1]
    *,
    page_size: int,
    total_num_pages: int,
    q_token_offset: int | jax.Array = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build ``(valid_kv_lens [T_q], physical_indices [T_q, topk], token_end [])``."""
    T_q, topk = topk_indices.shape
    num_seqs = seq_lens.shape[0]
    q_off = jnp.asarray(q_token_offset, dtype=jnp.int32)

    tok = jnp.arange(T_q, dtype=jnp.int32) + q_off
    seg = jnp.clip(
        jnp.searchsorted(cu_q_lens, tok, side="right").astype(jnp.int32) - 1,
        0,
        max(0, num_seqs - 1),
    )
    q_start = cu_q_lens[seg]
    q_end = cu_q_lens[seg + 1]
    kv_len = seq_lens[seg]
    total_q = cu_q_lens[num_seqs]
    active_q = (tok >= 0) & (tok < total_q) & (tok >= q_start) & (tok < q_end) & (kv_len > 0)

    safe_topk = jnp.maximum(topk_indices, 0)
    seq_page, token_offset = _floor_divmod_const(safe_topk, page_size)
    pt_base = cu_kv_lens[seg] // page_size
    pt_end = cu_kv_lens[seg + 1] // page_size
    pt_unclipped = pt_base[:, None] + seq_page
    valid_slot = (
        active_q[:, None]
        & (topk_indices >= 0)
        & (topk_indices < kv_len[:, None])
        & (pt_unclipped < pt_end[:, None])
        & (pt_unclipped < page_indices.shape[0])
    )

    pt_idx = jnp.clip(pt_unclipped, 0, max(0, page_indices.shape[0] - 1))
    phys_page = jnp.clip(
        gather_pt_1d_pallas(page_indices, pt_idx),
        0,
        max(0, total_num_pages - 1),
    )

    flat_idx = jnp.arange(T_q * topk, dtype=jnp.int32).reshape(T_q, topk)
    _, scattered_page = _floor_divmod_const(flat_idx * 104729, max(1, total_num_pages))
    _, scattered_offset = _floor_divmod_const(flat_idx * 15485863, page_size)

    page_ids = jnp.where(valid_slot, phys_page, scattered_page)
    tok_offs = jnp.where(valid_slot, token_offset, scattered_offset)
    physical_indices = (page_ids * page_size + tok_offs).astype(jnp.int32)

    valid_kv_lens = jnp.sum(valid_slot, axis=-1).astype(jnp.int32)
    token_end = jnp.clip(total_q - q_off, 0, T_q).astype(jnp.int32)
    return valid_kv_lens, physical_indices, token_end


# ---------------------------------------------------------------------------
# Unified 576 B/tok Native SparseCore KV Pack / Scatter / Gather / MLA
# ---------------------------------------------------------------------------


@register_pytree_node_class
class NativeSCKVBuffer:
    """Split native-SparseCore 576 B/tok MLA KV buffer pair ``(nope, rope)``.

    ``nope`` has shape ``u32[Pn, page_size, 128]`` (physical HBM tile ``(8, 128)``)
    and ``rope`` has shape ``u32[Pn, page_size // 8, 128]`` (reshaped to ``(-1, 16)``
    with physical HBM tile ``(8, 16)`` inside SparseCore scatter/gather). Keeping
    two separate ``jax.Array`` leaves in a registered PyTree avoids conflicting HBM
    tilings on a single buffer while preserving 1 entry per layer in ``kv_buffer``
    and automatic JAX buffer donation / ``shard_map`` partitioning.
    """

    __slots__ = ("nope", "rope")

    def __init__(self, nope: Any, rope: Any):
        self.nope = nope
        self.rope = rope

    def tree_flatten(self):
        return (self.nope, self.rope), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(children[0], children[1])

    @property
    def dtype(self):
        return self.nope.dtype

    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nope.shape[0], self.nope.shape[1] + self.rope.shape[1], 128)

    @property
    def nbytes(self) -> int:
        return int(self.nope.nbytes) + int(self.rope.nbytes)

    @property
    def sharding(self):
        return getattr(self.nope, "sharding", None)

    def __getitem__(self, idx):
        return NativeSCKVBuffer(self.nope[idx], self.rope[idx])


def pack_tokens_u32(values: jax.Array, token_bytes: int) -> jax.Array:
    """Pack ``[T, token_bytes]`` FP8 tokens into 4-band ``[T, token_bytes // 4]`` ``uint32``."""
    if token_bytes % _WORD_BYTES != 0:
        raise ValueError(f"token_bytes ({token_bytes}) must be divisible by {_WORD_BYTES}")
    u8 = lax.bitcast_convert_type(values.astype(jnp.float8_e4m3fn), jnp.uint8)
    if u8.shape[-1] < token_bytes:
        u8 = jnp.pad(u8, ((0, 0), (0, token_bytes - u8.shape[-1])))
    elif u8.shape[-1] > token_bytes:
        u8 = u8[:, :token_bytes]
    words_per_tok = token_bytes // _WORD_BYTES
    bands = u8.reshape(u8.shape[0], _WORD_BYTES, words_per_tok).astype(jnp.uint32)
    return (
        bands[:, 0]
        | (bands[:, 1] << 8)
        | (bands[:, 2] << 16)
        | (bands[:, 3] << 24)
    )


def unpack_tokens_u32_to_fp8(words: jax.Array) -> jax.Array:
    """Pure-JAX inverse of ``pack_tokens_u32`` for CPU parity tests."""
    b0 = (words & jnp.uint32(0xFF)).astype(jnp.uint8)
    b1 = ((words >> 8) & jnp.uint32(0xFF)).astype(jnp.uint8)
    b2 = ((words >> 16) & jnp.uint32(0xFF)).astype(jnp.uint8)
    b3 = ((words >> 24) & jnp.uint32(0xFF)).astype(jnp.uint8)
    u8 = jnp.concatenate([b0, b1, b2, b3], axis=-1)
    return lax.bitcast_convert_type(u8, jnp.float8_e4m3fn)


def _unpack_unified_native_sc_kv_to_2d(
    cache_kv: NativeSCKVBuffer | jax.Array,
    *,
    page_size: int,
) -> jax.Array:
    """Unpack native-SC ``uint32`` cache into ``[Pn * page_size, 640]`` FP8 (CPU ref)."""
    if isinstance(cache_kv, NativeSCKVBuffer):
        pn = cache_kv.nope.shape[0]
        nope_u32 = cache_kv.nope.reshape(pn * page_size, 128)
        rope_u32 = cache_kv.rope.reshape(pn * page_size, 16)
    else:
        pn = cache_kv.shape[0]
        nope_u32 = cache_kv[:, :page_size, :].reshape(pn * page_size, 128)
        rope_u32 = cache_kv[:, page_size:, :].reshape(pn * page_size, 16)
    nope_fp8 = unpack_tokens_u32_to_fp8(nope_u32)  # [Pn*ps, 512]
    rope_fp8 = unpack_tokens_u32_to_fp8(rope_u32)  # [Pn*ps, 64]
    pad_zeros = jnp.zeros((pn * page_size, 64), dtype=jnp.float8_e4m3fn)
    return jnp.concatenate([nope_fp8, rope_fp8, pad_zeros], axis=-1)


class _ScatterPlan(NamedTuple):
    lanes_per_block: int
    cores_per_block: int
    num_cores: int
    steps_per_core: int

    @property
    def lanes_per_core(self) -> int:
        return self.lanes_per_block // self.cores_per_block

    @property
    def blocks_per_round(self) -> int:
        return self.num_cores // self.cores_per_block

    @property
    def total_tokens(self) -> int:
        return self.steps_per_core * self.blocks_per_round * self.lanes_per_block


def _make_scatter_plan(num_tokens: int, num_lanes: int, num_cores: int) -> _ScatterPlan:
    num_blocks = pl.cdiv(max(1, num_tokens), num_lanes)
    spare = min(num_cores // num_blocks, num_lanes)
    cores_per_block = 1 << (int(spare).bit_length() - 1) if spare > 0 else 1
    return _ScatterPlan(
        lanes_per_block=num_lanes,
        cores_per_block=cores_per_block,
        num_cores=num_cores,
        steps_per_core=pl.cdiv(num_blocks, num_cores // cores_per_block),
    )


def _native_sc_scatter_kernel(
    dst_hbm_ref: Any,
    src_ref: Any,
    cache_ref: Any,
    sem: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    plan: _ScatterPlan,
) -> None:
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))
    num_rows = cache_ref.shape[0]
    block_slot = core_index // plan.cores_per_block
    lane_group = lax.rem(core_index, plan.cores_per_block)

    def block_index(step):
        return step * plan.blocks_per_round + block_slot

    def body(dst_ref, src_buf):
        dst_rows = dst_ref[...]
        copies = []
        for i in range(plan.lanes_per_block):
            dst = dst_rows[i]
            write = jnp.logical_and(dst >= 0, dst < num_rows)
            if plan.cores_per_block > 1:
                write = jnp.logical_and(write, lane_group == i // plan.lanes_per_core)
            row = jnp.clip(dst, 0, num_rows - 1)
            copy = pltpu.make_async_copy(
                src_buf.at[pl.ds(i, 1)],
                cache_ref.at[pl.ds(row, 1)],
                sem.at[i],
            )
            copies.append((write, copy))

            @pl.when(write)
            def _(copy=copy):
                copy.start()

        for write, copy in copies:

            @pl.when(write)
            def _(copy=copy):
                copy.wait()

    pltpu.emit_pipeline(
        body,
        grid=(plan.steps_per_core,),
        tiling=pltpu.Tiling.SPARSE_CORE,
        in_specs=(
            pl.BlockSpec((plan.lanes_per_block,), lambda r: (block_index(r),)),
            pl.BlockSpec(
                (plan.lanes_per_block, src_ref.shape[-1]),
                lambda r: (block_index(r), 0),
            ),
        ),
    )(dst_hbm_ref, src_ref)


@functools.partial(jax.jit, donate_argnums=(0,), static_argnames=("name",))
def scatter_rows_native_sc(
    cache: jax.Array,  # u32[Pn, ..., 128]
    src_2d: jax.Array,  # u32[num_tokens, row_width]
    dst_rows: jax.Array,  # i32[num_tokens]
    *,
    name: str = "scatter_native_sc_kv",
) -> jax.Array:
    """In-place SparseCore DMA scatter of ``uint32`` rows with ``use_tc_tiling_on_sc=False``."""
    row_width = src_2d.shape[-1]
    if not is_tpu_runtime() or pltpu.get_tpu_info().sparse_core is None:
        cache_2d = cache.reshape(-1, row_width)
        valid = (dst_rows >= 0) & (dst_rows < cache_2d.shape[0])
        safe_dst = jnp.where(valid, dst_rows, cache_2d.shape[0])
        updated = cache_2d.at[safe_dst].set(
            src_2d, mode="drop", wrap_negative_indices=False
        )
        return updated.reshape(cache.shape)

    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    num_cores = sc_info.num_cores * sc_info.num_subcores
    plan = _make_scatter_plan(src_2d.shape[0], sc_info.num_lanes, num_cores)

    pad_tokens = plan.total_tokens - src_2d.shape[0]
    if pad_tokens > 0:
        dst_rows = jnp.pad(dst_rows, (0, pad_tokens), constant_values=_SKIP_ROW)
        src_2d = jnp.pad(src_2d, ((0, pad_tokens), (0, 0)))

    new_ref_fn = jax.new_ref if hasattr(jax, "new_ref") else jax_core.new_ref
    cache_ref = new_ref_fn(cache.reshape(-1, row_width))

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    pl.kernel(
        functools.partial(
            _native_sc_scatter_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            plan=plan,
        ),
        out_type=(),
        scratch_types=(pltpu.SemaphoreType.DMA((plan.lanes_per_block,)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=False,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name=name,
    )(dst_rows, src_2d.reshape(-1, row_width), cache_ref)

    frozen = jax.freeze(cache_ref) if hasattr(jax, "freeze") else cache_ref[...]
    return frozen.reshape(cache.shape)


def scatter_unified_native_sc_kv(
    cache_kv: NativeSCKVBuffer | jax.Array,
    new_kv_c: jax.Array,  # [T_w, 512]
    new_k_pe: jax.Array,  # [T_w, 64]
    loc: jax.Array,  # i32[T_w]
    *,
    page_size: int,
) -> NativeSCKVBuffer | jax.Array:
    """Pack and scatter ``(new_kv_c, new_k_pe)`` into native 576 B/tok ``uint32`` cache."""
    if isinstance(cache_kv, NativeSCKVBuffer):
        pn = cache_kv.nope.shape[0]
        loc_i32 = loc.astype(jnp.int32)
        safe_loc = jnp.maximum(loc_i32, 0)
        page, slot = _floor_divmod_const(safe_loc, page_size)
        valid = (loc_i32 >= 0) & (page < pn)
        dst_rows = jnp.where(valid, page * page_size + slot, _SKIP_ROW).astype(jnp.int32)

        src_nope = pack_tokens_u32(new_kv_c, 512)  # u32[T_w, 128]
        src_rope = pack_tokens_u32(new_k_pe.reshape(new_kv_c.shape[0], -1), 64)  # u32[T_w, 16]

        new_nope = scatter_rows_native_sc(
            cache_kv.nope,
            src_nope,
            dst_rows,
            name="scatter_nope_native_sc",
        )
        new_rope = scatter_rows_native_sc(
            cache_kv.rope,
            src_rope,
            dst_rows,
            name="scatter_rope_native_sc",
        )
        return NativeSCKVBuffer(new_nope, new_rope)

    orig_shape = cache_kv.shape
    pn = orig_shape[0]
    rows_per_page = page_size + page_size // 8
    loc_i32 = loc.astype(jnp.int32)
    safe_loc = jnp.maximum(loc_i32, 0)
    page, slot = _floor_divmod_const(safe_loc, page_size)
    valid = (loc_i32 >= 0) & (page < pn)
    dst_nope = jnp.where(valid, page * rows_per_page + slot, _SKIP_ROW).astype(jnp.int32)
    dst_rope = jnp.where(
        valid,
        page * (rows_per_page * 8) + page_size * 8 + slot,
        _SKIP_ROW,
    ).astype(jnp.int32)
    src_nope = pack_tokens_u32(new_kv_c, 512)
    src_rope = pack_tokens_u32(new_k_pe.reshape(new_kv_c.shape[0], -1), 64)
    cache_128 = scatter_rows_native_sc(cache_kv.reshape(-1, 128), src_nope, dst_nope)
    cache_16 = scatter_rows_native_sc(cache_128.reshape(-1, 16), src_rope, dst_rope)
    return cache_16.reshape(orig_shape)


def _delta_swap(x, y, shift, mask):
    tmp = jnp.bitwise_and(jnp.bitwise_xor(jnp.right_shift(x, shift), y), mask)
    return (
        jnp.bitwise_xor(x, jnp.left_shift(tmp, shift)),
        jnp.bitwise_xor(y, tmp),
    )


def transpose_quad(data: list[Any]) -> list[Any]:
    """Four tokens' banded words -> four bands of token-interleaved words."""
    data = list(data)
    low_16 = jnp.uint32(0x0000FFFF)
    low_byte_each_half = jnp.uint32(0x00FF00FF)
    data[0], data[2] = _delta_swap(data[0], data[2], 16, low_16)
    data[1], data[3] = _delta_swap(data[1], data[3], 16, low_16)
    data[0], data[1] = _delta_swap(data[0], data[1], 8, low_byte_each_half)
    data[2], data[3] = _delta_swap(data[2], data[3], 8, low_byte_each_half)
    return data


def _unified_native_sc_gather_kernel(
    nope_hbm_ref: Any,
    rope_hbm_ref: Any,
    indices_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    sem_ref: Any,
    live_vmem_ref: Any = None,
    live_sem_ref: Any = None,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_owners: int,
    atoms_per_batch: int,
    batches_per_owner: int,
    wait_mode: str,
    serve_pipelined: bool,
    live_hbm_ref: Any = None,
    guard_tail: bool = False,
    rope_words: int = 16,
):
    owner = lax.axis_index((core_axis_name, subcore_axis_name))

    live_batches = None
    if live_hbm_ref is not None:
        cp = pltpu.make_async_copy(live_hbm_ref, live_vmem_ref, live_sem_ref.at[0])
        cp.start()
        cp.wait()
        live_batches = live_vmem_ref[pl.ds(0, _SC_NUM_LANES)][0]

    def serve_batch(indices_ref):
        owner_batch = pl.program_id(0)
        logical_batch = owner + owner_batch * num_owners
        occurrence_base = logical_batch * atoms_per_batch * _OCCURRENCES_PER_ATOM

        def values(atom, stream):
            return indices_ref[0, atom, stream, :]

        def destinations(atom, stream):
            return pl.ds(
                occurrence_base + atom * _OCCURRENCES_PER_ATOM + stream * _SC_NUM_LANES,
                _SC_NUM_LANES,
            )

        def body(*refs):
            atom = pl.program_id(0)
            nope_inputs = refs[:_ROPE_PACKING]
            rope_inputs = refs[_ROPE_PACKING : 2 * _ROPE_PACKING]
            rope_output = refs[-1]

            nope_copies = []
            for stream in range(_ROPE_PACKING):
                copy = pltpu.make_async_copy(
                    nope_inputs[stream],
                    nope_out_hbm_ref.at[destinations(atom, stream), :],
                    sem_ref.at[stream],
                )
                copy.start()
                nope_copies.append(copy)
            if wait_mode == "before_rope":
                for copy in nope_copies:
                    copy.wait()

            for row in range(_QUADS_PER_ATOM):
                data = []
                for token_slot in range(_ROPE_PACKING):
                    logical_rank = row * _ROPE_PACKING + token_slot
                    input_stream = logical_rank // _SC_NUM_LANES
                    input_lane = logical_rank % _SC_NUM_LANES
                    data.append(
                        rope_inputs[input_stream][
                            pl.ds(input_lane, 1), pl.ds(0, rope_words)
                        ]
                    )

                data = transpose_quad(data)
                for band in range(_ROPE_PACKING):
                    rope_output[
                        pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                    ] = data[band]
                for band in range(_ROPE_PACKING, 128 // rope_words):
                    rope_output[
                        pl.ds(row, 1), pl.ds(band * rope_words, rope_words)
                    ] = jnp.zeros((1, rope_words), jnp.uint32)

            if wait_mode == "overlap_rope":
                for copy in nope_copies:
                    copy.wait()

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(_SC_NUM_LANES), 128),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(_ROPE_PACKING)
        )
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(_SC_NUM_LANES), rope_words),
                lambda atom, stream=stream: (values(atom, stream), 0),
            )
            for stream in range(_ROPE_PACKING)
        )
        rope_out_spec = pl.BlockSpec(
            (_QUADS_PER_ATOM, 128),
            lambda atom: (logical_batch * atoms_per_batch + atom, 0),
        )

        def run_inner():
            pltpu.emit_pipeline(
                body,
                grid=(atoms_per_batch,),
                in_specs=nope_specs + rope_specs,
                out_specs=(rope_out_spec,),
                no_pipelining=not serve_pipelined,
            )(
                *([nope_hbm_ref] * _ROPE_PACKING),
                *([rope_hbm_ref] * _ROPE_PACKING),
                rope_out_hbm_ref,
            )

        if live_batches is not None and guard_tail:
            pl.when(logical_batch < live_batches)(run_inner)
        else:
            run_inner()

    indices_spec = pl.BlockSpec(
        (1, atoms_per_batch, _ROPE_PACKING, _SC_NUM_LANES),
        lambda batch: (owner + batch * num_owners, 0, 0, 0),
    )
    if live_batches is None:
        pltpu.emit_pipeline(
            serve_batch,
            grid=(batches_per_owner,),
            in_specs=(indices_spec,),
        )(indices_hbm_ref)
    else:
        owner_batches = live_vmem_ref[pl.ds(_SC_NUM_LANES, _SC_NUM_LANES)][0]

        @pl.when(owner_batches > 0)
        def _serve_live():
            pltpu.emit_pipeline(
                serve_batch,
                grid=(owner_batches,),
                in_specs=(indices_spec,),
            )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_size",
        "atoms_per_batch",
        "wait_mode",
        "serve_pipelined",
        "live_multiple",
    ),
)
def dsa_gather_unified_native_sc(
    nope_cache: jax.Array,  # u32[Pn, page_size, 128]
    rope_cache: jax.Array,  # u32[Pn, page_size // 8, 128]
    indices: jax.Array,  # i32[cur_chunk, topk]
    *,
    out_size: int,
    atoms_per_batch: int = 16,
    wait_mode: str = "overlap_rope",
    serve_pipelined: bool = True,
    num_live_indices: jax.Array | None = None,
    live_multiple: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Gather NOPE + SWAR-transposed ROPE from split native-SC 576 B/tok ``uint32`` caches."""
    nope_rows_view = nope_cache.reshape(-1, 128)
    num_tokens = nope_rows_view.shape[0]
    rope_words = rope_cache.size // num_tokens
    rope_rows_view = rope_cache.reshape(-1, rope_words)

    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core if tpu_info is not None else None
    num_cores = sc_info.num_cores if sc_info is not None else _SC_NUM_CORES
    num_subcores = sc_info.num_subcores if sc_info is not None else _SC_NUM_SUBCORES
    num_owners = num_cores * num_subcores

    max_units = out_size // (num_owners * _OCCURRENCES_PER_ATOM)
    atoms_per_batch = _largest_divisor(max_units, atoms_per_batch)
    occurrences_per_batch = atoms_per_batch * _OCCURRENCES_PER_ATOM
    owner_batch_size = num_owners * occurrences_per_batch
    num_batches = out_size // occurrences_per_batch
    batches_per_owner = num_batches // num_owners

    live = num_live_indices is not None
    scratch_types: dict[str, Any] = {"sem_ref": pltpu.SemaphoreType.DMA((_ROPE_PACKING,))}
    operands = [
        nope_rows_view,
        rope_rows_view,
        indices.reshape(num_batches, atoms_per_batch, _ROPE_PACKING, _SC_NUM_LANES),
    ]
    guard_tail = False
    if live:
        live_b = jnp.clip(
            (jnp.asarray(num_live_indices, jnp.int32) + occurrences_per_batch - 1)
            // occurrences_per_batch,
            0,
            num_batches,
        )
        owner_b = (live_b + num_owners - 1) // num_owners
        live_vec = jnp.concatenate(
            [
                jnp.full((_SC_NUM_LANES,), live_b, jnp.int32),
                jnp.full((_SC_NUM_LANES,), owner_b, jnp.int32),
            ]
        )
        operands.append(live_vec)
        scratch_types["live_vmem_ref"] = pltpu.VMEM((2 * _SC_NUM_LANES,), jnp.int32)
        scratch_types["live_sem_ref"] = pltpu.SemaphoreType.DMA((1,))
        guard_tail = live_multiple % owner_batch_size != 0

    mesh = plsc.VectorSubcoreMesh(
        num_cores=num_cores,
        num_subcores=num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    body = functools.partial(
        _unified_native_sc_gather_kernel,
        core_axis_name=mesh.core_axis_name,
        subcore_axis_name=mesh.subcore_axis_name,
        num_owners=num_owners,
        atoms_per_batch=atoms_per_batch,
        batches_per_owner=batches_per_owner,
        wait_mode=wait_mode,
        serve_pipelined=serve_pipelined,
        guard_tail=guard_tail,
        rope_words=rope_words,
    )
    if live:
        inner = body

        def body(nope, rope, idx, live_hbm, nope_o, rope_o, **scr):
            return inner(nope, rope, idx, nope_o, rope_o, live_hbm_ref=live_hbm, **scr)

    return jax_api.jit(
        pl.kernel(
            body,
            out_type=(
                jax.ShapeDtypeStruct((out_size, 128), jnp.uint32),
                jax.ShapeDtypeStruct((out_size // _ROPE_PACKING, 128), jnp.uint32),
            ),
            compiler_params=pltpu.CompilerParams(
                use_tc_tiling_on_sc=False,
                needs_layout_passes=True,
                disable_bounds_checks=True,
            ),
            scratch_types=scratch_types,
            mesh=mesh,
            name=(
                f"sc_unified_native_gather_a{atoms_per_batch}_{wait_mode}_"
                f"{'pipe' if serve_pipelined else 'nopipe'}{'_live' if live else ''}"
            ),
        )
    )(*operands)


def _sc_native_mla_attention_kernel(
    kv_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    q_nope_hbm_ref,  # bf16[T_q, H_pad, 512]
    q_pe_hbm_ref,  # bf16[T_q, H_pad, 64]
    gathered_nope_hbm_ref,  # u32[chunk_size, topk, 128]
    gathered_rope_hbm_ref,  # u32[chunk_size, topk // 4, 128]
    o_hbm_ref,  # bf16[T_q, H_pad, 512]
    bkv_nope_x2_ref,  # u32[2, batch_size, topk, 128]
    bkv_rope_x2_ref,  # u32[2, batch_size, topk // 4, 128]
    bq_nope_x2_ref,  # bf16[2, batch_size, H_pad, 512]
    bq_pe_x2_ref,  # bf16[2, batch_size, H_pad, 64]
    bo_x2_ref,  # bf16[2, batch_size, H_pad, 512]
    sems,  # DMA((5, 2))
    *,
    sm_scale: float,
    batch_size: int,
):
    """Double-buffered split-Q TensorCore MLA over native-SC ``uint32`` gathered KV."""
    _, _, o_dim = q_nope_hbm_ref.shape
    _, topk, _ = gathered_nope_hbm_ref.shape
    kv_dim_pad = 640
    q_dtype = q_nope_hbm_ref.dtype
    f32_min = jnp.finfo(jnp.float32).min

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def _async_copy(src, dst, sem, wait: bool):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem_n = sems.at[0, sem_idx]
        sem_r = sems.at[1, sem_idx]
        dst_n = bkv_nope_x2_ref.at[sem_idx, :]
        dst_r = bkv_rope_x2_ref.at[sem_idx, :]
        local_start = seq_idx_start - start_seq_idx
        if not wait:
            _async_copy(
                gathered_nope_hbm_ref.at[pl.ds(local_start, batch_size)], dst_n, sem_n, False
            )
            _async_copy(
                gathered_rope_hbm_ref.at[pl.ds(local_start, batch_size)], dst_r, sem_r, False
            )
        else:
            _async_copy(dst_n, dst_n, sem_n, True)
            _async_copy(dst_r, dst_r, sem_r, True)

    def _fetch_bq_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem_n = sems.at[2, sem_idx]
        sem_p = sems.at[3, sem_idx]
        dst_n = bq_nope_x2_ref.at[sem_idx, :]
        dst_p = bq_pe_x2_ref.at[sem_idx, :]
        if not wait:
            _async_copy(q_nope_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], dst_n, sem_n, False)
            _async_copy(q_pe_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], dst_p, sem_p, False)
        else:
            _async_copy(dst_n, dst_n, sem_n, True)
            _async_copy(dst_p, dst_p, sem_p, True)

    def _send_bo_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem = sems.at[4, sem_idx]
        src = bo_x2_ref.at[sem_idx, :]
        if not wait:
            _async_copy(src, o_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], sem, False)
        else:
            _async_copy(src, src, sem, True)

    def _load_bq(sem_idx, batch_idx):
        q_n = bq_nope_x2_ref.at[sem_idx, batch_idx][...]
        q_p = bq_pe_x2_ref.at[sem_idx, batch_idx][...]
        q_cat = jnp.concatenate([q_n, q_p], axis=-1)
        if q_cat.shape[-1] < kv_dim_pad:
            q_cat = jnp.pad(q_cat, ((0, 0), (0, kv_dim_pad - q_cat.shape[-1])))
        return q_cat

    def _load_bkv(sem_idx, batch_idx, kv_len):
        bkv_nope = pltpu.bitcast(bkv_nope_x2_ref.at[sem_idx, batch_idx][...], jnp.uint8)
        bkv_nope = pltpu.bitcast(bkv_nope.reshape(topk, 512), jnp.float8_e4m3fn)
        bkv_rope = pltpu.bitcast(bkv_rope_x2_ref.at[sem_idx, batch_idx][...], jnp.uint8)
        bkv_rope = pltpu.bitcast(bkv_rope.reshape(topk, 128), jnp.float8_e4m3fn)
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        return jnp.where(k_span < kv_len, bkv, 0)

    def _step1_qk_softmax(q, bkv, kv_len):
        s = jnp.einsum("nd,md->nm", q, bkv, preferred_element_type=jnp.float32)
        s = s * sm_scale
        kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(kv_span < kv_len, s, f32_min)
        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.sum(p, axis=1, keepdims=True)
        return p, l_sum

    def _step2_pv(p, bkv, l_sum, kv_len):
        bkv_v = bkv[:, :o_dim]
        pv = jnp.einsum("nm,md->nd", p, bkv_v, preferred_element_type=jnp.float32)
        l_safe = jnp.maximum(l_sum, 1e-30)
        if q_dtype == jnp.float32:
            out = lax.div(pv, l_safe)
        else:
            out = (pv * pl.reciprocal(l_safe, approx=True)).astype(q_dtype)
        return jnp.where(kv_len > 0, out, jnp.zeros_like(out))

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def _prologue():
        _fetch_bq_batch(batch_start_seq_idx, 0)
        _fetch_bkv_batch(batch_start_seq_idx, 0)

    bi_sem_idx = sem_ids_ref[0]
    next_seq_idx = batch_start_seq_idx + batch_size
    next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

    @pl.when(next_seq_idx < end_seq_idx)
    def _prefetch_next():
        sem_ids_ref[0] = next_bi_sem_idx
        _fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
        _fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

    bo_sem_idx = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

    _fetch_bq_batch(batch_start_seq_idx, bi_sem_idx, wait=True)
    _fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx, wait=True)

    @pl.when(pl.program_id(0) >= 2)
    def _wait_prev_send():
        _send_bo_batch(0, bo_sem_idx, wait=True)

    prev_p = None
    prev_bkv = None
    prev_l_sum = None
    prev_kv_len = None
    prev_out = None

    for batch_idx in range(batch_size):
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        bkv = _load_bkv(bi_sem_idx, batch_idx, kv_len)
        bq = _load_bq(bi_sem_idx, batch_idx)

        if prev_out is not None:
            prev_pad = (
                jnp.pad(prev_out, ((0, 0), (0, kv_dim_pad - o_dim)))
                if o_dim < kv_dim_pad
                else prev_out
            )
            bq = jnp.where(prev_pad == jnp.inf, prev_pad, bq)

        p, l_sum = _step1_qk_softmax(bq, bkv, kv_len)

        if prev_p is not None:
            out = _step2_pv(prev_p, prev_bkv, prev_l_sum, prev_kv_len)
            bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
            prev_out = out

        prev_p = p
        prev_bkv = bkv
        prev_l_sum = l_sum
        prev_kv_len = kv_len

    out_last = _step2_pv(prev_p, prev_bkv, prev_l_sum, prev_kv_len)
    bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out_last
    _send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def _epilogue():
        _send_bo_batch(0, 0, wait=True)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_odd():
            _send_bo_batch(0, 1, wait=True)


# ---------------------------------------------------------------------------
# Legacy 4D TensorCore-tiled SparseCore Gather Fallback
# ---------------------------------------------------------------------------


def _calculate_sc_col_And_streams(hidden_size: int, packing: int) -> tuple[int, int]:
    """Choose ``(col_size, num_streams)`` within SparseCore subcore VMEM limits."""
    tpu_info = pltpu.get_tpu_info()
    sc_info = tpu_info.sparse_core
    assert sc_info is not None
    lanes = sc_info.num_lanes

    match tpu_info.generation:
        case 6:
            vmem_budget = int((256 * 1024) * 0.82)
        case 7:
            vmem_budget = int((512 * 1024) * 0.82)
        case _:
            vmem_budget = int((256 * 1024) * 0.80)

    rows_per_stream = lanes // packing
    min_streams = max(1, cdiv(8, rows_per_stream))

    env_streams = int(os.environ.get("DSA_SC_GATHER_STREAMS", "0"))
    if env_streams > 0:
        cand_streams = [max(min_streams, env_streams)]
    elif tpu_info.generation >= 7 and packing == 4:
        cand_streams = [4, 2]
    elif tpu_info.generation >= 7 and packing == 2:
        cand_streams = [2, 1]
    else:
        cand_streams = [min_streams]

    for num_streams in cand_streams:
        if (num_streams * rows_per_stream) % 8 != 0:
            continue
        bytes_per_col = 2 * num_streams * (lanes + rows_per_stream) * 4
        max_safe_col = (vmem_budget // bytes_per_col // 128) * 128
        if max_safe_col >= hidden_size and hidden_size % 128 == 0:
            return hidden_size, num_streams
        start_col = (min(hidden_size, max_safe_col) // 128) * 128
        for c in range(start_col, 127, -128):
            if hidden_size % c == 0:
                return c, num_streams

    bytes_per_col = 2 * min_streams * (lanes + rows_per_stream) * 4
    max_safe_col = max(128, (vmem_budget // bytes_per_col // 128) * 128)
    for c in range(min(hidden_size, max_safe_col) // 128 * 128, 127, -128):
        if hidden_size % c == 0:
            return c, min_streams
    return max_safe_col, min_streams


def _sc_dsa_kv_gather_kernel(
    in_hbm_ref,
    indices_hbm_ref,
    valid_indices_ref,
    out_hbm_ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
    col_size: int,
):
    sc_info = pltpu.get_tpu_info().sparse_core
    assert sc_info is not None
    num_simd_lanes = sc_info.num_lanes
    num_cores = lax.axis_size((core_axis_name, subcore_axis_name))
    core_index = lax.axis_index((core_axis_name, subcore_axis_name))

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores
    num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)

    raw_dtype = out_hbm_ref.dtype
    dtype_bits = jax.dtypes.itemsize_bits(raw_dtype)
    packing = 32 // dtype_bits
    rows_per_stream = row_subchunk_size // packing
    shift_amount = packing.bit_length() - 1
    mask = (1 << dtype_bits) - 1
    shift_multiplier = dtype_bits.bit_length() - 1

    in_hbm_i32 = in_hbm_ref.bitcast(jnp.int32)
    out_hbm_i32 = out_hbm_ref.bitcast(jnp.int32)
    hidden_size = in_hbm_ref.shape[-1]
    num_cols = pl.cdiv(hidden_size, col_size)

    def outer_pipeline(idx_ref, valid_ref):
        valid_indices = valid_ref[pl.ds(0, row_subchunk_size)][0]
        b = pl.program_id(0)
        out_row_base = (b * num_cores + core_index) * num_row_subchunks
        core_start_idx = out_row_base * row_subchunk_size

        def subchunk(r, s):
            return r * num_streams + s

        def idx_window(r, s):
            return idx_ref[pl.ds(subchunk(r, s) * row_subchunk_size, row_subchunk_size)]

        def _inner_body(*refs):
            r = pl.program_id(0)
            gather_refs = refs[:num_streams]
            out_ref = refs[num_streams]

            for s in range(num_streams):
                g_ref = gather_refs[s]
                out_base_s = s * rows_per_stream
                if packing == 1:
                    g_dt = g_ref.bitcast(raw_dtype)
                    o_dt = out_ref.bitcast(raw_dtype)
                    o_dt[pl.ds(out_base_s, rows_per_stream), :] = g_dt[:, :]
                else:
                    sub_idxs = idx_window(r, s)
                    idx_rem = jnp.bitwise_and(sub_idxs, packing - 1)

                    def _col_step(col_base, g_ref=g_ref, idx_rem=idx_rem, out_base_s=out_base_s):
                        col_slice = pl.ds(col_base, 128)
                        for rg in range(rows_per_stream):
                            packed_row = jnp.zeros((1, 128), dtype=jnp.int32)
                            for sub in range(packing):
                                k = rg * packing + sub
                                dyn_shift = jnp.left_shift(idx_rem[k], shift_multiplier)
                                val = jnp.bitwise_right_shift(
                                    g_ref[pl.ds(k, 1), col_slice], dyn_shift
                                )
                                val = jnp.bitwise_and(val, mask)
                                packed_row = jnp.bitwise_or(
                                    packed_row, jnp.left_shift(val, sub * dtype_bits)
                                )
                            out_ref[pl.ds(out_base_s + rg, 1), col_slice] = packed_row

                    plsc.parallel_loop(0, col_size, step=128)(_col_step)

        in_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(row_subchunk_size), col_size),
                lambda r, col_id, s=s: (
                    jnp.bitwise_right_shift(idx_window(r, s), shift_amount),
                    col_id,
                ),
            )
            for s in range(num_streams)
        )
        out_spec = pl.BlockSpec(
            (num_streams * rows_per_stream, col_size),
            lambda r, col_id: (out_row_base // num_streams + r, col_id),
        )

        @pl.when(core_start_idx < valid_indices)
        def _run_gather():
            pltpu.emit_pipeline(
                _inner_body,
                grid=(num_row_subchunks // num_streams, num_cols),
                in_specs=in_specs,
                out_specs=(out_spec,),
            )(*([in_hbm_i32] * num_streams), out_hbm_i32)

    pltpu.emit_pipeline(
        outer_pipeline,
        grid=(num_blocks,),
        in_specs=(
            pl.BlockSpec(
                (row_chunk_size,),
                lambda b: (b * num_cores + core_index,),
            ),
            pl.BlockSpec(
                (row_subchunk_size,),
                lambda b: (0,),
            ),
        ),
    )(indices_hbm_ref, valid_indices_ref)


@jax.jit
def sc_dsa_kv_gather(
    cache_2d: jax.Array,  # [total_tokens, hidden_size]
    indices: jax.Array,  # int32[out_size]
    num_valid_indices: jax.Array | None = None,
) -> jax.Array:
    """Multi-stream SparseCore gather over ``cache_2d`` with dead-row skipping."""
    assert cache_2d.ndim == 2, f"cache_2d must be 2D, got {cache_2d.shape}"
    assert indices.ndim == 1, f"indices must be 1D, got {indices.shape}"

    orig_dtype = cache_2d.dtype
    dtype_bits = jax.dtypes.itemsize_bits(orig_dtype)
    if dtype_bits not in (8, 16, 32):
        raise ValueError(f"Unsupported KV cache dtype {orig_dtype} ({dtype_bits} bits)")

    if not is_tpu_runtime():
        safe = jnp.clip(indices, 0, max(0, cache_2d.shape[0] - 1))
        return cache_2d[safe]

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        safe = jnp.clip(indices, 0, max(0, cache_2d.shape[0] - 1))
        return cache_2d[safe]

    if dtype_bits == 8:
        raw_dtype = jnp.uint8
        cache_raw = lax.bitcast_convert_type(cache_2d, raw_dtype)
    elif dtype_bits == 16:
        raw_dtype = jnp.uint16
        cache_raw = lax.bitcast_convert_type(cache_2d, raw_dtype)
    else:
        raw_dtype = orig_dtype
        cache_raw = cache_2d

    hidden_size = cache_raw.shape[-1]
    out_size = indices.size
    packing = 32 // dtype_bits
    col_size, num_streams = _calculate_sc_col_And_streams(hidden_size, packing)
    aligned_hidden_size = align_to(hidden_size, col_size)
    if aligned_hidden_size != hidden_size:
        cache_raw = jnp.pad(cache_raw, ((0, 0), (0, aligned_hidden_size - hidden_size)))

    num_simd_lanes = sc_info.num_lanes
    num_cores = sc_info.num_cores * sc_info.num_subcores
    base_block_size = num_simd_lanes * num_cores

    raw_subchunks = max(num_streams, min(32, cdiv(out_size, base_block_size)))
    num_row_subchunks = min(32, align_to(raw_subchunks, num_streams))

    row_subchunk_size = num_simd_lanes
    row_chunk_size = row_subchunk_size * num_row_subchunks
    block_size = row_chunk_size * num_cores

    out_pad_size = align_to(out_size, block_size) - out_size
    if out_pad_size > 0:
        indices = jnp.pad(indices, ((0, out_pad_size),))

    if num_valid_indices is None:
        valid_vec = jnp.full((row_subchunk_size,), out_size, dtype=jnp.int32)
    else:
        valid_vec = jnp.broadcast_to(
            jnp.asarray(num_valid_indices, dtype=jnp.int32).reshape(()),
            (row_subchunk_size,),
        )

    vector_mesh = plsc.VectorSubcoreMesh(
        num_cores=sc_info.num_cores,
        num_subcores=sc_info.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out_raw = core_map_helper.kernel(
        functools.partial(
            _sc_dsa_kv_gather_kernel,
            core_axis_name=vector_mesh.core_axis_name,
            subcore_axis_name=vector_mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
            col_size=col_size,
        ),
        out_type=jax.ShapeDtypeStruct((out_size + out_pad_size, aligned_hidden_size), raw_dtype),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=vector_mesh,
        name="sc_dsa_kv_gather",
    )(cache_raw, indices, valid_vec)[:out_size, :hidden_size]

    if raw_dtype != orig_dtype:
        return lax.bitcast_convert_type(out_raw, orig_dtype)
    return out_raw


def _sc_gathered_mla_attention_kernel(
    kv_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    q_nope_hbm_ref,
    q_pe_hbm_ref,
    gathered_kv_hbm_ref,
    o_hbm_ref,
    bkv_x2_ref,
    bq_nope_x2_ref,
    bq_pe_x2_ref,
    bo_x2_ref,
    sems,
    *,
    sm_scale: float,
    batch_size: int,
):
    """Double-buffered split-Q TensorCore MLA kernel over 4D-gathered KV."""
    _, _, o_dim = q_nope_hbm_ref.shape
    _, _, kv_dim_pad = gathered_kv_hbm_ref.shape
    q_dtype = q_nope_hbm_ref.dtype
    f32_min = jnp.finfo(jnp.float32).min

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def _async_copy(src, dst, sem, wait: bool):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem = sems.at[0, sem_idx]
        dst = bkv_x2_ref.at[sem_idx, :]
        local_start = seq_idx_start - start_seq_idx
        if not wait:
            _async_copy(gathered_kv_hbm_ref.at[pl.ds(local_start, batch_size)], dst, sem, False)
        else:
            _async_copy(dst, dst, sem, True)

    def _fetch_bq_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem_nope = sems.at[1, sem_idx]
        sem_pe = sems.at[2, sem_idx]
        dst_nope = bq_nope_x2_ref.at[sem_idx, :]
        dst_pe = bq_pe_x2_ref.at[sem_idx, :]
        if not wait:
            _async_copy(q_nope_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], dst_nope, sem_nope, False)
            _async_copy(q_pe_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], dst_pe, sem_pe, False)
        else:
            _async_copy(dst_nope, dst_nope, sem_nope, True)
            _async_copy(dst_pe, dst_pe, sem_pe, True)

    def _send_bo_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem = sems.at[3, sem_idx]
        src = bo_x2_ref.at[sem_idx, :]
        if not wait:
            _async_copy(src, o_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], sem, False)
        else:
            _async_copy(src, src, sem, True)

    def _load_bq(sem_idx, batch_idx):
        q_n = bq_nope_x2_ref.at[sem_idx, batch_idx][...]
        q_p = bq_pe_x2_ref.at[sem_idx, batch_idx][...]
        q_cat = jnp.concatenate([q_n, q_p], axis=-1)
        if q_cat.shape[-1] < kv_dim_pad:
            q_cat = jnp.pad(q_cat, ((0, 0), (0, kv_dim_pad - q_cat.shape[-1])))
        return q_cat

    def _load_bkv(sem_idx, batch_idx, kv_len):
        bkv = bkv_x2_ref.at[sem_idx, batch_idx][...]
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        return jnp.where(k_span < kv_len, bkv, 0)

    def _step1_qk_softmax(q, bkv, kv_len):
        s = jnp.einsum("nd,md->nm", q, bkv, preferred_element_type=jnp.float32)
        s = s * sm_scale
        kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(kv_span < kv_len, s, f32_min)
        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.sum(p, axis=1, keepdims=True)
        return p, l_sum

    def _step2_pv(p, bkv, l_sum, kv_len):
        bkv_v = bkv[:, :o_dim]
        pv = jnp.einsum("nm,md->nd", p, bkv_v, preferred_element_type=jnp.float32)
        l_safe = jnp.maximum(l_sum, 1e-30)
        if q_dtype == jnp.float32:
            out = lax.div(pv, l_safe)
        else:
            out = (pv * pl.reciprocal(l_safe, approx=True)).astype(q_dtype)
        return jnp.where(kv_len > 0, out, jnp.zeros_like(out))

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def _prologue():
        _fetch_bq_batch(batch_start_seq_idx, 0)
        _fetch_bkv_batch(batch_start_seq_idx, 0)

    bi_sem_idx = sem_ids_ref[0]
    next_seq_idx = batch_start_seq_idx + batch_size
    next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

    @pl.when(next_seq_idx < end_seq_idx)
    def _prefetch_next():
        sem_ids_ref[0] = next_bi_sem_idx
        _fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
        _fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

    bo_sem_idx = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

    _fetch_bq_batch(batch_start_seq_idx, bi_sem_idx, wait=True)
    _fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx, wait=True)

    @pl.when(pl.program_id(0) >= 2)
    def _wait_prev_send():
        _send_bo_batch(0, bo_sem_idx, wait=True)

    prev_p = None
    prev_bkv = None
    prev_l_sum = None
    prev_kv_len = None
    prev_out = None

    for batch_idx in range(batch_size):
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        bkv = _load_bkv(bi_sem_idx, batch_idx, kv_len)
        bq = _load_bq(bi_sem_idx, batch_idx)

        if prev_out is not None:
            prev_pad = (
                jnp.pad(prev_out, ((0, 0), (0, kv_dim_pad - o_dim)))
                if o_dim < kv_dim_pad
                else prev_out
            )
            bq = jnp.where(prev_pad == jnp.inf, prev_pad, bq)

        p, l_sum = _step1_qk_softmax(bq, bkv, kv_len)

        if prev_p is not None:
            out = _step2_pv(prev_p, prev_bkv, prev_l_sum, prev_kv_len)
            bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
            prev_out = out

        prev_p = p
        prev_bkv = bkv
        prev_l_sum = l_sum
        prev_kv_len = kv_len

    out_last = _step2_pv(prev_p, prev_bkv, prev_l_sum, prev_kv_len)
    bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out_last
    _send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def _epilogue():
        _send_bo_batch(0, 0, wait=True)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_odd():
            _send_bo_batch(0, 1, wait=True)


def _cpu_reference_gathered_attention(
    ql_nope: jax.Array,  # [T_q, H, 512]
    q_pe: jax.Array,  # [T_q, H, 64]
    cache_2d: jax.Array,  # [Pn * page_size, 640]
    valid_kv_lens: jax.Array,  # [T_q]
    physical_indices: jax.Array,  # [T_q, topk]
    token_end: jax.Array,  # []
    *,
    sm_scale: float,
    chunk_size: int,
) -> jax.Array:
    """Exact reference path used on CPU (`JAX_PLATFORMS=cpu`) for unit parity tests."""
    T_q, H, o_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    topk = physical_indices.shape[-1]
    q_cat = jnp.concatenate([ql_nope, q_pe], axis=-1).astype(jnp.float32)
    f32_min = jnp.finfo(jnp.float32).min

    outs = []
    for start in range(0, T_q, chunk_size):
        end = min(start + chunk_size, T_q)
        idx_chunk = jnp.clip(physical_indices[start:end], 0, max(0, cache_2d.shape[0] - 1))
        kv_chunk = (
            cache_2d[idx_chunk.reshape(-1)].reshape(end - start, topk, -1).astype(jnp.float32)
        )
        k_chunk = kv_chunk[:, :, : o_dim + rope_dim]
        v_chunk = kv_chunk[:, :, :o_dim]
        kvl = valid_kv_lens[start:end]
        row_ids = jnp.arange(start, end, dtype=jnp.int32)
        active = (row_ids < token_end) & (kvl > 0)

        s = jnp.einsum("thd,tkd->thk", q_cat[start:end], k_chunk) * sm_scale
        mask = jnp.arange(topk, dtype=jnp.int32)[None, None, :] < kvl[:, None, None]
        s = jnp.where(mask, s, f32_min)
        m = jnp.max(s, axis=-1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.maximum(jnp.sum(p, axis=-1, keepdims=True), 1e-30)
        pv = jnp.einsum("thk,tkd->thd", p, v_chunk) / l_sum
        pv = jnp.where(active[:, None, None], pv, 0.0)
        outs.append(pv.astype(ql_nope.dtype))
    return jnp.concatenate(outs, axis=0)


def sparse_mla_sc_token_gather_attention(
    ql_nope: jax.Array,  # [T_q, H, 512]
    q_pe: jax.Array,  # [T_q, H, 64]
    new_kv_c: jax.Array | None,  # [T_write, 512] (None if cache already written)
    new_k_pe: jax.Array | None,  # [T_write, 64]
    cache_kv: jax.Array,  # u32[Pn, ps+ps//8, 128] OR fp8/bf16[Pn, pspk, pk, 640]
    loc: jax.Array | None,  # i32[T_write] flat slot per new token (-1 = dropped)
    valid_kv_lens: jax.Array,  # i32[T_q]
    physical_indices: jax.Array,  # i32[T_q, topk]
    token_end: jax.Array,  # i32[]
    *,
    page_size: int,
    kv_lora_rank: int = 512,
    sm_scale: float,
    chunk_size: int = DEFAULT_SC_GATHER_CHUNK_SIZE,
    attention_kernel_batch_size: int = DEFAULT_ATTN_KERNEL_BATCH_SIZE,
    small_chunk_tokens: int = 16,
    small_chunk_batch_size: int = 1,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array]:
    """Self-write new KV into ``cache_kv``, then run chunked SC gather + TC MLA."""
    is_cpu = not is_tpu_runtime()
    is_native_sc_kv = isinstance(cache_kv, NativeSCKVBuffer) or (
        cache_kv.dtype == jnp.uint32 and cache_kv.ndim == 3
    )

    if is_native_sc_kv:
        if new_kv_c is not None and new_k_pe is not None and loc is not None:
            cache_kv = scatter_unified_native_sc_kv(
                cache_kv,
                new_kv_c,
                new_k_pe,
                loc,
                page_size=page_size,
            )

        if is_cpu or pltpu.get_tpu_info().sparse_core is None:
            cache_2d_ref = _unpack_unified_native_sc_kv_to_2d(cache_kv, page_size=page_size)
            out = _cpu_reference_gathered_attention(
                ql_nope,
                q_pe,
                cache_2d_ref,
                valid_kv_lens,
                physical_indices,
                token_end,
                sm_scale=float(sm_scale),
                chunk_size=max(1, min(chunk_size, ql_nope.shape[0])),
            )
            return out, cache_kv

        if not isinstance(cache_kv, NativeSCKVBuffer):
            raise ValueError(
                "TPU native-SC MLA requires NativeSCKVBuffer(nope, rope) so NOPE (8x128) "
                "and ROPE (8x16) carry independent HBM tilings."
            )

        T_q, actual_heads, o_dim = ql_nope.shape
        rope_dim = q_pe.shape[-1]
        topk = physical_indices.shape[-1]
        if o_dim != 512:
            raise ValueError(f"native-SC MLA requires o_dim == 512, got {o_dim}")
        if topk % _ROPE_PACKING != 0:
            raise ValueError(f"topk ({topk}) must be divisible by {_ROPE_PACKING}")

        H_pad = align_to(actual_heads, 8)
        if H_pad != actual_heads:
            ql_work = jnp.pad(ql_nope, ((0, 0), (0, H_pad - actual_heads), (0, 0)))
            qpe_work = jnp.pad(q_pe, ((0, 0), (0, H_pad - actual_heads), (0, 0)))
        else:
            ql_work = ql_nope
            qpe_work = q_pe

        eff_vmem_limit = max(int(vmem_limit_bytes), DEFAULT_VMEM_LIMIT_BYTES)
        num_chunks = cdiv(T_q, chunk_size)
        for c_idx in range(num_chunks):
            start_pos = c_idx * chunk_size
            end_pos = min(start_pos + chunk_size, T_q)
            cur_chunk = end_pos - start_pos

            is_small = small_chunk_tokens > 0 and cur_chunk <= small_chunk_tokens
            batch_cap = small_chunk_batch_size if is_small else attention_kernel_batch_size
            k_bsz = _largest_divisor(cur_chunk, batch_cap)
            gather_atoms = 1 if is_small else 16

            valid_rows = jnp.maximum(0, jnp.minimum(token_end, end_pos) - start_pos)
            batch_end = start_pos + cdiv(valid_rows, k_bsz) * k_bsz
            num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

            gathered_nope, gathered_rope = dsa_gather_unified_native_sc(
                cache_kv.nope,
                cache_kv.rope,
                physical_indices[start_pos:end_pos],
                out_size=cur_chunk * topk,
                atoms_per_batch=gather_atoms,
                wait_mode="overlap_rope",
                serve_pipelined=True,
                num_live_indices=num_valid_indices,
                live_multiple=k_bsz * topk,
            )
            gathered_nope = gathered_nope.reshape(cur_chunk, topk, 128)
            gathered_rope = gathered_rope.reshape(cur_chunk, topk // _ROPE_PACKING, 128)

            end_seq_idx = jnp.maximum(start_pos, batch_end)
            grid = (cdiv(end_seq_idx - start_pos, k_bsz),)
            scratch_shapes = [
                pltpu.VMEM((2, k_bsz, topk, 128), jnp.uint32),
                pltpu.VMEM((2, k_bsz, topk // _ROPE_PACKING, 128), jnp.uint32),
                pltpu.VMEM((2, k_bsz, H_pad, o_dim), ql_work.dtype),
                pltpu.VMEM((2, k_bsz, H_pad, rope_dim), qpe_work.dtype),
                pltpu.VMEM((2, k_bsz, H_pad, o_dim), ql_work.dtype),
                pltpu.SemaphoreType.DMA((5, 2)),
            ]
            scalar_prefetches = (
                valid_kv_lens,
                jnp.stack([jnp.int32(start_pos), end_seq_idx.astype(jnp.int32)]),
                jnp.zeros((2,), jnp.int32),
            )
            scope_name = f"MLA-NativeSC-tk_{topk}-bz_{k_bsz}-cz_{cur_chunk}"
            ql_work = jax.named_scope(scope_name)(
                pl.pallas_call(
                    functools.partial(
                        _sc_native_mla_attention_kernel,
                        sm_scale=float(sm_scale),
                        batch_size=k_bsz,
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=len(scalar_prefetches),
                        in_specs=[
                            pl.BlockSpec(memory_space=pltpu.HBM),
                            pl.BlockSpec(memory_space=pltpu.HBM),
                            pl.BlockSpec(memory_space=pltpu.HBM),
                            pl.BlockSpec(memory_space=pltpu.HBM),
                        ],
                        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                        grid=grid,
                        scratch_shapes=scratch_shapes,
                    ),
                    compiler_params=pltpu.CompilerParams(
                        dimension_semantics=("arbitrary",),
                        vmem_limit_bytes=eff_vmem_limit,
                        disable_bounds_checks=True,
                    ),
                    out_shape=jax.ShapeDtypeStruct(ql_work.shape, ql_work.dtype),
                    input_output_aliases={len(scalar_prefetches): 0},
                    name=scope_name,
                )
            )(*scalar_prefetches, ql_work, qpe_work, gathered_nope, gathered_rope)

        out = ql_work[:, :actual_heads, :o_dim]
        row_ids = lax.broadcasted_iota(jnp.int32, (T_q, 1, 1), 0)
        out = jnp.where(row_ids < token_end, out, jnp.zeros_like(out))
        return out, cache_kv

    if new_kv_c is not None and new_k_pe is not None and loc is not None:
        T_w = new_kv_c.shape[0]
        rope_w = new_k_pe.shape[-1]
        row = jnp.zeros((T_w, cache_kv.shape[3]), dtype=cache_kv.dtype)
        row = row.at[:, :kv_lora_rank].set(new_kv_c.astype(cache_kv.dtype))
        row = row.at[:, kv_lora_rank : kv_lora_rank + rope_w].set(
            new_k_pe.reshape(T_w, rope_w).astype(cache_kv.dtype)
        )
        cache_kv = paged_write_back(
            cache_kv,
            row,
            loc.astype(jnp.int32),
            page_size=page_size,
            interpret=is_cpu,
            small_rows_scatter=(T_w <= 64),
        )

    cache_2d = cache_kv.reshape(-1, cache_kv.shape[3])
    if is_cpu or pltpu.get_tpu_info().sparse_core is None:
        out = _cpu_reference_gathered_attention(
            ql_nope,
            q_pe,
            cache_2d,
            valid_kv_lens,
            physical_indices,
            token_end,
            sm_scale=float(sm_scale),
            chunk_size=max(1, min(chunk_size, ql_nope.shape[0])),
        )
        return out, cache_kv

    T_q, actual_heads, o_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    topk = physical_indices.shape[-1]
    if o_dim % 128 != 0:
        raise ValueError(f"ql_nope dim ({o_dim}) must be a multiple of 128")

    H_pad = align_to(actual_heads, 8)
    if H_pad != actual_heads:
        ql_work = jnp.pad(ql_nope, ((0, 0), (0, H_pad - actual_heads), (0, 0)))
        qpe_work = jnp.pad(q_pe, ((0, 0), (0, H_pad - actual_heads), (0, 0)))
    else:
        ql_work = ql_nope
        qpe_work = q_pe

    kv_bytes = jax.dtypes.itemsize_bits(cache_kv.dtype) // 8
    max_batch_for_vmem = max(1, min(attention_kernel_batch_size, 16 // max(1, kv_bytes)))

    num_chunks = cdiv(T_q, chunk_size)
    for c_idx in range(num_chunks):
        start_pos = c_idx * chunk_size
        end_pos = min(start_pos + chunk_size, T_q)
        cur_chunk = end_pos - start_pos

        is_small = small_chunk_tokens > 0 and cur_chunk <= small_chunk_tokens
        batch_cap = small_chunk_batch_size if is_small else max_batch_for_vmem
        k_bsz = _largest_divisor(cur_chunk, batch_cap)

        valid_rows = jnp.maximum(0, jnp.minimum(token_end, end_pos) - start_pos)
        batch_end = start_pos + cdiv(valid_rows, k_bsz) * k_bsz
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        idx_chunk = physical_indices[start_pos:end_pos].reshape(-1)
        gathered_kv = sc_dsa_kv_gather(
            cache_2d,
            idx_chunk,
            num_valid_indices=num_valid_indices,
        ).reshape(cur_chunk, topk, cache_kv.shape[3])

        end_seq_idx = jnp.maximum(start_pos, batch_end)
        grid = (cdiv(end_seq_idx - start_pos, k_bsz),)

        scratch_shapes = [
            pltpu.VMEM((2, k_bsz, topk, cache_kv.shape[3]), cache_kv.dtype),
            pltpu.VMEM((2, k_bsz, H_pad, o_dim), ql_work.dtype),
            pltpu.VMEM((2, k_bsz, H_pad, rope_dim), qpe_work.dtype),
            pltpu.VMEM((2, k_bsz, H_pad, o_dim), ql_work.dtype),
            pltpu.SemaphoreType.DMA((4, 2)),
        ]
        scalar_prefetches = (
            valid_kv_lens,
            jnp.stack([jnp.int32(start_pos), end_seq_idx.astype(jnp.int32)]),
            jnp.zeros((2,), jnp.int32),
        )
        scope_name = f"MLA-SCGather-tk_{topk}-bz_{k_bsz}-cz_{cur_chunk}"
        ql_work = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _sc_gathered_mla_attention_kernel,
                    sm_scale=float(sm_scale),
                    batch_size=k_bsz,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=[
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ],
                    out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(ql_work.shape, ql_work.dtype),
                input_output_aliases={len(scalar_prefetches): 0},
                name=scope_name,
            )
        )(*scalar_prefetches, ql_work, qpe_work, gathered_kv)

    out = ql_work[:, :actual_heads, :o_dim]
    row_ids = lax.broadcasted_iota(jnp.int32, (T_q, 1, 1), 0)
    out = jnp.where(row_ids < token_end, out, jnp.zeros_like(out))
    return out, cache_kv
