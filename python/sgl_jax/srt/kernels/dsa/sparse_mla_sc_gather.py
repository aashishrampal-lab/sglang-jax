"""Token-level SparseCore KV Gather + Double-Buffered TensorCore MLA Attention.

Ports the highest-impact sparse MLA execution path from ``vllm-torchtpu`` into
``sglang-jax``'s fused 4D MLA KV cache layout ``[Pn, page_size // pk, pk, 640]``:

1. ``prepare_sc_gather_metadata``:
   Translates logical ``topk_indices [T_q, topk]`` through ``sglang-jax``'s
   packed ``page_indices`` + ``cu_kv_lens`` table into flat physical token rows,
   hashing ``-1`` / padded slots with large primes (``104729``, ``15485863``) so
   SparseCore indirect gathers never hit HBM bank-0 conflicts. Precomputes once
   on each full DSA layer and reuses across all 57 IndexShare shared layers.
2. ``sc_dsa_kv_gather``:
   Multi-stream SparseCore ``pl.Indirect`` gather over the zero-copy 2D view
   ``cache_kv.reshape(-1, 640)``, skipping dead subcores/chunks via
   ``@pl.when(core_start_idx < valid_indices)``.
3. ``_sc_gathered_mla_attention_kernel``:
   Double-buffered split-Q TensorCore Pallas MLA kernel that joins ``(ql, qpe)``
   in VMEM, masks ``-1`` padding slots in both scores and keys (preventing
   ``0 * fp8_NaN = NaN``), runs single-block softmax, and contracts ``PV`` over
   ``bkv[:, :512]`` (4 tiles instead of 5).
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import paged_write_back
from sgl_jax.srt.kernels.sparse_core import core_map_helper

DEFAULT_SC_GATHER_CHUNK_SIZE = int(os.environ.get("DSA_SC_GATHER_CHUNK_SIZE", "64"))
DEFAULT_ATTN_KERNEL_BATCH_SIZE = int(os.environ.get("DSA_SC_ATTN_BATCH_SIZE", "16"))
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


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
    """Build ``(valid_kv_lens [T_q], physical_indices [T_q, topk], token_end [])``.

    Supports both standard head-TP (``q_token_offset = 0``) and token-sharded
    attention (``q_token_offset = tp_rank * T_q``).
    """
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
    phys_page = jnp.clip(page_indices[pt_idx], 0, max(0, total_num_pages - 1))

    # Scatter invalid / -1 padding slots across physical pages & offsets so
    # SparseCore indirect gathers never bottleneck on HBM bank 0.
    flat_idx = jnp.arange(T_q * topk, dtype=jnp.int32).reshape(T_q, topk)
    _, scattered_page = _floor_divmod_const(flat_idx * 104729, max(1, total_num_pages))
    _, scattered_offset = _floor_divmod_const(flat_idx * 15485863, page_size)

    page_ids = jnp.where(valid_slot, phys_page, scattered_page)
    tok_offs = jnp.where(valid_slot, token_offset, scattered_offset)
    physical_indices = (page_ids * page_size + tok_offs).astype(jnp.int32)

    valid_kv_lens = jnp.sum(valid_slot, axis=-1).astype(jnp.int32)
    token_end = jnp.clip(total_q - q_off, 0, T_q).astype(jnp.int32)
    return valid_kv_lens, physical_indices, token_end


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

    # Merged output block height (num_streams * (lanes // packing)) must be a
    # multiple of 8 int32 rows to satisfy TensorCore tiling on SparseCore.
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
        # Double-buffered: num_streams * lanes gather i32 rows + num_streams * rows_per_stream out i32 rows
        bytes_per_col = 2 * num_streams * (lanes + rows_per_stream) * 4
        max_safe_col = (vmem_budget // bytes_per_col // 128) * 128
        if max_safe_col >= hidden_size and hidden_size % 128 == 0:
            return hidden_size, num_streams
        start_col = (min(hidden_size, max_safe_col) // 128) * 128
        for c in range(start_col, 127, -128):
            if hidden_size % c == 0:
                return c, num_streams

    # Safe fallback
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

    if jax.default_backend() != "tpu":
        safe = jnp.clip(indices, 0, max(0, cache_2d.shape[0] - 1))
        return cache_2d[safe]

    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        safe = jnp.clip(indices, 0, max(0, cache_2d.shape[0] - 1))
        return cache_2d[safe]

    # Bitcast fp8/bf16 to unsigned integers so SparseCore bitwise unpack operates
    # on raw integer lanes without dtype conversion.
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
    # Scalar prefetches
    kv_lens_ref,  # i32[max_num_tokens]
    start_end_seq_idx_ref,  # i32[2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # i32[2] (bi_sem_idx, bo_sem_idx)
    # HBM inputs
    q_nope_hbm_ref,  # bf16[max_num_tokens, H_pad, nope_dim (512)]
    q_pe_hbm_ref,  # bf16[max_num_tokens, H_pad, rope_dim (64)]
    gathered_kv_hbm_ref,  # (fp8|bf16)[chunk_size, topk, kv_dim_pad (640)]
    # HBM output (aliased with q_nope_hbm_ref)
    o_hbm_ref,  # bf16[max_num_tokens, H_pad, nope_dim (512)]
    # VMEM scratch
    bkv_x2_ref,  # [2, batch_size, topk, 640]
    bq_nope_x2_ref,  # [2, batch_size, H_pad, 512]
    bq_pe_x2_ref,  # [2, batch_size, H_pad, 64]
    bo_x2_ref,  # [2, batch_size, H_pad, 512]
    sems,  # DMA((3, 2)): 0=bkv, 1=bq, 2=bo
    *,
    sm_scale: float,
    batch_size: int,
):
    """Double-buffered split-Q TensorCore MLA kernel over SparseCore-gathered KV."""
    _, num_q_heads, o_dim = q_nope_hbm_ref.shape
    _, topk, kv_dim_pad = gathered_kv_hbm_ref.shape
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
        sem = sems.at[1, sem_idx]
        dst_nope = bq_nope_x2_ref.at[sem_idx, :]
        dst_pe = bq_pe_x2_ref.at[sem_idx, :]
        if not wait:
            _async_copy(q_nope_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], dst_nope, sem, False)
            _async_copy(q_pe_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], dst_pe, sem, False)
        else:
            _async_copy(dst_nope, dst_nope, sem, True)
            _async_copy(dst_pe, dst_pe, sem, True)

    def _send_bo_batch(seq_idx_start, sem_idx, *, wait: bool = False):
        sem = sems.at[2, sem_idx]
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
        # Zero rows >= kv_len so scattered -1 padding slots can never introduce
        # fp8 NaN (0 * NaN = NaN) into the PV matmul.
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
        # Contract only over kv_lora_rank (512 = 4 tiles) instead of all 640 cols!
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
            # Prevent LLO QK runahead so MXU (PV) and VPU (exp/sum) interleave.
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
    q_cat = jnp.concatenate([ql_nope, q_pe], axis=-1).astype(jnp.float32)  # [T_q, H, 576]
    f32_min = jnp.finfo(jnp.float32).min

    outs = []
    for start in range(0, T_q, chunk_size):
        end = min(start + chunk_size, T_q)
        idx_chunk = jnp.clip(physical_indices[start:end], 0, max(0, cache_2d.shape[0] - 1))
        kv_chunk = cache_2d[idx_chunk.reshape(-1)].reshape(end - start, topk, -1).astype(jnp.float32)
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
    cache_kv: jax.Array,  # [Pn, pspk, pk, 640]
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
    """Self-write new KV into 4D ``cache_kv``, then run chunked SC gather + TC MLA."""
    is_cpu = jax.default_backend() == "cpu"

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

    # Pad heads to a multiple of 8 so 4D VMEM buffer indexing is sublane-aligned
    # even at high TP (e.g. H=4 on tp=16).
    H_pad = align_to(actual_heads, 8)
    if H_pad != actual_heads:
        ql_work = jnp.pad(ql_nope, ((0, 0), (0, H_pad - actual_heads), (0, 0)))
        qpe_work = jnp.pad(q_pe, ((0, 0), (0, H_pad - actual_heads), (0, 0)))
    else:
        # Copy/identity ensures safe aliasing if ql_nope shares buffer upstream.
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
            pltpu.SemaphoreType.DMA((3, 2)),
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

    return ql_work[:, :actual_heads, :o_dim], cache_kv
