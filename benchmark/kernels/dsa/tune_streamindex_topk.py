"""
Tune DSA Lightning Indexer kernels (`streamindex_topk` and `streamindex_page_topk`)
for GLM-5.2 / DeepSeek-V3.2 on TPU.

Sweeps:
1. Decode (`streamindex_topk`):
   - `num_kv_pages_per_block` (bkv_p) in {4, 8, 16, 32, 64, 128}
   - `decode_req_batch_size` (seq_batch_size) in {1, 2, 4, 8}
   - `topk_backend` in {"sc", "xla"}
   - Plus comparison against `streamindex_topk_ref`
2. Prefill (`streamindex_page_topk`):
   - `num_queries_per_block` (bq_sz) in {64, 128, 256, 512, 1024} (with bkv_p=128)
   - Plus comparison against `streamindex_page_topk_ref`

Usage:
    python -m benchmark.kernels.dsa.tune_streamindex_topk
"""

from __future__ import annotations

import argparse
import functools
import time
from math import inf

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.ref import streamindex_page_topk_ref, streamindex_topk_ref
from sgl_jax.srt.kernels.dsa.streamindex_topk import (
    streamindex_page_topk,
    streamindex_topk,
)

# GLM-5.2 Lightning Indexer constants
H_I = 32  # index_n_heads
D = 128  # index_head_dim
K = 2048  # index_topk
PAGE_SIZE = 128
KV_PACKING = 2  # bfloat16


def _time_fn(fn, args, warmup: int = 2, iters: int = 5) -> float:
    """Compile + warmup + time `fn(*args)` in milliseconds."""
    try:
        for _ in range(warmup):
            out = fn(*args)
            jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
            jax.block_until_ready(out)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / iters
    except Exception as e:
        print(f"    [SKIP/ERR] {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        return inf


def tune_decode(
    batch_sizes: list[int],
    kv_lens: list[int],
    bkv_p_candidates: list[int],
    req_bs_candidates: list[int],
    backends: list[str],
):
    print("=" * 100)
    print(
        f"DECODE TUNING: streamindex_topk (H_I={H_I}, D={D}, k={K}, page_size={PAGE_SIZE}, dtype=bf16)"
    )
    print("=" * 100)

    max_kv = max(kv_lens)
    max_pages_per_seq = (max_kv + PAGE_SIZE - 1) // PAGE_SIZE
    max_b = max(batch_sizes)
    pool_pages = min(max_b * max_pages_per_seq + 16, 8192)

    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    cache_kv = jax.random.normal(
        k1, (pool_pages, PAGE_SIZE // KV_PACKING, KV_PACKING, D), dtype=jnp.bfloat16
    )
    jax.block_until_ready(cache_kv)

    results = []
    for B in batch_sizes:
        q = jax.random.normal(k2, (B, H_I, D), dtype=jnp.bfloat16)
        w = jax.random.normal(k3, (B, H_I), dtype=jnp.bfloat16)
        cu_q_lens = jnp.arange(B + 1, dtype=jnp.int32)
        dist = jnp.array([B, B, B], dtype=jnp.int32)
        # Fixed-stride page_indices [B * max_pages_per_seq], matching production
        # DSASparseAttentionBackend._fixed_stride_pages so XLA compiles once per B!
        fixed_page_indices = (
            jnp.arange(B * max_pages_per_seq, dtype=jnp.int32) % (pool_pages - 1)
        ) + 1

        # Pre-build jitted callables once per B
        ker_fns = {}
        for backend in backends:
            for bkv_p in bkv_p_candidates:
                for req_bs in req_bs_candidates:
                    if req_bs > B:
                        continue
                    ker_fns[(bkv_p, req_bs, backend)] = jax.jit(
                        functools.partial(
                            streamindex_topk,
                            k=K,
                            compression_ratio=1,
                            num_kv_pages_per_block=bkv_p,
                            num_queries_per_block=1,
                            decode_req_batch_size=req_bs,
                            topk_backend=backend,
                        )
                    )

        for kv_len in kv_lens:
            pages_per_seq = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
            seq_lens = jnp.full((B,), kv_len, dtype=jnp.int32)

            ref_ms = inf
            if B * kv_len <= 16 * 100352:
                page_indices_ref = (
                    jnp.arange(B * pages_per_seq, dtype=jnp.int32) % (pool_pages - 1)
                ) + 1
                cu_kv_lens = jnp.arange(B + 1, dtype=jnp.int32) * (pages_per_seq * PAGE_SIZE)
                ref_fn = jax.jit(
                    functools.partial(
                        streamindex_topk_ref,
                        k=K,
                        pages_per_seq=pages_per_seq,
                        one_token_per_seq=True,
                    )
                )
                cache3d = cache_kv.reshape(pool_pages, PAGE_SIZE, D)
                ref_ms = _time_fn(
                    ref_fn,
                    (q, w, cache3d, seq_lens, page_indices_ref, cu_q_lens, cu_kv_lens, dist),
                )

            best_ms = inf
            best_cfg = None

            for (bkv_p, req_bs, backend), ker_fn in ker_fns.items():
                if bkv_p > pages_per_seq and bkv_p > 64:
                    continue
                ms = _time_fn(
                    ker_fn,
                    (q, w, cache_kv, seq_lens, fixed_page_indices, cu_q_lens, dist),
                )
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = (bkv_p, req_bs, backend)

            speedup = (ref_ms / best_ms) if (ref_ms < inf and best_ms < inf) else float("nan")
            print(
                f"B={B:2d}, kv_len={kv_len:6d} ({pages_per_seq:3d} pages) | "
                f"ref={ref_ms:7.3f} ms | best={best_ms:7.3f} ms "
                f"(bkv_p={best_cfg[0] if best_cfg else -1:3d}, "
                f"req_bs={best_cfg[1] if best_cfg else -1:2d}, "
                f"backend={best_cfg[2] if best_cfg else 'N/A':>3s}) | "
                f"speedup={speedup:5.2f}x"
            )
            results.append((B, kv_len, ref_ms, best_ms, best_cfg))

    return results


def tune_prefill(
    q_lens: list[int],
    kv_lens: list[int],
    bq_candidates: list[int],
):
    print("\n" + "=" * 100)
    print(
        f"PREFILL TUNING: streamindex_page_topk (H_I={H_I}, D={D}, k_pages={K // PAGE_SIZE}, page_size={PAGE_SIZE}, bkv_p=128)"
    )
    print("=" * 100)

    k_pages = K // PAGE_SIZE
    bkv_p = 128
    max_kv = max(kv_lens)
    max_pages = (max_kv + PAGE_SIZE - 1) // PAGE_SIZE
    pool_pages = max_pages + 16

    rng = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(rng, 3)
    cache_kv = jax.random.normal(
        k1, (pool_pages, PAGE_SIZE // KV_PACKING, KV_PACKING, D), dtype=jnp.bfloat16
    )
    jax.block_until_ready(cache_kv)
    fixed_page_indices = jnp.arange(max_pages, dtype=jnp.int32) + 1

    results = []
    for q_len in q_lens:
        q = jax.random.normal(k2, (q_len, H_I, D), dtype=jnp.bfloat16)
        w = jax.random.normal(k3, (q_len, H_I), dtype=jnp.bfloat16)
        cu_q_lens = jnp.array([0, q_len], dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], dtype=jnp.int32)
        num_seqs = jnp.int32(1)

        ker_fns = {}
        for bq_sz in bq_candidates:
            if bq_sz > q_len:
                continue
            ker_fns[bq_sz] = jax.jit(
                functools.partial(
                    streamindex_page_topk,
                    k_pages=k_pages,
                    num_kv_pages_per_block=bkv_p,
                    num_queries_per_block=bq_sz,
                )
            )

        for kv_len in kv_lens:
            if q_len > kv_len:
                continue
            pages_per_seq = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
            seq_lens = jnp.array([kv_len], dtype=jnp.int32)
            page_indices_ref = jnp.arange(pages_per_seq, dtype=jnp.int32) + 1
            cu_kv_lens = jnp.array([0, pages_per_seq * PAGE_SIZE], dtype=jnp.int32)

            ref_fn = jax.jit(
                functools.partial(
                    streamindex_page_topk_ref,
                    k_pages=k_pages,
                    pages_per_seq=pages_per_seq,
                    one_token_per_seq=False,
                )
            )
            cache3d = cache_kv.reshape(pool_pages, PAGE_SIZE, D)
            ref_ms = _time_fn(
                ref_fn,
                (q, w, cache3d, seq_lens, page_indices_ref, cu_q_lens, cu_kv_lens, dist),
            )

            best_ms = inf
            best_bq = None
            for bq_sz, ker_fn in ker_fns.items():
                ms = _time_fn(
                    ker_fn,
                    (q, w, cache_kv, seq_lens, fixed_page_indices, cu_q_lens, num_seqs),
                )
                if ms < best_ms:
                    best_ms = ms
                    best_bq = bq_sz

            speedup = (ref_ms / best_ms) if (ref_ms < inf and best_ms < inf) else float("nan")
            print(
                f"q_len={q_len:5d}, kv_len={kv_len:6d} ({pages_per_seq:3d} pages) | "
                f"ref={ref_ms:7.3f} ms | best={best_ms:7.3f} ms "
                f"(bq_sz={best_bq if best_bq else -1:4d}, bkv_p=128) | "
                f"speedup={speedup:5.2f}x"
            )
            results.append((q_len, kv_len, ref_ms, best_ms, best_bq))

    return results


def main():
    parser = argparse.ArgumentParser(description="Tune DSA streamindex_topk kernels")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--kv-lens",
        type=int,
        nargs="+",
        default=[8192, 32768, 65536, 100352],
    )
    parser.add_argument(
        "--q-lens",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
    )
    parser.add_argument(
        "--bkv-p",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--req-bs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=["sc", "xla"],
    )
    parser.add_argument(
        "--bq-sz",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )
    parser.add_argument(
        "--prefill-first",
        action="store_true",
        help="Run prefill tuning before decode tuning",
    )
    args = parser.parse_args()

    devices = jax.local_devices()
    print(f"Local devices ({len(devices)}): {devices[0].device_kind}")
    with jax.default_device(devices[0]):
        if args.prefill_first:
            tune_prefill(
                q_lens=args.q_lens,
                kv_lens=args.kv_lens,
                bq_candidates=args.bq_sz,
            )
            tune_decode(
                batch_sizes=args.batch_sizes,
                kv_lens=args.kv_lens,
                bkv_p_candidates=args.bkv_p,
                req_bs_candidates=args.req_bs,
                backends=args.backends,
            )
        else:
            tune_decode(
                batch_sizes=args.batch_sizes,
                kv_lens=args.kv_lens,
                bkv_p_candidates=args.bkv_p,
                req_bs_candidates=args.req_bs,
                backends=args.backends,
            )
            tune_prefill(
                q_lens=args.q_lens,
                kv_lens=args.kv_lens,
                bq_candidates=args.bq_sz,
            )


if __name__ == "__main__":
    main()
