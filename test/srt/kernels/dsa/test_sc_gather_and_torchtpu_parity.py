"""Parity & regression tests for vllm-torchtpu optimizations ported to sglang-jax:

1. LIBTPU_INIT_ARGS collective offload tuning + JAX Pallas fori_loop lowering patch.
2. Token-level SparseCore MLA KV gather (`sc_dsa_kv_gather`) + double-buffered
   split-Q TensorCore MLA attention (`sparse_mla_sc_token_gather_attention`).
3. Causal short-sequence early-exit in `streamindex_topk` and `streamindex_page_topk`.
4. Direct 4D `paged_write_back` (`_write_idx_4d`) for `idx_cache` without 3D relayout.
5. `GlmDsaIndexer` BF16 Hadamard dtype preservation, fused `(query, key)` Hadamard,
   and fused `[wk, weights_proj]` projection (`fuse_wk_weights_proj`).
6. End-to-end `DSASparseAttentionBackend` full + IndexShare shared-layer forward
   for both prefill (`EXTEND`) and `DECODE` with `DSA_SC_TOKEN_GATHER=1`.
"""

from __future__ import annotations

import os
import types
import unittest

os.environ.setdefault("DSA_SC_TOKEN_GATHER", "1")
os.environ.setdefault("DSA_PREFILL_SPARSE", "1")
os.environ.setdefault("DSA_PALLAS_IDX_WRITE", "1")
os.environ.setdefault("DSA_STREAMIDX_EARLY_EXIT", "1")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsa.sparse_mla_sc_gather import (
    prepare_sc_gather_metadata,
    sc_dsa_kv_gather,
    sparse_mla_sc_token_gather_attention,
)
from sgl_jax.srt.kernels.dsa.streamindex_topk import (
    streamindex_page_topk,
    streamindex_topk,
)
from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    DSASparseAttentionBackend,
    _scatter_paged,
    _spec_token_slots,
)
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.tpu_runtime_tuning import (
    configure_libtpu_init_args,
    patch_jax_pallas_fori_lowering,
)


def _exact_gathered_mla_ref(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    cache_kv: jax.Array,
    topk_indices: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    page_size: int,
    sm_scale: float,
    q_token_offset: int = 0,
) -> jax.Array:
    """Pure-JAX exact token-level sparse MLA reference over 4D `cache_kv`."""
    T_q, H, o_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    topk = topk_indices.shape[-1]
    cache_2d = cache_kv.reshape(-1, cache_kv.shape[-1])
    num_seqs = seq_lens.shape[0]
    total_q = int(cu_q_lens[num_seqs])

    outs = []
    q_cat_all = jnp.concatenate([ql_nope, q_pe], axis=-1).astype(jnp.float32)
    for r in range(T_q):
        tok = r + q_token_offset
        if tok >= total_q:
            outs.append(jnp.zeros((H, o_dim), dtype=ql_nope.dtype))
            continue
        seg = int(np.searchsorted(np.asarray(cu_q_lens), tok, side="right")) - 1
        seg = max(0, min(seg, num_seqs - 1))
        q_s = int(cu_q_lens[seg])
        q_e = int(cu_q_lens[seg + 1])
        kvl = int(seq_lens[seg])
        if not (q_s <= tok < q_e and kvl > 0):
            outs.append(jnp.zeros((H, o_dim), dtype=ql_nope.dtype))
            continue
        pt_base = int(cu_kv_lens[seg]) // page_size
        row_tk = np.asarray(topk_indices[r])
        valid_pos = [int(p) for p in row_tk if 0 <= int(p) < kvl]
        if not valid_pos:
            outs.append(jnp.zeros((H, o_dim), dtype=ql_nope.dtype))
            continue
        phys = [
            int(page_indices[pt_base + (p // page_size)]) * page_size + (p % page_size)
            for p in valid_pos
        ]
        kv_rows = cache_2d[jnp.asarray(phys, dtype=jnp.int32)].astype(jnp.float32)
        k_rows = kv_rows[:, : o_dim + rope_dim]
        v_rows = kv_rows[:, :o_dim]
        logits = (q_cat_all[r] @ k_rows.T) * sm_scale
        m = jnp.max(logits, axis=-1, keepdims=True)
        p = jnp.exp(logits - m)
        out_r = (p @ v_rows) / jnp.sum(p, axis=-1, keepdims=True)
        outs.append(out_r.astype(ql_nope.dtype))
    return jnp.stack(outs, axis=0)


class TestScGatherAndTorchTpuParity(unittest.TestCase):
    def test_01_tpu_runtime_tuning_and_fori_patch(self):
        prev = os.environ.get("LIBTPU_INIT_ARGS")
        try:
            os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_enable_async_collective_fusion=true"
            merged = configure_libtpu_init_args()
            self.assertIn("--xla_tpu_enable_async_collective_fusion=true", merged)
            self.assertIn("--xla_tpu_use_dynamic_smem_negotiation=true", merged)
            self.assertIn(
                "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864", merged
            )
            self.assertIn(
                "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864", merged
            )
        finally:
            if prev is None:
                os.environ.pop("LIBTPU_INIT_ARGS", None)
            else:
                os.environ["LIBTPU_INIT_ARGS"] = prev

        applied = patch_jax_pallas_fori_lowering()
        self.assertTrue(applied, "patch_jax_pallas_fori_lowering should apply cleanly")

    def test_02_sc_gather_metadata_and_attention_parity(self):
        rng = np.random.default_rng(42)
        page_size = 128
        total_pages = 32
        topk = 64
        sm_scale = 1.0 / np.sqrt(576.0)

        for kv_dtype, atol in ((jnp.bfloat16, 2.5e-2), (jnp.float8_e4m3fn, 6e-2)):
            pk = 32 // jax.dtypes.itemsize_bits(kv_dtype)
            pspk = page_size // pk
            cache_f32 = rng.standard_normal((total_pages, pspk, pk, 640), dtype=np.float32) * 0.25
            cache_kv = jnp.asarray(cache_f32).astype(kv_dtype)

            # Ragged 2-seq + 1-padded-seq batch with T=32 padded queries (24 active)
            seq_lens = jnp.array([300, 190, 0, 0], dtype=jnp.int32)
            cu_q_lens = jnp.array([0, 16, 24, 24, 24], dtype=jnp.int32)
            cu_kv_lens = jnp.array([0, 384, 640, 640, 640], dtype=jnp.int32)
            page_indices = jnp.array([3, 7, 11, 15, 19, 2, 0, 0], dtype=jnp.int32)

            T_q = 32
            tk_np = np.full((T_q, topk), -1, dtype=np.int32)
            for r in range(16):
                vis = 300 - 16 + r + 1
                cnt = min(topk - 8, vis)
                tk_np[r, :cnt] = rng.choice(vis, size=cnt, replace=False)
            for r in range(16, 24):
                vis = 190 - 8 + (r - 16) + 1
                cnt = min(topk - 12, vis)
                tk_np[r, :cnt] = rng.choice(vis, size=cnt, replace=False)
            topk_indices = jnp.asarray(tk_np, dtype=jnp.int32)

            vkvl, phys_idx, tok_end = prepare_sc_gather_metadata(
                topk_indices,
                seq_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                page_size=page_size,
                total_num_pages=total_pages,
            )
            self.assertEqual(int(tok_end), 24)
            self.assertTrue(bool(jnp.all((phys_idx >= 0) & (phys_idx < total_pages * page_size))))
            # Verify -1 padding slots scatter across multiple pages (not all bank 0)
            pad_pages = np.asarray(phys_idx[24:]) // page_size
            self.assertGreater(len(np.unique(pad_pages)), 8)

            # Test both H=4 (tp=16 head count, exercises H_pad=8 padding) and H=16
            for H in (4, 16):
                ql = jnp.asarray(rng.standard_normal((T_q, H, 512)) * 0.2).astype(jnp.bfloat16)
                qpe = jnp.asarray(rng.standard_normal((T_q, H, 64)) * 0.2).astype(jnp.bfloat16)
                new_kvc = jnp.asarray(rng.standard_normal((T_q, 512)) * 0.2).astype(kv_dtype)
                new_kpe = jnp.asarray(rng.standard_normal((T_q, 64)) * 0.2).astype(kv_dtype)
                loc = _spec_token_slots(seq_lens, cu_q_lens, cu_kv_lens, page_indices, T_q, page_size)

                out, updated_cache = sparse_mla_sc_token_gather_attention(
                    ql,
                    qpe,
                    new_kvc,
                    new_kpe,
                    cache_kv,
                    loc,
                    vkvl,
                    phys_idx,
                    tok_end,
                    page_size=page_size,
                    kv_lora_rank=512,
                    sm_scale=sm_scale,
                    chunk_size=16,
                )
                ref = _exact_gathered_mla_ref(
                    ql,
                    qpe,
                    updated_cache,
                    topk_indices,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    page_size=page_size,
                    sm_scale=sm_scale,
                )
                self.assertFalse(bool(jnp.any(jnp.isnan(out))))
                diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
                self.assertLess(diff, atol, f"dtype={kv_dtype} H={H} max_abs={diff}")

                # Also verify direct sc_dsa_kv_gather parity on TPU/CPU
                flat_cache = updated_cache.reshape(-1, 640)
                gathered = sc_dsa_kv_gather(flat_cache, phys_idx.reshape(-1), jnp.int32(24 * topk))
                expected_gather = flat_cache[phys_idx.reshape(-1)][: 24 * topk]
                np.testing.assert_array_equal(
                    np.asarray(gathered[: 24 * topk].astype(jnp.float32)),
                    np.asarray(expected_gather.astype(jnp.float32)),
                )

    def test_03_streamindex_early_exit_parity(self):
        T, H, D = 16, 8, 128
        page_size = 128
        total_pages = 16
        cache = jnp.zeros((total_pages, page_size // 2, 2, D), dtype=jnp.bfloat16)
        q = jnp.ones((T, H, D), dtype=jnp.bfloat16)
        w = jnp.ones((T, H), dtype=jnp.float32)
        seq_lens = jnp.array([10, 4, 0, 0], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 8, 12, 12, 12], dtype=jnp.int32)
        page_indices = jnp.arange(16, dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)

        tk = streamindex_topk(
            q,
            w,
            cache,
            seq_lens,
            page_indices,
            cu_q_lens,
            dist,
            k=32,
            compression_ratio=1,
            num_kv_pages_per_block=4,
            num_queries_per_block=8,
        )
        # First query of seq 0 is at q_pos = 10 - 8 + 0 = 2 -> visible {0, 1, 2}
        np.testing.assert_array_equal(np.asarray(tk[0, :3]), np.array([0, 1, 2]))
        self.assertTrue(np.all(np.asarray(tk[0, 3:]) == -1))
        # Padded query row 12 -> all -1
        self.assertTrue(np.all(np.asarray(tk[12]) == -1))

        tp = streamindex_page_topk(
            q,
            w,
            cache,
            seq_lens,
            jnp.arange(4 * 128, dtype=jnp.int32) % total_pages,
            cu_q_lens,
            jnp.int32(2),
            k_pages=4,
            compression_ratio=1,
            num_kv_pages_per_block=128,
            num_queries_per_block=8,
            pages_per_seq=128,
        )
        self.assertEqual(int(tp[0, 0]), 0)
        self.assertTrue(np.all(np.asarray(tp[0, 1:]) == -1))
        self.assertTrue(np.all(np.asarray(tp[12]) == -1))

    def test_04_write_idx_4d_parity(self):
        rng = np.random.default_rng(7)
        page_size = 128
        pk = 2
        pspk = page_size // pk
        total_pages = 24
        idx_dim = 128
        T = 20

        cache4d = jnp.asarray(
            rng.standard_normal((total_pages, pspk, pk, idx_dim)), dtype=jnp.bfloat16
        )
        k_new = jnp.asarray(rng.standard_normal((T, idx_dim)), dtype=jnp.bfloat16)
        seq_lens = jnp.array([140, 8, 0, 0], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 12, 16, 16, 16], dtype=jnp.int32)
        cu_kv_lens = jnp.array([0, 256, 384, 384, 384], dtype=jnp.int32)
        page_indices = jnp.array([4, 9, 13, 0, 0, 0, 0, 0], dtype=jnp.int32)
        pages_per_seq = 2

        backend = DSASparseAttentionBackend(
            index_topk=64,
            index_head_dim=128,
            index_n_heads=8,
            skip_offset=0,
            full_slot={0: 0},
            num_attn_heads=8,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            page_size=page_size,
            mesh=None,
            attention_data_partition_axis="data",
        )
        loc = _spec_token_slots(seq_lens, cu_q_lens, cu_kv_lens, page_indices, T, page_size)
        out_4d, _ = backend._write_idx_4d(
            cache4d, k_new, seq_lens, page_indices, cu_q_lens, cu_kv_lens, "data", loc_=loc
        )
        ref_3d = _scatter_paged(
            cache4d.reshape(total_pages, page_size, idx_dim),
            k_new,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            pages_per_seq,
        ).reshape(cache4d.shape)

        # Live allocated pages [1..total_pages) must match bit-for-bit (page 0 is sentinel)
        np.testing.assert_array_equal(
            np.asarray(out_4d[1:].astype(jnp.float32)),
            np.asarray(ref_3d[1:].astype(jnp.float32)),
        )

    def test_05_glm_dsa_indexer_hadamard_and_fused_wk_weights_parity(self):
        from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer, _hadamard_matrix

        cfg = types.SimpleNamespace(
            hidden_size=256,
            q_lora_rank=128,
            qk_rope_head_dim=64,
            index_head_dim=128,
            index_n_heads=8,
            index_topk=64,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            rope_parameters=None,
            max_position_embeddings=4096,
            quantization_config=None,
        )
        devices = jax.devices()
        mesh = Mesh(np.array(devices[:1]).reshape(1, 1), ("data", "tensor"))
        rngs = nnx.Rngs(0)
        with jax.set_mesh(mesh):
            indexer = GlmDsaIndexer(cfg, layer_id=0, mesh=mesh, rngs=rngs, dtype=jnp.bfloat16)

            T = 12
            hidden = jax.random.normal(jax.random.PRNGKey(1), (T, 256), dtype=jnp.bfloat16)
            q_lora = jax.random.normal(jax.random.PRNGKey(2), (T, 128), dtype=jnp.bfloat16)
            pos = jnp.arange(T, dtype=jnp.int32)

            # Reference: separate Hadamard per tensor + unfused wk / weights_proj
            q_ref0, k_ref0, w_ref0 = indexer(pos, hidden, q_lora)
            self.assertEqual(q_ref0.dtype, jnp.bfloat16, "q_idx must stay bf16 (no fp32 promotion)")
            self.assertEqual(k_ref0.dtype, jnp.bfloat16, "k_idx must stay bf16 (no fp32 promotion)")

            # Check fused wk_weights_proj matches unfused
            fused_ok = indexer.fuse_wk_weights_proj()
            self.assertTrue(fused_ok)
            q_fused, k_fused, w_fused = indexer(pos, hidden, q_lora)
            self.assertEqual(q_fused.dtype, jnp.bfloat16)
            self.assertEqual(k_fused.dtype, jnp.bfloat16)

            q_diff = float(jnp.max(jnp.abs(q_fused.astype(jnp.float32) - q_ref0.astype(jnp.float32))))
            k_diff = float(jnp.max(jnp.abs(k_fused.astype(jnp.float32) - k_ref0.astype(jnp.float32))))
            w_diff = float(jnp.max(jnp.abs(w_fused.astype(jnp.float32) - w_ref0.astype(jnp.float32))))
            self.assertLess(q_diff, 1e-3)
            self.assertLess(k_diff, 2e-2)
            self.assertLess(w_diff, 2e-2)

            # Verify Hadamard matrix orthogonality
            hm = _hadamard_matrix(128, jnp.float32)
            ident = hm @ hm.T
            self.assertLess(float(jnp.max(jnp.abs(ident - jnp.eye(128)))), 1e-5)

    def test_06_dsa_backend_e2e_prefill_and_decode_indexshare(self):
        devices = jax.devices()
        mesh = Mesh(np.array(devices[:1]).reshape(1, 1), ("data", "tensor"))
        page_size = 128
        total_pages = 8
        H = 8
        index_topk = 64

        backend = DSASparseAttentionBackend(
            index_topk=index_topk,
            index_head_dim=128,
            index_n_heads=8,
            skip_offset=0,
            full_slot={0: 0},
            enable_coop_prefill=False,
            num_attn_heads=H,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            page_size=page_size,
            mesh=mesh,
            attention_data_partition_axis="data",
        )

        class _DummyPool:
            def __init__(self, kv0, kv1, idx0):
                self.kvs = {0: kv0, 1: kv1}
                self.idxs = {0: idx0}

            def get_fused_kv_buffer(self, lid):
                return self.kvs[lid]

            def get_indexer_key_buffer(self, slot):
                return self.idxs[slot]

        with jax.set_mesh(mesh):
            sh4 = NamedSharding(mesh, P("data", None, None, None))
            sh3 = NamedSharding(mesh, P("data", "tensor", None))
            sh3_rep = NamedSharding(mesh, P("data", None, None))
            sh2 = NamedSharding(mesh, P("data", None))
            sh1 = NamedSharding(mesh, P("data"))

            kv0 = jax.device_put(jnp.zeros((total_pages, 64, 2, 640), dtype=jnp.bfloat16), sh4)
            kv1 = jax.device_put(jnp.zeros((total_pages, 64, 2, 640), dtype=jnp.bfloat16), sh4)
            idx0 = jax.device_put(jnp.zeros((total_pages, 64, 2, 128), dtype=jnp.bfloat16), sh4)
            pool = _DummyPool(kv0, kv1, idx0)

            # Prefill T=16 tokens into page 1
            T = 16
            md = MLAAttentionMetadata(
                seq_lens=jax.device_put(jnp.array([16, 0], dtype=jnp.int32), sh1),
                cu_q_lens=jax.device_put(jnp.array([0, 16, 16], dtype=jnp.int32), sh1),
                cu_kv_lens=jax.device_put(jnp.array([0, 128, 128], dtype=jnp.int32), sh1),
                page_indices=jax.device_put(jnp.array([1, 0], dtype=jnp.int32), sh1),
                distribution=jax.device_put(jnp.array([0, 0, 1], dtype=jnp.int32), sh1),
            )
            backend.forward_metadata = md

            fb = types.SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                positions=jax.device_put(jnp.arange(T, dtype=jnp.int32), sh1),
                out_cache_loc=jax.device_put(jnp.arange(128, 128 + T, dtype=jnp.int32), sh1),
            )
            layer0 = types.SimpleNamespace(layer_id=0, scaling=1.0 / np.sqrt(576.0), sliding_window_size=None, logit_cap=None)
            layer1 = types.SimpleNamespace(layer_id=1, scaling=1.0 / np.sqrt(576.0), sliding_window_size=None, logit_cap=None)

            ql = jax.device_put(jax.random.normal(jax.random.PRNGKey(10), (T, H, 512), dtype=jnp.bfloat16) * 0.2, sh3)
            qpe = jax.device_put(jax.random.normal(jax.random.PRNGKey(11), (T, H, 64), dtype=jnp.bfloat16) * 0.2, sh3)
            kvc = jax.device_put(jax.random.normal(jax.random.PRNGKey(12), (T, 1, 512), dtype=jnp.bfloat16) * 0.2, sh3_rep)
            kpe = jax.device_put(jax.random.normal(jax.random.PRNGKey(13), (T, 1, 64), dtype=jnp.bfloat16) * 0.2, sh3_rep)
            q_idx = jax.device_put(jax.random.normal(jax.random.PRNGKey(14), (T, 8, 128), dtype=jnp.bfloat16) * 0.2, sh3_rep)
            k_idx = jax.device_put(jax.random.normal(jax.random.PRNGKey(15), (T, 128), dtype=jnp.bfloat16) * 0.2, sh2)
            w_idx = jax.device_put(jnp.ones((T, 8), dtype=jnp.float32) * 0.1, sh2)

            o0, c0 = backend(
                ql, kvc, kvc, layer0, fb, pool,
                q_rope=qpe, k_rope=kpe, indexer_type="full",
                q_idx=q_idx, k_idx=k_idx, idx_weights=w_idx,
            )
            self.assertIsNotNone(c0.topk, "Full prefill layer must emit token topk for IndexShare")
            self.assertEqual(c0.topk.shape, (T, index_topk))
            self.assertTrue(hasattr(fb, "_sc_gather_meta_cache"))

            o1, c1 = backend(
                ql, kvc, kvc, layer1, fb, pool,
                q_rope=qpe, k_rope=kpe, indexer_type="shared",
                dsa_topk_in=c0.topk, dsa_topk_pages_in=c0.topk_pages,
            )
            self.assertIsNone(c1.topk)
            self.assertFalse(bool(jnp.any(jnp.isnan(o0))))
            self.assertFalse(bool(jnp.any(jnp.isnan(o1))))
            # Since layer0 and layer1 saw identical Q/K and seq_len (16) <= index_topk (64),
            # full layer and shared IndexShare layer outputs must match exactly!
            np.testing.assert_allclose(
                np.asarray(o0.astype(jnp.float32)),
                np.asarray(o1.astype(jnp.float32)),
                atol=1e-3,
            )

            # Now run 1-step DECODE at seq_len=17 on the updated caches!
            pool.kvs[0] = c0.kv
            pool.kvs[1] = c1.kv
            pool.idxs[0] = c0.idx
            md_dec = MLAAttentionMetadata(
                seq_lens=jax.device_put(jnp.array([17, 0], dtype=jnp.int32), sh1),
                cu_q_lens=jax.device_put(jnp.array([0, 1, 2], dtype=jnp.int32), sh1),
                cu_kv_lens=jax.device_put(jnp.array([0, 128, 128], dtype=jnp.int32), sh1),
                page_indices=jax.device_put(jnp.array([1, 0], dtype=jnp.int32), sh1),
                distribution=jax.device_put(jnp.array([1, 1, 1], dtype=jnp.int32), sh1),
            )
            backend.forward_metadata = md_dec
            fb_dec = types.SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                positions=jax.device_put(jnp.array([16, 0], dtype=jnp.int32), sh1),
                out_cache_loc=np.empty((1,), dtype=np.int32),  # exercises decode slot derivation!
            )
            ql_d = jax.device_put(ql[:2], sh3)
            qpe_d = jax.device_put(qpe[:2], sh3)
            kvc_d = jax.device_put(kvc[:2], sh3_rep)
            kpe_d = jax.device_put(kpe[:2], sh3_rep)
            qi_d = jax.device_put(q_idx[:2], sh3_rep)
            ki_d = jax.device_put(k_idx[:2], sh2)
            wi_d = jax.device_put(w_idx[:2], sh2)

            od0, cd0 = backend(
                ql_d, kvc_d, kvc_d, layer0, fb_dec, pool,
                q_rope=qpe_d, k_rope=kpe_d, indexer_type="full",
                q_idx=qi_d, k_idx=ki_d, idx_weights=wi_d,
            )
            od1, _ = backend(
                ql_d, kvc_d, kvc_d, layer1, fb_dec, pool,
                q_rope=qpe_d, k_rope=kpe_d, indexer_type="shared",
                dsa_topk_in=cd0.topk, dsa_topk_pages_in=cd0.topk_pages,
            )
            self.assertFalse(bool(jnp.any(jnp.isnan(od0))))
            np.testing.assert_allclose(
                np.asarray(od0[0].astype(jnp.float32)),
                np.asarray(od1[0].astype(jnp.float32)),
                atol=1e-3,
            )
            # Padded decode slot 1 (seq_len == 0) must output zeros
            np.testing.assert_allclose(
                np.asarray(od0[1].astype(jnp.float32)),
                np.zeros((H, 512), dtype=np.float32),
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
