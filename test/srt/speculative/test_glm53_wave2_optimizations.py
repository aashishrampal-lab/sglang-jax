"""Unit tests for GLM-5.3 Wave-2 speculative & model optimizations:
1. Horizontal fusion of replicated block-quant `q_a_proj` + `kv_a_proj_with_mqa` (+ `indexer.wk`)
   into a single `(3072, 6144)` `QuantizedLinear` (`fuse_qkv_a_proj` / `fuse_glm5_qkv_a_projections`).
2. Direct 4D `idx_cache` scatter (`use_fast_4d`), hoisted `_fixed_stride_pages_sharded`, and
   `DSA_SKIP_VERIFY_PREWRITE=1` in `DSASparseAttentionBackend`.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import sgl_jax.srt.layers.attention.dsa_sparse_backend as dsa_mod
from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSASparseAttentionBackend
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata
from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.glm5_moe import Glm5Attention, fuse_glm5_qkv_a_projections


def _make_cpu_mesh() -> Mesh:
    devs = np.array(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devs, ("data", "tensor"), axis_types=(AxisType.Explicit, AxisType.Explicit))


def _dequant_blockwise_matmul(x: jax.Array, w_q: jax.Array, w_scale: jax.Array) -> jax.Array:
    """CPU reference matching `xla_quantized_matmul_local` blockwise layout `[in_blocks, 1, n_out]`."""
    in_blocks = w_scale.shape[0]
    n_out, n_in = w_q.shape
    block_k = n_in // in_blocks
    w_f32 = w_q.astype(jnp.float32).reshape(n_out, in_blocks, block_k)
    # w_scale is [in_blocks, 1, n_out] -> transpose to [n_out, in_blocks, 1]
    s_f32 = jnp.transpose(w_scale[:, 0, :].astype(jnp.float32), (1, 0))[:, :, None]
    w_deq = (w_f32 * s_f32).reshape(n_out, n_in)
    return jnp.matmul(x.astype(jnp.float32), w_deq.T, precision=jax.lax.Precision.HIGHEST).astype(
        x.dtype
    )


def _fake_blockwise_kernel(x, w_q, w_scale, block_size, x_q_dtype, tuned_value=None):
    return _dequant_blockwise_matmul(x, w_q, w_scale)


def _populate_static_fp8_linear(
    ql: QuantizedLinear, rng: np.random.Generator, out_dim: int, in_dim: int, block: int = 128
):
    mesh = ql.mesh
    w_np = rng.uniform(-1.0, 1.0, size=(out_dim, in_dim)).astype(np.float32)
    w_q = jnp.asarray(w_np, dtype=jnp.float8_e4m3fn)
    in_blocks = in_dim // block
    out_blocks = (out_dim + block - 1) // block
    s_2d = rng.uniform(0.01, 0.2, size=(out_blocks, in_blocks)).astype(np.float32)
    s_exp = np.repeat(s_2d, repeats=block, axis=0)[:out_dim]  # [out_dim, in_blocks]
    s_3d = jnp.asarray(np.transpose(s_exp, (1, 0))[:, None, :], dtype=jnp.float32)
    ql.weight_q.value = jax.device_put(w_q, NamedSharding(mesh, P(None, None)))
    ql.weight_scale.value = jax.device_put(s_3d, NamedSharding(mesh, P(None, None, None)))


def _build_quantized_glm5_attn(mesh: Mesh, *, has_indexer: bool, seed: int = 0) -> Glm5Attention:
    rng = np.random.default_rng(seed)
    with jax.set_mesh(mesh):
        attn = Glm5Attention(
            hidden_size=6144,
            num_heads=64,
            num_kv_heads=1,
            max_position_embeddings=4096,
            mesh=mesh,
            head_dim=256,
            rms_norm_eps=1e-5,
            layer_id=0 if has_indexer else 1,
            dtype=jnp.bfloat16,
            use_absorbed=True,
            has_indexer=has_indexer,
            indexer_type="full" if has_indexer else "shared",
            use_dsa_sparse=True,
        )
    # Replace q_a_proj, kv_a_proj_with_mqa, and (if present) indexer.wk with static FP8 QuantizedLinear
    attn.q_a_proj = QuantizedLinear.from_linear(
        attn.q_a_proj,
        weight_dtype=jnp.float8_e4m3fn,
        activation_dtype=None,
        is_static_input=False,
        weight_block_size=(128, 128),
        allow_narrow_n_blockwise=True,
    )
    _populate_static_fp8_linear(attn.q_a_proj, rng, 2048, 6144)

    attn.kv_a_proj_with_mqa = QuantizedLinear.from_linear(
        attn.kv_a_proj_with_mqa,
        weight_dtype=jnp.float8_e4m3fn,
        activation_dtype=None,
        is_static_input=False,
        weight_block_size=(128, 128),
        allow_narrow_n_blockwise=True,
    )
    _populate_static_fp8_linear(attn.kv_a_proj_with_mqa, rng, 576, 6144)

    if has_indexer:
        attn.indexer.wk = QuantizedLinear.from_linear(
            attn.indexer.wk,
            weight_dtype=jnp.float8_e4m3fn,
            activation_dtype=None,
            is_static_input=False,
            weight_block_size=(128, 128),
            allow_narrow_n_blockwise=True,
        )
        _populate_static_fp8_linear(attn.indexer.wk, rng, 128, 6144)
    return attn


def test_fuse_qkv_a_proj_parity_full_and_shared():
    mesh = _make_cpu_mesh()
    rng = np.random.default_rng(42)
    T = 10
    x = jax.device_put(
        jnp.asarray(rng.standard_normal((T, 6144)), dtype=jnp.bfloat16),
        NamedSharding(mesh, P("data", None)),
    )
    positions = jax.device_put(jnp.arange(T, dtype=jnp.int32), NamedSharding(mesh, P("data")))

    with mock.patch(
        "sgl_jax.srt.kernels.quantized_matmul.kernel.get_blockwise_kernel",
        return_value=_fake_blockwise_kernel,
    ):
        for has_indexer in (True, False):
            attn_ref = _build_quantized_glm5_attn(mesh, has_indexer=has_indexer, seed=7)
            attn_fused = _build_quantized_glm5_attn(mesh, has_indexer=has_indexer, seed=7)

            # Record unfused outputs
            captured_ref = {}

            def _capture_mqa_ref(q_nope, q_rope, compressed, k_rope, fb, pool, **kw):
                captured_ref["q_nope"] = q_nope
                captured_ref["q_rope"] = q_rope
                captured_ref["compressed"] = compressed
                captured_ref["k_rope"] = k_rope
                captured_ref["kw"] = kw
                return jnp.zeros((T, attn_ref.num_heads * attn_ref.v_head_dim), jnp.bfloat16), None

            attn_ref._forward_mqa = _capture_mqa_ref
            fb = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_kvshare_readonly=False,
                spec_info=SimpleNamespace(draft_token_num=5),
                attn_backend=SimpleNamespace(
                    forward_kv_only=lambda c_kv, k_pe, layer, fb_, pool_, **kw: (c_kv, k_pe, kw)
                ),
            )
            with jax.set_mesh(mesh):
                out_ref, _ = attn_ref(positions, x, fb, None)
                kv_only_ref = attn_ref.forward_kv_only(positions, x, fb, None)

            # Fuse projections on attn_fused
            count = fuse_glm5_qkv_a_projections(attn_fused)
            assert count == 1
            assert attn_fused.qkv_a_proj is not None
            assert attn_fused.qkv_a_proj.weight_q.value.shape == (3072, 6144)
            assert attn_fused.qkv_a_proj.weight_scale.value.shape == (48, 1, 3072)
            assert attn_fused.q_a_proj is None
            assert attn_fused.kv_a_proj_with_mqa is None
            if has_indexer:
                assert attn_fused._qkv_a_has_wk is True
                assert attn_fused.indexer.wk is None
                assert attn_fused._qkv_a_offsets == (2048, 2048, 2624, 2816, 2944)
            else:
                assert attn_fused._qkv_a_has_wk is False
                assert attn_fused._qkv_a_offsets == (2048, 2048, 2624, 0, 0)

            # Verify nnx.split + nnx.merge + jax.jit graph traversal works cleanly
            graphdef, state = nnx.split(attn_fused)
            flat_paths = {".".join(map(str, p)) for p, _ in state.flat_state()}
            assert "qkv_a_proj.weight_q" in flat_paths
            assert "qkv_a_proj.weight_scale" in flat_paths
            assert "q_a_proj.weight_q" not in flat_paths
            assert "kv_a_proj_with_mqa.weight_q" not in flat_paths
            if has_indexer:
                assert "indexer.wk.weight_q" not in flat_paths

            captured_fused = {}

            def _capture_mqa_fused(q_nope, q_rope, compressed, k_rope, fb, pool, **kw):
                captured_fused["q_nope"] = q_nope
                captured_fused["q_rope"] = q_rope
                captured_fused["compressed"] = compressed
                captured_fused["k_rope"] = k_rope
                captured_fused["kw"] = kw
                return (
                    jnp.zeros((T, attn_fused.num_heads * attn_fused.v_head_dim), jnp.bfloat16),
                    None,
                )

            attn_fused = nnx.merge(graphdef, state)
            attn_fused._forward_mqa = _capture_mqa_fused
            with jax.set_mesh(mesh):
                out_fused, _ = attn_fused(positions, x, fb, None)
                kv_only_fused = attn_fused.forward_kv_only(positions, x, fb, None)

            np.testing.assert_array_equal(np.asarray(out_ref), np.asarray(out_fused))
            for k in ("q_nope", "q_rope", "compressed", "k_rope"):
                np.testing.assert_array_equal(
                    np.asarray(captured_ref[k]),
                    np.asarray(captured_fused[k]),
                    err_msg=f"Mismatch on {k} (has_indexer={has_indexer})",
                )
            if has_indexer:
                for k in ("q_idx", "k_idx", "idx_weights"):
                    np.testing.assert_array_equal(
                        np.asarray(captured_ref["kw"][k]),
                        np.asarray(captured_fused["kw"][k]),
                        err_msg=f"Mismatch on indexer {k}",
                    )
                np.testing.assert_array_equal(
                    np.asarray(kv_only_ref[2]["k_idx"]),
                    np.asarray(kv_only_fused[2]["k_idx"]),
                )
            np.testing.assert_array_equal(np.asarray(kv_only_ref[0]), np.asarray(kv_only_fused[0]))
            np.testing.assert_array_equal(np.asarray(kv_only_ref[1]), np.asarray(kv_only_fused[1]))


def test_fast_4d_idx_scatter_and_skip_verify_prewrite_parity():
    mesh = _make_cpu_mesh()
    page_size = 64
    pk = 2
    idx_dim = 128
    h_idx = 32
    pages_per_seq = 4
    num_seqs = 4  # 3 active (including 1 page-boundary straddle at kv_len=66, G=5) + 1 padded seq
    G = 5
    T = num_seqs * G
    num_pages = 32

    backend = DSASparseAttentionBackend(
        index_topk=256,
        index_head_dim=idx_dim,
        index_n_heads=h_idx,
        skip_offset=0,
        full_slot={0: 0},
        num_attn_heads=64,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        page_size=page_size,
        mesh=mesh,
    )

    rng = np.random.default_rng(123)
    # seq 0: kv_len=66 (straddles page 0 -> page 1 with G=5: slots 61,62,63,64,65)
    # seq 1: kv_len=40 (single page)
    # seq 2: kv_len=130 (straddles page 1 -> page 2: slots 125,126,127,128,129)
    # seq 3: kv_len=0 (padded inactive slot)
    seq_lens = jax.device_put(jnp.array([66, 40, 130, 0], jnp.int32), NamedSharding(mesh, P("data")))
    cu_q_lens = jax.device_put(
        jnp.array([0, 5, 10, 15, 15], jnp.int32), NamedSharding(mesh, P("data"))
    )
    cu_kv_lens = jax.device_put(
        jnp.arange(num_seqs + 1, dtype=jnp.int32) * (pages_per_seq * page_size),
        NamedSharding(mesh, P("data")),
    )
    # Assign disjoint physical pages >= 1 to all sequences
    page_indices = jax.device_put(
        jnp.arange(1, 1 + num_seqs * pages_per_seq, dtype=jnp.int32),
        NamedSharding(mesh, P("data")),
    )
    dist = jax.device_put(jnp.array([0, 0, 3], jnp.int32), NamedSharding(mesh, P("data")))
    md = MLAAttentionMetadata(
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        distribution=dist,
    )

    # With _SPEC_VERIFY_LAST_Q_ONLY=True, q_idx and idx_weights can be [num_seqs, H, D]
    q_idx = jax.device_put(
        jnp.asarray(rng.standard_normal((num_seqs, h_idx, idx_dim)), dtype=jnp.bfloat16),
        NamedSharding(mesh, P("data", None, None)),
    )
    k_idx = jax.device_put(
        jnp.asarray(rng.standard_normal((T, idx_dim)), dtype=jnp.bfloat16),
        NamedSharding(mesh, P("data", None)),
    )
    idx_w = jax.device_put(
        jnp.asarray(rng.uniform(0.1, 1.0, (num_seqs, h_idx)), dtype=jnp.bfloat16),
        NamedSharding(mesh, P("data", None)),
    )
    init_cache = jax.device_put(
        jnp.asarray(
            rng.standard_normal((num_pages, page_size // pk, pk, idx_dim)), dtype=jnp.bfloat16
        ),
        NamedSharding(mesh, P("data", None, None, None)),
    )

    fb = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_kvshare_readonly=False,
        spec_info=SimpleNamespace(draft_token_num=G),
    )

    # Stub `streamindex_page_topk` with causal reference scoring on CPU so we can test
    # `_INDEXER_KERNEL_PREFILL=True` end-to-end on CPU!
    def _cpu_streamindex_page_topk(
        q_, w_, cache4d_, sl_, fixed_pi_, cuq_, num_valid_seqs_, *, k_pages, pages_per_seq, **_
    ):
        c3d = cache4d_.reshape(cache4d_.shape[0], page_size, cache4d_.shape[3])
        S_loc = sl_.shape[0]
        cukv_fixed = jnp.arange(S_loc + 1, dtype=jnp.int32) * (pages_per_seq * page_size)
        dist_fixed = jnp.stack(
            [jnp.int32(0), jnp.int32(0), num_valid_seqs_.astype(jnp.int32)]
        )
        return dsa_mod.streamindex_page_topk_ref(
            q_,
            w_,
            c3d,
            sl_,
            fixed_pi_,
            cuq_,
            cukv_fixed,
            dist_fixed,
            k_pages=k_pages,
            pages_per_seq=pages_per_seq,
            one_token_per_seq=False,
        )

    with (
        jax.set_mesh(mesh),
        mock.patch.object(dsa_mod, "_INDEXER_KERNEL_PREFILL", True),
        mock.patch.object(dsa_mod, "_IDX_VERIFY_BATCHED", False),
        mock.patch.object(dsa_mod, "_SPEC_VERIFY_QGROUP", True),
        mock.patch.object(dsa_mod, "_SPEC_VERIFY_LAST_Q_ONLY", True),
        mock.patch.object(dsa_mod, "streamindex_page_topk", _cpu_streamindex_page_topk),
    ):
        spec_loc = backend._spec_token_slots_sharded(md, T, "data")
        fixed_pi = backend._fixed_stride_pages_sharded(md, "data")

        # Baseline (no fast_4d, no precomputed_fixed_pi)
        with mock.patch.dict(os.environ, {"DSA_SPEC_FAST_4D_IDX": "0"}):
            cache_ref, pages_ref = backend._maybe_index_prefill_pages(
                True,
                q_idx,
                k_idx,
                idx_w,
                init_cache,
                "data",
                md,
                compute_pages=True,
                num_queries_per_block=16,
                q_group=G,
                forward_batch=fb,
                precomputed_loc=None,
                precomputed_fixed_pi=None,
            )

        # Optimized (use_fast_4d + precomputed_loc + precomputed_fixed_pi)
        with mock.patch.dict(os.environ, {"DSA_SPEC_FAST_4D_IDX": "1"}):
            cache_opt, pages_opt = backend._maybe_index_prefill_pages(
                True,
                q_idx,
                k_idx,
                idx_w,
                init_cache,
                "data",
                md,
                compute_pages=True,
                num_queries_per_block=16,
                q_group=G,
                forward_batch=fb,
                precomputed_loc=spec_loc,
                precomputed_fixed_pi=fixed_pi,
            )

        # Live pages 1..num_pages-1 must be bit-identical (page 0 is the unused sentinel page)
        np.testing.assert_array_equal(np.asarray(cache_ref[1:]), np.asarray(cache_opt[1:]))
        np.testing.assert_array_equal(np.asarray(pages_ref), np.asarray(pages_opt))

        # Verify DSA_SKIP_VERIFY_PREWRITE=1 skips _prewrite_spec_kv on grouped TARGET_VERIFY
        # while still calling _prewrite_spec_kv on DRAFT_EXTEND step 0 (reuse=False).
        q_nope = jax.device_put(
            jnp.zeros((T, 64, 512), jnp.bfloat16), NamedSharding(mesh, P("data", "tensor", None))
        )
        q_rope = jax.device_put(
            jnp.zeros((T, 64, 64), jnp.bfloat16), NamedSharding(mesh, P("data", "tensor", None))
        )
        new_kvc = jax.device_put(
            jnp.ones((T, 512), jnp.bfloat16), NamedSharding(mesh, P("data", None))
        )
        new_kpe = jax.device_put(
            jnp.ones((T, 64), jnp.bfloat16), NamedSharding(mesh, P("data", None))
        )
        kv_cache = jax.device_put(
            jnp.zeros((num_pages, page_size // pk, pk, 640), jnp.bfloat16),
            NamedSharding(mesh, P("data", None, None, None)),
        )

        prewrite_calls = []
        orig_prewrite = backend._prewrite_spec_kv

        def _spy_prewrite(*a, **kw):
            prewrite_calls.append(1)
            return orig_prewrite(*a, **kw)

        with (
            mock.patch.object(backend, "_prewrite_spec_kv", side_effect=_spy_prewrite),
            mock.patch.object(
                backend,
                "_run_sparse",
                side_effect=lambda ql, qpe, kvc, kpe, cache, *a, **kw: (ql, cache),
            ),
        ):
            with mock.patch.dict(os.environ, {"DSA_SKIP_VERIFY_PREWRITE": "1"}):
                backend._run_spec_as_decode(
                    q_nope,
                    q_rope,
                    new_kvc,
                    new_kpe,
                    kv_cache,
                    init_cache,
                    q_idx,
                    k_idx,
                    idx_w,
                    True,
                    None,
                    None,
                    False,
                    1.0,
                    "data",
                    md,
                    forward_batch=fb,
                )
            assert len(prewrite_calls) == 0, "TARGET_VERIFY grouped should skip _prewrite_spec_kv"

            # DRAFT_EXTEND step 0 (reuse=False, ungrouped) MUST still prewrite!
            fb_draft = SimpleNamespace(
                forward_mode=ForwardMode.DRAFT_EXTEND,
                spec_kvshare_readonly=False,
                seq_lens=seq_lens,
            )
            q_idx_full = jax.device_put(
                jnp.zeros((T, h_idx, idx_dim), jnp.bfloat16),
                NamedSharding(mesh, P("data", None, None)),
            )
            idx_w_full = jax.device_put(
                jnp.ones((T, h_idx), jnp.bfloat16),
                NamedSharding(mesh, P("data", None)),
            )
            with mock.patch.dict(os.environ, {"DSA_SKIP_VERIFY_PREWRITE": "1"}):
                backend._run_spec_as_decode(
                    q_nope,
                    q_rope,
                    new_kvc,
                    new_kpe,
                    kv_cache,
                    init_cache,
                    q_idx_full,
                    k_idx,
                    idx_w_full,
                    True,
                    None,
                    None,
                    False,
                    1.0,
                    "data",
                    md,
                    forward_batch=fb_draft,
                )
            assert len(prewrite_calls) == 1, "DRAFT_EXTEND step 0 (reuse=False) must still prewrite"


def test_pd_pin_prefix_slots_prevents_eviction_on_decode_and_prefill():
    from sgl_jax.srt.disaggregation.base.transfer import AdmissionState
    from sgl_jax.srt.disaggregation.decode import (
        DecodeBookkeeping,
        DecodePreallocQueue,
        DecodeTransferQueue,
        SchedulerDisaggregationDecodeMixin,
    )

    page_size = 128

    class _FakeAllocator:
        def __init__(self, total_tokens: int):
            self.page_size = page_size
            self._free = list(range(total_tokens))

        def available_size(self, dp_rank: int = 0) -> int:
            return len(self._free)

        def alloc(self, n: int, dp_rank: int = 0):
            if n > len(self._free):
                return None
            out = np.asarray(self._free[:n], dtype=np.int32)
            self._free = self._free[n:]
            return out

        def release(self, arr):
            self._free.extend(int(x) for x in np.asarray(arr).tolist())

    class _FakeDecodeSched(SchedulerDisaggregationDecodeMixin):
        def __init__(self, total_tokens: int):
            self.token_to_kv_pool_allocator = _FakeAllocator(total_tokens)
            self.server_args = SimpleNamespace(
                disaggregation_num_reserved_decode_tokens=128,
                disaggregation_max_inflight_transfers=32,
                disaggregation_pull_timeout_seconds=3600.0,
            )
            self.dp_size = 1
            self.disagg_prealloc_queue = DecodePreallocQueue()
            self.disagg_transfer_queue = DecodeTransferQueue()
            self.waiting_queue = []
            self.running_batch = None
            self.per_dp_max_running_requests = 256
            self.spec_algorithm = SimpleNamespace(is_none=lambda: False)
            self.enable_overlap = False
            self._pd_decode_prefix_cache = {}
            self.disagg_kv_manager = SimpleNamespace(
                poll_decode_metadata=lambda ctx: True,
                try_start_decode=lambda ctx: SimpleNamespace(
                    state=AdmissionState.ADMITTED, receiver=SimpleNamespace()
                ),
            )

        def _release_decode_kv_indices(self, indices, dp_rank=0):
            self.token_to_kv_pool_allocator.release(indices)

        def _build_kv_spec_for_req(self, req):
            return None

        def _commit_direct_received_kv(self, req, indices, expected):
            return None

        def _pd_mark_time(self, req, stage):
            return None

        def _req_pull_bytes_per_device(self, req):
            return 0

        def _inflight_pull_bytes_per_device(self):
            return 0

        def _pull_budget_bytes_per_device(self):
            return 1 << 60

    def _make_req(rid: str, trunk_id: int, suffix_seed: int, total_len: int = 4096, prefix_len: int = 3200):
        prefix = [1000 + trunk_id] * prefix_len
        suffix = [50000 + suffix_seed + (i % 97) for i in range(total_len - prefix_len)]
        return SimpleNamespace(
            rid=rid,
            origin_input_ids=prefix + suffix,
            dp_rank=0,
            disagg_prefill_dp_rank=0,
            disagg_transfer_id=rid,
            bootstrap_room=1,
            cached_tokens=0,
            sampling_params=SimpleNamespace(max_new_tokens=128, ignore_eos=False),
            output_ids=[],
        )

    def _prefill_compute_skip(slot_list: list[list[int]], req, max_slots: int, pin: bool) -> int:
        seqlen = len(req.origin_input_ids)
        target_prefix_len = 262144
        min_prefix_skip = 2048
        skip_tokens = 0
        matched_idx = -1
        cap_cmp = min(target_prefix_len, ((seqlen - page_size) // page_size) * page_size)
        for s_idx in range(len(slot_list) - 1, -1, -1):
            prev_prefix_ids = slot_list[s_idx]
            cmp_len = min(len(prev_prefix_ids), cap_cmp + 1)
            if cmp_len >= min_prefix_skip:
                arr_cur = np.asarray(req.origin_input_ids[:cmp_len], dtype=np.int32)
                arr_prev = np.asarray(prev_prefix_ids[:cmp_len], dtype=np.int32)
                diff = np.nonzero(arr_cur != arr_prev)[0]
                lcp_tokens = int(diff[0]) if len(diff) > 0 else cmp_len
                usable_lcp = max(0, lcp_tokens - 1) if lcp_tokens < cmp_len else min(lcp_tokens, cap_cmp)
                cand_skip = (usable_lcp // page_size) * page_size
                if cand_skip >= min_prefix_skip and seqlen >= cand_skip + page_size and cand_skip > skip_tokens:
                    skip_tokens = cand_skip
                    matched_idx = s_idx
        if skip_tokens > 0 and matched_idx >= 0:
            matched_ids = slot_list.pop(matched_idx)
            slot_list.append(matched_ids)
        elif skip_tokens == 0:
            cap_len = min(target_prefix_len, ((seqlen - page_size) // page_size) * page_size)
            if cap_len >= min_prefix_skip and not (pin and len(slot_list) >= max_slots):
                while len(slot_list) >= max_slots and slot_list:
                    slot_list.pop(0)
                slot_list.append(list(req.origin_input_ids[: cap_len + 1]))
        return skip_tokens

    sched = _FakeDecodeSched(total_tokens=32768)
    prefill_slots: list[list[int]] = []

    with mock.patch.dict(
        os.environ,
        {
            "SGL_PD_MAX_PREFIX_SLOTS": "4",
            "SGL_PD_PIN_PREFIX_SLOTS": "1",
            "SGL_PD_MIN_PREFIX_SKIP_TOKENS": "2048",
        },
    ):
        # Stage 1 warmup: pin trunks 0..3 (even if prefill and decode see reversed order within 0..3!)
        for k in [0, 1, 2, 3]:
            r_p = _make_req(f"w_p_{k}", trunk_id=k, suffix_seed=1)
            assert _prefill_compute_skip(prefill_slots, r_p, max_slots=4, pin=True) == 0
        for k in [3, 1, 2, 0]:
            r_d = _make_req(f"w_d_{k}", trunk_id=k, suffix_seed=1)
            bk = DecodeBookkeeping(req_id=r_d.rid, req=r_d, p_info={})
            sched.disagg_prealloc_queue.add(bk)
            sched._admit_decode_prealloc()
            sched.disagg_transfer_queue.abort_matching(r_d.rid, False)
            assert r_d._pd_transfer_skip_tokens == 0
            # Simulate warmup request finishing (ref_count drops to 0, prefix stays pinned)
            r_d._pd_shared_prefix_entry["ref_count"] -= 1
            sched.token_to_kv_pool_allocator.release(bk.kv_indices[r_d._pd_shared_prefix_len :])

        assert len(prefill_slots) == 4
        assert len(sched._pd_decode_prefix_cache[0]) == 4

        # Stage 2 warmup & Phase 1: send unpinned trunks 4..9 and interleaved pinned trunks 0..3
        # Verify 100% skip_tokens agreement between prefill and decode, and zero eviction of 0..3!
        for turn, k in enumerate([4, 5, 0, 6, 2, 7, 3, 1, 8, 0]):
            r_p = _make_req(f"t_{turn}_p", trunk_id=k, suffix_seed=100 + turn)
            r_d = _make_req(f"t_{turn}_d", trunk_id=k, suffix_seed=100 + turn)
            skip_p = _prefill_compute_skip(prefill_slots, r_p, max_slots=4, pin=True)
            bk = DecodeBookkeeping(req_id=r_d.rid, req=r_d, p_info={})
            sched.disagg_prealloc_queue.add(bk)
            sched._admit_decode_prealloc()
            sched.disagg_transfer_queue.abort_matching(r_d.rid, False)
            skip_d = r_d._pd_transfer_skip_tokens

            expected_skip = 3072 if k < 4 else 0
            assert skip_p == expected_skip, f"Prefill skip mismatch for trunk {k}: {skip_p}"
            assert skip_d == expected_skip, f"Decode skip mismatch for trunk {k}: {skip_d}"
            if r_d._pd_shared_prefix_entry is not None:
                r_d._pd_shared_prefix_entry["ref_count"] -= 1
                sched.token_to_kv_pool_allocator.release(bk.kv_indices)
            else:
                sched.token_to_kv_pool_allocator.release(bk.kv_indices)

        assert len(sched._pd_decode_prefix_cache[0]) == 4

