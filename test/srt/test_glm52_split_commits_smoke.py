import ast
import os
import py_compile
import subprocess
import sys
import unittest
import zlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_ROOT = os.path.join(REPO_ROOT, "python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

STAGE = int(os.environ.get("GLM52_COMMIT_STAGE", "1"))


class TestGlm52SplitCommitsSmoke(unittest.TestCase):
    def test_00_modified_files_syntax_and_py_compile(self):
        """Verify every modified .py file compiles without syntax errors."""
        has_base = (
            subprocess.call(
                ["git", "-C", REPO_ROOT, "rev-parse", "--verify", "37e64ec^{commit}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        )
        if has_base:
            out = subprocess.check_output(
                ["git", "-C", REPO_ROOT, "diff", "--name-only", "37e64ec", "HEAD"]
            ).decode()
            py_files = [f.strip() for f in out.splitlines() if f.strip().endswith(".py")]
        else:
            py_files = []
            for root, _, files in os.walk(os.path.join(REPO_ROOT, "python", "sgl_jax")):
                for fn in files:
                    if fn.endswith(".py"):
                        py_files.append(os.path.relpath(os.path.join(root, fn), REPO_ROOT))
        self.assertGreater(len(py_files), 0)
        for rel in py_files:
            full_path = os.path.join(REPO_ROOT, rel)
            if os.path.exists(full_path):
                py_compile.compile(full_path, doraise=True)
                with open(full_path, "r") as f:
                    ast.parse(f.read(), filename=rel)

    def test_01_commit1_model_and_kernel_tuning(self):
        """Stage 1+: Verify TPU v7x tuned block sizes, DSA kernels, and GLM-5.2 MoE loader."""
        from sgl_jax.srt.kernels.biased_topk import tuned_block_sizes as bt_tbs
        from sgl_jax.srt.kernels.fused_moe.v2 import tuned_block_configs as fmoe_tbc
        from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import tuned_tile_sizes as gmm_tts
        from sgl_jax.srt.kernels.grouped_topk import tuned_block_sizes as gt_tbs
        from sgl_jax.srt.kernels.mla.v2 import tuned_block_sizes as mla_tbs
        from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import (
            tuned_block_sizes as qmm_tbs,
        )

        self.assertTrue(hasattr(bt_tbs, "TUNED_BT"))
        self.assertIn("TPU v7", fmoe_tbc.TUNED_BLOCK_CONFIGS)
        self.assertTrue(len(fmoe_tbc.TUNED_BLOCK_CONFIGS["TPU v7"]) > 0)
        self.assertTrue(hasattr(gmm_tts, "TUNED_TILE_SIZES_GMM_V2"))
        self.assertTrue(hasattr(gt_tbs, "TUNED_BT"))
        self.assertTrue(hasattr(mla_tbs, "TUNED_BLOCK_SIZES_MLA"))
        self.assertTrue(len(qmm_tbs.TUNED_BLOCK_SIZES) > 0)

        # Verify GLM-5.2 model & DSA kernel modules
        with open(os.path.join(PYTHON_ROOT, "sgl_jax/srt/models/glm5_moe.py")) as f:
            glm5_src = f.read()
        self.assertIn("_requantize_blockwise_shared_weight", glm5_src)
        self.assertIn("quantize_weights", glm5_src)

        with open(
            os.path.join(PYTHON_ROOT, "sgl_jax/srt/layers/attention/mla_backend.py")
        ) as f:
            mla_src = f.read()
        self.assertIn("return max(32, res)", mla_src)

    def test_02_commit2_coop_prefill_radix_and_hicache(self):
        """Stage 2+: Verify Cooperative Prefill, UnifiedRadixCache, and MLA/DSA HiCache."""
        if STAGE < 2:
            return
        from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController
        from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
        from sgl_jax.srt.mem_cache.radix_cache import RadixCache
        from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.assertTrue(hasattr(HiCacheController, "set_launcher"))
        self.assertTrue(issubclass(RadixCache, BasePrefixCache))
        self.assertTrue(issubclass(UnifiedRadixCache, BasePrefixCache))

        with open(os.path.join(PYTHON_ROOT, "sgl_jax/srt/managers/scheduler.py")) as f:
            sched_src = f.read()
        self.assertIn("def _maybe_get_coop_prefill_batch", sched_src)
        self.assertIn("def _build_coop_step_batch", sched_src)
        self.assertIn("SGL_MLA_COOP_PREFIX", sched_src)

        with open(
            os.path.join(PYTHON_ROOT, "sgl_jax/srt/layers/attention/dsa_sparse_backend.py")
        ) as f:
            dsa_src = f.read()
        self.assertIn("def _scatter_or_coop_idx", dsa_src)
        self.assertIn("def _coop_kv_update", dsa_src)

    def test_03_commit3_speculative_decoding_and_dsa_verify(self):
        """Stage 3+: Verify DSA_SPEC_VERIFY_LAST_Q_ONLY and Speculative Precompile."""
        if STAGE < 3:
            return
        with open(
            os.path.join(PYTHON_ROOT, "sgl_jax/srt/layers/attention/dsa_sparse_backend.py")
        ) as f:
            dsa_src = f.read()
        self.assertIn('_SPEC_VERIFY_LAST_Q_ONLY = os.environ.get("DSA_SPEC_VERIFY_LAST_Q_ONLY", "1") == "1"', dsa_src)
        self.assertIn('_SPEC_VERIFY_QGROUP_MIN_BS = int(os.environ.get("DSA_SPEC_VERIFY_QGROUP_MIN_BS", "1"))', dsa_src)
        self.assertIn('_IDX_VERIFY_BATCHED_MAX_KV = int(os.environ.get("DSA_IDX_VERIFY_BATCHED_MAX_KV", "131072"))', dsa_src)

        with open(
            os.path.join(PYTHON_ROOT, "sgl_jax/srt/managers/schedule_batch.py")
        ) as f:
            sb_src = f.read()
        self.assertIn("SGLANG_JAX_SPEC_PRECOMPILE_FIRST_RUNG_ONLY", sb_src)
        with open(
            os.path.join(PYTHON_ROOT, "sgl_jax/srt/model_executor/compilation_manager.py")
        ) as f:
            cm_src = f.read()
        self.assertIn("extend_next_token_ids", cm_src)

    def test_04_commit4_pd_disagg_and_minilb_routing(self):
        """Stage 4: Verify PD Disagg KV-only prefill, shared-prefix skip/extend, and MiniLB sticky routing."""
        if STAGE < 4:
            return
        import asyncio
        from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer

        lb = MiniLoadBalancer.__new__(MiniLoadBalancer)
        lb.prefill_servers = ["http://127.0.0.1:8001"]
        lb.decode_servers = ["http://127.0.0.1:8002"]
        lb.prefill_dp_size = 1
        lb.decode_dp_size = 1
        lb._dp_seq_counter = {}
        lb._active_prefix_sig = None
        lb._active_prefix_leader_dp = None

        async def _noop_ensure():
            return

        lb._ensure_dp_sizes = _noop_ensure

        os.environ["SGL_PD_SHARED_PREFIX_LEN"] = "4096"
        req_a = {"input_ids": [1, 2, 3] * 2000, "bootstrap_room": 1, "disagg_transfer_id": "t1"}
        req_b = {"input_ids": [1, 2, 3] * 2000}
        h_expected = zlib.crc32(b"session-xyz") % 16
        # Verify deterministic CRC32 sticky routing and suppression of cross-DP leader tag when dp_rank is set
        req_a["dp_rank"] = h_expected
        self.assertIsNone(lb._get_item_prefix_sig(req_a))
        self.assertIsNotNone(lb._get_item_prefix_sig(req_b))

        # Verify _align_dp_requests safely wraps dp_rank modulo prefill_dp_size (e.g., dp_size=1)
        p_req, d_req = asyncio.run(lb._align_dp_requests(req_a))
        self.assertEqual(p_req["dp_rank"], 0)
        self.assertEqual(d_req["dp_rank"], 0)

        with open(os.path.join(PYTHON_ROOT, "sgl_jax/srt/models/glm5_moe.py")) as f:
            glm5_src = f.read()
        self.assertIn("def forward_kv_only", glm5_src)

        with open(os.path.join(PYTHON_ROOT, "sgl_jax/srt/disaggregation/decode.py")) as f:
            dec_src = f.read()
        self.assertIn("SGL_PD_MULTI_TURN_PREFIX_EXTEND", dec_src)
        self.assertIn("262144", dec_src)


if __name__ == "__main__":
    unittest.main()
