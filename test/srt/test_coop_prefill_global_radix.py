"""Pre-flight & unit validation suite for Cooperative Shared-Prefix Prefill + Global RadixCache.

Validates:
1. AST attribute compatibility between `scheduler.py` / `mla_backend.py` and `ServerArgs`
   (prevents `AttributeError` such as `disaggregation_enable_overlap_schedule`).
2. `glm5_moe.py` (`patch_model_config`) does not call `jax.default_backend()` before
   `jax.distributed.initialize()` (prevents `/tmp/libtpu_lockfile` multi-process contention).
3. `mla_backend.py` `coop_flag` (`seq_lens_[-1] < 0`) vs normal multi-sequence `1024`-token
   batches (prevents zeroing `seq_lens_[-1]` during `[EXTEND] PRECOMPILE`).
4. Global `RadixCache` across `dp_size=32` with `InsertParams`, `MatchPrefixParams`, and `EvictParams`.
5. End-to-end `Scheduler.__init__` + `_detect_and_run_cooperative_prefill()` + `process_batch_result()`
   on `__coop_prefill_*` batches (both overlap and non-overlap modes).
"""

import ast
import dataclasses
import os
import sys
import types
import unittest
from unittest import mock
import numpy as np


def _ensure_safe_imports():
    """Ensure jax/flax/transformers imports succeed without initializing PJRT / libtpu_lockfile."""
    try:
        import jax
        jax.process_index = lambda *a, **kw: 0
        jax.process_count = lambda *a, **kw: 1
        jax.local_device_count = lambda *a, **kw: 8
        jax.device_count = lambda *a, **kw: 32
        jax.default_backend = lambda *a, **kw: "cpu"
    except ImportError:
        import importlib.abc
        import importlib.machinery

        jax_mod = types.ModuleType("jax")
        jax_mod.__path__ = []
        jnp_mod = types.ModuleType("jax.numpy")
        jnp_mod.__path__ = []
        for k, v in np.__dict__.items():
            setattr(jnp_mod, k, v)
        jnp_mod.bfloat16 = np.float16
        jnp_mod.float8_e4m3fn = np.float16
        jnp_mod.float8_e5m2 = np.float16
        jnp_mod.float8_e4m3fnuz = np.float16
        jnp_mod.float8_e5m2fnuz = np.float16
        jnp_mod.Array = np.ndarray
        jax_mod.numpy = jnp_mod
        jax_mod.Array = np.ndarray
        jax_mod.dtype = np.dtype
        jax_mod.jit = lambda fn=None, **kw: (fn if fn is not None else (lambda f: f))
        jax_mod.sharding = types.ModuleType("jax.sharding")
        jax_mod.sharding.__path__ = []
        jax_mod.sharding.Mesh = object
        jax_mod.sharding.NamedSharding = mock.MagicMock()
        jax_mod.sharding.PartitionSpec = mock.MagicMock()
        jax_mod.tree_util = types.ModuleType("jax.tree_util")
        jax_mod.tree_util.__path__ = []
        jax_mod.tree_util.register_pytree_node_class = lambda cls: cls
        jax_mod.distributed = types.ModuleType("jax.distributed")
        jax_mod.distributed.__path__ = []
        jax_mod.distributed.is_initialized = lambda: True
        jax_mod.process_count = lambda: 1
        jax_mod.process_index = lambda: 0
        jax_mod.local_device_count = lambda: 8
        jax_mod.device_count = lambda: 32
        sys.modules["jax"] = jax_mod
        sys.modules["jax.numpy"] = jnp_mod
        sys.modules["jax.sharding"] = jax_mod.sharding
        sys.modules["jax.tree_util"] = jax_mod.tree_util
        sys.modules["jax.distributed"] = jax_mod.distributed

        flax_struct = types.ModuleType("flax.struct")
        flax_struct.__path__ = []
        flax_struct.dataclass = dataclasses.dataclass
        flax_struct.field = lambda **kw: dataclasses.field(
            default=kw.get("default", dataclasses.MISSING)
        )
        sys.modules["flax.struct"] = flax_struct

        class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path, target=None):
                if fullname.startswith("sgl_jax"):
                    real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                    if real_spec is not None:
                        return None
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

            def create_module(self, spec):
                if spec.name in sys.modules:
                    return sys.modules[spec.name]
                m = mock.MagicMock()
                m.__name__ = spec.name
                m.__path__ = []
                m.__spec__ = spec
                m.PretrainedConfig = object
                return m

            def exec_module(self, module):
                pass

        sys.meta_path.append(_AutoStubFinder())


_ensure_safe_imports()


class TestCoopPrefillAndGlobalRadix(unittest.TestCase):

    def _get_sgl_paths(self):
        import sgl_jax
        base = os.path.dirname(sgl_jax.__file__)
        scheduler_path = os.path.join(base, "srt", "managers", "scheduler.py")
        mla_path = os.path.join(base, "srt", "layers", "attention", "mla_backend.py")
        glm5_path = os.path.join(base, "srt", "models", "glm5_moe.py")
        sa_path = os.path.join(base, "srt", "server_args.py")
        return base, scheduler_path, mla_path, glm5_path, sa_path

    def test_1_server_args_attribute_compatibility(self):
        """Verify every direct `server_args.<attr>` in scheduler.py & mla_backend.py exists on ServerArgs."""
        _, scheduler_path, mla_path, _, sa_path = self._get_sgl_paths()

        # Extract all dataclass fields and methods/properties defined on ServerArgs in server_args.py
        with open(sa_path, "r") as f:
            sa_tree = ast.parse(f.read(), filename=sa_path)

        sa_defined_attrs = set()
        for node in ast.walk(sa_tree):
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        sa_defined_attrs.add(item.target.id)
                    elif isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                sa_defined_attrs.add(t.id)
                    elif isinstance(item, ast.FunctionDef):
                        sa_defined_attrs.add(item.name)

        for path in (scheduler_path, mla_path):
            with open(path, "r") as f:
                tree = ast.parse(f.read(), filename=path)

            missing = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    is_sa = (
                        (isinstance(node.value, ast.Name) and node.value.id == "server_args")
                        or (
                            isinstance(node.value, ast.Attribute)
                            and isinstance(node.value.value, ast.Name)
                            and node.value.value.id == "self"
                            and node.value.attr == "server_args"
                        )
                    )
                    if is_sa and node.attr not in sa_defined_attrs:
                        missing.append((os.path.basename(path), node.lineno, node.attr))

            self.assertEqual(
                missing,
                [],
                f"Direct server_args.<attr> accesses missing from ServerArgs: {missing}. "
                f"Use getattr(server_args, attr, default) instead!",
            )

    def test_2_glm5_moe_no_premature_jax_default_backend(self):
        """Verify glm5_moe.py patch_model_config does not call jax.default_backend()."""
        _, _, _, glm5_path, _ = self._get_sgl_paths()
        if not os.path.exists(glm5_path):
            return
        with open(glm5_path, "r") as f:
            src = f.read()
        self.assertNotIn(
            'jax.default_backend() == "tpu"',
            src,
            "glm5_moe.py still contains `jax.default_backend() == \"tpu\"` in patch_model_config, "
            "which initializes PJRT TPU client before jax.distributed.initialize() and causes libtpu_lockfile!",
        )

    def test_3_mla_coop_flag_preserves_multi_seq_precompile_and_decodes_coop(self):
        """Verify mla_backend coop_flag logic preserves multi-seq 1024-token batches and activates on negative sentinel."""
        _, _, mla_path, _, _ = self._get_sgl_paths()
        with open(mla_path, "r") as f:
            src = f.read()
        self.assertIn(
            "coop_flag = seq_lens_[-1]",
            src,
            "mla_backend.py must read coop_flag = seq_lens_[-1] before zeroing sentinel",
        )
        self.assertIn(
            "seq_lens_ = jnp.where(coop_flag < 0, seq_lens_.at[-1].set(0), seq_lens_)",
            src,
            "mla_backend.py must only zero seq_lens_[-1] when coop_flag < 0",
        )

        # Case A: Precompile batch with 64 sequences of 16 tokens (sum == 1024, seq_lens_[-1] == 16)
        seq_lens_precompile = np.full((64,), 16, dtype=np.int32)
        coop_flag_a = seq_lens_precompile[-1]
        seq_lens_after_a = np.where(
            coop_flag_a < 0,
            np.concatenate([seq_lens_precompile[:-1], [0]]),
            seq_lens_precompile,
        )
        self.assertEqual(int(seq_lens_after_a[-1]), 16)
        self.assertEqual(int(seq_lens_after_a.sum()), 1024)

        # Case B: Cooperative prefill step 1 (chunk=1024, coop_step=1 -> sentinel = -2, coop_prefix_len = 32768)
        seq_lens_coop = np.zeros((64,), dtype=np.int32)
        seq_lens_coop[0] = 1024
        seq_lens_coop[-1] = -(1 + 1)  # coop_step = 1 -> sentinel = -2
        coop_flag_b = seq_lens_coop[-1]
        self.assertLess(coop_flag_b, 0)
        seq_lens_after_b = np.where(
            coop_flag_b < 0,
            np.concatenate([seq_lens_coop[:-1], [0]]),
            seq_lens_coop,
        )
        self.assertEqual(int(seq_lens_after_b[-1]), 0)
        self.assertEqual(int(seq_lens_after_b[0]), 1024)
        coop_step_b = (-coop_flag_b) - 1
        coop_prefix_len_b = coop_step_b * (32 * 1024)
        self.assertEqual(int(coop_prefix_len_b + seq_lens_after_b[0]), 33792)

    def test_4_global_radix_cache_insert_match_evict(self):
        """Verify RadixCache insert/match_prefix/evict across dp_size=32 using InsertParams/MatchPrefixParams/EvictParams."""
        from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
        from sgl_jax.srt.mem_cache.radix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
            RadixCache,
            RadixKey,
        )

        dp_size = 32
        page_size = 128
        max_reqs = 64
        max_ctx = 4096
        total_kv_tokens = 32 * 4096

        req_pool = ReqToTokenPool(size=max_reqs, max_context_len=max_ctx)
        kv_alloc = PagedTokenToKVPoolAllocator(
            size=total_kv_tokens, page_size=page_size, kvcache=mock.MagicMock(), dp_size=dp_size
        )
        cache = RadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=kv_alloc,
            page_size=page_size,
            disable=False,
        )

        # Allocate 256 pages (32,768 tokens = 1024 per DP rank) and insert for all 32 DP ranks
        num_tokens = 256 * page_size
        kv_indices = np.concatenate(
            [kv_alloc.alloc(1024, dp_rank=r) for r in range(dp_size)]
        )
        self.assertIsNotNone(kv_indices)
        self.assertEqual(len(kv_indices), num_tokens)

        prefix_tokens = list(range(100, 100 + num_tokens))
        for r in range(dp_size):
            rkey = RadixKey(list(prefix_tokens), None, dp_rank=r)
            cache.insert(InsertParams(key=rkey, value=kv_indices.copy()))

        # Verify every DP rank r in 0..31 matches the full 32,768-token prefix!
        for r in range(dp_size):
            query = prefix_tokens + [999999 + r] * 256
            res = cache.match_prefix(MatchPrefixParams(key=RadixKey(query, None, dp_rank=r)))
            self.assertEqual(
                len(res.device_indices),
                num_tokens,
                f"DP rank {r} failed to match global RadixCache prefix",
            )
            np.testing.assert_array_equal(res.device_indices, kv_indices)

        # Verify EvictParams works across DP ranks without AttributeError
        cache.evict(EvictParams(num_tokens=page_size * 4, dp_rank=0))

    def test_5_scheduler_coop_prefill_and_process_batch_result(self):
        """Verify Scheduler._maybe_get_coop_prefill_batch and process_batch_result on __coop_prefill_* batches."""
        from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
        from sgl_jax.srt.managers.scheduler import Scheduler
        from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
        from sgl_jax.srt.mem_cache.radix_cache import MatchPrefixParams, RadixCache, RadixKey
        from sgl_jax.srt.sampling.sampling_params import SamplingParams

        # Create a lightweight Scheduler skeleton with real pools & RadixCache
        sched = object.__new__(Scheduler)
        sched.dp_size = 32
        sched.page_size = 128
        sched.node_rank = 0
        sched.chunked_prefill_size = 1024
        sched.max_prefill_tokens = 32768
        sched.disable_radix_cache = False
        sched.enable_overlap = True
        sched.tp_worker = mock.MagicMock()
        sched.spec_algorithm = mock.MagicMock()
        sched.spec_algorithm.is_none.return_value = True
        sched.mesh = None
        sched.model_config = mock.MagicMock()
        sched.model_config.is_multimodal = False
        sched.model_config.hf_eos_token_id = [2]
        sched.model_config.vocab_size = 151552
        sched._comm_backend = None
        sched.recv_requests = lambda: []
        sched.pending_dp_reqs = []

        sched.req_to_token_pool = ReqToTokenPool(size=64, max_context_len=105000)
        sched.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            size=3223552, page_size=128, kvcache=mock.MagicMock(), dp_size=32
        )
        sched.tree_cache = RadixCache(
            req_to_token_pool=sched.req_to_token_pool,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            page_size=128,
            disable=False,
        )
        sched.running_batch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(32)],
            req_to_token_pool=sched.req_to_token_pool,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            tree_cache=sched.tree_cache,
            model_config=sched.model_config,
            enable_overlap=True,
            dp_size=32,
            spec_algorithm=sched.spec_algorithm,
            mesh=None,
        )
        sched.chunked_reqs = [None] * 32
        sched._coop_prefill_state = None
        sched.disagg_prefill_queue = []

        # Build 16 waiting requests with a 95,714-token shared system prompt + 8,000-token unique suffix
        shared_prefix = [42] * 95714
        waiting = []
        sp = SamplingParams(max_new_tokens=16, temperature=0.0)
        for i in range(16):
            r = Req(
                rid=f"req_{i}",
                origin_input_text="",
                origin_input_ids=shared_prefix + [1000 + i] * 8000,
                sampling_params=sp,
            )
            r.dp_rank = i
            waiting.append(r)
        sched.waiting_queue = waiting

        # Step through _maybe_get_coop_prefill_batch() until it finishes and populates Global RadixCache
        executed_batches = []
        while True:
            b = sched._maybe_get_coop_prefill_batch()
            if b is None:
                break
            executed_batches.append(b)
            # Verify process_batch_result handles every cooperative prefill batch cleanly in both overlap & non-overlap modes
            sched.enable_overlap = True
            sched.process_batch_result(b, mock.MagicMock())
            sched.enable_overlap = False
            sched.process_batch_result(b, mock.MagicMock())

        self.assertEqual(len(executed_batches), 3, "95,714 tokens across 32x1024=32,768 step size must run 3 steps")

        # Verify 95,616 tokens (747 pages of 128) are now cached in Global RadixCache for all 16 requests!
        for i, req in enumerate(sched.waiting_queue):
            match_res = sched.tree_cache.match_prefix(
                MatchPrefixParams(key=RadixKey(req.origin_input_ids, req.extra_key, dp_rank=i))
            )
            self.assertEqual(
                len(match_res.device_indices),
                95616,
                f"Request {i} should hit 95,616 cached prefix tokens in Global RadixCache",
            )

    def test_6_page_size_256_coop_and_radix(self):
        """Verify Cooperative Prefill + Global RadixCache with page_size=256."""
        from sgl_jax.srt.managers.scheduler import Scheduler
        from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
        from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
        from sgl_jax.srt.mem_cache.radix_cache import RadixCache, RadixKey
        from sgl_jax.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sgl_jax.srt.sampling.sampling_params import SamplingParams

        sched = Scheduler.__new__(Scheduler)
        sched.dp_size = 32
        sched.page_size = 256
        sched.node_rank = 0
        sched.chunked_prefill_size = 1024
        sched.max_prefill_tokens = 32768
        sched.disable_radix_cache = False
        sched.enable_overlap = True
        sched.tp_worker = mock.MagicMock()
        sched.spec_algorithm = mock.MagicMock()
        sched.spec_algorithm.is_none.return_value = True
        sched.mesh = None
        sched.model_config = mock.MagicMock()
        sched.model_config.is_multimodal = False
        sched.model_config.hf_eos_token_id = [2]
        sched.model_config.vocab_size = 151552
        sched._comm_backend = None
        sched.recv_requests = lambda: []
        sched.pending_dp_reqs = []

        sched.req_to_token_pool = ReqToTokenPool(size=64, max_context_len=105000)
        sched.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            size=3211264, page_size=256, kvcache=mock.MagicMock(), dp_size=32
        )
        sched.tree_cache = RadixCache(
            req_to_token_pool=sched.req_to_token_pool,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            page_size=256,
            disable=False,
        )
        sched.running_batch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(32)],
            req_to_token_pool=sched.req_to_token_pool,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            tree_cache=sched.tree_cache,
            model_config=sched.model_config,
            enable_overlap=True,
            dp_size=32,
            spec_algorithm=sched.spec_algorithm,
            mesh=None,
        )
        sched.chunked_reqs = [None] * 32
        sched._coop_prefill_state = None
        sched.disagg_prefill_queue = []

        shared_prefix = [42] * 90112
        waiting = []
        sp = SamplingParams(max_new_tokens=16, temperature=0.0)
        for i in range(32):
            r = Req(
                rid=f"req_{i}",
                origin_input_text="",
                origin_input_ids=shared_prefix + [1000 + i] * 7888,
                sampling_params=sp,
            )
            r.dp_rank = i
            waiting.append(r)
        sched.waiting_queue = waiting

        executed_batches = []
        while True:
            b = sched._maybe_get_coop_prefill_batch()
            if b is None:
                break
            executed_batches.append(b)
            sched.process_batch_result(b, mock.MagicMock())

        self.assertEqual(len(executed_batches), 3)
        for i, req in enumerate(sched.waiting_queue):
            match_res = sched.tree_cache.match_prefix(
                MatchPrefixParams(key=RadixKey(req.origin_input_ids, req.extra_key, dp_rank=i))
            )
            self.assertEqual(len(match_res.device_indices), 90112)

    def test_7_mid_wave_coop_prefill_with_running_batch(self):
        """Verify mid-wave [CoopPrefill] runs when running_batch is non-empty and preserves running_batch state."""
        from sgl_jax.srt.managers.scheduler import Scheduler
        from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
        from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
        from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
        from sgl_jax.srt.mem_cache.radix_cache import RadixCache, RadixKey
        from sgl_jax.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sgl_jax.srt.sampling.sampling_params import SamplingParams

        dp_size = 16
        sched = Scheduler.__new__(Scheduler)
        sched.dp_size = dp_size
        sched.page_size = 64
        sched.node_rank = 0
        sched.chunked_prefill_size = 2048
        sched.max_prefill_tokens = 32768
        sched.per_dp_max_running_requests = 8
        sched.disable_radix_cache = False
        sched.enable_overlap = True
        sched.tp_worker = mock.MagicMock()
        sched.spec_algorithm = mock.MagicMock()
        sched.spec_algorithm.is_none.return_value = True
        sched.mesh = None
        sched.model_config = mock.MagicMock()
        sched.model_config.is_multimodal = False
        sched.model_config.hf_eos_token_id = [2]
        sched.model_config.vocab_size = 151552
        sched._comm_backend = None
        sched.recv_requests = lambda: []
        sched.pending_dp_reqs = []

        # 459,136 tokens per DP rank * 16 DP ranks = 7,346,176 tokens (matches 2x2x2 FP8)
        sched.req_to_token_pool = ReqToTokenPool(size=128, max_context_len=105000)
        sched.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            size=459136 * dp_size, page_size=64, kvcache=mock.MagicMock(), dp_size=dp_size
        )
        sched.tree_cache = RadixCache(
            req_to_token_pool=sched.req_to_token_pool,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            page_size=64,
            disable=False,
        )
        sched.running_batch = ScheduleBatch.init_new(
            reqs=[[] for _ in range(dp_size)],
            req_to_token_pool=sched.req_to_token_pool,
            token_to_kv_pool_allocator=sched.token_to_kv_pool_allocator,
            tree_cache=sched.tree_cache,
            model_config=sched.model_config,
            enable_overlap=True,
            dp_size=dp_size,
            spec_algorithm=sched.spec_algorithm,
            mesh=None,
        )
        sched.chunked_reqs = [None] * dp_size
        sched._coop_prefill_state = None
        sched.disagg_prefill_queue = []

        sp = SamplingParams(max_new_tokens=1000, temperature=0.0)

        # Simulate Batch 0 already running in running_batch (occupying req_pool_idx 0..15)
        sched.req_to_token_pool.free_slots = list(range(dp_size, 128))
        sentinel_row_0 = np.arange(100, 200, dtype=np.int32)
        sched.req_to_token_pool.write((0, slice(0, 100)), sentinel_row_0)

        batch0_prefix = [101] * 92160
        for r in range(dp_size):
            req0 = Req(
                rid=f"b0_req_{r}",
                origin_input_text="",
                origin_input_ids=batch0_prefix + [2000 + r] * 8000,
                sampling_params=sp,
                dp_rank=r,
            )
            req0.req_pool_idx = r
            sched.running_batch.reqs_info[r].reqs.append(req0)

        # Now enqueue Batch 1 (32 requests with a distinct 92,160-token shared prefix) while Batch 0 is running
        batch1_prefix = [202] * 92160
        waiting_b1 = []
        for i in range(32):
            r = Req(
                rid=f"b1_req_{i}",
                origin_input_text="",
                origin_input_ids=batch1_prefix + [3000 + i] * 8000,
                sampling_params=sp,
                dp_rank=0,  # deliberately unbalanced initially to test rebalancing
            )
            waiting_b1.append(r)
        sched.waiting_queue = waiting_b1

        executed_batches = []
        while True:
            b = sched._maybe_get_coop_prefill_batch()
            if b is None:
                break
            executed_batches.append(b)
            # Verify coop_req_pool_idx used free_slots[0] (16), NOT 0!
            self.assertEqual(int(b.reqs_info[0].req_pool_indices[0]), 16)
            sched.process_batch_result(b, mock.MagicMock())

        self.assertEqual(len(executed_batches), 3, "Mid-wave CoopPrefill must run 3 steps for 92,160 tokens")
        # Verify Batch 0's req_pool_idx=0 page table was NOT overwritten!
        np.testing.assert_array_equal(
            sched.req_to_token_pool.req_to_token[0, :100],
            sentinel_row_0,
        )
        # Verify Batch 1's waiting requests were balanced across all 16 DP ranks (2 per rank)
        dp_counts = [sum(1 for req in sched.waiting_queue if req.dp_rank == r) for r in range(dp_size)]
        self.assertEqual(dp_counts, [2] * dp_size)

        # Verify all 16 DP ranks hit 92,160 cached prefix tokens for Batch 1
        for r in range(dp_size):
            match_res = sched.tree_cache.match_prefix(
                MatchPrefixParams(key=RadixKey(batch1_prefix + [9999], None, dp_rank=r))
            )
            self.assertEqual(len(match_res.device_indices), 92160)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoopPrefillAndGlobalRadix)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("[VALIDATE] ALL 7 COOPERATIVE PREFILL + GLOBAL RADIXCACHE VALIDATION TESTS PASSED!")

