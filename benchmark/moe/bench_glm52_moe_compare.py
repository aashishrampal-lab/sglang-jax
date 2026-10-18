"""Head-to-head benchmark of EPMoE vs FusedEPMoEV2 on GLM-5.2 shapes."""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.layers.moe import EPMoE, FusedEPMoEV2, TopK


def create_mesh() -> Mesh:
    devices = jax.devices()
    n = len(devices)
    mesh_devices = np.array(devices).reshape(n, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def bench_fn(fn, *args, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) * 1000.0 / iters


def run_suite(
    mesh: Mesh,
    num_experts: int,
    token_counts: list[int],
    hidden_size: int = 6144,
    intermediate_dim: int = 2048,
    top_k: int = 8,
    routed_scaling_factor: float = 2.5,
):
    ep_size = mesh.shape["data"] * mesh.shape["tensor"]
    local_experts = num_experts // ep_size
    print("=" * 110)
    print(
        f"GLM-5.2 MoE Benchmark | devices={ep_size} | E={num_experts} ({local_experts} local/dev) | "
        f"H={hidden_size}, I={intermediate_dim}, top_k={top_k}, shared_experts=1, dtype=bf16"
    )
    print("=" * 110, flush=True)

    with jax.set_mesh(mesh):
        topk_layer = TopK(
            topk=top_k,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            routed_scaling_factor=routed_scaling_factor,
            layer_id=3,
            mesh=mesh,
        )

        ep_moe = EPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            intermediate_dim=intermediate_dim,
            mesh=mesh,
            ep_size=ep_size,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            layer_id=3,
        )
        rep_sharding = NamedSharding(mesh, P(None, None))
        se_w_gu = jax.device_put(
            jnp.ones((hidden_size, 2 * intermediate_dim), dtype=jnp.bfloat16) * 0.01,
            rep_sharding,
        )
        se_w_d = jax.device_put(
            jnp.ones((intermediate_dim, hidden_size), dtype=jnp.bfloat16) * 0.01,
            rep_sharding,
        )

        fused_v2 = FusedEPMoEV2(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            intermediate_dim=intermediate_dim,
            mesh=mesh,
            ep_size=ep_size,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            layer_id=3,
            renormalize_topk_logits=True,
            routed_scaling_factor=routed_scaling_factor,
            use_grouped_topk=False,
            num_groups=1,
            top_k_groups=1,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=intermediate_dim,
        )
        f_ep_sharding = NamedSharding(mesh, P(("data", "tensor"), None, None))
        fused_v2.w1.value = jax.device_put(
            jnp.ones((num_experts, hidden_size, intermediate_dim), dtype=jnp.bfloat16) * 0.01,
            f_ep_sharding,
        )
        fused_v2.w3.value = jax.device_put(
            jnp.ones((num_experts, hidden_size, intermediate_dim), dtype=jnp.bfloat16) * 0.01,
            f_ep_sharding,
        )
        fused_v2.w2.value = jax.device_put(
            jnp.ones((num_experts, intermediate_dim, hidden_size), dtype=jnp.bfloat16) * 0.01,
            f_ep_sharding,
        )
        fused_v2.w1_shared.value = jax.device_put(
            jnp.ones((hidden_size, intermediate_dim), dtype=jnp.bfloat16) * 0.01,
            rep_sharding,
        )
        fused_v2.w3_shared.value = jax.device_put(
            jnp.ones((hidden_size, intermediate_dim), dtype=jnp.bfloat16) * 0.01,
            rep_sharding,
        )
        fused_v2.w2_shared.value = jax.device_put(
            jnp.ones((intermediate_dim, hidden_size), dtype=jnp.bfloat16) * 0.01,
            rep_sharding,
        )

        topk_def, topk_state = nnx.split(topk_layer)
        topk_leaves, topk_sdef = jax.tree_util.tree_flatten(topk_state)

        ep_def, ep_state = nnx.split(ep_moe)
        ep_leaves, ep_sdef = jax.tree_util.tree_flatten(ep_state)

        fv2_def, fv2_state = nnx.split(fused_v2)
        fv2_leaves, fv2_sdef = jax.tree_util.tree_flatten(fv2_state)

        @jax.jit
        def run_ep_moe(x, logits, tk_l, ep_l, w_gu, w_d):
            gu = jnp.dot(x, w_gu)
            g, u = jnp.split(gu, 2, axis=-1)
            se_out = jnp.dot(jax.nn.silu(g) * u, w_d)
            tk = nnx.merge(topk_def, jax.tree_util.tree_unflatten(topk_sdef, tk_l))
            ep = nnx.merge(ep_def, jax.tree_util.tree_unflatten(ep_sdef, ep_l))
            tw, tids = tk(logits)
            routed_out = ep(x, tw, tids)
            return routed_out + se_out

        @jax.jit
        def run_fused_v2(x, logits, tk_l, fv2_l):
            tk = nnx.merge(topk_def, jax.tree_util.tree_unflatten(topk_sdef, tk_l))
            fv2 = nnx.merge(fv2_def, jax.tree_util.tree_unflatten(fv2_sdef, fv2_l))
            tw, tids = tk(logits)
            return fv2(x, tw, tids)

        print(
            f"{'num_tokens':>10s} | {'tok/dev':>7s} | {'EPMoE+SE (ms)':>14s} | "
            f"{'FusedMoEV2 (ms)':>16s} | {'Speedup (EPMoE/FusedV2)':>24s}",
            flush=True,
        )
        print("-" * 85, flush=True)

        rng = jax.random.PRNGKey(42)
        tok_sharding = NamedSharding(mesh, P("data", None))
        for nt in token_counts:
            x = jax.device_put(
                jax.random.normal(rng, (nt, hidden_size), dtype=jnp.bfloat16),
                tok_sharding,
            )
            logits = jax.device_put(
                jax.random.normal(rng, (nt, num_experts), dtype=jnp.float32),
                tok_sharding,
            )
            try:
                t_ep = bench_fn(run_ep_moe, x, logits, topk_leaves, ep_leaves, se_w_gu, se_w_d)
            except Exception as e:
                t_ep = float("nan")
                print(f"  [EPMoE err at nt={nt}]: {str(e)[:160]}", flush=True)

            try:
                t_fv2 = bench_fn(run_fused_v2, x, logits, topk_leaves, fv2_leaves)
            except Exception as e:
                t_fv2 = float("nan")
                print(f"  [FusedV2 err at nt={nt}]: {str(e)[:160]}", flush=True)

            speedup = t_ep / t_fv2 if (t_ep > 0 and t_fv2 > 0) else float("nan")
            print(
                f"{nt:10d} | {nt // ep_size:7d} | {t_ep:14.3f} | "
                f"{t_fv2:16.3f} | {speedup:23.2f}x",
                flush=True,
            )


def main():
    import os
    if int(os.environ.get("NNODES", "1")) > 1:
        jax.distributed.initialize()
    mesh = create_mesh()
    ep_size = len(jax.devices())
    token_counts = [
        ep_size * 1,
        ep_size * 2,
        ep_size * 4,
        ep_size * 8,
        ep_size * 16,
        ep_size * 32,
        ep_size * 64,
        ep_size * 128,
        ep_size * 256,
        ep_size * 512,
        ep_size * 1024,
    ]
    if ep_size == 32:
        run_suite(mesh, num_experts=256, token_counts=token_counts)
    else:
        # Case 1: E=64 on 8 devices -> 8 local experts/device (matches E=256 on 32 devices!)
        run_suite(mesh, num_experts=64, token_counts=token_counts)
        # Case 2: E=256 on 8 devices -> 32 local experts/device
        run_suite(mesh, num_experts=256, token_counts=token_counts[:10])


if __name__ == "__main__":
    main()
