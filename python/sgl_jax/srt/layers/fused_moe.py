"""Fused Expert-Parallel MoE layer using Pallas kernel."""

import os

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe
from sgl_jax.srt.utils.quantization.quantization_utils import (
    is_int4_dtype,
    quantize_tensor,
)


def _expand_moe_block_scale(scale_3d: jax.Array, n_out: int, block_n: int) -> jax.Array:
    """Expand compact 2D MoE block scales to the kernel's fast 1D-ready layout."""
    scale_per_channel = jnp.repeat(scale_3d, block_n, axis=2)[..., :n_out]
    return scale_per_channel[:, :, None, :]


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: w1/w3 are (num_experts, hidden_size, intermediate_size) for gate/up proj
      and w2 is (num_experts, intermediate_size, hidden_size) for down proj
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        hidden_size: Hidden size of the model
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to select per token (top_k)
        ep_size: Expert parallel size (number of devices to shard experts across)
        mesh: JAX mesh for distributed execution
        intermediate_dim: Intermediate dimension for expert FFN
        weight_dtype: Data type for weights
        dtype: Data type for computation
        activation: Activation function ("silu", "gelu", "swigluoai")
        layer_id: Layer index (for debugging)
        renormalize_topk_logits: Whether to renormalize top-k weights
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c: Tile size parameters (auto-selected if None)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        quantization_config=None,
        enable_act_quant: bool = True,
        # Profiling / ablation flags (primarily for microbenching).
        disable_a2a: bool = False,
        disable_dynamic_ffn1: bool = False,
        disable_dynamic_ffn2: bool = False,
        disable_weight_load: bool = False,
        disable_a2a_s_tile_read: bool = False,
        disable_a2a_s_acc_tile_write: bool = False,
        disable_shared_expert: bool = False,
        disable_all_reduce_metadata: bool = False,
        disable_sync_barrier: bool = False,
        use_jax_allreduce_metadata: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.use_grouped_topk = use_grouped_topk
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.renormalize_topk_logits = renormalize_topk_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.num_shared_experts = num_shared_experts
        self.moe_shared_expert_intermediate_size = (
            moe_shared_expert_intermediate_size or intermediate_dim
        )
        self.mesh = mesh
        self.disable_a2a = disable_a2a
        self.disable_dynamic_ffn1 = disable_dynamic_ffn1
        self.disable_dynamic_ffn2 = disable_dynamic_ffn2
        self.disable_weight_load = disable_weight_load
        self.disable_a2a_s_tile_read = disable_a2a_s_tile_read
        self.disable_a2a_s_acc_tile_write = disable_a2a_s_acc_tile_write
        self.disable_shared_expert = disable_shared_expert
        self.disable_all_reduce_metadata = disable_all_reduce_metadata
        self.disable_sync_barrier = disable_sync_barrier
        self.use_jax_allreduce_metadata = use_jax_allreduce_metadata

        metadata = get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )
        # Optional explicit disable for in-kernel activation quantization. The
        # positive signal comes from quantization_config.moe_activation_dtype.
        self.enable_act_quant_cfg = enable_act_quant

        # Initialize weights.
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )
        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(1),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, intermediate_dim, hidden_size),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w1_scale = None
        self.w3_scale = None
        self.w2_scale = None

        if self.num_shared_experts > 0:
            se_inter_dim = self.moe_shared_expert_intermediate_size * self.num_shared_experts

            self.w1_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w2_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (se_inter_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w3_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )
        else:
            self.w1_shared = None
            self.w3_shared = None
            self.w2_shared = None

        self.w1_shared_scale = None
        self.w3_shared_scale = None
        self.w2_shared_scale = None

        # Read block-wise quantization settings from config.
        weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )
        if weight_block_size is not None and len(weight_block_size) == 2:
            self.quant_block_k = int(weight_block_size[1])  # block_k
            self.quant_block_n = int(weight_block_size[0])  # block_n
        else:
            self.quant_block_k = None
            self.quant_block_n = None

    def quantize_weights(self, is_static: bool = False, *, abstract: bool = False):
        """Quantize MoE weights in-place. Call once after model loading."""
        if self.quantized_dtype is None:
            return
        if is_static and is_int4_dtype(self.quantized_dtype):
            raise ValueError("Static INT4 checkpoints require moe_backend='epmoe'.")
        if (
            not is_static
            and self.w1_scale is not None
            and getattr(self.w1.value, "dtype", None) == self.quantized_dtype
        ):
            return

        # Determine quant_block_k. The v1 kernel requires a block size when
        # scales are provided, so per-channel fp8 must tile to block-256; the v2
        # kernel accepts per-channel (None).
        wsz = (
            self.quant_block_k
            if self.quant_block_k is not None
            else (None if isinstance(self, FusedEPMoEV2) else 256)
        )
        if hasattr(self, "quant_block_k"):
            del self.quant_block_k
        self.quant_block_k = wsz

        mesh_context = (
            jax.sharding.use_abstract_mesh(self.mesh.abstract_mesh)
            if abstract
            else jax.set_mesh(self.mesh)
        )
        with mesh_context:
            if is_static:
                names = ["w1", "w3", "w2"]
                if self.num_shared_experts > 0:
                    names += ["w1_shared", "w3_shared", "w2_shared"]
                for name in names:
                    param = getattr(self, name)
                    if isinstance(param.value, jax.ShapeDtypeStruct):
                        param.value = jax.ShapeDtypeStruct(
                            param.value.shape, self.quantized_dtype, sharding=param.value.sharding
                        )
                ep_scale_sharding = P(("data", "tensor"), None, None, None)

                if wsz is None:
                    # Per-channel: scale shape (E, 1, 1, N)
                    w1_scale_shape = (
                        self.num_experts,
                        1,
                        1,
                        self.intermediate_dim,
                    )
                    w3_scale_shape = w1_scale_shape
                    w2_scale_shape = (
                        self.num_experts,
                        1,
                        1,
                        self.hidden_size,
                    )
                else:
                    # Block-wise: scale shape (E, K//block_k, 1, N)
                    w1_scale_shape = (
                        self.num_experts,
                        self.hidden_size // wsz,
                        1,
                        self.intermediate_dim,
                    )
                    w3_scale_shape = w1_scale_shape
                    w2_scale_shape = (
                        self.num_experts,
                        self.intermediate_dim // wsz,
                        1,
                        self.hidden_size,
                    )

                if hasattr(self, "w1_scale"):
                    del self.w1_scale
                self.w1_scale = nnx.Param(
                    jnp.zeros(w1_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w3_scale"):
                    del self.w3_scale
                self.w3_scale = nnx.Param(
                    jnp.zeros(w3_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w2_scale"):
                    del self.w2_scale
                self.w2_scale = nnx.Param(
                    jnp.zeros(w2_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if self.num_shared_experts > 0:
                    # fused kernel expects per-channel shared scale (1, 1, se_inter)
                    # — see _validate_fused_ep_moe_args / kernel.py:455-470
                    shared_scale_sharding = P(None, None, None)
                    se_inter = self.moe_shared_expert_intermediate_size * self.num_shared_experts

                    if hasattr(self, "w1_shared_scale"):
                        del self.w1_shared_scale
                    self.w1_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, se_inter), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                    if hasattr(self, "w3_shared_scale"):
                        del self.w3_shared_scale
                    self.w3_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, se_inter), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                    if hasattr(self, "w2_shared_scale"):
                        del self.w2_shared_scale
                    self.w2_shared_scale = nnx.Param(
                        jnp.zeros((1, 1, self.hidden_size), dtype=jnp.float32),
                        out_sharding=shared_scale_sharding,
                    )

                return

            # NOTE: Fused MoE shards the expert dimension across EP=(data*tensor).
            ep_sharding = P(("data", "tensor"), None, None)
            ep_scale_sharding = P(("data", "tensor"), None, None, None)

            # Replace original weights with quantized versions one by one so
            # each BF16 expert tensor is freed before quantizing the next.
            if self.quant_block_n is not None:
                # 2D block-wise quantization: scale shape (E, K//block_k, N//block_n)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                self.w1 = nnx.Param(w1_value, out_sharding=ep_sharding)
                del w1_value
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                self.w3 = nnx.Param(w3_value, out_sharding=ep_sharding)
                del w3_value
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                self.w2 = nnx.Param(w2_value, out_sharding=ep_sharding)
                del w2_value
            else:
                # 1D sub-channel quantization: scale shape (E, K//wsz, N)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                self.w1 = nnx.Param(w1_value, out_sharding=ep_sharding)
                del w1_value
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                self.w3 = nnx.Param(w3_value, out_sharding=ep_sharding)
                del w3_value
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                self.w2 = nnx.Param(w2_value, out_sharding=ep_sharding)
                del w2_value

            # Update scales (reshape to 4D for GMM kernel)
            if self.quant_block_n is not None:
                # 2D block-wise: expand block scales once so forward can run
                # through the fast 1D kernel path without changing semantics.
                w1_scale_4d = _expand_moe_block_scale(
                    w1_scale, self.intermediate_dim, self.quant_block_n
                )
                w3_scale_4d = _expand_moe_block_scale(
                    w3_scale, self.intermediate_dim, self.quant_block_n
                )
                w2_scale_4d = _expand_moe_block_scale(
                    w2_scale, self.hidden_size, self.quant_block_n
                )
            else:
                if self.quant_block_k is None:
                    # Per-channel reduction over K returns [E, N].
                    w1_scale_4d = w1_scale[:, None, None, :]
                    w3_scale_4d = w3_scale[:, None, None, :]
                    w2_scale_4d = w2_scale[:, None, None, :]
                else:
                    # Sub-channel [E, K//wsz, N] -> [E, K//wsz, 1, N].
                    w1_scale_4d = w1_scale[:, :, None, :]
                    w3_scale_4d = w3_scale[:, :, None, :]
                    w2_scale_4d = w2_scale[:, :, None, :]

            if hasattr(self, "w1_scale"):
                del self.w1_scale
            self.w1_scale = nnx.Param(
                w1_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w3_scale"):
                del self.w3_scale
            self.w3_scale = nnx.Param(
                w3_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w2_scale"):
                del self.w2_scale
            self.w2_scale = nnx.Param(
                w2_scale_4d,
                out_sharding=ep_scale_sharding,
            )

            if self.w1_shared is not None:
                w1_shared_value, w1_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1_shared.value,
                    axis=0,
                )
                w3_shared_value, w3_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3_shared.value,
                    axis=0,
                )
                w2_shared_value, w2_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2_shared.value,
                    axis=0,
                )

                self.w1_shared = nnx.Param(w1_shared_value, out_sharding=P(None, None))
                self.w3_shared = nnx.Param(w3_shared_value, out_sharding=P(None, None))
                self.w2_shared = nnx.Param(w2_shared_value, out_sharding=P(None, None))

                if hasattr(self, "w1_shared_scale"):
                    del self.w1_shared_scale
                self.w1_shared_scale = nnx.Param(
                    w1_shared_scale.reshape(
                        1,
                        1,
                        w1_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w3_shared_scale"):
                    del self.w3_shared_scale
                self.w3_shared_scale = nnx.Param(
                    w3_shared_scale.reshape(
                        1,
                        1,
                        w3_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w2_shared_scale"):
                    del self.w2_shared_scale
                self.w2_shared_scale = nnx.Param(
                    w2_shared_scale.reshape(
                        1,
                        1,
                        w2_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,
        out_sharding: jax.sharding.Sharding | None = None,
    ) -> jax.Array:
        """Forward pass through the fused MoE layer."""
        assert hidden_states.ndim == 2
        orig_num_tokens = hidden_states.shape[0]
        mesh_ep_size = self.mesh.shape["data"] * self.mesh.shape["tensor"]
        kernel_sharding = jax.sharding.NamedSharding(self.mesh, P(("data", "tensor"), None))
        if orig_num_tokens % mesh_ep_size != 0:
            padded_num_tokens = (
                (orig_num_tokens + mesh_ep_size - 1) // mesh_ep_size
            ) * mesh_ep_size
            pad_t = padded_num_tokens - orig_num_tokens

            def _pad_and_reshard(hs_in, tw_in, ti_in):
                hs_p = jnp.pad(hs_in, ((0, pad_t), (0, 0)))
                tw_p = jnp.pad(tw_in, ((0, pad_t), (0, 0)), constant_values=0.0)
                ti_p = jnp.pad(ti_in, ((0, pad_t), (0, 0)), constant_values=-1)
                return (
                    jax.sharding.reshard(hs_p, kernel_sharding),
                    jax.sharding.reshard(tw_p, kernel_sharding),
                    jax.sharding.reshard(ti_p, kernel_sharding),
                )

            if getattr(self.mesh, "_any_axis_explicit", False):
                hidden_states, topk_weights, topk_ids = jax.sharding.auto_axes(
                    _pad_and_reshard,
                    out_sharding=(kernel_sharding, kernel_sharding, kernel_sharding),
                )(hidden_states, topk_weights, topk_ids)
            else:
                hidden_states, topk_weights, topk_ids = _pad_and_reshard(
                    hidden_states, topk_weights, topk_ids
                )
        else:
            hidden_states = jax.sharding.reshard(hidden_states, kernel_sharding)
            topk_weights = jax.sharding.reshard(topk_weights, kernel_sharding)
            topk_ids = jax.sharding.reshard(topk_ids, kernel_sharding)

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None
        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None
        w1_shared_scale = self.w1_shared_scale.value if self.w1_shared_scale is not None else None
        w3_shared_scale = self.w3_shared_scale.value if self.w3_shared_scale is not None else None
        w2_shared_scale = self.w2_shared_scale.value if self.w2_shared_scale is not None else None

        quant_block_k = self.quant_block_k if self.quant_block_k is not None else None

        kernel_sharding = jax.sharding.NamedSharding(self.mesh, P(("data", "tensor"), None))
        hidden_states = jax.sharding.reshard(hidden_states, kernel_sharding)
        topk_weights = jax.sharding.reshard(topk_weights, kernel_sharding)
        topk_ids = jax.sharding.reshard(topk_ids, kernel_sharding)

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            w3=self.w3.value,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=self.num_experts_per_tok,
            use_grouped_topk=self.use_grouped_topk,
            num_groups=self.num_groups,
            top_k_groups=self.top_k_groups,
            renormalize_topk_logits=self.renormalize_topk_logits,
            routed_scaling_factor=self.routed_scaling_factor,
            act_fn=self.activation,
            block_config=block_config,
            disable_a2a=self.disable_a2a,
            disable_dynamic_ffn1=self.disable_dynamic_ffn1,
            disable_dynamic_ffn2=self.disable_dynamic_ffn2,
            disable_weight_load=self.disable_weight_load,
            disable_a2a_s_tile_read=self.disable_a2a_s_tile_read,
            disable_a2a_s_acc_tile_write=self.disable_a2a_s_acc_tile_write,
            disable_shared_expert=self.disable_shared_expert,
            disable_all_reduce_metadata=self.disable_all_reduce_metadata,
            disable_sync_barrier=self.disable_sync_barrier,
            use_jax_allreduce_metadata=self.use_jax_allreduce_metadata,
            # Optional parameters (not used in basic case)
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            b1=None,
            b2=None,
            b3=None,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )

        if out_sharding is None:
            out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * output.ndim)))
        output = jax.sharding.reshard(output, out_sharding)
        if orig_num_tokens != output.shape[0]:
            output = output[:orig_num_tokens]
        return output


class FusedEPMoEV2(FusedEPMoE):
    """V2 fused EP-MoE layer using the Strix-style double-buffer kernel.

    Inherits weight init and quantization from FusedEPMoE. Overrides __call__
    to dispatch to fused_ep_moe_v2 with v2-specific flags, with an optional
    small-batch megablox gmm_v2 fast path (SGL_MOE_FUSED_MIN_TOKENS) for decode.
    """

    def _forward_small_batch_ep_gmm(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        w1_scale: jax.Array | None,
        w2_scale: jax.Array | None,
        w3_scale: jax.Array | None,
        w1_shared: jax.Array | None,
        w2_shared: jax.Array | None,
        w3_shared: jax.Array | None,
        w1_shared_scale: jax.Array | None,
        w2_shared_scale: jax.Array | None,
        w3_shared_scale: jax.Array | None,
        enable_act_quant: bool,
        swiglu_limit: float | None,
        shared_swiglu_limit: float | None,
    ) -> jax.Array:
        """Small-batch EP-MoE path using all-gather + megablox gmm_v2 + reduce-scatter.

        Avoids fused_ep_moe_v2's SparseCore sync barrier + dynamic ragged-all-to-all
        overhead on small decode batches (T < SGL_MOE_FUSED_MIN_TOKENS) while reusing
        FusedEPMoEV2's exact EP-sharded weights (zero extra HBM).
        """
        from sgl_jax.srt.kernels.fused_moe.v2.kernel import activation_fn
        from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

        ep_axis = ("data", "tensor")
        dp_size = int(self.mesh.shape["data"])
        tp_size = int(self.mesh.shape["tensor"])
        mesh_ep_size = dp_size * tp_size
        top_k = self.num_experts_per_tok
        num_experts = self.num_experts
        act_fn_name = self.activation
        act_q_dtype = self.activation_quantized_dtype if enable_act_quant else None

        ep_tok_spec = P(ep_axis, None)
        ep_w_spec = P(ep_axis, None, None)
        ep_s_spec = P(ep_axis, None, None, None)
        rep_2d_spec = P(None, None)

        def _shard_fn(
            hs_local,
            tw_local,
            ti_local,
            w1_loc,
            w2_loc,
            w3_loc,
            w1_s_loc,
            w2_s_loc,
            w3_s_loc,
            w1_se,
            w2_se,
            w3_se,
            w1_se_s,
            w2_se_s,
            w3_se_s,
        ):
            # 1. Compute shared expert locally on each device's [T_local, H] slice
            #    (zero collectives; overlaps on MXU with ICI all-gather of hs_local).
            shared_local = None
            if w1_se is not None:
                se_w_quant = w1_se_s is not None
                if enable_act_quant and se_w_quant:
                    hs_f32 = hs_local.astype(jnp.float32)
                    hs_amax = jnp.max(jnp.abs(hs_f32), axis=-1, keepdims=True)
                    hs_sc = jnp.maximum(hs_amax / jnp.float32(448.0), jnp.float32(1e-12))
                    hs_fp8 = (hs_f32 / hs_sc).astype(jnp.float8_e4m3fn)
                    gate_se = jnp.dot(
                        hs_fp8, w1_se, preferred_element_type=jnp.float32
                    ) * (hs_sc * w1_se_s)
                    up_se = jnp.dot(
                        hs_fp8, w3_se, preferred_element_type=jnp.float32
                    ) * (hs_sc * w3_se_s)
                    act_se = activation_fn(
                        gate_se, up_se, act_fn_name, shared_swiglu_limit
                    )
                    act_amax = jnp.max(jnp.abs(act_se), axis=-1, keepdims=True)
                    act_sc = jnp.maximum(act_amax / jnp.float32(448.0), jnp.float32(1e-12))
                    act_fp8 = (act_se / act_sc).astype(jnp.float8_e4m3fn)
                    shared_local = (
                        jnp.dot(act_fp8, w2_se, preferred_element_type=jnp.float32)
                        * (act_sc * w2_se_s)
                    ).astype(hs_local.dtype)
                elif se_w_quant:
                    w1_bf16 = (w1_se.astype(jnp.float32) * w1_se_s).astype(hs_local.dtype)
                    w3_bf16 = (w3_se.astype(jnp.float32) * w3_se_s).astype(hs_local.dtype)
                    gate_se = jnp.dot(
                        hs_local, w1_bf16, preferred_element_type=jnp.float32
                    )
                    up_se = jnp.dot(
                        hs_local, w3_bf16, preferred_element_type=jnp.float32
                    )
                    act_se = activation_fn(
                        gate_se, up_se, act_fn_name, shared_swiglu_limit
                    )
                    w2_bf16 = (w2_se.astype(jnp.float32) * w2_se_s).astype(hs_local.dtype)
                    shared_local = jnp.dot(
                        act_se.astype(hs_local.dtype),
                        w2_bf16,
                        preferred_element_type=jnp.float32,
                    ).astype(hs_local.dtype)
                else:
                    gate_se = jnp.dot(hs_local, w1_se, preferred_element_type=jnp.float32)
                    up_se = jnp.dot(hs_local, w3_se, preferred_element_type=jnp.float32)
                    act_se = activation_fn(
                        gate_se, up_se, act_fn_name, shared_swiglu_limit
                    )
                    shared_local = jnp.dot(
                        act_se.astype(hs_local.dtype),
                        w2_se,
                        preferred_element_type=jnp.float32,
                    ).astype(hs_local.dtype)

            # 2. All-gather tokens and routing metadata across the EP mesh.
            if mesh_ep_size > 1:
                my_ep = jax.lax.axis_index("data") * tp_size + jax.lax.axis_index("tensor")
                hs_all = jax.lax.all_gather(hs_local, axis_name=ep_axis, axis=0, tiled=True)
                tw_all = jax.lax.all_gather(tw_local, axis_name=ep_axis, axis=0, tiled=True)
                ti_all = jax.lax.all_gather(ti_local, axis_name=ep_axis, axis=0, tiled=True)
            else:
                my_ep = jnp.int32(0)
                hs_all = hs_local
                tw_all = tw_local
                ti_all = ti_local

            # Map invalid/padded routing slots (e.g. topk_ids == -1) to expert 0
            # with zero routing weight so argsort and bincount stay 1:1 aligned.
            valid_slot = (ti_all >= 0) & (ti_all < num_experts)
            safe_ids = jnp.where(valid_slot, ti_all, 0)
            safe_weights = jnp.where(valid_slot, tw_all, 0.0)

            flat_ids = jnp.ravel(safe_ids)
            sorted_pos = jnp.argsort(flat_ids, stable=True)
            token_indices = sorted_pos // top_k
            group_sizes = jnp.bincount(flat_ids, length=num_experts).astype(jnp.int32)

            lhs = hs_all[token_indices]
            local_num_experts = w1_loc.shape[0]
            group_offset = jnp.asarray(my_ep * local_num_experts, dtype=jnp.int32)

            gmm_kwargs = dict(
                group_sizes=group_sizes,
                preferred_element_type=hs_local.dtype,
                group_offset=group_offset,
                maybe_quantize_lhs=enable_act_quant,
                acc_dtype=jnp.float32,
                activation_quantized_dtype=act_q_dtype,
            )
            gate_out = gmm(
                lhs=lhs,
                rhs=w1_loc,
                rhs_scale=w1_s_loc,
                zero_initialize=False,
                **gmm_kwargs,
            )
            up_out = gmm(
                lhs=lhs,
                rhs=w3_loc,
                rhs_scale=w3_s_loc,
                zero_initialize=False,
                **gmm_kwargs,
            )
            inter = activation_fn(
                gate_out.astype(jnp.float32),
                up_out.astype(jnp.float32),
                act_fn_name,
                swiglu_limit,
            ).astype(hs_local.dtype)
            down_out = gmm(
                lhs=inter,
                rhs=w2_loc,
                rhs_scale=w2_s_loc,
                zero_initialize=True,
                **gmm_kwargs,
            )

            total_routed = flat_ids.shape[0]
            inv_pos = (
                jnp.zeros(total_routed, dtype=jnp.int32)
                .at[sorted_pos]
                .set(jnp.arange(total_routed, dtype=jnp.int32))
            )
            unsorted = jnp.take(down_out, indices=inv_pos, axis=0)
            routed_all = jnp.einsum(
                "tkh,tk->th",
                unsorted.reshape(hs_all.shape[0], top_k, -1).astype(jnp.float32),
                safe_weights.astype(jnp.float32),
            ).astype(hs_local.dtype)

            if mesh_ep_size > 1:
                routed_local = jax.lax.psum_scatter(
                    routed_all, axis_name=ep_axis, scatter_dimension=0, tiled=True
                )
            else:
                routed_local = routed_all

            return routed_local + shared_local if shared_local is not None else routed_local

        in_specs = (
            ep_tok_spec,
            ep_tok_spec,
            ep_tok_spec,
            ep_w_spec,
            ep_w_spec,
            ep_w_spec,
            ep_s_spec if w1_scale is not None else None,
            ep_s_spec if w2_scale is not None else None,
            ep_s_spec if w3_scale is not None else None,
            rep_2d_spec if w1_shared is not None else None,
            rep_2d_spec if w2_shared is not None else None,
            rep_2d_spec if w3_shared is not None else None,
            rep_2d_spec if w1_shared_scale is not None else None,
            rep_2d_spec if w2_shared_scale is not None else None,
            rep_2d_spec if w3_shared_scale is not None else None,
        )
        return jax.shard_map(
            _shard_fn,
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=ep_tok_spec,
            check_vma=False,
        )(
            hidden_states,
            topk_weights,
            topk_ids,
            self.w1.value,
            self.w2.value,
            self.w3.value,
            w1_scale,
            w2_scale,
            w3_scale,
            w1_shared,
            w2_shared,
            w3_shared,
            w1_shared_scale,
            w2_shared_scale,
            w3_shared_scale,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config=None,
        out_sharding: jax.sharding.Sharding | None = None,
        swiglu_limit: float | None = None,
        shared_swiglu_limit: float | None = None,
    ) -> jax.Array:
        from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2
        from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
            get_tuned_fused_moe_v2_block_config,
        )

        assert hidden_states.ndim == 2
        orig_num_tokens = hidden_states.shape[0]
        mesh_ep_size = self.mesh.shape["data"] * self.mesh.shape["tensor"]
        kernel_sharding = jax.sharding.NamedSharding(self.mesh, P(("data", "tensor"), None))
        if orig_num_tokens % mesh_ep_size != 0:
            padded_num_tokens = (
                (orig_num_tokens + mesh_ep_size - 1) // mesh_ep_size
            ) * mesh_ep_size
            pad_t = padded_num_tokens - orig_num_tokens

            def _pad_and_reshard(hs_in, tw_in, ti_in):
                hs_p = jnp.pad(hs_in, ((0, pad_t), (0, 0)))
                tw_p = jnp.pad(tw_in, ((0, pad_t), (0, 0)), constant_values=0.0)
                ti_p = jnp.pad(ti_in, ((0, pad_t), (0, 0)), constant_values=-1)
                return (
                    jax.sharding.reshard(hs_p, kernel_sharding),
                    jax.sharding.reshard(tw_p, kernel_sharding),
                    jax.sharding.reshard(ti_p, kernel_sharding),
                )

            if getattr(self.mesh, "_any_axis_explicit", False):
                hidden_states, topk_weights, topk_ids = jax.sharding.auto_axes(
                    _pad_and_reshard,
                    out_sharding=(kernel_sharding, kernel_sharding, kernel_sharding),
                )(hidden_states, topk_weights, topk_ids)
            else:
                hidden_states, topk_weights, topk_ids = _pad_and_reshard(
                    hidden_states, topk_weights, topk_ids
                )
        else:
            hidden_states = jax.sharding.reshard(hidden_states, kernel_sharding)
            topk_weights = jax.sharding.reshard(topk_weights, kernel_sharding)
            topk_ids = jax.sharding.reshard(topk_ids, kernel_sharding)

        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None

        # SE per-channel scales are stored 3D (1, 1, out); the v2 kernel reads them
        # 2D (1, out). Squeeze here (not in quantize_weights, which the v1 path and
        # the weight mapping consume in 3D form). None for bf16 SE weights (Mode 3).
        w1_shared_scale = (
            self.w1_shared_scale.value[:, 0, :] if self.w1_shared_scale is not None else None
        )
        w3_shared_scale = (
            self.w3_shared_scale.value[:, 0, :] if self.w3_shared_scale is not None else None
        )
        w2_shared_scale = (
            self.w2_shared_scale.value[:, 0, :] if self.w2_shared_scale is not None else None
        )
        # In-kernel act-quant (fp8 token, Mode 1) needs both a quant-config
        # activation dtype and fp8 weights. Weight-only fp8 checkpoints stay in
        # Mode 2 (bf16 token x fp8 weight).
        enable_act_quant = (
            self.activation_quantized_dtype is not None and self.enable_act_quant_cfg is not False
        ) and (w1_scale is not None)
        quant_block_k = self.quant_block_k if hasattr(self, "quant_block_k") else None
        quant_mode = (
            "none"
            if w1_scale is None
            else ("per_channel" if quant_block_k is None else "blockwise")
        )

        fused_min_tokens = int(os.environ.get("SGL_MOE_FUSED_MIN_TOKENS", "0") or "0")
        if (
            fused_min_tokens > 0
            and hidden_states.shape[0] < fused_min_tokens
            and self.num_experts % mesh_ep_size == 0
        ):
            output = self._forward_small_batch_ep_gmm(
                hidden_states,
                topk_weights,
                topk_ids,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w3_scale=w3_scale,
                w1_shared=w1_shared_val,
                w2_shared=w2_shared_val,
                w3_shared=w3_shared_val,
                w1_shared_scale=w1_shared_scale,
                w2_shared_scale=w2_shared_scale,
                w3_shared_scale=w3_shared_scale,
                enable_act_quant=enable_act_quant,
                swiglu_limit=swiglu_limit,
                shared_swiglu_limit=shared_swiglu_limit,
            )
        else:
            if block_config is None:
                block_config = get_tuned_fused_moe_v2_block_config(
                    num_tokens=hidden_states.shape[0],
                    num_experts=self.num_experts,
                    top_k=self.num_experts_per_tok,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_dim,
                    dtype=hidden_states.dtype,
                    weight_dtype=self.w1.value.dtype,
                    ep_size=self.ep_size,
                    use_shared_expert=self.w1_shared is not None,
                    use_grouped_topk=self.use_grouped_topk,
                    enable_act_quant=enable_act_quant,
                    quant_mode=quant_mode,
                )

            direct_scaled_dot = w1_scale is not None
            output = fused_ep_moe_v2(
                self.mesh,
                hidden_states,
                self.w1.value,
                self.w2.value,
                self.w3.value,
                topk_weights,
                topk_ids,
                self.num_experts_per_tok,
                act_fn=self.activation,
                swiglu_limit=swiglu_limit,
                shared_swiglu_limit=shared_swiglu_limit,
                block_config=block_config,
                quant_block_k=quant_block_k,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w3_scale=w3_scale,
                w1_shared=w1_shared_val,
                w2_shared=w2_shared_val,
                w3_shared=w3_shared_val,
                w1_shared_scale=w1_shared_scale,
                w2_shared_scale=w2_shared_scale,
                w3_shared_scale=w3_shared_scale,
                enable_act_quant=enable_act_quant,
                direct_scaled_dot=direct_scaled_dot,
                dp_axis_name="data",
                tp_axis_name="tensor",
            )

        # Reshard the MoE output to the caller-requested layout. Under sequence
        # parallelism out_sharding carries the SP-aware reduce_sharding
        # (('data','tensor') on the scatter dim); without it we fall back to the
        # plain DP layout P('data', None). b79f9951 dropped this arg and hardcoded
        # the DP layout, which silently broke SP for MoE layers — restored here.
        if out_sharding is not None:
            target_out_sharding = out_sharding
        else:
            default_spec = (
                P("data", None)
                if orig_num_tokens % self.mesh.shape["data"] == 0
                else P(None, None)
            )
            target_out_sharding = jax.sharding.NamedSharding(self.mesh, default_spec)

        if orig_num_tokens != output.shape[0]:
            if getattr(self.mesh, "_any_axis_explicit", False):
                output = jax.sharding.auto_axes(
                    lambda y: jax.sharding.reshard(y[:orig_num_tokens], target_out_sharding),
                    out_sharding=target_out_sharding,
                )(output)
            else:
                output = jax.sharding.reshard(output, target_out_sharding)
                output = output[:orig_num_tokens]
        else:
            output = jax.sharding.reshard(output, target_out_sharding)
        return output
