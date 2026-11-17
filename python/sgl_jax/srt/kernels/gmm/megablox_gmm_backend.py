from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm as gmm_v1_kernel
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import gmm_v2 as gmm_v2_kernel
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import is_supported_by_gmm_v2
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple


def gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    rhs_scale: jnp.ndarray | None = None,
    rhs_bias: jnp.ndarray | None = None,
    tiling: tuple[int, int, int] | None = None,
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    acc_dtype: jnp.dtype | None = None,
    activation_quantized_dtype: jnp.dtype | None = None,
    v2_tile_info: Any = None,
    rhs_up: jnp.ndarray | None = None,
    rhs_up_scale: jnp.ndarray | None = None,
    rhs_up_bias: jnp.ndarray | None = None,
    fuse_act: str | None = None,
    swiglu_limit: float | None = None,
) -> jax.Array:
    """Dispatch GMM to v2 or v1, with optional activation quantization.

    When ``activation_quantized_dtype`` is provided and v2 is **not** used, the
    LHS is quantized before the kernel call and the output is rescaled
    afterwards, mirroring the sglang-gpu per-token activation quantization
    scheme.  When v2 **is** used the kernel handles quantization internally.

    Args:
        lhs: [m, k] input matrix.
        rhs: [num_groups, k, n] weight matrix.
        group_sizes: [num_groups] group sizes.
        preferred_element_type: Output dtype.
        rhs_scale: Optional [num_groups, num_blocks, 1, n] scale.
        rhs_bias: Optional [num_groups, 1, n] bias.
        tiling: Optional (tm, tk, tn) tile sizes for v1; None for auto.
        group_offset: Optional group offset for sharding.
        existing_out: Optional existing output to accumulate into.
        interpret: If True, run in interpret mode (CPU); disables v2.
            Defaults to auto-detect (True when not running on TPU).
        maybe_quantize_lhs: If True, gmm_v2 will quantize lhs when rhs is
            quantized. Set to False to keep activations unquantized.
        zero_initialize: Whether to initialize unvisited output elements to zero.
        acc_dtype: Optional accumulation dtype for gmm_v2 (e.g. jnp.float32).
        activation_quantized_dtype: When set, quantize the LHS activations to
            this dtype before the kernel call (v1 path only) and rescale the
            output afterwards.
        v2_tile_info: Optional TileSizes or TileFn for v2 kernel tiling.
            Defaults to calculate_tiling (auto-tiler) when None.
    """
    if interpret is None:
        interpret = not is_tpu_runtime()

    use_gmm_v2 = not interpret and is_supported_by_gmm_v2(
        rhs_scale, maybe_quantize_lhs=maybe_quantize_lhs
    )
    if rhs_up is not None and not is_supported_by_gmm_v2(
        rhs_up_scale, maybe_quantize_lhs=maybe_quantize_lhs
    ):
        use_gmm_v2 = False

    if not use_gmm_v2 and fuse_act is not None:
        from sgl_jax.srt.kernels.fused_moe.v2.kernel import activation_fn

        if rhs_up is None:
            half_n = rhs.shape[-1] // 2
            w_gate, w_up = rhs[..., :half_n], rhs[..., half_n:]
            s_gate = None if rhs_scale is None else rhs_scale[..., :half_n]
            s_up = None if rhs_scale is None else rhs_scale[..., half_n:]
            b_gate = None if rhs_bias is None else rhs_bias[..., :half_n]
            b_up = None if rhs_bias is None else rhs_bias[..., half_n:]
        else:
            w_gate, w_up = rhs, rhs_up
            s_gate, s_up = rhs_scale, rhs_up_scale
            b_gate, b_up = rhs_bias, rhs_up_bias
        gate_out = gmm(
            lhs,
            w_gate,
            group_sizes,
            preferred_element_type=jnp.float32,
            rhs_scale=s_gate,
            rhs_bias=b_gate,
            tiling=tiling,
            group_offset=group_offset,
            existing_out=existing_out,
            interpret=interpret,
            maybe_quantize_lhs=maybe_quantize_lhs,
            zero_initialize=zero_initialize,
            acc_dtype=acc_dtype,
            activation_quantized_dtype=activation_quantized_dtype,
            v2_tile_info=v2_tile_info,
        )
        up_out = gmm(
            lhs,
            w_up,
            group_sizes,
            preferred_element_type=jnp.float32,
            rhs_scale=s_up,
            rhs_bias=b_up,
            tiling=tiling,
            group_offset=group_offset,
            existing_out=existing_out,
            interpret=interpret,
            maybe_quantize_lhs=maybe_quantize_lhs,
            zero_initialize=zero_initialize,
            acc_dtype=acc_dtype,
            activation_quantized_dtype=activation_quantized_dtype,
            v2_tile_info=v2_tile_info,
        )
        return activation_fn(gate_out, up_out, fuse_act, swiglu_limit).astype(
            preferred_element_type
        )

    # gmm_v2 pads lhs internally to sublane tiling without mutating group_sizes;
    # keep 128-row alignment only for legacy gmm_v1.
    m = lhs.shape[0]
    pad_size = 0
    if not use_gmm_v2 and not interpret and m % 128 != 0:
        pad_size = 128 - (m % 128)
        lhs = jax.lax.pad(lhs, jnp.zeros((), dtype=lhs.dtype), ((0, pad_size, 0), (0, 0, 0)))
        group_sizes = group_sizes.at[-1].add(pad_size)

    lhs_scale = None
    if not use_gmm_v2 and activation_quantized_dtype is not None:
        lhs_q, lhs_scale = quantize_tensor_simple(lhs, activation_quantized_dtype, dim=-1)
        lhs = lhs_q

    if use_gmm_v2:
        v2_kwargs = {}
        if v2_tile_info is not None:
            v2_kwargs["tile_info"] = v2_tile_info
        out = gmm_v2_kernel(
            lhs,
            rhs,
            group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            group_offset=group_offset,
            rhs_up=rhs_up,
            rhs_up_scale=rhs_up_scale,
            rhs_up_bias=rhs_up_bias,
            fuse_act=fuse_act,
            swiglu_limit=swiglu_limit,
            preferred_element_type=preferred_element_type,
            maybe_quantize_lhs=maybe_quantize_lhs,
            zero_initialize=zero_initialize,
            acc_dtype=acc_dtype,
            **v2_kwargs,
        )
    else:
        out = gmm_v1_kernel(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=preferred_element_type,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
            tiling=tiling,
            group_offset=group_offset,
            existing_out=existing_out,
            interpret=interpret,
        )

    if lhs_scale is not None:
        out = out * lhs_scale

    # Slice output back to original size if padded
    if pad_size > 0:
        out = out[:m]

    return out
