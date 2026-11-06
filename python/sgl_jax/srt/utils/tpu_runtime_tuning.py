"""TPU runtime environment and JAX Pallas lowering optimizations.

Mirrors the highest-impact startup tunings from ``vllm-torchtpu``:
1. ``LIBTPU_INIT_ARGS`` collective offload & dynamic SMEM negotiation:
   - ``--xla_tpu_use_dynamic_smem_negotiation=true``
   - ``--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864`` (64 MiB on TPU7x)
   - ``--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864`` (64 MiB on TPU7x)
   Raising the SparseCore collective offload threshold to 64 MiB keeps per-layer
   AR/AG collectives on TensorCore ICI links instead of contending with SparseCore
   KV / MoE gathers.
2. ``patch_jax_pallas_fori_lowering()``:
   Elides the dead remainder ``scf.for`` loop emitted by JAX Pallas' Mosaic
   lowering for dynamic-bound ``jax.lax.fori_loop`` when ``unroll == 1``.
"""

from __future__ import annotations

import inspect
import os
import textwrap
import urllib.request

_DEFAULT_TPU7X_SC_OFFLOAD_BYTES = 67108864  # 64 MiB

_CHIP_SC_OFFLOAD_BYTES = {
    "tpu7x": 67108864,
    "v7x": 67108864,
    "v6e": 16777216,
    "trillium": 16777216,
}


def _user_has_libtpu_flag(existing: str, flag_name: str) -> bool:
    for tok in existing.split():
        if tok == flag_name or tok.startswith(flag_name + "="):
            return True
    return False


def _detect_sc_offload_threshold_bytes() -> int | None:
    override = os.environ.get("SGL_TPU_SC_OFFLOAD_MIN_BYTES")
    if override is not None:
        val = int(override)
        return val if val > 0 else None

    accel = (
        os.environ.get("TPU_ACCELERATOR_TYPE", "")
        or os.environ.get("TPU_TYPE", "")
    ).lower()
    for prefix, sz in _CHIP_SC_OFFLOAD_BYTES.items():
        if prefix in accel:
            return sz

    # Fast local check: only query GCE metadata if TPU VFIO devices are mounted.
    if os.path.exists("/dev/vfio"):
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=0.3) as resp:
                meta_accel = resp.read().decode("utf-8", errors="ignore").strip().lower()
            for prefix, sz in _CHIP_SC_OFFLOAD_BYTES.items():
                if prefix in meta_accel:
                    return sz
        except Exception:
            pass
        return _DEFAULT_TPU7X_SC_OFFLOAD_BYTES

    return None


def configure_libtpu_init_args() -> None:
    """Inject optimal ``LIBTPU_INIT_ARGS`` before ``jax`` initializes ``libtpu``."""
    if os.environ.get("SGL_DISABLE_AUTO_LIBTPU_TUNING", "0") == "1":
        return
    if os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
        return

    existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
    extra: list[str] = []

    if not _user_has_libtpu_flag(existing, "--xla_tpu_use_dynamic_smem_negotiation"):
        extra.append("--xla_tpu_use_dynamic_smem_negotiation=true")

    sc_offload_bytes = _detect_sc_offload_threshold_bytes()
    if sc_offload_bytes is not None:
        for flag in (
            "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes",
            "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes",
        ):
            if not _user_has_libtpu_flag(existing, flag):
                extra.append(f"{flag}={sc_offload_bytes}")

    if extra:
        assembled = " ".join(extra)
        if existing:
            assembled = f"{assembled} {existing}"
        os.environ["LIBTPU_INIT_ARGS"] = assembled


def patch_jax_pallas_fori_lowering() -> bool:
    """Elide the dead remainder ``scf.for`` from Pallas ``fori_loop`` lowering when ``unroll == 1``."""
    if os.environ.get("SGL_DISABLE_PALLAS_FORI_PATCH", "0") == "1":
        return False
    try:
        from jax._src.pallas.mosaic import lowering as _lowering
    except Exception:
        return False

    if getattr(_lowering, "_sgl_fori_patch_applied", False) or getattr(
        _lowering, "_torchtpu_fori_patch_applied", False
    ):
        return True

    orig = getattr(_lowering, "_lower_jaxpr_to_for_loop", None)
    if orig is None:
        return False

    try:
        src = textwrap.dedent(inspect.getsource(orig))
    except Exception:
        return False

    needle = "elif has_dynamic_remainder:"
    fix = "elif has_dynamic_remainder and unroll != 1:"
    if needle not in src or fix in src:
        return False

    patched_src = src.replace(needle, fix, 1)
    ns: dict = {"__name__": _lowering.__name__}
    try:
        exec(compile(patched_src, _lowering.__file__, "exec"), _lowering.__dict__, ns)
    except Exception:
        return False

    new_fn = ns.get("_lower_jaxpr_to_for_loop")
    if new_fn is None:
        return False

    _lowering._lower_jaxpr_to_for_loop = new_fn
    _lowering._sgl_fori_patch_applied = True
    return True
