# Adapted from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/sparse_core/core_map_helper.py
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run single-mesh SparseCore kernels through ``pl.kernel``.

Delegates directly to ``pl.kernel`` wrapped in ``@jax.jit`` rather than
pre-allocating output buffers via ``lax.empty`` + ``new_ref`` + ``core_map``,
avoiding zero-operand ``AllocateBuffer`` CSE merging and ``CopyInsertion`` HBM
copies across multi-layer graphs.
"""

from jax._src import api
from jax.experimental import pallas as pl


def kernel(
    body,
    *,
    out_type,
    mesh,
    scratch_types=(),
    compiler_params=None,
    interpret=False,
    cost_estimate=None,
    debug=False,
    name=None,
    metadata=None,
):
    """JIT-wrapped ``pl.kernel`` for single-mesh SparseCore kernels."""
    return api.jit(
        pl.kernel(
            body,
            out_type=out_type,
            mesh=mesh,
            scratch_types=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
    )
