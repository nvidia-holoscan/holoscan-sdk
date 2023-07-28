# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We import cli, core and gxf to make sure they're available before other modules that rely on them
from . import cli, core, gxf

__all__ = ["as_tensor", "cli", "core", "gxf"]


def as_tensor(obj):
    if hasattr(obj, "__cuda_array_interface__") or hasattr(obj, "__array_interface__"):
        # Workaround for bug in CuPy<13.0.0a1 where strides of ndarray with all but 1 dimension a
        # singleton, could become order='F' strides instead of the expected order='C' strides.
        # See:
        #   https://github.com/cupy/cupy/pull/7438
        #   https://github.com/cupy/cupy/pull/7457
        #   https://forums.developer.nvidia.com/t/known-issue-in-cupy-affecting-tensor-interop/244423
        #
        # Here we force any array that is both C and F contiguous to have C-ordered strides
        # We do this for both NumPy and CuPy arrays for consistency.

        # nd array, but with only 1 non-singleton dimension
        nd_singleton = (obj.ndim > 1) and (sum(tuple(s > 1 for s in obj.shape)) == 1)
        if nd_singleton and obj.flags.forc:
            # determine expected strides for a C-contiguous array
            expected_strides = [
                1,
            ] * obj.ndim
            expected_strides[obj.ndim - 1] = obj.itemsize
            for i in range(obj.ndim - 2, -1, -1):
                expected_strides[i] = expected_strides[i + 1] * obj.shape[i + 1]
            expected_strides = tuple(expected_strides)

            # make a copy to force stride update if they do not match
            if obj.strides != expected_strides:
                if hasattr(obj, "__cuda_array_interface__"):
                    try:
                        import cupy as cp

                        if isinstance(obj, cp.ndarray):
                            # use as_strided to avoid a copy
                            obj = cp.lib.stride_tricks.as_strided(
                                obj, shape=obj.shape, strides=expected_strides
                            )
                            return core.Tensor.as_tensor(obj)
                    except ImportError:
                        pass
                elif hasattr(obj, "__array_interface__"):
                    try:
                        import numpy as np

                        if isinstance(obj, np.ndarray):
                            # use as_strided to avoid a copy
                            obj = np.lib.stride_tricks.as_strided(
                                obj, shape=obj.shape, strides=expected_strides
                            )
                            return core.Tensor.as_tensor(obj)
                    except ImportError:
                        pass
                # update strides by making an explicit copy
                try:
                    obj = obj.copy(order="C")
                except (AttributeError, TypeError):
                    import warnings

                    warnings.warn(
                        "Unexpected strides encountered during call to `as_tensor` and no copy "
                        "method was available. Leaving the strides unchanged."
                    )

    return core.Tensor.as_tensor(obj)


as_tensor.__doc__ = core.Tensor.as_tensor.__doc__


# Other modules are exposed to the public API but will only be lazily loaded
_EXTRA_MODULES = [
    "conditions",
    "executors",
    "graphs",
    "logger",
    "operators",
    "resources",
    "schedulers",
]
__all__.extend(_EXTRA_MODULES)


# Autocomplete
def __dir__():
    return __all__


# Lazily load extra modules
def __getattr__(name):
    import importlib
    import sys

    if name in _EXTRA_MODULES:
        module_name = f"{__name__}.{name}"
        module = importlib.import_module(module_name)  # import
        sys.modules[module_name] = module  # cache
        return module
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
