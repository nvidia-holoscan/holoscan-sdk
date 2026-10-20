# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Inference owns HoloInfer, InferenceOp, its extension, and Python facade. Core
# remains in the companion Core wheel; CUDA and TensorRT remain external runtime
# prerequisites supplied by the deployment environment.
set(_inference_wheel_python_files
    __init__.py
    operators/__init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _inference_wheel_python_files
        __init__.pyi
        _inference.pyi
        py.typed
        operators/__init__.pyi)
endif()
holoscan_add_python_wheel_component(
    MODULE inference
    PACKAGE holoscan_inference
    PYTHON_DIRECTORY "${_inference_package}"
    PYTHON_FILES ${_inference_wheel_python_files}
    EXTENSION_TARGET holoscan_inference_python
    NATIVE_TARGETS holoinfer_utils holoinfer op_inference
    LIBRARY_RUNPATH "$ORIGIN/../../holoscan/lib"
    EXTENSION_RUNPATH "$ORIGIN/../holoscan/lib")
