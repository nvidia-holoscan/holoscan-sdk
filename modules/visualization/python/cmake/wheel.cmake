# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Visualization owns its renderer, operator, extension, and Python facade. Core
# remains in the companion Core wheel and is reached through sibling runpaths.
set(_visualization_wheel_python_files
    __init__.py
    operators/__init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _visualization_wheel_python_files
        __init__.pyi
        _visualization.pyi
        py.typed
        operators/__init__.pyi)
endif()
holoscan_add_python_wheel_component(
    MODULE visualization
    PACKAGE holoscan_visualization
    PYTHON_DIRECTORY "${_visualization_package}"
    PYTHON_FILES ${_visualization_wheel_python_files}
    EXTENSION_TARGET holoscan_visualization_python
    NATIVE_TARGETS holoviz op_holoviz
    LIBRARY_RUNPATH "$ORIGIN/../../holoscan/lib"
    EXTENSION_RUNPATH "$ORIGIN/../holoscan/lib")
