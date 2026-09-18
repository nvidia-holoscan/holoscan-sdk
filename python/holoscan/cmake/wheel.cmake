# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Share the actual Core declaration with packaging tests using inert producers.
# This file is included standalone by those tests, so it carries its own default.
if(NOT DEFINED HOLOSCAN_BUILD_PYTHON_STUBS)
    set(HOLOSCAN_BUILD_PYTHON_STUBS ON)
endif()
set(_holoscan_python_files __init__.py _module_discovery.py core/__init__.py)
# Ship typing only as a complete set. py.typed without the generated _core.pyi would
# advertise a typed package whose symbols no type checker can resolve.
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _holoscan_python_files
        __init__.pyi py.typed core/__init__.pyi core/_core.pyi)
endif()

_holoscan_get_iceoryx2_ffi_library(_holoscan_iceoryx2_ffi_library)
holoscan_add_python_wheel_component(
    MODULE core PACKAGE holoscan
    PYTHON_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}"
    PYTHON_FILES ${_holoscan_python_files}
    EXTENSION_TARGET holoscan_core_python EXTENSION_DIRECTORY core
    NATIVE_TARGETS core logger holomq iceoryx2-cxx::shared-lib-cxx
    NATIVE_FILES "${_holoscan_iceoryx2_ffi_library}")
unset(_holoscan_iceoryx2_ffi_library)
