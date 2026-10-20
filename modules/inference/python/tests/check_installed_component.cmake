# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_INSTALL_PREFIX
        HOLOSCAN_INSTALL_LIBDIR
        HOLOSCAN_PYTHON_EXECUTABLE
        HOLOSCAN_PYTHON_VERSION
        HOLOSCAN_BUILD_PYTHON_STUBS)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()
unset(required_variable)

file(REMOVE_RECURSE "${HOLOSCAN_INSTALL_PREFIX}")

function(_holoscan_install_component component)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${HOLOSCAN_INSTALL_PREFIX}"
        --component "${component}")
    if(DEFINED HOLOSCAN_BUILD_TYPE AND NOT HOLOSCAN_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${HOLOSCAN_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endfunction()

# Stage Core, its native dependencies, and native Inference without the
# independently owned Python component. The provider must not leak through
# either native package.
foreach(component IN ITEMS
        holoscan-dependencies
        iceoryx2-lib
        holoscan-holomq
        holoscan-core
        holoscan-config
        holoscan-config-io
        holoscan-params
        holoscan-sensor-io
        holoscan-python_libs
        holoscan-inference)
    _holoscan_install_component("${component}")
endforeach()
unset(component)

string(CONCAT site_packages
    "${HOLOSCAN_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIBDIR}/python${HOLOSCAN_PYTHON_VERSION}"
    "/site-packages")
set(inference_package "${site_packages}/holoscan_inference")
if(EXISTS "${inference_package}")
    message(FATAL_ERROR
        "Native component staging unexpectedly installed ${inference_package}")
endif()

set(python_environment
    "PYTHONPATH=${site_packages}"
    "LD_LIBRARY_PATH=${HOLOSCAN_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIBDIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${python_environment}
        "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "import holoscan; import importlib.util; assert importlib.util.find_spec('holoscan_inference') is None"
    RESULT_VARIABLE native_import_result
    ERROR_VARIABLE native_import_error)
if(NOT native_import_result EQUAL 0)
    message(FATAL_ERROR "Native-only Python import failed:\n${native_import_error}")
endif()

_holoscan_install_component(holoscan-inference-python_libs)
set(runtime_paths
        __init__.py
        _inference.abi3.so
        operators/__init__.py)
foreach(required_path IN LISTS runtime_paths)
    if(NOT EXISTS "${inference_package}/${required_path}")
        message(FATAL_ERROR "Installed Inference package is missing ${required_path}")
    endif()
endforeach()
unset(required_path)
unset(runtime_paths)

set(typing_paths
        __init__.pyi
        _inference.pyi
        operators/__init__.pyi
        py.typed)
foreach(typing_path IN LISTS typing_paths)
    if(HOLOSCAN_BUILD_PYTHON_STUBS AND NOT EXISTS "${inference_package}/${typing_path}")
        message(FATAL_ERROR "Installed Inference package is missing ${typing_path}")
    elseif(NOT HOLOSCAN_BUILD_PYTHON_STUBS AND EXISTS "${inference_package}/${typing_path}")
        message(FATAL_ERROR
            "Untyped Inference package unexpectedly installed ${typing_path}")
    endif()
endforeach()
unset(typing_path)
unset(typing_paths)

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${python_environment}
        "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "from holoscan_inference.operators import InferenceOp; assert InferenceOp.name == 'holoscan.inference.InferenceOp'"
    RESULT_VARIABLE inference_import_result
    ERROR_VARIABLE inference_import_error)
if(NOT inference_import_result EQUAL 0)
    message(FATAL_ERROR "Installed Inference Python import failed:\n${inference_import_error}")
endif()

# The component also installs the distribution metadata that registers the
# holoscan.inference alias with Core's module discovery.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${python_environment}
        "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "import holoscan.inference, holoscan_inference; assert holoscan.inference is holoscan_inference"
    RESULT_VARIABLE alias_import_result
    ERROR_VARIABLE alias_import_error)
if(NOT alias_import_result EQUAL 0)
    message(FATAL_ERROR "Installed Inference alias import failed:\n${alias_import_error}")
endif()
