# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This private bridge passes C++ values between extensions. Hash a conservative superset of its
# declarations, including nanobind's internals identity. This detects stale interface/build pairs;
# it is not a general proof of C++ ABI compatibility or a substitute for package dependencies.
function(holoscan_configure_binding_services_compatibility sdk_include core_headers nanobind_root output_dir)
    set(_identity "holoscan-python-binding-services-v2\n")
    foreach(_setting IN ITEMS
            HOLOSCAN_BUILD_VERSION CMAKE_SYSTEM_NAME CMAKE_SYSTEM_PROCESSOR CMAKE_SIZEOF_VOID_P
            CMAKE_CXX_COMPILER_ID CMAKE_CXX_COMPILER_VERSION CMAKE_CXX_STANDARD
            CMAKE_CXX_FLAGS CMAKE_BUILD_TYPE CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
        string(APPEND _identity "${_setting}=${${_setting}}\n")
    endforeach()
    # Absolute checkout paths and the build/runtime Python minor are deliberately excluded:
    # cp312-abi3 modules must keep their identity when loaded by Python 3.13 or 3.14.
    foreach(_group IN ITEMS sdk_include core_headers nanobind_root)
        if(_group STREQUAL "nanobind_root")
            file(GLOB_RECURSE _headers CONFIGURE_DEPENDS RELATIVE "${${_group}}"
                "${${_group}}/include/*.h" "${${_group}}/src/*.h")
        else()
            file(GLOB_RECURSE _headers CONFIGURE_DEPENDS RELATIVE "${${_group}}"
                "${${_group}}/*.hpp" "${${_group}}/*.h")
        endif()
        if(NOT _headers)
            message(FATAL_ERROR "Missing binding-services compatibility headers: ${_group}")
        endif()
        list(SORT _headers)
        foreach(_header IN LISTS _headers)
            set(_path "${${_group}}/${_header}")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_path}")
            file(SHA256 "${_path}" _digest)
            string(APPEND _identity "${_group}/${_header}=${_digest}\n")
        endforeach()
    endforeach()
    string(SHA256 HOLOSCAN_PYTHON_BINDING_SERVICES_HASH "${_identity}")
    file(MAKE_DIRECTORY "${output_dir}")
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/binding_services_compatibility.hpp.in"
        "${output_dir}/binding_services_compatibility.hpp" @ONLY)
endfunction()
