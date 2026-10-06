# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(HoloscanLoggingPolicy)

# Return whether the active C++ driver accepts the GNU-style ELF options used by the SDK. Checking
# both platform and frontend keeps clang-cl out even though its compiler id is Clang.
function(holoscan_is_linux_gnu_like_frontend output_variable)
    set(_holoscan_is_linux_gnu_like_frontend FALSE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT MSVC AND
       CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
       NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(_holoscan_is_linux_gnu_like_frontend TRUE)
    endif()
    set(${output_variable} ${_holoscan_is_linux_gnu_like_frontend} PARENT_SCOPE)
endfunction()

# Return whether the target platform provides the Linux file-descriptor primitives used by public
# Channel readiness handles. Keep this separate from compiler classification: the implementation
# relies on epoll and eventfd even when another platform uses a GNU-like frontend.
function(holoscan_platform_has_native_descriptor_readiness output_variable)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(${output_variable} 1 PARENT_SCOPE)
    else()
        set(${output_variable} 0 PARENT_SCOPE)
    endif()
endfunction()

# Select the implementation source for a native descriptor readiness handle. Unsupported platforms
# retain the exported API through a checked kNotSupported implementation rather than compiling the
# Linux system headers or fabricating a Windows HANDLE-to-int conversion.
function(holoscan_select_native_descriptor_readiness_source
         output_variable linux_source unsupported_source)
    holoscan_platform_has_native_descriptor_readiness(_holoscan_has_descriptor_readiness)
    if(_holoscan_has_descriptor_readiness)
        set(${output_variable} "${linux_source}" PARENT_SCOPE)
    else()
        set(${output_variable} "${unsupported_source}" PARENT_SCOPE)
    endif()
    unset(_holoscan_has_descriptor_readiness)
endfunction()

# Configure the ELF origin-relative build and install lookup paths only for the frontend/platform
# lane that owns that loader convention. CMAKE_LIBRARY_OUTPUT_DIRECTORY is the configured build-tree
# library location; the install layout remains independent. Directory-scope variables are returned
# to the caller so one top-level invocation applies consistently to subsequently created targets.
function(holoscan_configure_linux_origin_rpath)
    holoscan_is_linux_gnu_like_frontend(_holoscan_linux_origin_rpath)
    if(_holoscan_linux_origin_rpath)
        set(CMAKE_BUILD_RPATH "\$ORIGIN:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" PARENT_SCOPE)
        set(CMAKE_INSTALL_RPATH "\$ORIGIN" PARENT_SCOPE)
    endif()
    unset(_holoscan_linux_origin_rpath)
endfunction()

# Apply the portable C++ frontend contract without leaking one compiler's flags to another. The
# MSVC-conforming preprocessor is required by the public expected-propagation macros and applies to
# both native MSVC and clang-cl frontends that advertise MSVC command-line semantics.
function(holoscan_configure_cxx20_target target_name visibility)
    target_compile_features(${target_name} ${visibility} cxx_std_20)
    if(MSVC)
        target_compile_options(${target_name} ${visibility}
            "$<$<COMPILE_LANGUAGE:CXX>:/permissive->"
            "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>"
            "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        )
    endif()
endfunction()

# ELF linker hardening flags are meaningful only on Linux GNU-like driver lanes. In particular,
# CMAKE_CXX_COMPILER_ID=Clang also covers clang-cl, where these options are invalid.
function(holoscan_configure_linux_shared_link target_name)
    holoscan_is_linux_gnu_like_frontend(_holoscan_linux_elf_link_options)
    if(BUILD_SHARED_LIBS AND _holoscan_linux_elf_link_options)
        target_link_options(${target_name} PRIVATE
            LINKER:--exclude-libs,ALL
            LINKER:--no-undefined
            LINKER:--gc-sections
        )
    endif()
    unset(_holoscan_linux_elf_link_options)
endfunction()

# Hide one static dependency archive only on Linux GNU-like driver lanes. This is separate from
# shared-library hardening because an INTERFACE dependency such as spdlog must propagate the option
# to the final SDK link without exporting an incompatible option on MSVC-semantic frontends.
function(holoscan_configure_linux_archive_hiding target_name archive_name)
    holoscan_is_linux_gnu_like_frontend(_holoscan_linux_archive_hiding)
    if(_holoscan_linux_archive_hiding)
        target_link_options(${target_name} INTERFACE "LINKER:--exclude-libs,${archive_name}")
    endif()
    unset(_holoscan_linux_archive_hiding)
endfunction()
