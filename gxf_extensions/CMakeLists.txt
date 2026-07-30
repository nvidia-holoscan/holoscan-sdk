# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install target to a sub directory that has the same folder structure as the source
function(install_gxf_extension target_name)
    cmake_parse_arguments(PARSE_ARGV 1 INSTALL_GXF_EXTENSION "" "COMPONENT" "")

    # If 'COMPONENT' option is not defined
    if(NOT DEFINED INSTALL_GXF_EXTENSION_COMPONENT)
        set(INSTALL_GXF_EXTENSION_COMPONENT "holoscan-gxf_extensions")
    endif()

    install(TARGETS "${target_name}"
        DESTINATION lib/gxf_extensions
        COMPONENT "${INSTALL_GXF_EXTENSION_COMPONENT}"
    )

    # If dependent '_lib' target is defined, add it to the install target
    if(TARGET "${target_name}_lib")
        install(TARGETS "${target_name}_lib"
            DESTINATION lib/gxf_extensions
            COMPONENT "${INSTALL_GXF_EXTENSION_COMPONENT}"
        )
    endif()
endfunction()

list(APPEND CMAKE_INSTALL_RPATH_LIST
    \$ORIGIN                                    # load lib${extension}_lib.so
    \$ORIGIN/..                                 # load 3rd-party dependency libs
    \$ORIGIN/../../../nvidia/cuda_runtime/lib   # for python wheels
    \$ORIGIN/../../../tensorrt                  # for python wheels
    \$ORIGIN/../HoloscanWrapperExtension        # For HoloscanWrapperExtension
    \$ORIGIN/../UcxHoloscanExtension            # For UcxHoloscanExtension
    )
list(JOIN CMAKE_INSTALL_RPATH_LIST ":" CMAKE_INSTALL_RPATH)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/gxf_extensions)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/gxf_extensions)

add_subdirectory(ucx)
