# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holoscan_configure_example_install target)
    set(one_value_args RUNTIME_DESTINATION)
    cmake_parse_arguments(HCEI "" "${one_value_args}" "" ${ARGN})
    if(HCEI_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_configure_example_install received unknown arguments: "
            "${HCEI_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT HCEI_RUNTIME_DESTINATION)
        file(RELATIVE_PATH HCEI_EXAMPLE_RELATIVE_DIR
            "${HOLOSCAN_EXAMPLES_SOURCE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}")
        set(HCEI_RUNTIME_DESTINATION
            "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/${HCEI_EXAMPLE_RELATIVE_DIR}")
    endif()
    # Installed C++ binaries mirror their source directory under share/examples:
    #
    #   <prefix>/share/examples/<example path>/cpp/<binary>
    #
    # The SDK libraries live in <prefix>/<libdir>. Compute the path from the
    # runtime destination so each ../ corresponds to one layout component and
    # nested examples remain relocatable without LD_LIBRARY_PATH.
    file(RELATIVE_PATH HCEI_LIBDIR_RELATIVE_PATH
        "/${HCEI_RUNTIME_DESTINATION}"
        "/${CMAKE_INSTALL_LIBDIR}")
    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN/${HCEI_LIBDIR_RELATIVE_PATH}")
endfunction()
