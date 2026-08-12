# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holoscan_configure_example_install target)
    set(one_value_args INSTALL_RPATH)
    cmake_parse_arguments(HCEI "" "${one_value_args}" "" ${ARGN})
    if(HCEI_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_configure_example_install received unknown arguments: "
            "${HCEI_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT HCEI_INSTALL_RPATH)
        set(HCEI_INSTALL_RPATH "$ORIGIN/../../../lib")
    endif()
    # Installed example binaries live under examples/<example>/cpp while SDK
    # shared libraries live under lib. Deeper example layouts can override the
    # relative path without requiring users to set LD_LIBRARY_PATH.
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "${HCEI_INSTALL_RPATH}")
endfunction()
