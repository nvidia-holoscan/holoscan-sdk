# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(holoscan_configure_example_test_workspace target)
    # Preserve a caller-supplied test launcher inside our workspace scope.
    get_property(launcher TARGET ${target} PROPERTY TEST_LAUNCHER)
    if(UNIX)
        set_property(TARGET ${target} PROPERTY TEST_LAUNCHER
            sh "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/run_with_workspace.sh" ${launcher})
    elseif(WIN32)
        # The Windows helper inherits ACLs; retain its documented build-tree fallback.
        set_property(TARGET ${target} PROPERTY TEST_LAUNCHER
            "${CMAKE_COMMAND}" -E env
            "HOLOSCAN_EXAMPLE_WORKSPACE_ROOT=${CMAKE_CURRENT_BINARY_DIR}" ${launcher})
    endif()
endfunction()
