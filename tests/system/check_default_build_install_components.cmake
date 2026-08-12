# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RUN_SCRIPT OR NOT EXISTS "${HOLOSCAN_RUN_SCRIPT}")
    message(FATAL_ERROR "HOLOSCAN_RUN_SCRIPT must name the public build wrapper")
endif()

file(READ "${HOLOSCAN_RUN_SCRIPT}" run_script)
string(FIND "${run_script}"
    "install_component_if_present holoscan-graph-artifacts"
    graph_artifact_install_position)
if(NOT graph_artifact_install_position EQUAL -1)
    message(FATAL_ERROR
        "The default build install must not execute the optional graph-artifact component")
endif()
