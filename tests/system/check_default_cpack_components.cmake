# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_CPACK_CONFIG OR NOT EXISTS "${HOLOSCAN_CPACK_CONFIG}")
    message(FATAL_ERROR "HOLOSCAN_CPACK_CONFIG must name the generated CPack configuration")
endif()

include("${HOLOSCAN_CPACK_CONFIG}")

list(FIND CPACK_COMPONENTS_ALL "holoscan-graph-artifacts" graph_artifact_component_index)
if(NOT graph_artifact_component_index EQUAL -1)
    message(FATAL_ERROR
        "The default SDK package must not execute the optional graph-artifact component")
endif()
