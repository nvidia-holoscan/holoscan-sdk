# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(sequential)
add_subdirectory(sequential_with_subgraph)
add_subdirectory(conditional)
add_subdirectory(conditional_routing)
add_subdirectory(fork_join)
add_subdirectory(for_loop)
add_subdirectory(mixed_flows)
add_subdirectory(streaming)
add_subdirectory(streaming_with_monitor)
add_subdirectory(state_based_monitoring)

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(
  FILES README.md
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
