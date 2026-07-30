# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(models)
if(HOLOSCAN_CPP_EXAMPLES)
  add_subdirectory(cpp)
  add_dependencies(activation_map activation_map_dummy_models)
endif()

if(HOLOSCAN_PYTHON_EXAMPLES)
  add_subdirectory(python)
  add_dependencies(python_activation_map activation_map_dummy_models)
endif()

install(
  FILES README.md
  DESTINATION "examples/activation_map"
  COMPONENT "holoscan-examples"
)
