# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(HOLOSCAN_PYTHON_EXAMPLES)
  add_subdirectory(python)
endif()
# Model may also be used in GPU Resident Inference and other examples
add_subdirectory(model)

install(
  FILES README.md
  DESTINATION "examples/bring_your_own_model"
  COMPONENT "holoscan-examples"
)
