#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(HOLOSCAN_BUILD_ORT)
  add_subdirectory(onnx)
endif()
if(HOLOSCAN_BUILD_LIBTORCH)
  add_subdirectory(torch)
endif()
