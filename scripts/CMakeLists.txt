# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install useful scripts for developers
install(
  FILES
    download_ngc_data
    get_cmake_cuda_archs.py
    graph_surgeon.py
    ctest_time_comparison.py
    test_pattern_validation.py
  DESTINATION "bin"
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
  COMPONENT "holoscan-core"
)

install(
  FILES README.md
  DESTINATION "bin"
  COMPONENT "holoscan-core"
)
