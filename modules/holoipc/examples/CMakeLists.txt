# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Holoscan IPC C++ examples under modules/holoipc/examples/. Included from holoipc/CMakeLists.txt whenever
# the IPC module is built (HOLOSCAN_BUILD_IPC ON; this file is not reached otherwise).

# fastdds_example and fastdds_benchmark use Fast DDS APIs; only build when HOLOSCAN_IPC_TRANSPORT_FASTDDS is ON
if(HOLOSCAN_IPC_TRANSPORT_FASTDDS)
  add_subdirectory(fastdds_example)
  add_subdirectory(fastdds_benchmark)
endif()
