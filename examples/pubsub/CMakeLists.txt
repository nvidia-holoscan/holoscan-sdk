# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(HOLOSCAN_BUILD_IPC AND HOLOSCAN_IPC_TRANSPORT_FASTDDS)
  add_subdirectory(ping_tensor)
  add_subdirectory(video_replayer)
endif()
