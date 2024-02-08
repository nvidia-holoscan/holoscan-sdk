# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RacerX sample data
add_library(racerx_data INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/racerx/racerx-small.mp4")
# Setup the installation rule
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/racerx DESTINATION data COMPONENT holoscan-data)

if(HOLOSCAN_DOWNLOAD_DATASETS)
  include(HoloscanDownloadData)

  set(RACERX_DATA_VERSION "20231009")
  set(RACERX_DATA_MD5 "86cd7e5477bb9eaa0cfc0547912d77b2")

  # Download the racerX sample data
  holoscan_download_data(racerx
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_racerx_video/versions/${RACERX_DATA_VERSION}/zip
    URL_MD5 ${RACERX_DATA_MD5}
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/racerx/racerx-small.mp4"
  )

  # Add dependency to force the download
  add_dependencies(racerx_data racerx_download)
endif()

# Multiai ultrasound sample data
add_library(multiai_ultrasound_data INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound/video/icardio_input1.avi")
# Setup the installation rule
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound DESTINATION data COMPONENT holoscan-data)

if(HOLOSCAN_DOWNLOAD_DATASETS)
  # Download the multiai sample data
  holoscan_download_data(multiai_ultrasound
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_multi_ai_ultrasound_sample_data/versions/20221201/zip
    URL_MD5 8096b080563e6a75c8e50c3b271364cd
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound/video/icardio_input1.avi"
  )

  # Add dependency to force the download
  add_dependencies(multiai_ultrasound_data multiai_ultrasound_download)
endif()
