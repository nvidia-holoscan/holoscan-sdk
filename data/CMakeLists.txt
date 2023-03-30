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

include(ExternalProject)

# Endoscopy sample data
add_library(endoscopy_data INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy/video/surgical_video.264")
# Setup the installation rule
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy DESTINATION data COMPONENT holoscan-data)

if(HOLOSCAN_DOWNLOAD_DATASETS)
  # Download the endoscopy sample data
  ExternalProject_Add(endoscopy_data_download
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_endoscopy_sample_data/versions/20230128/zip
    URL_MD5 9732a54944589f7ca4f1337e8adf0838
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230128.zip
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
    BUILD_BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy/video/surgical_video.264"
  )

  # Add dependency to force the download
  add_dependencies(endoscopy_data endoscopy_data_download)
endif()

# Multiai ultrasound sample data
add_library(multiai_ultrasound_data INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound/video/icardio_input1.avi")
# Setup the installation rule
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound DESTINATION data COMPONENT holoscan-data)

if(HOLOSCAN_DOWNLOAD_DATASETS)
  # Download the multiai sample data
  ExternalProject_Add(multiai_ultrasound_data_download
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_multi_ai_ultrasound_sample_data/versions/20221201/zip
    URL_MD5 8096b080563e6a75c8e50c3b271364cd
    DOWNLOAD_NAME holoscan_multi_ai_ultrasound_sample_data_20221201.zip
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
    BUILD_BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound/video/icardio_input1.avi"
  )

  # Add dependency to force the download
  add_dependencies(multiai_ultrasound_data multiai_ultrasound_data_download)
endif()
