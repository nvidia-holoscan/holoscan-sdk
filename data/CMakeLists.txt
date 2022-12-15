# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Download the endoscopy sample data
ExternalProject_Add(endoscopy_data
  URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_endoscopy_sample_data/versions/20221121/zip
  URL_MD5 d86b6145b9c72d0063d512875f613963
  DOWNLOAD_NAME holoscan_endoscopy_sample_data_20221121.zip
  DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  EXCLUDE_FROM_ALL 1
)

# Download the ultrasound sample data
ExternalProject_Add(ultrasound_data
  URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_ultrasound_sample_data/versions/20220608/zip
  URL_MD5 de88ccb8660f4e5787f726d868d77f69
  DOWNLOAD_NAME holoscan_ultrasound_sample_data_20220608
  DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ultrasound
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  EXCLUDE_FROM_ALL 1
)

# Download the multiai sample data
ExternalProject_Add(multiai_ultrasound_data
  URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_multi_ai_ultrasound_sample_data/versions/20221201/zip
  URL_MD5 8096b080563e6a75c8e50c3b271364cd
  DOWNLOAD_NAME holoscan_multi_ai_ultrasound_sample_data_20221201.zip
  DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  EXCLUDE_FROM_ALL 1
)

# Download the colonoscopy sample data
ExternalProject_Add(colonoscopy_data
  URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_colonoscopy_sample_data/versions/20221031/zip
  URL_MD5 d94aee8ee4d847535db85725f2c6b7ac
  DOWNLOAD_NAME holoscan_colonoscopy_sample_data_20221031.zip
  DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/colonoscopy
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  EXCLUDE_FROM_ALL 1
)

  # Setup the installation rules
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy DESTINATION data COMPONENT holoscan-data)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ultrasound DESTINATION data COMPONENT holoscan-data)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/colonoscopy DESTINATION data COMPONENT holoscan-data)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound DESTINATION data COMPONENT holoscan-data)
