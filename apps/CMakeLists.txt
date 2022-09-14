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

# Download the associated dataset if needed
if(HOLOSCAN_DOWNLOAD_DATASETS)
  include(ExternalProject)

  # Download the endoscopy sample data
  ExternalProject_Add(endoscopy_data
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_endoscopy_sample_data/versions/20220608/zip
    URL_MD5 741897d5a0aea87d6016ca87631b4eed
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20220608.zip
    DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/test_data
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/test_data/endoscopy
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
    DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/test_data
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/test_data/ultrasound
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    EXCLUDE_FROM_ALL 1
  )
endif()

add_subdirectory(endoscopy_tool_tracking)
add_subdirectory(experiments)

# Build the applications based on GXF if enabled
if(HOLOSCAN_BUILD_GXF_EXTENSIONS)
  add_subdirectory(endoscopy_tool_tracking_gxf)
  add_subdirectory(ultrasound_segmentation_gxf)

  if(HOLOSCAN_BUILD_HI_SPEED_ENDO_APP)
    add_subdirectory(hi_speed_endoscopy_gxf)
  endif()
endif()
