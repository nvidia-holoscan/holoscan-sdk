# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# RacerX sample data
add_library(racerx_data INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/racerx/racerx-small.mp4")
# Setup the installation rule
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/racerx DESTINATION data COMPONENT holoscan-data)

set(RACERX_DATA_VERSION "20231009")
set(RACERX_DATA_MD5 "b67492afea29610105995c4c27bd5a05")

if(HOLOSCAN_DOWNLOAD_DATASETS)
  include(HoloscanDownloadData)

  # Download the racerX sample data
  holoscan_download_data(racerx
    URL https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/data/racerx/racerx_${RACERX_DATA_VERSION}.zip
    URL_MD5 ${RACERX_DATA_MD5}
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    BYPRODUCTS "${CMAKE_CURRENT_SOURCE_DIR}/racerx/racerx-small.mp4"
  )

  # Add dependency to force the download
  add_dependencies(racerx_data racerx_download)
endif()
