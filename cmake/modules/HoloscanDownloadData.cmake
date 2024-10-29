# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Creates a target with the <dataname>_data
function(holoscan_download_data dataname)

  # If we already have the target we return
  if(TARGET "${dataname}_download")
    return()
  endif()

  cmake_parse_arguments(DATA "GENERATE_GXF_ENTITIES;ALL"
                             "URL;URL_MD5;DOWNLOAD_DIR;GXF_ENTITIES_WIDTH;GXF_ENTITIES_HEIGHT;GXF_ENTITIES_CHANNELS;GXF_ENTITIES_FRAMERATE"
                             "BYPRODUCTS" ${ARGN})

  if(NOT DATA_URL)
    message(FATAL "No URL set for holoscan_download_data. Please set the URL.")
  endif()

  if(NOT DATA_DOWNLOAD_DIR)
    message(FATAL "No DOWNLOAD_DIR set for holoscan_download_data. Please set the DOWNLOAD_DIR.")
  endif()

  if(DATA_URL_MD5)
    list(APPEND extra_data_options --md5 ${DATA_URL_MD5})
  endif()

  if(DATA_GENERATE_GXF_ENTITIES)
     list(APPEND extra_data_options --generate_gxf_entities)
  endif()

  if(DATA_GXF_ENTITIES_WIDTH)
    list(APPEND extra_data_options --gxf_entities_width ${DATA_GXF_ENTITIES_WIDTH})
  endif()

  if(DATA_GXF_ENTITIES_HEIGHT)
    list(APPEND extra_data_options --gxf_entities_height ${DATA_GXF_ENTITIES_HEIGHT})
  endif()

  if(DATA_GXF_ENTITIES_CHANNELS)
    list(APPEND extra_data_options --gxf_entities_channels ${DATA_GXF_ENTITIES_CHANNELS})
  endif()

  if(DATA_GXF_ENTITIES_FRAMERATE)
    list(APPEND extra_data_options --gxf_entities_framerate ${DATA_GXF_ENTITIES_FRAMERATE})
  endif()

  # Find the download_ngc_data script
  find_program(DOWNLOAD_NGC_DATA_SCRIPT download_ngc_data ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_SOURCE_DIR}/scripts NO_CMAKE_FIND_ROOT_PATH)
  if(NOT DOWNLOAD_NGC_DATA_SCRIPT)
    message(FATAL_ERROR "download_ngc_data not found")
  endif()


  # Using a custom_command attached to a custom target allows to run only the custom command
  # if the stamp is not generated
  add_custom_command(OUTPUT "${DATA_DOWNLOAD_DIR}/${dataname}/${dataname}.stamp"
     COMMAND ${DOWNLOAD_NGC_DATA_SCRIPT}
     --url ${DATA_URL}
     --download_dir ${DATA_DOWNLOAD_DIR}
     --download_name ${dataname}
     ${extra_data_options}
     BYPRODUCTS ${DATA_BYPRODUCTS}
  )

  # If the target should be run all the time
  set(ALL)
  if(DATA_ALL)
    set(ALL "ALL")
  endif()

  add_custom_target("${dataname}_download" ${ALL} DEPENDS "${DATA_DOWNLOAD_DIR}/${dataname}/${dataname}.stamp")

endfunction()
