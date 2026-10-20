# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Creates a target with the <dataname>_data
function(holoscan_download_data dataname)

  # If we already have the target we return
  if(TARGET "${dataname}_download")
    return()
  endif()

  cmake_parse_arguments(DATA "ALL"
                             "URL;URL_MD5;DOWNLOAD_DIR"
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

  # Find the script beside this module (legacy install layout), in the current
  # install bin directory, or relative to this module in the source tree.
  find_program(DOWNLOAD_NGC_DATA_SCRIPT download_ngc_data
    HINTS
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../../bin"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../scripts"
    NO_CMAKE_FIND_ROOT_PATH
  )
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
