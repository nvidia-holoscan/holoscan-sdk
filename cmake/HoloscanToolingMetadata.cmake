# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(holoscan_install_tooling_metadata template_name output_name component_name)
  set(_holoscan_tooling_metadata_dir
      "${CMAKE_CURRENT_BINARY_DIR}/tooling-metadata")
  file(MAKE_DIRECTORY "${_holoscan_tooling_metadata_dir}")
  configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/tooling/${template_name}"
      "${_holoscan_tooling_metadata_dir}/${output_name}"
      @ONLY
  )
  install(FILES "${_holoscan_tooling_metadata_dir}/${output_name}"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/holoscan/tooling/v1"
      COMPONENT "${component_name}"
  )
  unset(_holoscan_tooling_metadata_dir)
endfunction()
