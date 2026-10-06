# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is loaded by CPack after CPackConfig.cmake and once for each requested generator.
if(CPACK_GENERATOR STREQUAL "DEB")
  # Debian packages deliberately omit the version-specific Python binding payload. Python remains
  # available to source builds and archive generators, whose layouts are not constrained by the
  # system Python ABI.
  list(REMOVE_ITEM CPACK_COMPONENTS_ALL
    holoscan-python_libs
    holoscan-sensor-io-python_libs
    holoscan-vision-python_libs
    holoscan-visualization-python_libs
    holoscan-inference-python_libs)
endif()

if(HOLOSCAN_CPACK_MONOLITHIC_PACKAGE)
  # Preserve the historical single-package layout as an explicit fallback for downstream tooling
  # that has not yet migrated to the granular payloads.
  set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
  set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
else()
  set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
  set(CPACK_DEB_COMPONENT_INSTALL ON)
  set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)

  # CPack does not promote component dependencies across component groups. Add the public package
  # contract explicitly so each module requires the exact core version produced in this run.
  foreach(module_group IN ITEMS VISION INFERENCE VISUALIZATION TESTING)
    if(CPACK_DEBIAN_${module_group}_PACKAGE_DEPENDS)
      string(APPEND CPACK_DEBIAN_${module_group}_PACKAGE_DEPENDS ", ")
    endif()
    string(APPEND CPACK_DEBIAN_${module_group}_PACKAGE_DEPENDS
      "holoscan-core (= ${CPACK_PACKAGE_VERSION})")
  endforeach()

  # Component archives are useful developer artifacts, but an empty meta-package has no meaningful
  # archive equivalent. The Debian meta-package is generated separately with dpkg-deb.
  set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
  set(CPACK_ARCHIVE_CORE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-holoscan-core")
  set(CPACK_ARCHIVE_TESTING_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-holoscan-testing")
  set(CPACK_ARCHIVE_VISION_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-holoscan-vision")
  set(CPACK_ARCHIVE_INFERENCE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-holoscan-inference")
  set(CPACK_ARCHIVE_VISUALIZATION_FILE_NAME
    "${CPACK_PACKAGE_FILE_NAME}-holoscan-visualization")
endif()
