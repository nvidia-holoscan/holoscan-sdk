# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is loaded by CPack after CPackConfig.cmake and once for each requested generator.
if(HOLOSCAN_CPACK_INCLUDE_EXAMPLES)
  # Examples are excluded from packages by default.
  if(NOT CPACK_HOLOSCAN_EXAMPLE_COMPONENTS)
    message(FATAL_ERROR
      "HOLOSCAN_CPACK_INCLUDE_EXAMPLES requires a build configured with HOLOSCAN_BUILD_EXAMPLES=ON")
  endif()
  list(APPEND CPACK_COMPONENTS_ALL ${CPACK_HOLOSCAN_EXAMPLE_COMPONENTS})
  list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)
endif()

if(CPACK_GENERATOR STREQUAL "DEB")
  # Debian packages deliberately omit the version-specific Python binding payload. Python remains
  # available to source builds and archive generators, whose layouts are not constrained by the
  # system Python ABI. The archive-root license copy is also omitted; Debian packages install the
  # license under share/doc.
  list(REMOVE_ITEM CPACK_COMPONENTS_ALL
    holoscan-archive-license
    holoscan-python_libs
    holoscan-testing-python_libs
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
  # CPack falls back to the monolithic Recommends for any package without its own list.
  unset(CPACK_DEBIAN_PACKAGE_RECOMMENDS)

  # CPack does not promote component dependencies across component groups. Add the public package
  # contract explicitly so each module requires the exact core version produced in this run.
  foreach(module_group IN ITEMS VISION INFERENCE VISUALIZATION)
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
  set(CPACK_ARCHIVE_VISION_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-holoscan-vision")
  set(CPACK_ARCHIVE_INFERENCE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-holoscan-inference")
  set(CPACK_ARCHIVE_VISUALIZATION_FILE_NAME
    "${CPACK_PACKAGE_FILE_NAME}-holoscan-visualization")
endif()
