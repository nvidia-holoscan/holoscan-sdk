# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)
project(holoscan_dependency_package_config LANGUAGES NONE)

list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_SDK_SOURCE_DIR}/cmake/modules")
include("${HOLOSCAN_SDK_SOURCE_DIR}/cmake/HoloscanDependencyPackageConfig.cmake")
include("${HOLOSCAN_SDK_SOURCE_DIR}/cmake/setup_dependencies.cmake")

# Reuse the SDK's fetched sources for the grouped archive regression.
enable_language(CXX)
set(rapids-cmake-dir "${HOLOSCAN_TEST_RAPIDS_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
set(CPM_DOWNLOAD_LOCATION "${HOLOSCAN_TEST_CPM_FILE}")
set(CPM_tl-expected_SOURCE "${HOLOSCAN_TEST_EXPECTED_SOURCE_DIR}")
set(HOLOSCAN_PACKAGE_NAME holoscan)
rapids_cpm_init()
holoscan_add_dependency(expected_rapids)

# A sentinel keeps the grouped archive nonempty. tl-expected must add no installed files.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sentinel.txt" "external tl-expected\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/sentinel.txt"
    DESTINATION share/holoscan COMPONENT holoscan-config)

set(CPACK_PACKAGE_NAME holoscan-expected-fixture)
set(CPACK_PACKAGE_VERSION 1.0.0)
set(CPACK_PACKAGING_INSTALL_PREFIX /usr)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
set(CPACK_COMPONENTS_ALL holoscan-config holoscan-dependencies)
set(CPACK_ARCHIVE_CORE_FILE_NAME expected-core)
include(CPackComponent)
cpack_add_component_group(core)
cpack_add_component(holoscan-config GROUP core)
cpack_add_component(holoscan-dependencies GROUP core)
include(CPack)

get_directory_property(expected_excluded DIRECTORY "${tl-expected_SOURCE_DIR}"
    EXCLUDE_FROM_ALL)
if(NOT expected_excluded)
    message(FATAL_ERROR "CPM did not exclude upstream tl-expected install rules")
endif()
