# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(tl-expected 1.1.0
    GLOBAL_TARGETS tl::expected
    BUILD_EXPORT_SET "${HOLOSCAN_DEPEND_EXPORT_SET}"
    INSTALL_EXPORT_SET "${HOLOSCAN_DEPEND_EXPORT_SET}"

    CPM_ARGS

    GITHUB_REPOSITORY TartanLlama/expected
    GIT_TAG v1.1.0
    GIT_SHALLOW TRUE
    PATCH_COMMAND bash -c "patch -p1 -N --dry-run -i ${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected.patch >/dev/null && patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected.patch || patch -p1 -R --dry-run -i ${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected.patch >/dev/null"
    EXCLUDE_FROM_ALL

    OPTIONS
    "EXPECTED_BUILD_TESTS OFF"
    "EXPECTED_BUILD_PACKAGE OFF"
)

# tl-expected is a PUBLIC compile dependency of holoscan::core (expected.hpp includes
# <tl/expected.hpp>), so it is registered in the holoscan export set above and the installed
# holoscan-dependencies.cmake will find_dependency(tl-expected). EXCLUDE_FROM_ALL suppresses its own
# install rules, so install both the headers and a minimal find-package config that defines the
# tl::expected INTERFACE target pointing at those headers.
if(tl-expected_ADDED)
    include(GNUInstallDirs)
    install(DIRECTORY "${tl-expected_SOURCE_DIR}/include/tl"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT "${HOLOSCAN_DEPEND_INSTALL_COMPONENT}"
    )
    install(FILES
            "${CMAKE_CURRENT_LIST_DIR}/configs/tl-expected-config.cmake"
            "${CMAKE_CURRENT_LIST_DIR}/configs/tl-expected-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tl-expected"
        COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}
    )
endif()

# Set 'tl-expected_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(tl-expected_SOURCE_DIR ${tl-expected_SOURCE_DIR} PARENT_SCOPE)
