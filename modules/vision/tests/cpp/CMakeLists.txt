# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG QUIET)
endif()
if(NOT TARGET GTest::gtest_main)
    holoscan_add_dependency(gtest_rapids COMPONENT holoscan-dependencies)
endif()

get_target_property(_holoscan_vision_source_dir vision SOURCE_DIR)

add_executable(holoscan_vision_tests
    test_capacity_api.cpp
    test_pipeline.cpp
    test_operators.cpp
    "${_holoscan_vision_source_dir}/src/detail/cvcuda_tensor.cpp"
)
holoscan_configure_cxx20_target(holoscan_vision_tests PRIVATE)
target_include_directories(holoscan_vision_tests PRIVATE
    "${_holoscan_vision_source_dir}/src"
    "${holoscan_SOURCE_DIR}/src"
    "${holoscan_SOURCE_DIR}/tests/core"
)
target_link_libraries(holoscan_vision_tests
    PRIVATE
        holoscan::vision
        cvcuda
        nvcv_types
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(holoscan_vision_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS "vision;unit"
)

add_test(NAME VISION.installed_package_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${_holoscan_vision_source_dir}/tests/installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
        "-DHOLOSCAN_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
        "-DHOLOSCAN_EXTRA_BUILD_TARGETS=vision"
        "-DHOLOSCAN_EXTRA_INSTALL_COMPONENTS=holoscan-vision"
        "-DHOLOSCAN_CONSUMER_FIND_MODE=module"
        "-DHOLOSCAN_MODULE_PACKAGE_DIR=${CMAKE_INSTALL_LIBDIR}/cmake/holoscan"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P "${CMAKE_SOURCE_DIR}/tests/system/run_time_installed_consumer.cmake"
)
set_tests_properties(VISION.installed_package_consumer PROPERTIES
    RUN_SERIAL TRUE
    LABELS "vision;installed-package"
)

add_test(NAME VISION.installed_component_consumer
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DHOLOSCAN_CONSUMER_SOURCE_DIR=${_holoscan_vision_source_dir}/tests/installed_consumer"
        "-DHOLOSCAN_CONSUMER_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/installed_component_consumer"
        "-DCMAKE_BUILD_TYPE=$<CONFIG>"
        "-DHOLOSCAN_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
        "-DHOLOSCAN_EXTRA_BUILD_TARGETS=vision"
        "-DHOLOSCAN_EXTRA_INSTALL_COMPONENTS=holoscan-vision"
        "-DHOLOSCAN_CONSUMER_FIND_MODE=component"
        "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
        "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DHOLOSCAN_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DHOLOSCAN_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
        "-DHOLOSCAN_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P "${CMAKE_SOURCE_DIR}/tests/system/run_time_installed_consumer.cmake"
)
set_tests_properties(VISION.installed_component_consumer PROPERTIES
    RUN_SERIAL TRUE
    LABELS "vision;installed-package"
)

unset(_holoscan_vision_source_dir)
