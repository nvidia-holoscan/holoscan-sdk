# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${holoscan_SOURCE_DIR}/tests/cmake/consumer_test.cmake")

if(NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG QUIET)
endif()
if(NOT TARGET GTest::gtest_main)
    holoscan_add_dependency(gtest_rapids COMPONENT holoscan-dependencies)
endif()

get_target_property(_holoscan_vision_source_dir vision SOURCE_DIR)

add_executable(holoscan_vision_tests
    test_capacity_api.cpp
    test_operator_catalog.cpp
    test_pipeline.cpp
    test_operators.cpp
    test_image_payload.cpp
    test_borrowed_image.cpp
)

set(_holoscan_vision_test_targets holoscan_vision_tests)
# Keep native malloc interposition out of the general Vision test executable. A sanitizer runtime
# intercepts malloc itself and calls it before thread-local storage is ready, so sanitizer builds
# omit this executable.
if(NOT HOLOSCAN_ENABLE_GOOGLE_SANITIZER AND NOT HOLOSCAN_ENABLE_THREAD_SANITIZER)
    add_executable(holoscan_vision_nvcv_warmup_tests test_nvcv_warmup.cpp)
    list(APPEND _holoscan_vision_test_targets holoscan_vision_nvcv_warmup_tests)
endif()

include(GoogleTest)
foreach(_vision_test_target IN LISTS _holoscan_vision_test_targets)
    target_sources(${_vision_test_target} PRIVATE
        "${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp"
    )
    holoscan_configure_cxx20_target(${_vision_test_target} PRIVATE)
    holoscan_apply_test_optimization(${_vision_test_target})
    target_include_directories(${_vision_test_target} PRIVATE
        "${_holoscan_vision_source_dir}/src"
        "${holoscan_SOURCE_DIR}/src"
        "${holoscan_SOURCE_DIR}/tests/core"
        "${holoscan_SOURCE_DIR}/tests"
    )
    target_link_libraries(${_vision_test_target}
        PRIVATE
            holoscan::vision
            vision_detail_objects
            holoscan::sensor_io
            cvcuda
            nvcv_types
            GTest::gtest_main
    )
    gtest_discover_tests(${_vision_test_target}
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "vision;unit"
    )
endforeach()
unset(_vision_test_target)
unset(_holoscan_vision_test_targets)

holoscan_add_installed_consumer_test(
    VISION.installed_package_consumer
    SOURCE_DIRECTORY "${_holoscan_vision_source_dir}/tests/installed_consumer"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/installed_consumer"
    LABELS vision
    CMAKE_ARGS
        "-DHOLOSCAN_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
        "-DHOLOSCAN_EXTRA_BUILD_TARGETS=vision"
        "-DHOLOSCAN_EXTRA_INSTALL_COMPONENTS=holoscan-vision"
        "-DHOLOSCAN_CONSUMER_FIND_MODE=module"
        "-DHOLOSCAN_MODULE_PACKAGE_DIR=${CMAKE_INSTALL_LIBDIR}/cmake/holoscan"
)

holoscan_add_installed_consumer_test(
    VISION.installed_component_consumer
    SOURCE_DIRECTORY "${_holoscan_vision_source_dir}/tests/installed_consumer"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/installed_component_consumer"
    LABELS vision
    CMAKE_ARGS
        "-DHOLOSCAN_BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
        "-DHOLOSCAN_EXTRA_BUILD_TARGETS=vision"
        "-DHOLOSCAN_EXTRA_INSTALL_COMPONENTS=holoscan-vision"
        "-DHOLOSCAN_CONSUMER_FIND_MODE=component"
)

unset(_holoscan_vision_source_dir)
