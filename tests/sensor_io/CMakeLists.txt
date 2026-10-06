# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

# One target covers every component. Most of the surface is header-only; the exception is the Image
# metadata path that arrived with image.fbs, which is compiled into the library. Every test here
# exercises the installed public headers only, with no reach into public/src: the surface is meant
# to be usable by an out-of-tree operator, and a test that borrowed a private seam could pass on a
# surface that no such operator can compile against. Two tests stand up a full run session, because
# what they pin cannot be observed from a helper call: the passthrough test's metadata-forwarding
# idiom spans the descriptor encode path and the runtime envelope, and the lifecycle-guard session
# test needs Core's own stage schedule to show that a withheld stage body coexists with a run that
# terminated cleanly.
add_executable(sensor_io_test
    test_binary_schema.cpp
    test_clock_discipline.cpp
    test_descriptor_passthrough.cpp
    test_image_descriptor_scratch.cpp
    test_lifecycle_guards.cpp
    test_lifecycle_guards_session.cpp
    test_provenance.cpp
    test_schema_identity.cpp
    test_sequence_tracker.cpp
    ../common/allocation_counter.cpp
)
holoscan_configure_test_clang_tidy(sensor_io_test)

target_include_directories(sensor_io_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# holoscan::sensor_io now supplies the generated Image owning types and their trait specializations
# as well as the library under test, so the passthrough test's stand-in for a descriptor-backed sensor
# payload comes from the same dependency. holoscan::schema stays named because the test also reaches
# Core's tensor machinery directly, and because the transitive link should not be load-bearing here.
target_link_libraries(sensor_io_test
    PRIVATE
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        GTest::gtest_main
)

gtest_discover_tests(sensor_io_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX SENSOR_IO.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

include("${holoscan_SOURCE_DIR}/tests/cmake/consumer_test.cmake")

# The common SDK exports also reference the configuration and parameter libraries.
set(_sensor_io_consumer_targets core holoscan_schema sensor_io config config_io params params_io)
# iceoryx2-lib installs both C++ variants, regardless of the SDK's selected linkage.
# Imported dependencies have no local build rule and must not be passed to --target.
foreach(_dependency IN ITEMS spdlog shared-lib-cxx static-lib-cxx)
    if(TARGET ${_dependency})
        get_target_property(_imported ${_dependency} IMPORTED)
        if(NOT _imported)
            list(APPEND _sensor_io_consumer_targets ${_dependency})
        endif()
    endif()
endforeach()

if(BUILD_SHARED_LIBS)
    set(_iceoryx2_target iceoryx2-cxx::shared-lib-cxx)
else()
    set(_iceoryx2_target iceoryx2-cxx::static-lib-cxx)
endif()
get_target_property(_imported ${_iceoryx2_target} IMPORTED)
set(_sensor_io_consumer_packages "")
if(_imported)
    # Carry the selected prebuilt packages into the separate consumer configuration.
    # Source builds instead discover the packages installed in the fresh test prefix.
    foreach(_package IN ITEMS iceoryx2-cxx iceoryx2-c iceoryx2-bb-cxx)
        if(DEFINED ${_package}_DIR AND NOT "${${_package}_DIR}" STREQUAL "")
            list(APPEND _sensor_io_consumer_packages "-D${_package}_DIR=${${_package}_DIR}")
        endif()
    endforeach()
endif()

holoscan_add_installed_consumer_test(
    SENSOR_IO.installed_consumer
    SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/installed_consumer"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/installed_consumer"
    RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_installed_consumer.cmake"
    CMAKE_ARGS
        "-DHOLOSCAN_BUILD_TARGETS=${_sensor_io_consumer_targets}"
        ${_sensor_io_consumer_packages}
)
unset(_sensor_io_consumer_targets)
unset(_sensor_io_consumer_packages)
unset(_dependency)
unset(_package)
unset(_iceoryx2_target)
unset(_imported)
