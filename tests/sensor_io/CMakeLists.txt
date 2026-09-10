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
    test_clock_discipline.cpp
    test_descriptor_passthrough.cpp
    test_lifecycle_guards.cpp
    test_lifecycle_guards_session.cpp
    test_provenance.cpp
    test_schema_identity.cpp
    test_sequence_tracker.cpp
)
holoscan_configure_test_clang_tidy(sensor_io_test)

target_include_directories(sensor_io_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
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
