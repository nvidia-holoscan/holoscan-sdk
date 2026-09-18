# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Unit tests for the layer renderer API and its HoloViz backends. They link the backend object
# library directly (never libholoscan_op_holoviz): the seam headers they reach through are
# private to the operator and the object library is the only target that exposes them.
# target_link_libraries() on an OBJECT library both adds its object files to the executable and
# propagates its include roots (operators/src, the module-private src tree).
# Shared test infrastructure (gtest main with the skip protocol, image comparison helpers) is
# reused from the module's tests/unit tree.
set(_holoviz_module_unit_dir "${holoscan_visualization_SOURCE_DIR}/tests/unit")

function(holoviz_operator_add_unittest_target target_name)
    add_executable(${target_name}
        ${ARGN}
        "${_holoviz_module_unit_dir}/holoviz_test_main.cpp"
    )
    add_executable(holoviz::${target_name} ALIAS ${target_name})
    target_include_directories(${target_name} PRIVATE "${_holoviz_module_unit_dir}")
    target_link_libraries(${target_name}
        PRIVATE
            op_holoviz_backends
            holoscan::visualization::holoviz
            GTest::gtest
    )

    holoviz_copy_runtime_dlls(${target_name})
    add_test(NAME ${target_name} COMMAND ${target_name})
    # Same conventions as tests/unit/CMakeLists.txt: validation layer on, SKIP_RETURN_CODE 77.
    set_tests_properties(${target_name} PROPERTIES
        ENVIRONMENT "HOLOVIZ_VALIDATION=1"
        LABELS "holoviz;visualization"
        SKIP_RETURN_CODE 77
    )
    add_dependencies(holoscan_visualization_test ${target_name})
endfunction()

holoviz_operator_add_unittest_target(holoviz_test_layer_enums test_enums.cpp)
holoviz_operator_add_unittest_target(holoviz_test_layer_desc test_layer_desc.cpp)
holoviz_operator_add_unittest_target(holoviz_test_sink_desc test_sink_desc.cpp)
holoviz_operator_add_unittest_target(holoviz_test_render_backend test_render_backend.cpp)

# The backend seam tests reach through the operator-private headers ("holoviz_op/backends/...")
# and the module-private HoloViz seams ("vulkan/vk_swapchain_fault_injection.hpp",
# vk_swapchain_util.hpp); both include roots come from the object library's usage requirements.
holoviz_operator_add_unittest_target(holoviz_test_wsi_present_backend test_wsi_present_backend.cpp)
holoviz_operator_add_unittest_target(holoviz_test_wsi_present_fault_injection
    test_wsi_present_fault_injection.cpp)
# Fullscreen runtime behavior (monitor-sized creation, swapchain recovery) through the same
# fault-injection seam. Takes over a monitor, so every test skips unless
# HOLOVIZ_FULLSCREEN_TESTS=1 is set in the environment.
holoviz_operator_add_unittest_target(holoviz_test_fullscreen_window test_fullscreen_window.cpp)
holoviz_operator_add_unittest_target(holoviz_test_backend_render test_holoviz_render_backend.cpp)
holoviz_operator_add_unittest_target(holoviz_test_backend_text test_holoviz_backend_text.cpp)
holoviz_operator_add_unittest_target(holoviz_test_backend_geometry test_holoviz_backend_geometry.cpp)
holoviz_operator_add_unittest_target(holoviz_test_backend_image test_holoviz_backend_image.cpp)
holoviz_operator_add_unittest_target(holoviz_test_backend_lut test_holoviz_backend_lut.cpp)

# HolovizOp image-layer Tensor admission through the production operator path. The operator
# sources are compiled into the test alongside the backend object library (never the DSO, see
# above), which also gives the test the operator-private impl header. Needs holoscan::core's
# private synthetic operator fixture from the SDK tree, so it is SDK-integrated only; it owns a
# real window and swapchain like the WSI present backend tests.
if(NOT holoscan_visualization_IS_TOP_LEVEL)
    set(_holoviz_operator_src "${CMAKE_CURRENT_SOURCE_DIR}/../../src/holoviz_op")
    holoviz_operator_add_unittest_target(holoviz_test_holoviz_op_image_admission
        test_holoviz_op_image_admission.cpp
        "${_holoviz_operator_src}/error_translation.cpp"
        "${_holoviz_operator_src}/holoviz_op.cpp"
        "${_holoviz_operator_src}/holoviz_op_impl.cpp"
    )
    unset(_holoviz_operator_src)
    target_link_libraries(holoviz_test_holoviz_op_image_admission PRIVATE CUDA::cudart)
    target_include_directories(holoviz_test_holoviz_op_image_admission
        PRIVATE
            ${holoscan_SOURCE_DIR}/src
            ${holoscan_SOURCE_DIR}/tests/core
    )
    set_tests_properties(holoviz_test_holoviz_op_image_admission PROPERTIES RUN_SERIAL TRUE)
endif()

unset(_holoviz_module_unit_dir)
