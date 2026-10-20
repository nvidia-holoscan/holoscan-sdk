# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Operator-level HolovizOp tests that link the operator DSO like an application does (never the
# backend object library, so there is exactly one copy of HoloViz in the process). They need
# Holoscan Core, so they only exist in the SDK build.
function(holoviz_operator_add_system_test target_name)
    add_executable(${target_name} ${ARGN})
    if(COMMAND holoscan_apply_test_optimization)
        holoscan_apply_test_optimization(${target_name})
    endif()
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_link_libraries(${target_name}
        PRIVATE
            holoscan::core
            holoscan::visualization::op_holoviz
            GTest::gtest
    )
    holoviz_copy_runtime_dlls(${target_name})
    add_dependencies(holoscan_visualization_test ${target_name})
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
        ENVIRONMENT "HOLOVIZ_VALIDATION=1"
        LABELS "holoviz;visualization"
        SKIP_RETURN_CODE 77
    )
endfunction()

# start() sink dispatch without a Graph: unsupported sinks fail before any GPU resource exists.
holoviz_operator_add_system_test(holoviz_test_holoviz_op_sinks test_holoviz_op_sinks.cpp)
# The temporal contract as compiled (LayerTrigger -> OnEach, ClockTrigger -> OnClock) and the
# constructor's trigger validation; compile-only, no window or GPU.
holoviz_operator_add_system_test(holoviz_test_holoviz_op_contract test_holoviz_op_contract.cpp)
# HolovizOp's CUDA sink end to end: captured frames on the output port, pixel preservation, and a
# two-operator chain. Needs a CUDA device (skips otherwise), no window.
holoviz_operator_add_system_test(holoviz_test_holoviz_op_cuda_sink test_holoviz_op_cuda_sink.cpp)
target_link_libraries(holoviz_test_holoviz_op_cuda_sink
    PRIVATE CUDA::cuda_driver holoscan::visualization::holoviz)
