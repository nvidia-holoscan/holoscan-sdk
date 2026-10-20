# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(tensor_stream_registration_benchmark tensor_stream_registration_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets tensor_stream_registration_benchmark)
target_include_directories(tensor_stream_registration_benchmark PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(tensor_stream_registration_benchmark
    PRIVATE benchmark::benchmark_main holoscan::core CUDA::cudart)
if(BUILD_TESTING)
    add_test(NAME BENCHMARK_TENSOR_STREAM_REGISTRATION
        COMMAND tensor_stream_registration_benchmark --benchmark_min_time=0.01s)
    set_tests_properties(BENCHMARK_TENSOR_STREAM_REGISTRATION PROPERTIES LABELS "benchmark;runtime;tensor")
endif()

add_executable(runtime_dispatcher_benchmark
    runtime_dispatcher_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/support/direct_port_binding.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_dispatcher_benchmark)
target_include_directories(runtime_dispatcher_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
)
target_link_libraries(runtime_dispatcher_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan_core_private_archive
        holoscan::core
)

add_executable(runtime_channel_benchmark
    runtime_channel_benchmark.cpp
    $<TARGET_OBJECTS:holoscan_runtime_channel_test_access_objects>
)
add_dependencies(holoscan-benchmark-targets runtime_channel_benchmark)
target_include_directories(runtime_channel_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(runtime_channel_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan_core_private_archive
        holoscan::core
        Threads::Threads
)

add_executable(runtime_payload_benchmark
    runtime_payload_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_payload_benchmark holoscan_flatbuffer_schemas_generated)
add_dependencies(holoscan-benchmark-targets runtime_payload_benchmark)
target_include_directories(runtime_payload_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
)
target_link_libraries(runtime_payload_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan_core_private_archive
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        Threads::Threads
)

add_executable(runtime_descriptor_graph_benchmark
    runtime_descriptor_graph_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_descriptor_graph_benchmark)
target_include_directories(runtime_descriptor_graph_benchmark PRIVATE
    ${holoscan_SOURCE_DIR}/tests
    ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(runtime_descriptor_graph_benchmark PRIVATE
    benchmark::benchmark_main holoscan::core holoscan::sensor_io Threads::Threads
)
add_executable(runtime_descriptor_receive_benchmark
    runtime_descriptor_receive_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)

add_executable(runtime_prepared_schema_graph_benchmark
    runtime_prepared_schema_graph_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_prepared_schema_graph_benchmark)
target_include_directories(runtime_prepared_schema_graph_benchmark PRIVATE
    ${holoscan_SOURCE_DIR}/tests
    ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(runtime_prepared_schema_graph_benchmark PRIVATE
    benchmark::benchmark_main holoscan::core holoscan::schema holoscan::sensor_io
    holomq::holomq Threads::Threads
)
add_dependencies(holoscan-benchmark-targets runtime_descriptor_receive_benchmark)
target_include_directories(runtime_descriptor_receive_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/tests
)
target_link_libraries(runtime_descriptor_receive_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
        holoscan::sensor_io
)

set(RUNTIME_NATIVE_SCHEMA_BENCHMARK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_native_schema_generated")
holoscan_add_flatbuffer_schema(
    TARGET runtime_native_schema_benchmark_generated
    SCHEMAS runtime_native_schema_benchmark.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RUNTIME_NATIVE_SCHEMA_BENCHMARK_GENERATED_DIR}"
    TRAITS_HEADER runtime_native_schema_benchmark_traits.hpp
    MAX_SERIALIZED_SIZE 8192
    NATIVE_BOUNDS_FILE runtime_native_schema_bounds.json
)
add_executable(runtime_native_schema_benchmark
    runtime_native_schema_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_native_schema_benchmark runtime_native_schema_benchmark_generated)
add_dependencies(holoscan-benchmark-targets runtime_native_schema_benchmark)
target_include_directories(runtime_native_schema_benchmark PRIVATE
    ${holoscan_SOURCE_DIR}/tests
    "${RUNTIME_NATIVE_SCHEMA_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_native_schema_benchmark PRIVATE
    benchmark::benchmark_main holoscan::core holoscan::schema holoscan::sensor_io
)

set(RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_multi_tensor_generated")
holoscan_add_flatbuffer_schema(
    TARGET runtime_multi_tensor_benchmark_generated
    SCHEMAS runtime_multi_tensor_benchmark.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
    TRAITS_HEADER runtime_multi_tensor_benchmark_traits.hpp
    MAX_SERIALIZED_SIZE 512
)

target_sources(runtime_payload_benchmark PRIVATE runtime_tensor_projection_benchmark.cpp)
add_dependencies(runtime_payload_benchmark runtime_multi_tensor_benchmark_generated)
target_include_directories(runtime_payload_benchmark PRIVATE
    "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}")

set(RUNTIME_THIRD_PARTY_BENCHMARK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_third_party_generated")
holoscan_add_flatbuffer_schema(
    TARGET runtime_third_party_benchmark_generated
    SCHEMAS runtime_third_party_benchmark.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RUNTIME_THIRD_PARTY_BENCHMARK_GENERATED_DIR}"
    TRAITS_HEADER runtime_third_party_benchmark_traits.hpp
    MAX_SERIALIZED_SIZE 128
)

add_executable(runtime_graph_benchmark
    runtime_graph_benchmark.cpp
    runtime_graph_phase_trace.cpp
    runtime_graph_cuda_work.cu
    runtime_graph_cuda_work.hpp
    runtime_graph_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_graph_benchmark
    holoscan_flatbuffer_schemas_generated
    runtime_multi_tensor_benchmark_generated
    runtime_third_party_benchmark_generated
)
add_dependencies(holoscan-benchmark-targets runtime_graph_benchmark)
target_include_directories(runtime_graph_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
        "${RUNTIME_THIRD_PARTY_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_graph_benchmark
    PRIVATE
        flatbuffers::flatbuffers
        fmt::fmt-header-only
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        Threads::Threads
)

# Separate from the single-producer graph benchmark: each source owns a Runtime partition.
add_executable(runtime_fanout_scaling_benchmark runtime_fanout_scaling_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets runtime_fanout_scaling_benchmark)
target_include_directories(runtime_fanout_scaling_benchmark PRIVATE "${holoscan_SOURCE_DIR}/src")
target_link_libraries(runtime_fanout_scaling_benchmark PRIVATE holoscan::core Threads::Threads)

if(BUILD_TESTING)
    if(Python3_Interpreter_FOUND)
        add_test(NAME BENCHMARK_FANOUT_SCALING_RUNNER_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_fanout_scaling.py")
        set_tests_properties(BENCHMARK_FANOUT_SCALING_RUNNER_PYTHON_TESTS
            PROPERTIES LABELS "benchmark;runtime;fanout")
    endif()
    foreach(FANOUT_DELIVERY IN ITEMS baseline pump emit direct)
        foreach(FANOUT_LAYOUT IN ITEMS disjoint shared)
            set(FANOUT_LAYOUT_ARGUMENTS)
            if(FANOUT_LAYOUT STREQUAL "shared")
                list(APPEND FANOUT_LAYOUT_ARGUMENTS --shared-destinations)
            endif()
            add_test(
                NAME "BENCHMARK_FANOUT_SCALING_${FANOUT_DELIVERY}_${FANOUT_LAYOUT}"
                COMMAND runtime_fanout_scaling_benchmark
                    --delivery "${FANOUT_DELIVERY}" ${FANOUT_LAYOUT_ARGUMENTS}
                    --producers 2 --fanout 3 --samples 32 --warmup 8 --period-us 1000
                    --queue-depth 1 --stall-us 1000 --counters
                    --timeout-ms 10000
            )
            set_tests_properties("BENCHMARK_FANOUT_SCALING_${FANOUT_DELIVERY}_${FANOUT_LAYOUT}"
                PROPERTIES LABELS "benchmark;runtime;fanout" TIMEOUT 20)
        endforeach()
    endforeach()
    # A result that cannot be written fails the run; BENCHMARK_FANOUT_SCALING_baseline_disjoint is
    # the same run with its result on standard output.
    add_test(
        NAME BENCHMARK_FANOUT_SCALING_UNWRITABLE_RESULT_FAILS
        COMMAND runtime_fanout_scaling_benchmark
            --delivery baseline --producers 2 --fanout 3 --samples 32 --warmup 8 --period-us 1000
            --queue-depth 1 --stall-us 1000 --counters --timeout-ms 10000 --output /dev/full
    )
    set_tests_properties(BENCHMARK_FANOUT_SCALING_UNWRITABLE_RESULT_FAILS
        PROPERTIES LABELS "benchmark;runtime;fanout" TIMEOUT 20 WILL_FAIL TRUE)
endif()

# The robot control loop shape: two clocks, side inputs, and a cross-partition diamond.
set(RUNTIME_CONTROL_LOOP_BENCHMARK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_control_loop_generated")
holoscan_add_flatbuffer_schema(
    TARGET runtime_control_loop_benchmark_generated
    SCHEMAS runtime_control_loop_benchmark.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RUNTIME_CONTROL_LOOP_BENCHMARK_GENERATED_DIR}"
    TRAITS_HEADER runtime_control_loop_benchmark_traits.hpp
    MAX_SERIALIZED_SIZE 256
)

add_executable(runtime_control_loop_benchmark runtime_control_loop_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets runtime_control_loop_benchmark)
add_dependencies(runtime_control_loop_benchmark
    holoscan_flatbuffer_schemas_generated runtime_control_loop_benchmark_generated)
target_include_directories(runtime_control_loop_benchmark PRIVATE "${holoscan_SOURCE_DIR}/src"
    "${HOLOSCAN_SCHEMA_GENERATED_DIR}" "${RUNTIME_CONTROL_LOOP_BENCHMARK_GENERATED_DIR}")
target_link_libraries(runtime_control_loop_benchmark PRIVATE
    flatbuffers::flatbuffers holoscan::core holoscan::schema holoscan::sensor_io Threads::Threads
    ${CMAKE_DL_LIBS})

# LD_PRELOAD shim that counts (or, with CONTROL_LOOP_YIELD_SHIM=noop, drops) sched_yield calls.
add_library(control_loop_yield_shim SHARED control_loop_yield_shim.cpp)
add_dependencies(holoscan-benchmark-targets control_loop_yield_shim)
target_link_libraries(control_loop_yield_shim PRIVATE ${CMAKE_DL_LIBS})

if(BUILD_TESTING)
    if(Python3_Interpreter_FOUND)
        add_test(NAME BENCHMARK_CONTROL_LOOP_RUNNER_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_control_loop.py")
        # The binary lets the tests check that authored guard text stays valid JSON.
        set_tests_properties(BENCHMARK_CONTROL_LOOP_RUNNER_PYTHON_TESTS
            PROPERTIES LABELS "benchmark;runtime;control_loop"
            ENVIRONMENT "CONTROL_LOOP_BENCHMARK=$<TARGET_FILE:runtime_control_loop_benchmark>")
    endif()
    # A retained device frame decodes its own host copy however far the producer runs ahead.
    add_test(NAME BENCHMARK_CONTROL_LOOP_DEVICE_SHADOW_SELF_TEST
        COMMAND runtime_control_loop_benchmark --self-test-device-shadow)
    set_tests_properties(BENCHMARK_CONTROL_LOOP_DEVICE_SHADOW_SELF_TEST
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # Functional smoke tests only: no pinning, no FIFO, and no latency thresholds.
    foreach(CONTROL_LOOP_CASE IN ITEMS
            "split rcb" "split split-outputs" "split single" "split same-partition-port"
            "split rcb-no-target-input" "fused rcb" "no-framework rcb" "no-framework single")
        string(REPLACE " " ";" CONTROL_LOOP_ARGUMENTS "${CONTROL_LOOP_CASE}")
        list(GET CONTROL_LOOP_ARGUMENTS 0 CONTROL_LOOP_MODE)
        list(GET CONTROL_LOOP_ARGUMENTS 1 CONTROL_LOOP_TOPOLOGY)
        string(TOUPPER "${CONTROL_LOOP_MODE}_${CONTROL_LOOP_TOPOLOGY}" CONTROL_LOOP_NAME)
        string(REPLACE "-" "_" CONTROL_LOOP_NAME "${CONTROL_LOOP_NAME}")
        add_test(
            NAME "BENCHMARK_CONTROL_LOOP_${CONTROL_LOOP_NAME}"
            COMMAND runtime_control_loop_benchmark
                --mode "${CONTROL_LOOP_MODE}" --topology "${CONTROL_LOOP_TOPOLOGY}"
                --samples 200 --warmup 20 --timeout-ms 10000
        )
        set_tests_properties("BENCHMARK_CONTROL_LOOP_${CONTROL_LOOP_NAME}"
            PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    endforeach()
    # Counter guards (--expect) are deterministic structure checks, not timing: they fail when a
    # fast path is lost. Thresholds leave margin for a loaded CTest host.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_SPLIT_RCB_COUNTERS
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --expect plan.joint_state==drop_oldest:direct
            --expect phase.transport_mutex_acquisitions==0 --expect phase.pump_waits==0
            --expect per_tick.causal_frontier_accepted>=2.4
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_SPLIT_RCB_COUNTERS
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # A 50 ms plan stage overruns every 1 ms control tick, and the 137 Hz target lane keeps
    # publishing past the nominal run length its records are sized for. The run must count the
    # publications it could not record and report itself invalid.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_OVERRUN_REJECTS_LOST_TARGET_OBSERVATIONS
        COMMAND runtime_control_loop_benchmark
            --mode no-framework --topology rcb --samples 20 --warmup 5
            --work-ns plan=50000000 --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_OVERRUN_REJECTS_LOST_TARGET_OBSERVATIONS
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20
        PASS_REGULAR_EXPRESSION "\"valid\":false.*\"target_observation_overflows\":[1-9]")
    # Native ticks follow OnClock coalescing: after an overrun or a late wake, the newest due tick
    # runs at once.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_NATIVE_SCHEDULE_SELF_TEST
        COMMAND runtime_control_loop_benchmark --self-test-native-schedule
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_NATIVE_SCHEDULE_SELF_TEST
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # A target whose emit is still in progress at a tick is not a missed freshness for that tick,
    # and a consumer that reads a target inside its emit gets a nonnegative delivery latency.
    # A three-tick joint-state selection retains its exact age and identity without clamping.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_TARGET_FRESHNESS_SELF_TEST
        COMMAND runtime_control_loop_benchmark --self-test-target-freshness
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_TARGET_FRESHNESS_SELF_TEST
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The only native callback starts before the 20 ms timeout and ends after it, so the run must
    # report itself timed out and invalid, and exit nonzero; the two tests check both.
    foreach(CONTROL_LOOP_EXIT_CHECK IN ITEMS OUTPUT EXIT)
        add_test(
            NAME BENCHMARK_CONTROL_LOOP_NATIVE_FINAL_CALLBACK_TIMES_OUT_${CONTROL_LOOP_EXIT_CHECK}
            COMMAND runtime_control_loop_benchmark
                --mode no-framework --topology single --samples 1 --warmup 0
                --work-ns plan=50000000 --timeout-ms 20
        )
    endforeach()
    set_tests_properties(BENCHMARK_CONTROL_LOOP_NATIVE_FINAL_CALLBACK_TIMES_OUT_OUTPUT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20
        PASS_REGULAR_EXPRESSION "\"valid\":false.*\"timed_out\":true")
    set_tests_properties(BENCHMARK_CONTROL_LOOP_NATIVE_FINAL_CALLBACK_TIMES_OUT_EXIT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20 WILL_FAIL TRUE)
    # The target lane bounds the run too. The 20 control ticks end about 30 ms in, well inside the
    # 100 ms timeout, and the target's first callback starts before them; its 300 ms of work, or
    # its 1 s sleep, outlasts the timeout, so the run times out though every tick was done.
    set(CONTROL_LOOP_TARGET_WORK_ARGUMENTS --work-ns target=300000000)
    set(CONTROL_LOOP_TARGET_SLEEP_ARGUMENTS --target-rate-hz 1)
    foreach(CONTROL_LOOP_TARGET_CASE IN ITEMS WORK SLEEP)
        set(CONTROL_LOOP_TEST_PREFIX
            BENCHMARK_CONTROL_LOOP_NATIVE_TARGET_${CONTROL_LOOP_TARGET_CASE}_TIMES_OUT)
        foreach(CONTROL_LOOP_EXIT_CHECK IN ITEMS OUTPUT EXIT)
            set(CONTROL_LOOP_TEST ${CONTROL_LOOP_TEST_PREFIX}_${CONTROL_LOOP_EXIT_CHECK})
            add_test(
                NAME ${CONTROL_LOOP_TEST}
                COMMAND runtime_control_loop_benchmark
                    --mode no-framework --topology single --samples 20 --warmup 0
                    --work-profile zero ${CONTROL_LOOP_TARGET_${CONTROL_LOOP_TARGET_CASE}_ARGUMENTS}
                    --timeout-ms 100
            )
            if(CONTROL_LOOP_EXIT_CHECK STREQUAL "OUTPUT")
                set_tests_properties(${CONTROL_LOOP_TEST}
                    PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20
                    PASS_REGULAR_EXPRESSION "\"valid\":false.*\"timed_out\":true")
            else()
                set_tests_properties(${CONTROL_LOOP_TEST}
                    PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20 WILL_FAIL TRUE)
            endif()
        endforeach()
    endforeach()
    # Native lanes place themselves through a fixed cpu_set_t; a CPU beyond it is rejected.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_NATIVE_REJECTS_CPU_OUTSIDE_THE_CPU_SET
        COMMAND runtime_control_loop_benchmark
            --mode no-framework --topology single --samples 5 --warmup 0 --control-cpu 1000000
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_NATIVE_REJECTS_CPU_OUTSIDE_THE_CPU_SET
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20 WILL_FAIL TRUE)
    # A requested observations file that cannot be written fails the run.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_UNWRITABLE_OBSERVATIONS_FAIL
        COMMAND runtime_control_loop_benchmark
            --mode no-framework --topology single --samples 5 --warmup 0 --timeout-ms 10000
            --observations "${CMAKE_CURRENT_BINARY_DIR}/missing-directory/observations.csv"
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_UNWRITABLE_OBSERVATIONS_FAIL
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20 WILL_FAIL TRUE)
    # The HoloMQ A/B control for compile()'s direct delivery of the joint-state fanout.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_SPLIT_RCB_HOLOMQ_FANOUT
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --force-holomq-fanout
            --timeout-ms 10000 --expect plan.joint_state==drop_oldest:emit
            --expect phase.transport_mutex_acquisitions>0 --expect phase.pump_waits==0
            --expect per_tick.causal_frontier_accepted>=1.8
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_SPLIT_RCB_HOLOMQ_FANOUT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The pump-delivery A/B control for compile()'s delivery inside emit().
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_SPLIT_RCB_PUMP_DELIVERY
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --force-pump-delivery
            --timeout-ms 10000 --expect plan.joint_state==drop_oldest:pump
            --expect phase.transport_mutex_acquisitions>0 --expect phase.pump_waits>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_SPLIT_RCB_PUMP_DELIVERY
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The A/B control for continuing the causal chain into MotionPlanner (RUNTIME-084).
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_SPLIT_RCB_NO_SIDE_INPUT_CONTINUATION
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters
            --disable-side-input-continuation --timeout-ms 10000
            --expect plan.joint_state==drop_oldest:direct
            --expect per_tick.causal_frontier_attempts<=2.05
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_SPLIT_RCB_NO_SIDE_INPUT_CONTINUATION
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # Parity matrix: a blocking control chain makes the joint-state fanout a mixed-policy family,
    # which is delivered by direct retention after an all-or-nothing capacity check (CHAN-167).
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_BLOCKING
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --control-policy blocking --expect plan.joint_state==blocking:direct+drop_oldest:direct
            --expect phase.transport_mutex_acquisitions==0 --expect phase.pump_waits==0
            --expect per_tick.causal_frontier_accepted>=1.8
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_BLOCKING
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The HoloMQ-fanout control keeps the same mixed-policy family on HoloMQ: the blocking member at
    # the cut, the drop-oldest one inside emit().
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_BLOCKING_HOLOMQ_FANOUT
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --control-policy blocking --force-holomq-fanout
            --expect plan.joint_state==blocking:cut+drop_oldest:emit
            --expect phase.transport_mutex_acquisitions>0 --expect phase.pump_waits==0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_BLOCKING_HOLOMQ_FANOUT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # Parity matrix: an exported member keeps its own transported service, and the local recipients
    # of the joint-state family stay direct (CHAN-165).
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_EXPORT
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --export-joint-state --expect plan.joint_state==drop_oldest:direct
            --expect phase.pump_waits==0 --expect per_tick.causal_frontier_accepted>=2.4
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_EXPORT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The same mixed-locality family with an owning generated-schema joint state.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_EXPORT_SCHEMA
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --export-joint-state --joint-payload schema --expect plan.joint_state==drop_oldest:direct
            --expect phase.pump_waits==0 --expect per_tick.causal_frontier_accepted>=2.4
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_EXPORT_SCHEMA
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The HoloMQ A/B control for the exported family: the whole family stays on HoloMQ with event
    # wake, so the local recipients lose direct fanout and delivery inside emit() as well.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_EXPORT_HOLOMQ_FANOUT
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --export-joint-state --force-holomq-fanout --expect plan.joint_state==drop_oldest:pump
            --expect phase.transport_mutex_acquisitions>0 --expect phase.pump_waits>0
            --expect per_tick.causal_frontier_accepted>=1.8
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_EXPORT_HOLOMQ_FANOUT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # Parity matrix: capture-time side selectors. Producers stamp capture times in these arms; the
    # guards check that both side inputs select (or reuse) samples rather than idling.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_LATEST_WITHIN
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --side-selector latest-within --expect plan.joint_state==drop_oldest:direct
            --expect count.motion_side_selected>0 --expect count.target_side_selected>0
            --expect count.target_side_borrowed==0
            --expect per_tick.causal_frontier_accepted>=1.8
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_LATEST_WITHIN
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_REUSE_LAST
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --side-selector reuse-last --expect count.motion_side_reused>0
            --expect count.motion_side_absent==0 --expect count.target_side_selected>0
            --expect per_tick.causal_frontier_accepted>=1.8
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_REUSE_LAST
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # The selector rows keep the control partition's causal chain past MotionPlanner's freshness
    # side input (RUNTIME-086); this is their A/B control.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_LATEST_WITHIN_NO_BOUNDARY_CHAIN
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --side-selector latest-within --disable-boundary-causal-chain
            --expect count.motion_side_selected>0 --expect per_tick.causal_frontier_attempts==0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_LATEST_WITHIN_NO_BOUNDARY_CHAIN
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # A selector on the target lane alone leaves the control chain's continuation intact.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_LATEST_WITHIN_TARGET
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --side-selector latest-within --side-selector-scope target
            --expect count.target_side_selected>0 --expect per_tick.causal_frontier_accepted>=2.4
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_LATEST_WITHIN_TARGET
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # Parity matrix: an owning generated-schema joint state is a direct fanout family like POD.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_SCHEMA
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload schema --expect plan.joint_state==drop_oldest:direct
            --expect phase.transport_mutex_acquisitions==0
            --expect per_tick.causal_frontier_accepted>=2.4
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_SCHEMA
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # Unpinned CI permits four ticks of staleness while still rejecting the old 7-15-tick lag.
    # Require observations as well as the age guard: an empty distribution reports max zero.
    # Parity matrix: a Tensor joint state fans out by direct retention with one shared backing
    # (CHAN-162), and is direct when each output has one recipient (split-outputs). The
    # HoloMQ-fanout control delivers host Tensor inside emit() (RUNTIME-085), and the pump arm
    # keeps the transported staging path (CHAN-161) under guard.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_HOST
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload tensor-host --expect plan.joint_state==drop_oldest:direct
            --expect phase.pump_waits==0 --expect count.checksum_failures==0
            --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_HOST
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_HOST_EMIT
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload tensor-host --force-holomq-fanout
            --expect plan.joint_state==drop_oldest:emit
            --expect phase.pump_waits==0 --expect count.checksum_failures==0
            --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_HOST_EMIT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_HOST_PUMP
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload tensor-host --force-pump-delivery
            --expect plan.joint_state==drop_oldest:pump
            --expect phase.pump_waits>0 --expect count.checksum_failures==0
            --expect per_tick.pump_waits<=3 --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_HOST_PUMP
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_SPLIT_OUTPUTS
        COMMAND runtime_control_loop_benchmark
            --mode split --topology split-outputs --samples 200 --warmup 20 --counters
            --timeout-ms 10000 --joint-payload tensor-host
            --expect plan.joint_state==drop_oldest:direct --expect phase.pump_waits==0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_SPLIT_OUTPUTS
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_DEVICE
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload tensor-device --expect plan.joint_state==drop_oldest:direct
            --expect count.checksum_failures==0 --expect phase.pump_waits==0
            --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_TENSOR_DEVICE PROPERTIES
        LABELS "benchmark;runtime;control_loop;cuda" TIMEOUT 20 SKIP_RETURN_CODE 77)
    # Parity matrix: a descriptor-backed schema (ImageT) joint state, owning or published and
    # received the borrowing way the vision operators use. A fanout family stays on the pump; one
    # recipient per output is direct.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload descriptor --expect plan.joint_state==drop_oldest:direct
            --expect phase.pump_waits==0 --expect count.checksum_failures==0
            --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_EMIT
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload descriptor --force-holomq-fanout
            --expect plan.joint_state==drop_oldest:emit
            --expect phase.pump_waits==0 --expect count.checksum_failures==0
            --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_EMIT
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_PUMP
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --counters --timeout-ms 10000
            --joint-payload descriptor --force-pump-delivery
            --expect plan.joint_state==drop_oldest:pump
            --expect phase.pump_waits>0 --expect count.checksum_failures==0
            --expect per_tick.pump_waits<=3 --expect count.side_input_staleness_max<=4 --expect count.side_input_staleness_samples>0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_PUMP
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_BORROW
        COMMAND runtime_control_loop_benchmark
            --mode split --topology split-outputs --samples 200 --warmup 20 --counters
            --timeout-ms 10000 --joint-payload descriptor-borrow
            --expect plan.joint_state==drop_oldest:direct --expect count.checksum_failures==0
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_BORROW
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    # A borrowed descriptor joint state stays borrowed when the target lane selects one sample.
    foreach(CONTROL_LOOP_BORROW_SELECTOR IN ITEMS latest-within reuse-last)
        string(TOUPPER "${CONTROL_LOOP_BORROW_SELECTOR}" CONTROL_LOOP_BORROW_NAME)
        string(REPLACE "-" "_" CONTROL_LOOP_BORROW_NAME "${CONTROL_LOOP_BORROW_NAME}")
        add_test(
            NAME BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_BORROW_${CONTROL_LOOP_BORROW_NAME}
            COMMAND runtime_control_loop_benchmark
                --mode split --topology rcb --samples 200 --warmup 20 --counters
                --timeout-ms 10000 --joint-payload descriptor-borrow
                --side-selector ${CONTROL_LOOP_BORROW_SELECTOR} --side-selector-scope target
                --expect count.target_side_borrowed>0 --expect count.checksum_failures==0
        )
        set_tests_properties(
            BENCHMARK_CONTROL_LOOP_MATRIX_DESCRIPTOR_BORROW_${CONTROL_LOOP_BORROW_NAME}
            PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    endforeach()
    # A failing guard must fail the run and report itself: expect the direct joint state on pump.
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_GUARD_SELF_TEST
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --plan-only --expect plan.joint_state==drop_oldest:pump
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_GUARD_SELF_TEST PROPERTIES
        LABELS "benchmark;runtime;control_loop" TIMEOUT 20
        PASS_REGULAR_EXPRESSION "drop_oldest:direct\"..\"guards\":.\"passed\":false")
    # --plan-only writes its report where --output points: to a file, and failing on /dev/full.
    foreach(CONTROL_LOOP_PLAN_OUTPUT IN ITEMS FILE UNWRITABLE)
        set(CONTROL_LOOP_PLAN_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/control_loop_plan.json")
        if(CONTROL_LOOP_PLAN_OUTPUT STREQUAL "UNWRITABLE")
            set(CONTROL_LOOP_PLAN_OUTPUT_PATH /dev/full)
        endif()
        add_test(
            NAME BENCHMARK_CONTROL_LOOP_PLAN_ONLY_OUTPUT_${CONTROL_LOOP_PLAN_OUTPUT}
            COMMAND runtime_control_loop_benchmark
                --mode split --topology rcb --plan-only
                --expect plan.joint_state==drop_oldest:direct
                --output "${CONTROL_LOOP_PLAN_OUTPUT_PATH}"
        )
    endforeach()
    set_tests_properties(BENCHMARK_CONTROL_LOOP_PLAN_ONLY_OUTPUT_FILE
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20)
    set_tests_properties(BENCHMARK_CONTROL_LOOP_PLAN_ONLY_OUTPUT_UNWRITABLE
        PROPERTIES LABELS "benchmark;runtime;control_loop" TIMEOUT 20 WILL_FAIL TRUE)
    add_test(
        NAME BENCHMARK_CONTROL_LOOP_SPLIT_RCB_YIELD_SHIM
        COMMAND runtime_control_loop_benchmark
            --mode split --topology rcb --samples 200 --warmup 20 --require-yield-shim
            --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_CONTROL_LOOP_SPLIT_RCB_YIELD_SHIM PROPERTIES
        LABELS "benchmark;runtime;control_loop" TIMEOUT 20
        ENVIRONMENT "LD_PRELOAD=$<TARGET_FILE:control_loop_yield_shim>;CONTROL_LOOP_YIELD_SHIM=pass")
endif()

add_executable(runtime_graph_interprocess_benchmark
    runtime_graph_interprocess_benchmark.cpp
    runtime_graph_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_graph_interprocess_benchmark
    holoscan_flatbuffer_schemas_generated
    runtime_multi_tensor_benchmark_generated
)
add_dependencies(holoscan-benchmark-targets runtime_graph_interprocess_benchmark)
target_include_directories(runtime_graph_interprocess_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_graph_interprocess_benchmark
    PRIVATE
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        Threads::Threads
)

if(BUILD_TESTING)
    foreach(green_context_layout IN ITEMS
            same_partition dedicated_to_dedicated dedicated_to_residual)
        if(green_context_layout STREQUAL "same_partition")
            set(green_context_operator_lanes shared)
        else()
            set(green_context_operator_lanes separate)
        endif()
        string(TOUPPER "${green_context_layout}" green_context_layout_upper)
        add_test(
            NAME "BENCHMARK_GREEN_CONTEXT_${green_context_layout_upper}_SMOKE"
            COMMAND $<TARGET_FILE:runtime_graph_benchmark>
                    --scenario tensor
                    --tensor-storage device
                    --tensor-bytes 4096
                    --operator-lanes "${green_context_operator_lanes}"
                    --green-context-layout "${green_context_layout}"
                    --workload latency
                    --samples 8
                    --warmup 2
                    --queue-depth 1
                    --publisher-period-us 1000
                    --payload-validation all
                    --cuda-resource-counters
                    --timeout-ms 10000
                    --output
                    "${CMAKE_CURRENT_BINARY_DIR}/green_context_${green_context_layout}.json"
                    --plan-output
                    "${CMAKE_CURRENT_BINARY_DIR}/green_context_${green_context_layout}_plan.json"
        )
        set_tests_properties(
            "BENCHMARK_GREEN_CONTEXT_${green_context_layout_upper}_SMOKE"
            PROPERTIES
                LABELS "benchmark;ea2;runtime;cuda;green_context"
                SKIP_RETURN_CODE 77
                TIMEOUT 30
        )
    endforeach()

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_graph_benchmark.py"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_PYTHON_TESTS
            PROPERTIES
                LABELS "benchmark;runtime"
        )

        add_test(
            NAME BENCHMARK_PARTITION_COVER_COMPARISON_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_partition_cover_comparison.py"
        )
        set_tests_properties(BENCHMARK_PARTITION_COVER_COMPARISON_PYTHON_TESTS
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
        )

        foreach(PARTITION_COVER IN ITEMS automatic include_all hybrid)
            set(PARTITION_COVER_LANES separate)
            if(PARTITION_COVER STREQUAL "include_all")
                set(PARTITION_COVER_LANES shared)
            endif()
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_${PARTITION_COVER}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --partition-cover "${PARTITION_COVER}"
                        --operator-lanes "${PARTITION_COVER_LANES}"
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 15
                        --repetitions 1
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_partition_cover_${PARTITION_COVER}"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_${PARTITION_COVER}
                PROPERTIES
                    LABELS "benchmark;ea2;runtime;partitioning"
                    TIMEOUT 30
            )
        endforeach()

        # Source periods follow callbacks. A 20 ms sink behind a depth-one blocking edge makes the
        # 1 kHz source's publications time out and retry empty, repeating a publication sequence;
        # a three-sample burst puts a callback across the warmup boundary. Each run must report one
        # period per measured callback after the first.
        set(SOURCE_PERIOD_EMPTY_RETRY_ARGUMENTS
            --fanout 1 --queue-depth 1 --source-work-us 0 --sink-work-us 20000 --samples 4
            --warmup 0)
        set(SOURCE_PERIOD_BURST_WARMUP_BOUNDARY_ARGUMENTS
            --fanout 2 --queue-depth 4 --burst-size 3 --samples 7 --warmup 2)
        foreach(SOURCE_PERIOD_CASE IN ITEMS EMPTY_RETRY BURST_WARMUP_BOUNDARY)
            string(TOLOWER "${SOURCE_PERIOD_CASE}" SOURCE_PERIOD_DIRECTORY)
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_SOURCE_PERIOD_${SOURCE_PERIOD_CASE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --scenario pod
                        --workload latency
                        --operator-lanes separate
                        --queue-policy blocking
                        --publisher-period-us 1000
                        ${SOURCE_PERIOD_${SOURCE_PERIOD_CASE}_ARGUMENTS}
                        --measure-source-period
                        --session-timeout-ms 10000
                        --process-timeout-sec 15
                        --repetitions 1
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/source_period_${SOURCE_PERIOD_DIRECTORY}"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_PERIOD_${SOURCE_PERIOD_CASE}
                PROPERTIES LABELS "benchmark;runtime;fanout" TIMEOUT 30)
        endforeach()
        # The empty-retry row above must actually retry.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_SOURCE_PERIOD_EMPTY_RETRY_RETRIES
            COMMAND runtime_graph_benchmark
                    --scenario pod --workload latency --operator-lanes separate
                    --queue-policy blocking --publisher-period-us 1000
                    ${SOURCE_PERIOD_EMPTY_RETRY_ARGUMENTS} --measure-source-period
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_PERIOD_EMPTY_RETRY_RETRIES
            PROPERTIES LABELS "benchmark;runtime;fanout" TIMEOUT 30
                PASS_REGULAR_EXPRESSION "\"source_empty_retry_callbacks\":[1-9]")

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_SHARED_REJECTED
            COMMAND ${CMAKE_COMMAND}
                    "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                    -DREJECTION_CASE=automatic_shared
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/assert_runtime_graph_partition_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_SHARED_REJECTED
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
                TIMEOUT 15
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FORCE_HOLOMQ_REJECTED
            COMMAND ${CMAKE_COMMAND}
                    "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                    -DREJECTION_CASE=automatic_forced_holomq
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/assert_runtime_graph_partition_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FORCE_HOLOMQ_REJECTED
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
                TIMEOUT 15
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_SHARED_OWNER_CONFLICT_REJECTED
            COMMAND ${CMAKE_COMMAND}
                    "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                    -DREJECTION_CASE=shared_owner_conflict
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/assert_runtime_graph_partition_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SHARED_OWNER_CONFLICT_REJECTED
            PROPERTIES
                LABELS "benchmark;runtime;argument-validation;partitioning"
                TIMEOUT 15
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FANOUT_PHASE_TRACE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --partition-cover automatic
                    --operator-lanes separate
                    --workload notification_latency
                    --scenario pod
                    --fanout 16
                    --samples 256
                    --warmup 32
                    --publisher-period-us 1000
                    --payload-validation all
                    --runtime-phase-counters
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 3
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_partition_cover_automatic_fanout_phase_trace"
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FANOUT_PHASE_TRACE
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
                TIMEOUT 60
        )

        if(Python3_VERSION VERSION_GREATER_EQUAL 3.10)
            add_test(
                NAME BENCHMARK_GREEN_CONTEXT_QUALIFICATION_PYTHON_TESTS
                COMMAND ${Python3_EXECUTABLE} -m pytest -q
                        "${CMAKE_CURRENT_SOURCE_DIR}/test_run_green_context_qualification.py"
            )
            set_tests_properties(BENCHMARK_GREEN_CONTEXT_QUALIFICATION_PYTHON_TESTS
                PROPERTIES
                    LABELS "benchmark;ea2;runtime;cuda"
            )

            add_test(
                NAME BENCHMARK_CUDA_STREAM_POLICY_QUALIFICATION_PYTHON_TESTS
                COMMAND ${Python3_EXECUTABLE} -m pytest -q
                        "${CMAKE_CURRENT_SOURCE_DIR}/test_run_cuda_stream_policy_qualification.py"
            )
            set_tests_properties(BENCHMARK_CUDA_STREAM_POLICY_QUALIFICATION_PYTHON_TESTS
                PROPERTIES
                    LABELS "benchmark;runtime;cuda;partitioning"
            )
        endif()

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_smoke"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER
            PROPERTIES
                LABELS "benchmark;runtime"
                TIMEOUT 45
        )

        # Exercise the depth-one staged-ingress attribution through the Python runner so CI
        # validates the emitted counters, replacement evidence, and final delivery drain rather
        # than merely checking that the native benchmark exits successfully. This diagnostic arm
        # is intentionally distinct from counters-off performance baselines.
        add_test(
            NAME
                BENCHMARK_RUNTIME_GRAPH_RUNNER_DROP_OLDEST_DEPTH_ONE_HOLOMQ_FALLBACK_DIAGNOSTIC
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --operator-lanes separate
                    --queue-policy drop_oldest
                    --queue-depth 1
                    --notification-capacity 1
                    --samples 16
                    --warmup 4
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --test-gate-pod-sink-for-pressure
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 1
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_drop_oldest_depth_one_holomq_fallback_diagnostic"
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_RUNNER_DROP_OLDEST_DEPTH_ONE_HOLOMQ_FALLBACK_DIAGNOSTIC
            PROPERTIES
                LABELS "benchmark;runtime;lossy;diagnostic"
                TIMEOUT 30
        )

        foreach(THIRD_PARTY_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(THIRD_PARTY_DELIVERY_ARGUMENTS)
            if(THIRD_PARTY_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND THIRD_PARTY_DELIVERY_ARGUMENTS
                    --force-holomq-process-local-inline)
            endif()
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_THIRD_PARTY_${THIRD_PARTY_DELIVERY}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --scenario third_party_schema
                        --operator-lanes shared
                        --callback-work-us 5
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --payload-validation all
                        --runtime-phase-counters
                        ${THIRD_PARTY_DELIVERY_ARGUMENTS}
                        --session-timeout-ms 10000
                        --process-timeout-sec 15
                        --repetitions 1
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_third_party_${THIRD_PARTY_DELIVERY}"
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_RUNNER_THIRD_PARTY_${THIRD_PARTY_DELIVERY}
                PROPERTIES
                    LABELS "benchmark;runtime;schema;third-party"
                    TIMEOUT 30
            )
        endforeach()

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_RUNNER_PYTHON_TESTS
                COMMAND ${Python3_EXECUTABLE} -m pytest -q
                        "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_graph_interprocess_benchmark.py"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_RUNNER_PYTHON_TESTS
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
            )

            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TRACE_CORRELATION
                COMMAND $<TARGET_FILE:runtime_graph_interprocess_benchmark>
                        --run-id trace-correlation-self-test
                        --samples 1
                        --warmup 0
                        --trace-correlation-self-test 1
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TRACE_CORRELATION
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
            )

            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_START_BARRIER
                COMMAND $<TARGET_FILE:runtime_graph_interprocess_benchmark>
                        --run-id start-barrier-self-test
                        --samples 1
                        --warmup 0
                        --start-barrier-self-test 1
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_START_BARRIER
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
            )

            # Prove two independently compiled process-group workers exchange exact POD samples.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 1000
                        --payload-validation all
                        --runtime-phase-trace
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to1_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Exercise compiler-owned cross-process Tensor pools for each supported storage kind.
            foreach(tensor_storage IN ITEMS pageable_host pinned_host device)
                string(TOUPPER "${tensor_storage}" tensor_storage_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload tensor
                            --tensor-storage "${tensor_storage}"
                            --tensor-bytes 1048576
                            --workload latency
                            --samples 4
                            --warmup 1
                            --queue-depth 1
                            --publisher-period-us 1000
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_tensor_${tensor_storage}_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                # Keep transport-only throughput distinct from the full per-sample producer write.
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload tensor
                            --tensor-storage "${tensor_storage}"
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 8
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_tensor_${tensor_storage}_descriptor_only_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                if(tensor_storage STREQUAL "pageable_host")
                    add_test(
                        NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_PAGEABLE_HOST_DESCRIPTOR_ONLY_LATENCY_SMOKE
                        COMMAND ${Python3_EXECUTABLE}
                                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                                "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                                --payload tensor
                                --tensor-storage pageable_host
                                --tensor-bytes 1048576
                                --bulk-work-profile descriptor_only
                                --workload latency
                                --samples 16
                                --warmup 8
                                --queue-depth 1
                                --publisher-period-us 1000
                                --payload-validation final
                                --session-timeout-ms 10000
                                --process-timeout-sec 30
                                --output-dir
                                "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_tensor_pageable_host_descriptor_only_latency_smoke"
                    )
                    set_tests_properties(
                        BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_PAGEABLE_HOST_DESCRIPTOR_ONLY_LATENCY_SMOKE
                        PROPERTIES
                            LABELS "benchmark;runtime;interprocess;tensor;storage"
                            SKIP_RETURN_CODE 77
                            TIMEOUT 120
                    )
                endif()
            endforeach()

            # Keep Image metadata-plus-Tensor costs separate from raw Tensor transport, record the
            # intentional owning-receive copy allocations, and prove that every supported schema
            # provider preserves its declared memory kind.
            foreach(image_storage IN ITEMS pageable_host pinned_host device)
                string(TOUPPER "${image_storage}" image_storage_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_LATENCY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload image
                            --tensor-storage "${image_storage}"
                            --tensor-bytes 1048576
                            --bulk-work-profile full_write
                            --workload latency
                            --samples 4
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 1000
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_image_${image_storage}_latency_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_LATENCY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload image
                            --tensor-storage "${image_storage}"
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 8
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_image_${image_storage}_descriptor_only_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                if(image_storage STREQUAL "pageable_host" OR image_storage STREQUAL "device")
                    add_test(
                        NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_LATENCY_SMOKE"
                        COMMAND ${Python3_EXECUTABLE}
                                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                                "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                                --payload image
                                --tensor-storage "${image_storage}"
                                --tensor-bytes 1048576
                                --bulk-work-profile descriptor_only
                                --workload latency
                                --samples 8
                                --warmup 8
                                --queue-depth 1
                                --publisher-period-us 1000
                                --payload-validation final
                                --session-timeout-ms 10000
                                --process-timeout-sec 30
                                --output-dir
                                "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_image_${image_storage}_descriptor_only_latency_smoke"
                    )
                    set_tests_properties(
                        "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_LATENCY_SMOKE"
                        PROPERTIES
                            LABELS "benchmark;runtime;interprocess;tensor;schema;storage"
                            SKIP_RETURN_CODE 77
                            TIMEOUT 120
                    )
                endif()
            endforeach()

            # A CUDA-written pinned-host loan carries a cross-process readiness fence whether it is
            # a raw Tensor or an Image field. Check its bytes end to end, then its saturated
            # handoff, where every slot returns only after the consumer passes the fence.
            foreach(pinned_payload IN ITEMS tensor image)
                string(TOUPPER "${pinned_payload}" pinned_payload_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${pinned_payload_upper}_PINNED_HOST_CUDA_WRITER_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload "${pinned_payload}"
                            --tensor-storage pinned_host
                            --pinned-host-writer cuda
                            --tensor-bytes 1048576
                            --bulk-work-profile full_write
                            --workload latency
                            --samples 4
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 1000
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${pinned_payload}_pinned_host_cuda_writer_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${pinned_payload_upper}_PINNED_HOST_CUDA_WRITER_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;cuda"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${pinned_payload_upper}_PINNED_HOST_CUDA_WRITER_DESCRIPTOR_ONLY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload "${pinned_payload}"
                            --tensor-storage pinned_host
                            --pinned-host-writer cuda
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 8
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${pinned_payload}_pinned_host_cuda_writer_descriptor_only_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${pinned_payload_upper}_PINNED_HOST_CUDA_WRITER_DESCRIPTOR_ONLY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;cuda"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
            endforeach()

            # Exercise both multi-Tensor authoring paths through real Operator callbacks. The
            # excluded final warmup sample proves the direct bundle and staged-value callbacks are
            # allocation-free when the sink uses the activation-scoped borrowed batch view.
            foreach(multi_tensor_payload IN ITEMS multi_tensor_loan multi_tensor_staging)
                string(TOUPPER "${multi_tensor_payload}" multi_tensor_payload_upper)
                foreach(multi_tensor_storage IN ITEMS pageable_host pinned_host device)
                    string(TOUPPER "${multi_tensor_storage}" multi_tensor_storage_upper)
                    set(multi_tensor_storage_test
                        "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_${multi_tensor_storage_upper}_SMOKE")
                    add_test(
                            NAME "${multi_tensor_storage_test}"
                            COMMAND ${Python3_EXECUTABLE}
                                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                                    "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                                    --payload "${multi_tensor_payload}"
                                    --tensor-storage "${multi_tensor_storage}"
                                    --tensor-bytes 4096
                                    --bulk-work-profile descriptor_only
                                    --workload latency
                                    --samples 4
                                    --warmup 8
                                    --queue-depth 1
                                    --publisher-period-us 1000
                                    --payload-validation final
                                    --session-timeout-ms 10000
                                    --process-timeout-sec 30
                                    --output-dir
                                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${multi_tensor_payload}_${multi_tensor_storage}_smoke"
                    )
                    set_tests_properties(
                        "${multi_tensor_storage_test}"
                        PROPERTIES
                            LABELS "benchmark;runtime;interprocess;tensor;schema;multi_tensor;storage"
                            SKIP_RETURN_CODE 77
                            TIMEOUT 120
                    )
                endforeach()
                # Saturation exhausts the multi-field reservation metadata before the next
                # envelope loan; keep this case to prevent reintroducing that progress cycle.
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SATURATED_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload "${multi_tensor_payload}"
                            --tensor-storage pageable_host
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 16
                            --warmup 16
                            --queue-depth 2
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${multi_tensor_payload}_saturated_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SATURATED_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;schema;multi_tensor;saturated"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
            endforeach()

            # Exercise owning generated-schema encode/materialize across one process boundary and
            # require its fixed-field emit/receive callbacks to remain allocation-free.
            foreach(schema_workload IN ITEMS latency saturated)
                if(schema_workload STREQUAL "latency")
                    set(schema_publisher_period_us 1000)
                else()
                    set(schema_publisher_period_us 0)
                endif()
                string(TOUPPER "${schema_workload}" schema_workload_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_SCHEMA_TIME_${schema_workload_upper}_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload schema
                            --workload "${schema_workload}"
                            --samples 8
                            --warmup 2
                            --queue-depth 1
                            --publisher-period-us "${schema_publisher_period_us}"
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_schema_time_${schema_workload}_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_SCHEMA_TIME_${schema_workload_upper}_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
            endforeach()

            # Prove one source independently reaches two process-isolated consumers without loss.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --fanout 2
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 5000
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to2_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess;fanout"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Prove sustained coalesced fanout reports one logical publication and two exact,
            # ordered deliveries without partial visibility or duplicate retry.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SATURATED_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --fanout 2
                        --workload saturated
                        --samples 32
                        --warmup 8
                        --queue-depth 2
                        --publisher-period-us 0
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to2_saturated_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SATURATED_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess;fanout"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Prove one exact stream crosses source, relay, and sink process groups in order.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_CHAIN_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --topology chain
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 1000
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_chain_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_CHAIN_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess;chain"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Exercise the low-latency ceiling with both Operator lanes continuously active.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_BUSY_POLL_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 1000
                        --payload-validation all
                        --busy-poll
                        --runtime-phase-trace
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to1_busy_poll_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_BUSY_POLL_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )
        endif()

        # Run the complete native/runner evidence path with one shared Operator lane.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_SHARED_LANE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload notification_latency
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_shared_lane_smoke"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_SHARED_LANE
            PROPERTIES
                LABELS "benchmark;runtime"
                TIMEOUT 45
        )

        # Validate persisted role/lane and completion evidence for independently placed sinks.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_PER_SINK_LANES
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 4
                    --operator-lanes per_sink
                    --samples 8
                    --warmup 2
                    --queue-depth 2
                    --notification-capacity 2
                    --payload-validation all
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_per_sink_lanes_smoke"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_PER_SINK_LANES
            PROPERTIES
                LABELS "benchmark;runtime;fanout;placement"
                TIMEOUT 45
        )
    endif()

    string(CONCAT RUNTIME_CHANNEL_SMOKE_FILTER
        "BM_ChannelActivationCycleByGraphOwners/2"
        "|BM_ChannelReadinessBurst/1/1"
        "|BM_ChannelReadinessPollViaBusyLane/real_time"
        "|BM_ChannelDirectRetainedReadinessPoll/real_time"
        "|BM_ChannelDirectRetainedReadinessPollViaBusyLane/real_time"
        "|BM_ChannelDirectRetainedFusedPollConsumeCeiling/real_time"
        "|BM_ChannelDirectRetainedTwoDispatchPollConsume/real_time"
        "|BM_ChannelDirectRetainedResumablePollConsumeProbe/real_time"
        "|BM_ChannelDirectRetainedActivationCeiling/ordinary_sleeping/0/1/manual_time"
        "|BM_ChannelDirectRetainedActivationCeiling/receipt_sleeping/0/1/manual_time"
        "|BM_ChannelDirectRetainedActivationCeiling/composite_sleeping/0/1/manual_time"
        "|BM_ChannelLastWitnessRetirementByGraphInputs/1"
        "|BM_ChannelTypedPODCycle/real_time"
        "|BM_ChannelTypedPODDirectRetainedCycle/real_time"
        "|BM_ChannelTypedPOD64DirectRetainedCycle/real_time"
        "|BM_ChannelTypedPODPublishCut/manual_time"
        "|BM_ChannelTypedPODPublishRetire/manual_time"
        "|BM_ChannelTypedPODEmit/manual_time"
        "|BM_ChannelTypedPODDirectRetainedEmit/manual_time"
        "|BM_ChannelTypedPODFanoutEmit/4/manual_time"
        "|BM_ChannelTypedPODFanoutPublishCut/4/manual_time"
        "|BM_ChannelTypedPODFanoutConsumeRetire/4/manual_time"
        "|BM_ChannelTypedPODFanoutFusedPollConsumeCeiling/4/real_time"
        "|BM_ChannelTypedPODFanoutTwoDispatchConsumeRetire/4/real_time"
        "|BM_OutputTypedPODNoopBinding"
        "|BM_ChannelInlineTypedPODPublish/manual_time"
        "|BM_ChannelInlineTypedPODPublishNoEvent/manual_time"
        "|BM_ChannelInlineTypedPODPublishActivePolling/manual_time"
        "|BM_ChannelInlineTypedPODPublishDeferredReceive/manual_time"
        "|BM_ChannelInlineTypedPODPublishAsyncWaiter/manual_time"
        "|BM_ChannelTypedPODSourceCut/manual_time"
        "|BM_ChannelTypedPODConsumeRetire/manual_time"
    )

    add_test(
        NAME BENCHMARK_RUNTIME_DISPATCHER
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_dispatcher_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_DISPATCHER
        PROPERTIES
            LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_DISPATCHER_MANAGED_SIGNALS
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_dispatcher_benchmark>"
                "--benchmark_filter=^BM_RunSession(OneDirectActivation|SteadyDirectActivations)(PersistentManagedLifecycle|TransientManagedDefault)?(/real_time)?$"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_DISPATCHER_MANAGED_SIGNALS
        PROPERTIES
            LABELS "benchmark;runtime;signals"
    )

    add_test(
        NAME BENCHMARK_RUNTIME_CHANNEL
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_channel_benchmark>"
                "--benchmark_filter=${RUNTIME_CHANNEL_SMOKE_FILTER}"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_CHANNEL
        PROPERTIES
            LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_DESCRIPTOR_GRAPH
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_descriptor_graph_benchmark>"
                --benchmark_repetitions=1
    )
    set_tests_properties(BENCHMARK_RUNTIME_DESCRIPTOR_GRAPH PROPERTIES LABELS benchmark)

    add_test(
        NAME BENCHMARK_RUNTIME_DESCRIPTOR_RECEIVE
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_descriptor_receive_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_DESCRIPTOR_RECEIVE
        PROPERTIES LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_PAYLOAD
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_payload_benchmark>"
                "--benchmark_filter=BM_CompiledCatalog(CustomPod|Time|Imu)$|BM_CompiledCatalogImuTransitions"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_PAYLOAD
        PROPERTIES
            LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_NATIVE_SCHEMA
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_native_schema_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_NATIVE_SCHEMA
        PROPERTIES LABELS "benchmark;schema" TIMEOUT 60
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH
        COMMAND runtime_graph_benchmark
                --workload latency
                --samples 16
                --warmup 4
                --publisher-period-us 1000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_REJECTS_AMBIGUOUS_FANOUT_SINK_CPU
        COMMAND ${CMAKE_COMMAND}
                -DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_ambiguous_fanout_sink_cpu_failure.cmake"
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_REJECTS_AMBIGUOUS_FANOUT_SINK_CPU
        PROPERTIES
            LABELS "benchmark;runtime;argument-validation"
            TIMEOUT 5
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SATURATED
        COMMAND runtime_graph_benchmark
                --workload saturated
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SATURATED
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    foreach(LOSSY_DEPTH IN ITEMS 1 2)
        if(LOSSY_DEPTH EQUAL 1)
            set(LOSSY_DEPTH_SUFFIX _DEPTH_ONE)
        else()
            set(LOSSY_DEPTH_SUFFIX)
        endif()
        foreach(LOSSY_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(LOSSY_DELIVERY_ARGUMENTS)
            if(LOSSY_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND LOSSY_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
            endif()
            set(LOSSY_DIAGNOSTIC_ARGUMENTS)
            if(LOSSY_DEPTH EQUAL 1 AND LOSSY_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND LOSSY_DIAGNOSTIC_ARGUMENTS --runtime-phase-counters)
            endif()
            set(
                LOSSY_TEST_NAME
                BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST${LOSSY_DEPTH_SUFFIX}_${LOSSY_DELIVERY}
            )
            add_test(
                NAME ${LOSSY_TEST_NAME}
                COMMAND runtime_graph_benchmark
                        --workload saturated
                        --scenario pod
                        --queue-policy drop_oldest
                        --queue-depth ${LOSSY_DEPTH}
                        --notification-capacity ${LOSSY_DEPTH}
                        --samples 16
                        --warmup 4
                        --payload-validation all
                        ${LOSSY_DELIVERY_ARGUMENTS}
                        ${LOSSY_DIAGNOSTIC_ARGUMENTS}
                        --timeout-ms 10000
            )
            set_tests_properties(
                ${LOSSY_TEST_NAME}
                PROPERTIES
                    LABELS "benchmark;runtime;lossy"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(DROP_OLDEST_SCENARIO IN ITEMS
        schema
        third_party_schema
        tensor
        image
        multi_tensor_loan
        multi_tensor_staging
    )
        string(TOUPPER "${DROP_OLDEST_SCENARIO}" DROP_OLDEST_SCENARIO_UPPER)
        set(DROP_OLDEST_PAYLOAD_ARGUMENTS)
        set(DROP_OLDEST_WARMUP 4)
        if(NOT DROP_OLDEST_SCENARIO STREQUAL "schema" AND
           NOT DROP_OLDEST_SCENARIO STREQUAL "third_party_schema")
            list(APPEND DROP_OLDEST_PAYLOAD_ARGUMENTS
                --tensor-storage pageable_host
                --tensor-bytes 64
            )
        endif()
        if(DROP_OLDEST_SCENARIO STREQUAL "multi_tensor_loan")
            # Loan warmup must cover each compiled field pool: 9 slots with direct delivery and
            # 12 behind the HoloMQ fallback, whose drop-oldest subscriber stages.
            set(DROP_OLDEST_WARMUP 12)
        endif()
        foreach(DROP_OLDEST_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(DROP_OLDEST_DELIVERY_ARGUMENTS)
            if(DROP_OLDEST_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND
                    DROP_OLDEST_DELIVERY_ARGUMENTS
                    --force-holomq-process-local-inline
                )
            endif()
            add_test(
                NAME
                    BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST_${DROP_OLDEST_SCENARIO_UPPER}_${DROP_OLDEST_DELIVERY}
                COMMAND runtime_graph_benchmark
                        --workload saturated
                        --scenario ${DROP_OLDEST_SCENARIO}
                        --queue-policy drop_oldest
                        --queue-depth 2
                        --notification-capacity 2
                        --samples 16
                        --warmup ${DROP_OLDEST_WARMUP}
                        --payload-validation all
                        ${DROP_OLDEST_PAYLOAD_ARGUMENTS}
                        ${DROP_OLDEST_DELIVERY_ARGUMENTS}
                        --timeout-ms 10000
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST_${DROP_OLDEST_SCENARIO_UPPER}_${DROP_OLDEST_DELIVERY}
                PROPERTIES
                    LABELS "benchmark;runtime;lossy;${DROP_OLDEST_SCENARIO}"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(SATURATED_CAPACITY IN ITEMS 1 4 8)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_SATURATED_CAPACITY_${SATURATED_CAPACITY}
            COMMAND runtime_graph_benchmark
                    --workload saturated
                    --samples 16
                    --warmup 4
                    --queue-depth ${SATURATED_CAPACITY}
                    --notification-capacity ${SATURATED_CAPACITY}
                    --payload-validation all
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SATURATED_CAPACITY_${SATURATED_CAPACITY}
            PROPERTIES
                LABELS "benchmark;runtime;capacity"
                TIMEOUT 15
        )
    endforeach()

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY
        COMMAND runtime_graph_benchmark
                --workload notification_latency
                --samples 4
                --warmup 1
                --publisher-period-us 2000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    # Exercise the existing public placement contract with both Operators on one serial lane.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY_SHARED_LANE
        COMMAND runtime_graph_benchmark
                --workload notification_latency
                --scenario pod
                --operator-lanes shared
                --samples 4
                --warmup 1
                --publisher-period-us 2000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY_SHARED_LANE
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_NOTIFICATION_LATENCY
        COMMAND runtime_graph_benchmark
                --workload notification_latency
                --scenario source_only
                --samples 4
                --warmup 1
                --publisher-period-us 2000
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_NOTIFICATION_LATENCY
        PROPERTIES
            LABELS "benchmark;runtime"
            PASS_REGULAR_EXPRESSION
                "\"callback_thread_evidence\":\\{\"enabled\":true,\"valid\":true"
            TIMEOUT 15
    )

    # Bound notification pressure by completed source activations without a Channel edge.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_SATURATED
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario source_only
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_SATURATED
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SIGNAL
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario signal
                --samples 16
                --warmup 4
                --publisher-period-us 1000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SIGNAL
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    foreach(CHAIN_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
        set(CHAIN_DELIVERY_ARGUMENTS)
        if(CHAIN_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
            list(APPEND CHAIN_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_LINEAR_CHAIN_${CHAIN_DELIVERY}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario sample_forward
                    --relay-count 4
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --publisher-period-us 1000
                    --payload-validation all
                    ${CHAIN_DELIVERY_ARGUMENTS}
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_LINEAR_CHAIN_${CHAIN_DELIVERY}
            PROPERTIES
                LABELS "benchmark;runtime"
                TIMEOUT 15
        )
    endforeach()

    # Exercise the manually authored source/relay/sink partition cover under bounded downstream
    # pressure. Relay publication must preserve the owning Sample while retrying transient
    # backpressure instead of terminating the RunSession.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_BACKPRESSURE
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario sample_forward
                --relay-count 4
                --samples 32
                --warmup 8
                --publisher-period-us 200
                --queue-depth 1
                --notification-capacity 1
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_BACKPRESSURE
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    # A failing downstream callback must terminate a relay that is retrying a retained Sample
    # instead of leaving the benchmark process spinning until CTest kills it.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_FAILURE_IS_BOUNDED
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario sample_forward
                --relay-count 1
                --samples 8
                --warmup 0
                --publisher-period-us 1
                --queue-depth 1
                --notification-capacity 1
                --payload-validation none
                --test-fail-sink-after-relay-pressure
                --timeout-ms 1000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_FAILURE_IS_BOUNDED
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 5
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION "error=22"
            FAIL_REGULAR_EXPRESSION "runtime Graph benchmark timeout"
    )

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_FAILURE_DIAGNOSTIC_ARTIFACT
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_runtime_graph_failure_diagnostics.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_FAILURE_DIAGNOSTIC_ARTIFACT
            PROPERTIES
                LABELS "benchmark;runtime;diagnostics"
                TIMEOUT 10
        )
    endif()

    # Complete each logical sample only after all four transport-backed sinks receive it.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 4
                --operator-lanes shared
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT
        PROPERTIES
            LABELS "benchmark;runtime;fanout"
            TIMEOUT 15
    )

    foreach(POD_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_WAKE_ARGUMENTS)
        if(POD_FANOUT_WAKE_MODE STREQUAL "SUPPRESSED")
            # compile() delivers a blocking fanout family directly (CHAN-167); keep HoloMQ here.
            list(APPEND POD_FANOUT_WAKE_ARGUMENTS --force-holomq-fanout)
        elseif(POD_FANOUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-subscriber shared-lane POD service with compiler-selected
        # eventless delivery and with the explicit retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_EVENT_WAKE_${POD_FANOUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies exact two-recipient delivery and the requested
        # service-wide event-wake state without changing active fanout geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_EVENT_WAKE_${POD_FANOUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;event-wake"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: retain the Channel pump on an otherwise pump-free shared-lane eventless POD
    # HoloMQ row without selecting the independent forced-event-wake control.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_CHANNEL_PUMP_CONTROL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --operator-lanes shared
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --queue-depth 1
                --payload-validation all
                --force-holomq-process-local-inline
                --force-process-local-channel-pump
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: the runner requires event_wake=false, HoloMQ delivery, the explicit pump
    # provenance, and one entered Channel-pump support-thread role.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_CHANNEL_PUMP_CONTROL
        PROPERTIES
            LABELS "benchmark;runtime;holomq;eventless;channel-pump"
            TIMEOUT 15
    )

    foreach(POD_MULTI_OUTPUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_MULTI_OUTPUT_WAKE_ARGUMENTS)
        if(POD_MULTI_OUTPUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_MULTI_OUTPUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-output/one-subscriber-per-output shared-lane POD graph
        # with compiler-selected eventless delivery and with the exact retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_EVENT_WAKE_${POD_MULTI_OUTPUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario pod
                    --fanout 2
                    --multi-output
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_MULTI_OUTPUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies two HoloMQ services, one delivery from each output,
        # and the requested service-wide event-wake state without changing graph geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_EVENT_WAKE_${POD_MULTI_OUTPUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;multi-output;holomq;event-wake"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_MULTI_INPUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_MULTI_INPUT_WAKE_ARGUMENTS)
        if(POD_MULTI_INPUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_MULTI_INPUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-output/one-two-input-sink shared-lane POD graph with
        # compiler-selected eventless delivery and with the exact retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_INPUT_EVENT_WAKE_${POD_MULTI_INPUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario pod
                    --fanout 2
                    --multi-input
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_MULTI_INPUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies two HoloMQ services, two paired input reads per sink
        # callback, and the requested event-wake state without changing graph geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_MULTI_INPUT_EVENT_WAKE_${POD_MULTI_INPUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;multi-input;holomq;event-wake"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: run the two-output topology under sustained notification pressure with equal
    # explicit capacity on both independently published services.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_SATURATED_UNIFORM_DEPTH
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload saturated
                --scenario pod
                --fanout 2
                --multi-output
                --operator-lanes shared
                --samples 16
                --warmup 4
                --queue-depth 2
                --sink-queue-depths 2,2
                --notification-capacity 2
                --payload-validation all
                --force-holomq-process-local-inline
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: paired publication remains complete under pressure when both service queues
    # have equivalent geometry.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_SATURATED_UNIFORM_DEPTH
        PROPERTIES
            LABELS "benchmark;runtime;multi-output;holomq;queue-depth"
            TIMEOUT 15
    )

    # Key setup: pass heterogeneous service depths directly to the native parser for a
    # non-atomic paired-output topology.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_MIXED_DEPTH_REJECTED
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 2
                --multi-output
                --operator-lanes shared
                --samples 16
                --warmup 4
                --queue-depth 4
                --sink-queue-depths 4,2
                --notification-capacity 4
                --payload-validation all
                --force-holomq-process-local-inline
                --timeout-ms 10000
    )
    # Key assertion: native validation rejects geometry that can partially commit the first
    # physical publication before the second service reports backpressure.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_MIXED_DEPTH_REJECTED
        PROPERTIES
            LABELS "benchmark;runtime;multi-output;holomq;queue-depth"
            WILL_FAIL TRUE
            TIMEOUT 15
    )

    foreach(POD_MIXED_POLICY_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_MIXED_POLICY_WAKE_ARGUMENTS)
        if(POD_MIXED_POLICY_WAKE_MODE STREQUAL "SUPPRESSED")
            # compile() delivers a blocking fanout family directly (CHAN-167); keep HoloMQ here.
            list(APPEND POD_MIXED_POLICY_WAKE_ARGUMENTS --force-holomq-fanout)
        elseif(POD_MIXED_POLICY_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_MIXED_POLICY_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run one output through separate capacity-one blocking and drop-oldest HoloMQ
        # services on a shared lane with eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_MIXED_POLICY_EVENT_WAKE_${POD_MIXED_POLICY_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --mixed-policy-fanout
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --queue-depth 1
                    --notification-capacity 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_MIXED_POLICY_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies exact per-leg policies and delivery counts while the
        # pair changes only compiler-selected event wake and pump ownership.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_MIXED_POLICY_EVENT_WAKE_${POD_MIXED_POLICY_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;mixed-policy;event-wake"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_FANOUT_DEEP_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_DEEP_WAKE_ARGUMENTS)
        if(POD_FANOUT_DEEP_WAKE_MODE STREQUAL "SUPPRESSED")
            # compile() delivers a blocking fanout family directly (CHAN-167); keep HoloMQ here.
            list(APPEND POD_FANOUT_DEEP_WAKE_ARGUMENTS --force-holomq-fanout)
        elseif(POD_FANOUT_DEEP_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_DEEP_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same depth-eight, two-subscriber shared-lane POD service with
        # compiler-selected eventless delivery and with the explicit retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DEPTH_8_EVENT_WAKE_${POD_FANOUT_DEEP_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --queue-depth 8
                    --notification-capacity 8
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_DEEP_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner proves exact two-recipient delivery and the requested
        # service-wide wake state without changing active fanout or queue capacity.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DEPTH_8_EVENT_WAKE_${POD_FANOUT_DEEP_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;event-wake;queue-depth"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_FANOUT_MIXED_DEPTH_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS)
        if(POD_FANOUT_MIXED_DEPTH_WAKE_MODE STREQUAL "SUPPRESSED")
            # compile() delivers a blocking fanout family directly (CHAN-167); keep HoloMQ here.
            list(APPEND POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS --force-holomq-fanout)
        elseif(POD_FANOUT_MIXED_DEPTH_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run one depth-four and one depth-two shared-lane POD subscriber in the same
        # compiler-owned service with eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_MIXED_DEPTH_EVENT_WAKE_${POD_FANOUT_MIXED_DEPTH_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --queue-depth 4
                    --sink-queue-depths 4,2
                    --notification-capacity 4
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies ordered 4/2 subscriber geometry, exact delivery and
        # final sequences, readiness, and the requested service-wide event-wake state.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_MIXED_DEPTH_EVENT_WAKE_${POD_FANOUT_MIXED_DEPTH_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;event-wake;queue-depth"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_FANOUT_DROP_OLDEST_WAKE_MODE IN ITEMS DIRECT SUPPRESSED CONTROL)
        set(POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS)
        set(POD_FANOUT_DROP_OLDEST_WAKE_LABELS benchmark runtime fanout drop-oldest)
        if(POD_FANOUT_DROP_OLDEST_WAKE_MODE STREQUAL "SUPPRESSED")
            list(APPEND POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS --force-holomq-fanout)
        elseif(POD_FANOUT_DROP_OLDEST_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        if(NOT POD_FANOUT_DROP_OLDEST_WAKE_MODE STREQUAL "DIRECT")
            list(APPEND POD_FANOUT_DROP_OLDEST_WAKE_LABELS holomq event-wake)
        endif()
        # Key setup: clock four-publication POD bursts into two depth-two drop-oldest subscribers
        # at the same 20 kpublication/s average rate with compile()'s direct fanout, eventless
        # HoloMQ, and retained-wake HoloMQ delivery.
        # Use 16,384 samples (4,096 four-publication bursts), so the measurement spans 4,095
        # inter-burst periods (819 ms). The clock coalesces ticks that the source wakes too late
        # for, so lost ticks reflect platform wake latency and are bounded only loosely. Run these
        # cases serially so other CTests do not add lost ticks.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DROP_OLDEST_EVENT_WAKE_${POD_FANOUT_DROP_OLDEST_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --queue-policy drop_oldest
                    --queue-depth 2
                    --offered-rate-hz 20000
                    --offered-burst-size 4
                    --samples 16384
                    --warmup 15
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner requires one complete accepted burst for every clock tick that
        # fired, at most a quarter of the ticks lost, nonzero exact replacement accounting on both
        # legs, final-sequence delivery, the selected process-local delivery, and the requested
        # service wake state.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DROP_OLDEST_EVENT_WAKE_${POD_FANOUT_DROP_OLDEST_WAKE_MODE}
            PROPERTIES
                LABELS "${POD_FANOUT_DROP_OLDEST_WAKE_LABELS}"
                RUN_SERIAL TRUE
                TIMEOUT 15
        )
    endforeach()

    # Fanout attribution records HoloMQ admission and ready-hint checkpoints for every recipient.
    # The runner requires every checkpoint of every measured sample.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_ATTRIBUTION_HOLOMQ
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --fanout 2
                --operator-lanes per_sink
                --queue-policy blocking
                --samples 16
                --warmup 2
                --publisher-period-us 1000
                --fanout-attribution
                --force-holomq-fanout
                --repetitions 1
                --session-timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_ATTRIBUTION_HOLOMQ
        PROPERTIES LABELS "benchmark;runtime;fanout;holomq;attribution" TIMEOUT 20)
    # A blocking family that compile() delivers by direct retention never reaches those
    # checkpoints, so the native parser rejects the row instead of reporting incomplete timing.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_ATTRIBUTION_DIRECT_REJECTED
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario pod
                --fanout 2
                --operator-lanes per_sink
                --queue-policy blocking
                --samples 8
                --warmup 2
                --fanout-attribution
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_ATTRIBUTION_DIRECT_REJECTED
        PROPERTIES
            LABELS "benchmark;runtime;fanout;attribution"
            PASS_REGULAR_EXPRESSION "fanout attribution observes HoloMQ delivery"
            TIMEOUT 15)

    # Prove fanout can materialize one independently accepted execution lane per subscriber.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_PER_SINK_LANES
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 4
                --operator-lanes per_sink
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_PER_SINK_LANES
        PROPERTIES
            LABELS "benchmark;runtime;fanout;placement"
            TIMEOUT 15
    )

    # The same fanout under the pump-delivery control, which compile() no longer selects.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_PER_SINK_LANES_PUMP_DELIVERY
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 4
                --operator-lanes per_sink
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --force-process-local-pump-delivery
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_PER_SINK_LANES_PUMP_DELIVERY
        PROPERTIES
            LABELS "benchmark;runtime;fanout;placement"
            TIMEOUT 15
    )

    foreach(SCHEMA_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
        set(SCHEMA_DELIVERY_ARGUMENTS)
        if(SCHEMA_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
            list(APPEND SCHEMA_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_SCHEMA_${SCHEMA_DELIVERY}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario schema
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    ${SCHEMA_DELIVERY_ARGUMENTS}
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SCHEMA_${SCHEMA_DELIVERY}
            PROPERTIES
                LABELS "benchmark;runtime;schema"
                TIMEOUT 15
        )
    endforeach()

    foreach(SCHEMA_FANOUT_WAKE_SCENARIO IN ITEMS schema third_party_schema)
        string(TOUPPER "${SCHEMA_FANOUT_WAKE_SCENARIO}" SCHEMA_FANOUT_WAKE_SCENARIO_UPPER)
        foreach(SCHEMA_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(SCHEMA_FANOUT_WAKE_ARGUMENTS)
            if(SCHEMA_FANOUT_WAKE_MODE STREQUAL "SUPPRESSED")
                # compile() delivers a blocking fanout family directly (CHAN-167); keep HoloMQ here.
                list(APPEND SCHEMA_FANOUT_WAKE_ARGUMENTS --force-holomq-fanout)
            elseif(SCHEMA_FANOUT_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND SCHEMA_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: run the same two-subscriber SDK/third-party schema service with
            # compiler-selected eventless delivery and with the retained-wake control.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_${SCHEMA_FANOUT_WAKE_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${SCHEMA_FANOUT_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario "${SCHEMA_FANOUT_WAKE_SCENARIO}"
                        --fanout 2
                        --operator-lanes shared
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${SCHEMA_FANOUT_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies exact two-recipient schema delivery and the
            # requested service-wide wake state without changing active fanout geometry.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_${SCHEMA_FANOUT_WAKE_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${SCHEMA_FANOUT_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;schema;holomq;event-wake"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(IMAGE_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
        set(IMAGE_DELIVERY_ARGUMENTS)
        if(IMAGE_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
            list(APPEND IMAGE_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
        endif()
        foreach(IMAGE_ACCESSOR IN ITEMS OWNING BORROWED)
            set(IMAGE_ACCESSOR_ARGUMENTS)
            set(IMAGE_TEST_NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_${IMAGE_DELIVERY})
            if(IMAGE_ACCESSOR STREQUAL "BORROWED")
                list(APPEND IMAGE_ACCESSOR_ARGUMENTS --borrow-image-input)
                string(APPEND IMAGE_TEST_NAME "_BORROWED")
            endif()
            # REQ: OPERATOR-056
            add_test(
                NAME ${IMAGE_TEST_NAME}
                COMMAND runtime_graph_benchmark
                        --workload latency
                        --scenario image
                        --tensor-storage pageable_host
                        --tensor-bytes 64
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --payload-validation all
                        ${IMAGE_DELIVERY_ARGUMENTS}
                        ${IMAGE_ACCESSOR_ARGUMENTS}
                        --timeout-ms 10000
            )
            set_tests_properties(${IMAGE_TEST_NAME}
                PROPERTIES
                    LABELS "benchmark;runtime;schema;tensor"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_BORROW_REJECTS_OTHER_SCENARIOS
        COMMAND ${CMAKE_COMMAND}
                "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_image_borrow_scenario_failure.cmake"
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_IMAGE_BORROW_REJECTS_OTHER_SCENARIOS
        PROPERTIES LABELS "benchmark;runtime;argument-validation" TIMEOUT 5
    )

    foreach(MULTI_TENSOR_SCENARIO IN ITEMS multi_tensor_loan multi_tensor_staging)
        string(TOUPPER "${MULTI_TENSOR_SCENARIO}" MULTI_TENSOR_SCENARIO_UPPER)
        set(MULTI_TENSOR_WARMUP 2)
        if(MULTI_TENSOR_SCENARIO STREQUAL "multi_tensor_loan")
            set(MULTI_TENSOR_WARMUP 8)
        endif()
        foreach(MULTI_TENSOR_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(MULTI_TENSOR_DELIVERY_ARGUMENTS)
            if(MULTI_TENSOR_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND MULTI_TENSOR_DELIVERY_ARGUMENTS
                    --force-holomq-process-local-inline)
            endif()
            foreach(MULTI_TENSOR_VALIDATION IN ITEMS ALL FINAL)
                string(TOLOWER "${MULTI_TENSOR_VALIDATION}" MULTI_TENSOR_VALIDATION_LOWER)
                set(MULTI_TENSOR_TEST_NAME
                    BENCHMARK_RUNTIME_GRAPH_${MULTI_TENSOR_SCENARIO_UPPER}_${MULTI_TENSOR_DELIVERY}_${MULTI_TENSOR_VALIDATION})
                add_test(
                    NAME ${MULTI_TENSOR_TEST_NAME}
                    COMMAND runtime_graph_benchmark
                            --workload latency
                            --scenario "${MULTI_TENSOR_SCENARIO}"
                            --tensor-storage pageable_host
                            --tensor-bytes 64
                            --samples 8
                            --warmup ${MULTI_TENSOR_WARMUP}
                            --publisher-period-us 1000
                            --payload-validation "${MULTI_TENSOR_VALIDATION_LOWER}"
                            ${MULTI_TENSOR_DELIVERY_ARGUMENTS}
                            --timeout-ms 10000
                )
                set_tests_properties(
                    ${MULTI_TENSOR_TEST_NAME}
                    PROPERTIES
                        LABELS "benchmark;runtime;schema;tensor"
                        TIMEOUT 15
                )
            endforeach()
        endforeach()
    endforeach()

    # Require every compiled field-pool backing to be initialized, even when an unloaded run
    # could complete by repeatedly reusing a smaller inventory.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_LOAN_FULL_POOL_WARMUP
        COMMAND ${CMAKE_COMMAND}
                "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_runtime_graph_loan_warmup.cmake"
    )
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_LOAN_FULL_POOL_WARMUP
        PROPERTIES LABELS "benchmark;runtime;schema;tensor" TIMEOUT 30
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_LOAN_UNDERSIZED_WARMUP_REJECTED
        COMMAND ${CMAKE_COMMAND}
                "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_undersized_loan_warmup.cmake"
    )
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_LOAN_UNDERSIZED_WARMUP_REJECTED
        PROPERTIES
            LABELS "benchmark;runtime;schema;tensor;negative"
            TIMEOUT 15
    )

    # Key setup: exercise each one-deep shared-lane direct-retained Tensor payload shape with
    # causal continuation enabled.
    # REQ: RUNTIME-041
    foreach(TENSOR_DIRECT_CONTINUATION_SCENARIO IN ITEMS
        tensor
        image
        multi_tensor_loan
        multi_tensor_staging
    )
        string(TOUPPER
            "${TENSOR_DIRECT_CONTINUATION_SCENARIO}"
            TENSOR_DIRECT_CONTINUATION_SCENARIO_UPPER
        )
        set(TENSOR_DIRECT_CONTINUATION_BULK_WORK_ARGUMENTS)
        set(TENSOR_DIRECT_CONTINUATION_WARMUP 2)
        if(TENSOR_DIRECT_CONTINUATION_SCENARIO STREQUAL "multi_tensor_loan")
            set(TENSOR_DIRECT_CONTINUATION_WARMUP 8)
        endif()
        if(NOT TENSOR_DIRECT_CONTINUATION_SCENARIO STREQUAL "tensor")
            list(APPEND
                TENSOR_DIRECT_CONTINUATION_BULK_WORK_ARGUMENTS
                --bulk-work-profile descriptor_only
            )
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${TENSOR_DIRECT_CONTINUATION_SCENARIO_UPPER}_CONTINUATION_ENABLED
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${TENSOR_DIRECT_CONTINUATION_SCENARIO}"
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup ${TENSOR_DIRECT_CONTINUATION_WARMUP}
                    --publisher-period-us 1000
                    --payload-validation all
                    ${TENSOR_DIRECT_CONTINUATION_BULK_WORK_ARGUMENTS}
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner accepts continuation evidence while validating all Runtime
        # phase-counter invariants for Tensor-backed delivery.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${TENSOR_DIRECT_CONTINUATION_SCENARIO_UPPER}_CONTINUATION_ENABLED
            PROPERTIES
                LABELS "benchmark;runtime;tensor;causal-continuation"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: exercise each certified non-Tensor 1:1 HoloMQ path on a shared lane.
    foreach(HOLOMQ_ORDINARY_DISPATCH_SCENARIO IN ITEMS pod schema third_party_schema)
        string(TOUPPER
            "${HOLOMQ_ORDINARY_DISPATCH_SCENARIO}"
            HOLOMQ_ORDINARY_DISPATCH_SCENARIO_UPPER
        )
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${HOLOMQ_ORDINARY_DISPATCH_SCENARIO_UPPER}_HOLOMQ_NO_CONTINUATION_RECEIPT
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${HOLOMQ_ORDINARY_DISPATCH_SCENARIO}"
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires ordinary dispatch with no continuation receipts.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${HOLOMQ_ORDINARY_DISPATCH_SCENARIO_UPPER}_HOLOMQ_NO_CONTINUATION_RECEIPT
            PROPERTIES
                LABELS "benchmark;runtime;holomq;ordinary-dispatch"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: keep the forced-HoloMQ POD producer and consumer on separate partitions.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_CONTROL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --operator-lanes separate
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --force-holomq-process-local-inline
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: the runner requires event wake and ordinary dispatch for this control.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_CONTROL
        PROPERTIES
            LABELS "benchmark;runtime;holomq;event-wake"
            TIMEOUT 15
    )

    # Key setup: retain event wake on the one-deep shared-lane HoloMQ POD path.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_SAME_LANE_CONTROL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --operator-lanes shared
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --force-holomq-process-local-inline
                --force-holomq-event-wake
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: the runner requires ordinary dispatch and an active event-wake service.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_SAME_LANE_CONTROL
        PROPERTIES
            LABELS "benchmark;runtime;holomq;ordinary-dispatch;event-wake"
            TIMEOUT 15
    )

    foreach(TENSOR_WAKE_SCENARIO IN ITEMS
        tensor
        image
        multi_tensor_loan
        multi_tensor_staging
    )
        string(TOUPPER "${TENSOR_WAKE_SCENARIO}" TENSOR_WAKE_SCENARIO_UPPER)
        set(TENSOR_WAKE_BULK_WORK_ARGUMENTS)
        set(TENSOR_WAKE_WARMUP 2)
        if(TENSOR_WAKE_SCENARIO STREQUAL "multi_tensor_loan")
            set(TENSOR_WAKE_WARMUP 8)
        endif()
        if(NOT TENSOR_WAKE_SCENARIO STREQUAL "tensor")
            list(APPEND TENSOR_WAKE_BULK_WORK_ARGUMENTS --bulk-work-profile descriptor_only)
        endif()

        # Key setup: force an exact shared-lane Tensor payload edge through one-deep HoloMQ.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_SUPPRESSED
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${TENSOR_WAKE_SCENARIO}"
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup ${TENSOR_WAKE_WARMUP}
                    --publisher-period-us 1000
                    --payload-validation all
                    ${TENSOR_WAKE_BULK_WORK_ARGUMENTS}
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires compiler-selected event-wake suppression.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_SUPPRESSED
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;event-wake"
                TIMEOUT 15
        )

        # Key setup: select the private forced-wake compiler seam for the same Tensor graph.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_CONTROL
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${TENSOR_WAKE_SCENARIO}"
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup ${TENSOR_WAKE_WARMUP}
                    --publisher-period-us 1000
                    --payload-validation all
                    ${TENSOR_WAKE_BULK_WORK_ARGUMENTS}
                    --force-holomq-process-local-inline
                    --force-holomq-event-wake
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires the active event-wake service on the control row.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_CONTROL
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;event-wake"
                TIMEOUT 15
        )
        if(TENSOR_WAKE_SCENARIO STREQUAL "tensor")
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_TENSOR_HOLOMQ_EVENT_WAKE_SUPPRESSED
                BENCHMARK_RUNTIME_GRAPH_TENSOR_HOLOMQ_EVENT_WAKE_CONTROL
                PROPERTIES
                    SKIP_RETURN_CODE 77
            )
        endif()
    endforeach()

    foreach(RAW_TENSOR_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(RAW_TENSOR_FANOUT_WAKE_ARGUMENTS)
        if(RAW_TENSOR_FANOUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND RAW_TENSOR_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-recipient shared-lane raw Tensor service with
        # compiler-selected eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_EVENT_WAKE_${RAW_TENSOR_FANOUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario tensor
                    --fanout 2
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${RAW_TENSOR_FANOUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies both Tensor recipients, pool/release geometry, and
        # the requested service-wide wake state without changing active fanout geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_EVENT_WAKE_${RAW_TENSOR_FANOUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;tensor;event-wake"
                SKIP_RETURN_CODE 77
                TIMEOUT 15
        )
    endforeach()

    foreach(RAW_TENSOR_FANOUT_WIDTH IN ITEMS 4 8)
        foreach(RAW_TENSOR_WIDE_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(RAW_TENSOR_WIDE_FANOUT_WAKE_ARGUMENTS)
            if(RAW_TENSOR_WIDE_FANOUT_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND RAW_TENSOR_WIDE_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: scale the same shared-lane raw Tensor service to four and eight active
            # recipients with compiler-selected eventless and retained-wake delivery.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_WIDTH}_EVENT_WAKE_${RAW_TENSOR_WIDE_FANOUT_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario tensor
                        --fanout "${RAW_TENSOR_FANOUT_WIDTH}"
                        --operator-lanes shared
                        --tensor-storage pageable_host
                        --tensor-bytes 64
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${RAW_TENSOR_WIDE_FANOUT_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies every recipient, pool/release geometry, pointer
            # identity, final payload, and the requested service-wide wake state.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_WIDTH}_EVENT_WAKE_${RAW_TENSOR_WIDE_FANOUT_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;event-wake"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(RAW_TENSOR_FANOUT_STORAGE IN ITEMS pinned_host device)
        string(TOUPPER "${RAW_TENSOR_FANOUT_STORAGE}" RAW_TENSOR_FANOUT_STORAGE_UPPER)
        foreach(RAW_TENSOR_STORAGE_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(RAW_TENSOR_STORAGE_WAKE_ARGUMENTS)
            if(RAW_TENSOR_STORAGE_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND RAW_TENSOR_STORAGE_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: run the same two-recipient shared-lane raw Tensor service with pinned-host
            # and device pool backing under eventless and retained-wake delivery.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_STORAGE_UPPER}_EVENT_WAKE_${RAW_TENSOR_STORAGE_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario tensor
                        --fanout 2
                        --operator-lanes shared
                        --tensor-storage "${RAW_TENSOR_FANOUT_STORAGE}"
                        --tensor-bytes 64
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${RAW_TENSOR_STORAGE_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies the requested storage, both recipients, pool and
            # release geometry, pointer identity, final payload, and service-wide wake state.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_STORAGE_UPPER}_EVENT_WAKE_${RAW_TENSOR_STORAGE_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;storage;event-wake"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    # Key setup: sustain 1 MiB asynchronous device writes long enough to recycle the six-slot
    # raw Tensor pool while two shared-lane recipients retain each publication.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_DEVICE_POOL_RETRY
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload saturated
                --scenario tensor
                --fanout 2
                --operator-lanes shared
                --tensor-storage device
                --tensor-bytes 1048576
                --samples 2048
                --warmup 100
                --queue-depth 1
                --payload-validation final
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: transient device-pool exhaustion is retried until every publication reaches
    # both recipients with pointer identity and final payload validation intact.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_DEVICE_POOL_RETRY
        PROPERTIES
            LABELS "benchmark;runtime;fanout;tensor;storage;pool-retry"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )

    # Key setup: inject one allocation-boundary kResourceExhausted result that the
    # notification-driven raw-Tensor source consumes inside its first compute activation.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_IN_ACTIVATION_POOL_RETRY
        COMMAND $<TARGET_FILE:runtime_graph_benchmark>
                --workload saturated
                --scenario tensor
                --operator-lanes shared
                --samples 4
                --warmup 1
                --queue-depth 1
                --payload-validation all
                --runtime-phase-counters
                --test-source-tensor-pool-in-activation-retries 1
                --timeout-ms 10000
    )
    # Key assertion: the run completes six callbacks and reports the pool retry separately while
    # native phase validation and JSON retain zero Channel emit backpressure.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_IN_ACTIVATION_POOL_RETRY
        PROPERTIES
            LABELS "benchmark;runtime;tensor;pool-retry;phase-counters"
            PASS_REGULAR_EXPRESSION
                "\"source_callbacks\":6.*\"backpressured_emits\":0.*\"source_tensor_pool_in_activation_retries\":1"
            SKIP_RETURN_CODE 77
            TIMEOUT 15
    )

    # Key setup: inject one allocation-boundary kResourceExhausted result that the
    # notification-driven multi-Tensor source consumes inside its first compute activation.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_IN_ACTIVATION_POOL_RETRY
        COMMAND $<TARGET_FILE:runtime_graph_benchmark>
                --workload saturated
                --scenario multi_tensor
                --operator-lanes shared
                --samples 4
                --warmup 1
                --queue-depth 1
                --payload-validation all
                --runtime-phase-counters
                --test-source-tensor-pool-in-activation-retries 1
                --timeout-ms 10000
    )
    # Key assertion: the run completes six callbacks and reports the pool retry separately while
    # native phase validation and JSON retain zero Channel emit backpressure.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_IN_ACTIVATION_POOL_RETRY
        PROPERTIES
            LABELS "benchmark;runtime;multi_tensor;pool-retry;phase-counters"
            PASS_REGULAR_EXPRESSION
                "\"source_callbacks\":6.*\"backpressured_emits\":0.*\"source_tensor_pool_in_activation_retries\":1"
            SKIP_RETURN_CODE 77
            TIMEOUT 15
    )

    foreach(IMAGE_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(IMAGE_FANOUT_WAKE_ARGUMENTS)
        if(IMAGE_FANOUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND IMAGE_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-recipient shared-lane Image descriptor-sidecar service with
        # compiler-selected eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_FANOUT_EVENT_WAKE_${IMAGE_FANOUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario image
                    --fanout 2
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --bulk-work-profile descriptor_only
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${IMAGE_FANOUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies both Image envelopes, shared sidecar geometry, and
        # the requested service-wide wake state without changing active fanout geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_IMAGE_FANOUT_EVENT_WAKE_${IMAGE_FANOUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;tensor;schema;event-wake"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: two Image sinks on their own lanes validate every payload at once, each into its
    # own buffer.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_FANOUT_PER_SINK_LANES_VALIDATE_ALL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario image
                --fanout 2
                --operator-lanes per_sink
                --tensor-storage pageable_host
                --tensor-bytes 4096
                --bulk-work-profile descriptor_only
                --samples 32
                --warmup 2
                --publisher-period-us 1000
                --queue-depth 1
                --payload-validation all
                --repetitions 1
                --session-timeout-ms 10000
                --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_image_per_sink_lanes_validate_all"
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_IMAGE_FANOUT_PER_SINK_LANES_VALIDATE_ALL
        PROPERTIES LABELS "benchmark;runtime;fanout;holomq;tensor;schema" TIMEOUT 20)

    foreach(MULTI_TENSOR_FANOUT_SCENARIO IN ITEMS multi_tensor_loan multi_tensor_staging)
        string(TOUPPER
            "${MULTI_TENSOR_FANOUT_SCENARIO}"
            MULTI_TENSOR_FANOUT_SCENARIO_UPPER
        )
        set(MULTI_TENSOR_FANOUT_WARMUP 2)
        if(MULTI_TENSOR_FANOUT_SCENARIO STREQUAL "multi_tensor_loan")
            # Loan warmup must cover each compiled 10-slot field pool.
            set(MULTI_TENSOR_FANOUT_WARMUP 10)
        endif()
        foreach(MULTI_TENSOR_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(MULTI_TENSOR_FANOUT_WAKE_ARGUMENTS)
            if(MULTI_TENSOR_FANOUT_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND MULTI_TENSOR_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: run the same two-recipient shared-lane multi-Tensor descriptor service
            # with compiler-selected eventless delivery and with the retained-wake control.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_${MULTI_TENSOR_FANOUT_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${MULTI_TENSOR_FANOUT_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario "${MULTI_TENSOR_FANOUT_SCENARIO}"
                        --fanout 2
                        --operator-lanes shared
                        --tensor-storage pageable_host
                        --tensor-bytes 64
                        --bulk-work-profile descriptor_only
                        --samples 8
                        --warmup ${MULTI_TENSOR_FANOUT_WARMUP}
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${MULTI_TENSOR_FANOUT_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies both descriptor fields at both recipients,
            # pool/release geometry, and the requested service-wide wake state.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_${MULTI_TENSOR_FANOUT_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${MULTI_TENSOR_FANOUT_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;schema;event-wake"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(TENSOR_STORAGE IN ITEMS pageable_host pinned_host device)
        string(TOUPPER "${TENSOR_STORAGE}" TENSOR_STORAGE_UPPER)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_TENSOR_${TENSOR_STORAGE_UPPER}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario tensor
                    --tensor-storage "${TENSOR_STORAGE}"
                    --tensor-bytes 64
                    --samples 16
                    --warmup 4
                    --publisher-period-us 1000
                    --payload-validation final
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_TENSOR_${TENSOR_STORAGE_UPPER}
            PROPERTIES
                LABELS "benchmark;runtime;tensor"
                SKIP_RETURN_CODE 77
                TIMEOUT 15
        )
    endforeach()

    # Prove that eventless HoloMQ descriptor workloads retain identical no-write publication
    # geometry across storage arms. Image and staging reuse immutable stream-aware external
    # backings so device rows exercise Runtime's ordinary Tensor lifetime contract.
    foreach(DESCRIPTOR_STORAGE IN ITEMS pageable_host pinned_host device)
        string(TOUPPER "${DESCRIPTOR_STORAGE}" DESCRIPTOR_STORAGE_UPPER)
        add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario image
                        --fanout 2
                        --operator-lanes shared
                        --tensor-storage "${DESCRIPTOR_STORAGE}"
                        --tensor-bytes 4096
                        --bulk-work-profile descriptor_only
                        --samples 8
                        --warmup 8
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation final
                        --repetitions 1
                        --session-timeout-ms 10000
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_IMAGE_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;schema;storage"
                SKIP_RETURN_CODE 77
                TIMEOUT 30
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario multi_tensor
                    --operator-lanes shared
                    --tensor-storage "${DESCRIPTOR_STORAGE}"
                    --tensor-bytes 4096
                    --bulk-work-profile descriptor_only
                    --samples 8
                    --warmup 8
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation final
                    --force-holomq-process-local-inline
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;schema;multi_tensor;storage"
                SKIP_RETURN_CODE 77
                TIMEOUT 30
        )

        foreach(DESCRIPTOR_MODE IN ITEMS multi_tensor_loan multi_tensor_staging)
            string(TOUPPER "${DESCRIPTOR_MODE}" DESCRIPTOR_MODE_UPPER)
            set(DESCRIPTOR_MODE_DIAGNOSTIC_ARGUMENTS)
            set(DESCRIPTOR_MODE_WARMUP 8)
            if(DESCRIPTOR_MODE STREQUAL "multi_tensor_loan")
                # Preserve controller, lane, and channel progress if a runtime-owned field pool
                # fails to reach the first source callback before the session deadline.
                list(APPEND DESCRIPTOR_MODE_DIAGNOSTIC_ARGUMENTS --runtime-phase-counters)
                # Loan warmup must cover each compiled 10-slot field pool.
                set(DESCRIPTOR_MODE_WARMUP 10)
            endif()
            add_test(
                    NAME BENCHMARK_RUNTIME_GRAPH_${DESCRIPTOR_MODE_UPPER}_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_benchmark>"
                            --workload latency
                            --scenario "${DESCRIPTOR_MODE}"
                            --fanout 2
                            --operator-lanes shared
                            --tensor-storage "${DESCRIPTOR_STORAGE}"
                            --tensor-bytes 4096
                            --bulk-work-profile descriptor_only
                            --samples 8
                            --warmup ${DESCRIPTOR_MODE_WARMUP}
                            --publisher-period-us 1000
                            --queue-depth 1
                            --payload-validation final
                            ${DESCRIPTOR_MODE_DIAGNOSTIC_ARGUMENTS}
                            --repetitions 1
                            --session-timeout-ms 10000
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_${DESCRIPTOR_MODE_UPPER}_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;schema;multi_tensor;storage"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 30
            )
        endforeach()
    endforeach()

    # Prove the docs-oriented raw Tensor profile initializes every compiler-owned pool slot before
    # measurement and performs descriptor-only publications on the hot path.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_TENSOR_PINNED_HOST_DESCRIPTOR_ONLY
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload saturated
                --scenario tensor
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --bulk-work-profile descriptor_only
                --samples 16
                --warmup 8
                --queue-depth 2
                --notification-capacity 2
                --payload-validation final
                --session-timeout-ms 10000
                --process-timeout-sec 15
                --repetitions 1
                --output-dir
                "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_tensor_pinned_host_descriptor_only"
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_TENSOR_PINNED_HOST_DESCRIPTOR_ONLY
        PROPERTIES
            LABELS "benchmark;runtime;tensor"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )

    # Exercise both paced latency and saturated throughput for the generated native type with two
    # independently pooled Tensor fields used by the Core Runtime docs table.
    foreach(MULTI_TENSOR_WORKLOAD IN ITEMS latency saturated)
        string(TOUPPER "${MULTI_TENSOR_WORKLOAD}" MULTI_TENSOR_WORKLOAD_UPPER)
        if(MULTI_TENSOR_WORKLOAD STREQUAL "latency")
            set(MULTI_TENSOR_QUEUE_ARGUMENTS --queue-depth 1 --publisher-period-us 1000)
        else()
            set(MULTI_TENSOR_QUEUE_ARGUMENTS --queue-depth 2 --notification-capacity 2)
        endif()
        add_test(
            NAME "BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${MULTI_TENSOR_WORKLOAD_UPPER}"
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload "${MULTI_TENSOR_WORKLOAD}"
                    --scenario multi_tensor
                    --tensor-storage pageable_host
                    --tensor-bytes 4096
                    --bulk-work-profile descriptor_only
                    --samples 8
                    --warmup 8
                    ${MULTI_TENSOR_QUEUE_ARGUMENTS}
                    --payload-validation final
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 1
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_multi_tensor_${MULTI_TENSOR_WORKLOAD}"
        )
        set_tests_properties(
            "BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${MULTI_TENSOR_WORKLOAD_UPPER}"
            PROPERTIES
                LABELS "benchmark;runtime;tensor;schema;multi_tensor"
                TIMEOUT 30
        )
    endforeach()

    # Preserve one transport-backed Tensor control beside the default direct-retained rows.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_TENSOR_PAGEABLE_HOST_HOLOMQ_FALLBACK
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario tensor
                --tensor-storage pageable_host
                --tensor-bytes 64
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --force-holomq-process-local-inline
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_TENSOR_PAGEABLE_HOST_HOLOMQ_FALLBACK
        PROPERTIES
            LABELS "benchmark;runtime;tensor"
            TIMEOUT 15
    )
endif()
