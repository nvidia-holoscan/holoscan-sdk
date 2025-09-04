# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ##############################################################################
# # Common properties
# ##############################################################################
# TODO (grelee): Removed -Wold-style-cast due to it raising errors for yaml-cpp code in .cache.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror$<SEMICOLON>-Wall$<SEMICOLON>-Wextra>)

function(add_holoscan_library target)
    add_library(${target} SHARED ${ARGN})
    add_library(${HOLOSCAN_PACKAGE_NAME}::${target} ALIAS ${target})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${HOLOSCAN_PACKAGE_NAME}_${target}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
    )
    target_compile_features(${target}
        PUBLIC cxx_std_17
    )
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${HOLOSCAN_TOP}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    target_link_libraries(${target} PRIVATE holoscan_security_flags)
endfunction()

function(add_holoscan_operator operator)
    set(target op_${operator})
    add_library(${target} SHARED ${ARGN})
    add_library(${HOLOSCAN_PACKAGE_NAME}::ops::${operator} ALIAS ${target})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${HOLOSCAN_PACKAGE_NAME}_${target}
        EXPORT_NAME ops::${operator}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
    )
    target_link_libraries(${target}
        PUBLIC ${HOLOSCAN_PACKAGE_NAME}::core
    )
    target_link_libraries(${target} PRIVATE holoscan_security_flags)
endfunction()

function(add_holoscan_data_logger data_logger)
    set(target data_loggers_${data_logger})
    add_library(${target} SHARED ${ARGN})
    add_library(${HOLOSCAN_PACKAGE_NAME}::data_loggers::${data_logger} ALIAS ${target})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME ${HOLOSCAN_PACKAGE_NAME}_${target}
        EXPORT_NAME data_loggers::${data_logger}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
    )
    target_link_libraries(${target}
        PUBLIC ${HOLOSCAN_PACKAGE_NAME}::core
    )
    target_link_libraries(${target} PRIVATE holoscan_security_flags)
endfunction()

# ##############################################################################
# # Add library holoscan::spdlog_logger
# ##############################################################################
add_holoscan_library(spdlog_logger common/logger/spdlog_logger.cpp)
target_link_libraries(spdlog_logger
    PUBLIC
        fmt::fmt-header-only
        GXF::logger
    PRIVATE
        spdlog::spdlog_header_only
)
target_compile_definitions(spdlog_logger
    INTERFACE
        FMT_HEADER_ONLY=1
)

# ##############################################################################
# # Add library holoscan::logger
# ##############################################################################
add_holoscan_library(logger logger/logger.cpp)
target_link_libraries(logger
    PUBLIC
        fmt::fmt-header-only
        spdlog_logger
)
target_compile_definitions(logger
    INTERFACE
        FMT_HEADER_ONLY=1
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(logger
        PRIVATE HOLOSCAN_LOG_ACTIVE_LEVEL=1
    )
endif()

# ##############################################################################
# # Add library holoscan::matx
# ##############################################################################
# Create an INTERFACE library to wrap the matx dependency.
# This avoids a name collision with the 'matx' target created by rapids-cmake (CPM).
add_library(holoscan_matx INTERFACE)
add_library(holoscan::matx ALIAS holoscan_matx)

target_link_libraries(holoscan_matx
    INTERFACE
        matx::matx
)


# ##############################################################################
# # Add library holoscan::profiler
# ##############################################################################
add_holoscan_library(profiler
    profiler/profiler.cpp)
target_link_libraries(profiler
    PUBLIC
        nvtx3-cpp
)
target_include_directories(profiler
    PUBLIC
      $<BUILD_INTERFACE:${nvtx3_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty>
)

# ##############################################################################
# # Add library holoscan::pose_tree
# ##############################################################################
add_holoscan_library(pose_tree
    pose_tree/pose_tree_history.cpp
    pose_tree/pose_tree.cpp
    pose_tree/pose_tree_manager.cpp
    pose_tree/pose_tree_ucx_client.cpp
    pose_tree/pose_tree_ucx_server.cpp
)

# Note: holoscan::pose_tree depends on holoscan::core and holoscan::eigen3,
# but we can separate the core pose tree module (which doesn't depend on the Holoscan SDK)
# into `holoscan::pose_tree_core` later if needed.
target_link_libraries(pose_tree
    PUBLIC
        holoscan::core
        holoscan::eigen3
)

# ##############################################################################
# # Add library: holoscan::core
# ##############################################################################

set(CORE_PROTO_FILES
    core/distributed/proto/app_driver.proto
    core/distributed/proto/app_worker.proto
    core/distributed/proto/connection_item.proto
    core/distributed/proto/error_code.proto
    core/distributed/proto/multi_fragment_port_info.proto
    core/distributed/proto/result.proto
    core/distributed/proto/system_resource.proto
)

grpc_generate_cpp(CORE_GRPC_SRCS CORE_GRPC_HDRS ${CORE_PROTO_FILES})

add_holoscan_library(core
    core/analytics/csv_data_exporter.cpp
    core/analytics/data_exporter.cpp
    core/app_driver.cpp
    core/app_worker.cpp
    core/application.cpp
    core/arg.cpp
    core/argument_setter.cpp
    core/cli_options.cpp
    core/cli_parser.cpp
    core/component.cpp
    core/component_spec.cpp
    core/condition.cpp
    core/conditions/gxf/asynchronous.cpp
    core/conditions/gxf/boolean.cpp
    core/conditions/gxf/count.cpp
    core/conditions/gxf/cuda_buffer_available.cpp
    core/conditions/gxf/cuda_event.cpp
    core/conditions/gxf/cuda_stream.cpp
    core/conditions/gxf/downstream_affordable.cpp
    core/conditions/gxf/periodic.cpp
    core/conditions/gxf/memory_available.cpp
    core/conditions/gxf/message_available.cpp
    core/conditions/gxf/multi_message_available.cpp
    core/conditions/gxf/multi_message_available_timeout.cpp
    core/conditions/gxf/expiring_message.cpp
    core/config.cpp
    core/dataflow_tracker.cpp
    core/distributed/app_driver/client.cpp
    core/distributed/app_driver/service_impl.cpp
    core/distributed/app_driver/server.cpp
    core/distributed/app_worker/client.cpp
    core/distributed/app_worker/service_impl.cpp
    core/distributed/app_worker/server.cpp
    core/distributed/common/forward_op.cpp
    core/distributed/common/virtual_operator.cpp
    core/domain/tensor.cpp
    core/errors.cpp
    core/file_fifo_mutex.cpp
    core/executors/gxf/gxf_executor.cpp
    core/executors/gxf/gxf_logger.cpp
    core/executors/gxf/gxf_parameter_adaptor.cpp
    core/flow_tracking_annotation.cpp
    core/fragment.cpp
    core/fragment_scheduler.cpp
    core/fragment_service.cpp
    core/graphs/flow_graph.cpp
    core/gxf/codec_registry.cpp
    core/gxf/endpoint.cpp
    core/gxf/entity.cpp
    core/gxf/entity_group.cpp
    core/gxf/gxf_component.cpp
    core/gxf/gxf_component_info.cpp
    core/gxf/gxf_condition.cpp
    core/gxf/gxf_cuda.cpp
    core/gxf/gxf_execution_context.cpp
    core/gxf/gxf_extension_manager.cpp
    core/gxf/gxf_io_context.cpp
    core/gxf/gxf_network_context.cpp
    core/gxf/gxf_operator.cpp
    core/gxf/gxf_resource.cpp
    core/gxf/gxf_scheduler.cpp
    core/gxf/gxf_scheduling_term_wrapper.cpp
    core/gxf/gxf_utils.cpp
    core/gxf/gxf_wrapper.cpp
    core/io_spec.cpp
    core/messagelabel.cpp
    core/metadata.cpp
    core/network_context.cpp
    core/network_contexts/gxf/ucx_context.cpp
    core/operator.cpp
    core/operator_spec.cpp
    core/resource.cpp
    core/resources/async_data_logger.cpp
    core/resources/data_logger.cpp
    core/resources/gxf/allocator.cpp
    core/resources/gxf/annotated_double_buffer_receiver.cpp
    core/resources/gxf/annotated_double_buffer_transmitter.cpp
    core/resources/gxf/async_buffer_receiver.cpp
    core/resources/gxf/async_buffer_transmitter.cpp
    core/resources/gxf/block_memory_pool.cpp
    core/resources/gxf/clock.cpp
    core/resources/gxf/condition_combiner.cpp
    core/resources/gxf/cpu_thread.cpp
    core/resources/gxf/cuda_allocator.cpp
    core/resources/gxf/cuda_green_context_pool.cpp
    core/resources/gxf/cuda_green_context.cpp
    core/resources/gxf/cuda_stream_pool.cpp
    core/resources/gxf/double_buffer_receiver.cpp
    core/resources/gxf/double_buffer_transmitter.cpp
    core/resources/gxf/dfft_collector.cpp
    core/resources/gxf/gxf_component_resource.cpp
    core/resources/gxf/first_fit_allocator_base.cpp
    core/resources/gxf/holoscan_async_buffer_receiver.cpp
    core/resources/gxf/holoscan_async_buffer_transmitter.cpp
    core/resources/gxf/holoscan_ucx_receiver.cpp
    core/resources/gxf/holoscan_ucx_transmitter.cpp
    core/resources/gxf/manual_clock.cpp
    core/resources/gxf/realtime_clock.cpp
    core/resources/gxf/receiver.cpp
    core/resources/gxf/rmm_allocator.cpp
    core/resources/gxf/serialization_buffer.cpp
    core/resources/gxf/std_component_serializer.cpp
    core/resources/gxf/std_entity_serializer.cpp
    core/resources/gxf/stream_ordered_allocator.cpp
    core/resources/gxf/synthetic_clock.cpp
    core/resources/gxf/system_resources.cpp
    core/resources/gxf/transmitter.cpp
    core/resources/gxf/ucx_component_serializer.cpp
    core/resources/gxf/ucx_entity_serializer.cpp
    core/resources/gxf/ucx_holoscan_component_serializer.cpp
    core/resources/gxf/ucx_receiver.cpp
    core/resources/gxf/ucx_serialization_buffer.cpp
    core/resources/gxf/ucx_transmitter.cpp
    core/resources/gxf/unbounded_allocator.cpp
    core/scheduler.cpp
    core/schedulers/greedy_fragment_allocation.cpp
    core/schedulers/gxf/event_based_scheduler.cpp
    core/schedulers/gxf/greedy_scheduler.cpp
    core/schedulers/gxf/multithread_scheduler.cpp
    core/signal_handler.cpp
    core/system/cpu_resource_monitor.cpp
    core/system/gpu_resource_monitor.cpp
    core/system/network_utils.cpp
    core/system/system_resource_manager.cpp
    core/system/topology.cpp
    utils/cuda_stream_handler.cpp  # keep here instead of separate lib for backwards compatibility with 1.0
    utils/operator_runner.cpp
    ${CORE_GRPC_SRCS}
)

target_link_libraries(core
    PUBLIC
        ${CMAKE_DL_LIBS}  # for dlopen
        holoscan::logger
        holoscan::profiler
        CUDA::cuda_driver
        CUDA::cudart
        fmt::fmt-header-only
        GXF::app
        GXF::core
        GXF::cuda
        GXF::rmm
        GXF::serialization  # for nvidia::gxf::Endpoint
        GXF::std
        GXF::ucx
        yaml-cpp
        holoscan::eigen3
    INTERFACE
        ucxx::ucxx
    PRIVATE
        hwloc
        gRPC::grpc++
        gRPC::grpc++_reflection
        protobuf::libprotobuf
        ucxx::ucxx
)

target_include_directories(core
    PUBLIC
      $<BUILD_INTERFACE:${tl-expected_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${cli11_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${magic_enum_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/single_include>
      $<BUILD_INTERFACE:${concurrent_queue_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty>
)

# ##############################################################################
# # Add library: holoscan::infer_utils
# ##############################################################################
add_holoscan_library(infer_utils
    utils/holoinfer_utils.cpp
)
target_link_libraries(infer_utils
    PUBLIC
        holoscan::core
        holoscan::infer
)

# ##############################################################################
# # Add libraries: holoscan::ops::...
# ##############################################################################
add_subdirectory(operators)

# ##############################################################################
# # Add libraries: holoscan::data_loggers::...
# ##############################################################################
add_subdirectory(data_loggers)

