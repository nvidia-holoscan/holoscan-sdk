# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# TODO (grelee): I removed -Wold-style-cast due to it raising errors for ajantv2 code in .cache. After restructuring we can add it back in
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
endfunction()

# ##############################################################################
# # Add library holoscan::logger
# ##############################################################################
add_holoscan_library(logger logger/logger.cpp)
target_link_libraries(logger
    PUBLIC
    fmt::fmt-header-only

    PRIVATE
    spdlog::spdlog_header_only
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(logger
        PRIVATE HOLOSCAN_LOG_ACTIVE_LEVEL=1
    )
endif()

# ##############################################################################
# # Add library: holoscan::core
# ##############################################################################
add_holoscan_library(core
    core/application.cpp
    core/arg.cpp
    core/argument_setter.cpp
    core/component.cpp
    core/component_spec.cpp
    core/condition.cpp
    core/conditions/gxf/boolean.cpp
    core/conditions/gxf/count.cpp
    core/conditions/gxf/downstream_affordable.cpp
    core/conditions/gxf/message_available.cpp
    core/config.cpp
    core/domain/tensor.cpp
    core/executors/gxf/gxf_executor.cpp
    core/executors/gxf/gxf_parameter_adaptor.cpp
    core/fragment.cpp
    core/graphs/flow_graph.cpp
    core/gxf/entity.cpp
    core/gxf/gxf_condition.cpp
    core/gxf/gxf_execution_context.cpp
    core/gxf/gxf_extension_manager.cpp
    core/gxf/gxf_io_context.cpp
    core/gxf/gxf_operator.cpp
    core/gxf/gxf_resource.cpp
    core/gxf/gxf_tensor.cpp
    core/gxf/gxf_wrapper.cpp
    core/io_spec.cpp
    core/operator.cpp
    core/operator_spec.cpp
    core/resource.cpp
    core/resources/gxf/allocator.cpp
    core/resources/gxf/block_memory_pool.cpp
    core/resources/gxf/cuda_stream_pool.cpp
    core/resources/gxf/double_buffer_receiver.cpp
    core/resources/gxf/double_buffer_transmitter.cpp
    core/resources/gxf/receiver.cpp
    core/resources/gxf/std_component_serializer.cpp
    core/resources/gxf/transmitter.cpp
    core/resources/gxf/video_stream_serializer.cpp
)

target_link_libraries(core
    PUBLIC
        holoscan::logger
        CUDA::cudart
        fmt::fmt-header-only
        GXF::core
        GXF::cuda
        GXF::std
        yaml-cpp
)

target_include_directories(core
    PUBLIC
      $<BUILD_INTERFACE:${tl-expected_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
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
