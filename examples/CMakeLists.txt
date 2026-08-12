# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_examples LANGUAGES CXX CUDA)

include(CTest)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/HoloscanExampleInstall.cmake")

if(NOT COMMAND holoscan_add_graph_artifacts)
    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../cmake/HoloscanGraphArtifacts.cmake")
        include("${CMAKE_CURRENT_LIST_DIR}/../cmake/HoloscanGraphArtifacts.cmake")
    endif()
endif()

if(NOT TARGET holoscan::core)
    find_package(holoscan REQUIRED)
endif()

# Shipped examples opt into the SDK configuration matrix even when built from an installed source
# package. Ordinary consumers keep the logger header default unless they choose their own policy.
if(NOT COMMAND holoscan_apply_logging_policy)
    include("${holoscan_DIR}/HoloscanLoggingPolicy.cmake")
endif()

find_package(CUDAToolkit REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")

# CUDA ships CCCL's CMake config in a layout-dependent cmake dir; point at the
# candidates directly because CMake does not auto-search lib64 when
# CMAKE_LIBRARY_ARCHITECTURE is set. MatX's find_dependency(CCCL) reuses
# CMAKE_PREFIX_PATH (extended by holoscan-config.cmake), so CCCL must resolve
# before MatX.
find_package(CCCL 3.3.3 CONFIG REQUIRED
    PATHS
        "${CUDAToolkit_TARGET_DIR}/lib/cmake"
        "${CUDAToolkit_TARGET_DIR}/lib64/cmake"
        "${CUDAToolkit_LIBRARY_DIR}/cmake"
        "${CUDAToolkit_LIBRARY_ROOT}/lib64/cmake"
        "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake"
        /usr/local/cuda/lib64/cmake
)
find_package(matx CONFIG REQUIRED)

add_subdirectory(hello_world)
add_subdirectory(hello_core)
add_subdirectory(hello_logger)
add_subdirectory(hello_tensor)
add_subdirectory(hello_schema)
add_subdirectory(hello_matx)
add_subdirectory(hello_cccl)
add_subdirectory(ea1_slow_fast_fusion)
add_subdirectory(ea1_camera_imu_control)
add_subdirectory(ea1_external_clients)
add_subdirectory(first_pipeline)
add_subdirectory(first_operator)
add_subdirectory(tensors_101)
add_subdirectory(typed_parameters)
add_subdirectory(temporal_contracts)
add_subdirectory(errors_as_values)
add_subdirectory(time_and_clocks)
add_subdirectory(compile_and_check)
add_subdirectory(running_and_stopping)
add_subdirectory(retained_tensor_compositor)
add_subdirectory(embedded_device_tensor)
add_subdirectory(notification_bridge)
add_subdirectory(operator_anatomy)
add_subdirectory(parameter_schemas)
add_subdirectory(lifecycle_callbacks)
add_subdirectory(graph_composition)
add_subdirectory(reusable_graphs)
add_subdirectory(schema_payloads)
add_subdirectory(tensors_in_depth)
add_subdirectory(flatbuffer_schemas)
add_subdirectory(essential)
add_subdirectory(experimental)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION examples
        COMPONENT holoscan-examples
        FILES_MATCHING
            PATTERN "CMakeLists.txt"
            PATTERN "CMakeLists.min.txt"
            PATTERN "*.cmake"
            PATTERN "*.cpp"
            PATTERN "*.cu"
            PATTERN "*.hpp"
            PATTERN "*.fbs"
            PATTERN "*.json"
            PATTERN "*.yaml"
            PATTERN "*.md"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "build" EXCLUDE
        PATTERN "build-*" EXCLUDE
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "_deps" EXCLUDE
    )
endif()
