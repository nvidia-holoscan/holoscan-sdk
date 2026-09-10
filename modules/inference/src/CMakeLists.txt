#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project(holoinfer VERSION ${HOLOSCAN_BUILD_VERSION} LANGUAGES CXX CUDA)

add_subdirectory(include) # header-only interface for public holoinfer headers

set(holoinfer_utils_src
    utils/infer_utils.cpp
    utils/infer_buffer.cpp
    utils/work_queue.cpp
)
add_library(holoinfer_utils SHARED ${holoinfer_utils_src})
add_library(holoscan::holoinfer_utils ALIAS holoinfer_utils)

target_include_directories(holoinfer_utils
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(holoinfer_utils
    PUBLIC
        CUDA::cuda_driver
        CUDA::nvrtc
        holoscan::infer::headers
    PRIVATE
        CUDA::cudart
        holoscan_security_flags
        yaml-cpp::yaml-cpp
)

set_target_properties(holoinfer_utils PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

set(holoinfer_src
    infer/trt/core.cpp
    infer/trt/utils.cpp
    params/infer_param.cpp
    manager/infer_manager.cpp
    manager/infer_manager_setup.cpp
    manager/infer_manager_dispatch.cpp
    manager/infer_manager_fast_paths.cpp
    manager/infer_context.cpp
)

add_library(${PROJECT_NAME} SHARED ${holoinfer_src})
add_library(holoscan::infer ALIAS ${PROJECT_NAME})

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/manager>
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        CUDA::cuda_driver
        CUDA::nvrtc
        dlpack::dlpack
        holoscan::infer::headers
        holoscan::holoinfer_utils # [1]
        magic_enum::magic_enum
        tl::expected
    PRIVATE
        ${CMAKE_DL_LIBS} # to dlclose backend plugin handles
        CUDA::cudart
        holoscan::core # [2]
        holoscan::logger
        TensorRT::nvinfer_plugin
        TensorRT::nvonnxparser
        holoscan_security_flags
        yaml-cpp::yaml-cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME holoscan_infer
    EXPORT_NAME infer
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

# Generated export header (HOLOINFER_EXPORT) — see include/holoinfer/holoinfer_export.h.
# Replaces the hand-rolled `_HOLOSCAN_EXTERNAL_API_` visibility attribute.
include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME}
    BASE_NAME HOLOINFER
    EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/include/holoinfer/holoinfer_export.h
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/holoinfer/holoinfer_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoinfer
    COMPONENT holoscan-inference
)

install(TARGETS holoinfer_utils
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT holoscan-inference
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT holoscan-inference
)
