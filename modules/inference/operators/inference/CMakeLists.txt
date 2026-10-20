# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: there is no `add_holoscan_operator()` convenience macro in this tree yet; this target is
# wired directly with add_library() instead.

include(GenerateExportHeader)

add_library(op_inference
    holoinfer_utils.cpp
    inference.cpp
    ../operator_catalog.cpp
    inference_yaml.cpp
)
add_library(holoscan::op_inference ALIAS op_inference)

target_include_directories(op_inference
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../src/include
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>"
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(op_inference
    PUBLIC
        holoscan::core
        holoscan::infer
    PRIVATE
        CUDA::cudart
        fmt::fmt-header-only
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:
            $<INSTALL_INTERFACE:holoscan::_inference_yaml_private>>
)

set_target_properties(op_inference PROPERTIES
    OUTPUT_NAME holoscan_op_inference
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)
# Keep static dependencies such as yaml-cpp out of the exported symbols, as other SDK libraries do.
holoscan_configure_linux_shared_link(op_inference)
generate_export_header(op_inference
    BASE_NAME HOLOSCAN_OP_INFERENCE
    EXPORT_MACRO_NAME HOLOSCAN_OP_INFERENCE_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_OP_INFERENCE_NO_EXPORT
    INCLUDE_GUARD_NAME HOLOSCAN_OP_INFERENCE_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${CMAKE_BINARY_DIR}/include/holoscan/operators/inference/holoscan_op_inference_export.h"
)

install(TARGETS op_inference
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/holoscan/operators/inference
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/operators
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../include/holoscan/operators/inference
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/operators
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES
    "${CMAKE_BINARY_DIR}/include/holoscan/operators/inference/holoscan_op_inference_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/operators/inference
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
)
install(FILES ${CMAKE_SOURCE_DIR}/include/holoscan/utils/holoinfer_utils.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/utils
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
)
