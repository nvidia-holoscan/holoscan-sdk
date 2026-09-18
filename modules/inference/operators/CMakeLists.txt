# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: there is no `add_holoscan_operator()` convenience macro in this tree yet; this target is
# wired directly with add_library() instead.

add_library(op_inference
    holoinfer_utils.cpp
    inference.cpp
    inference_yaml.cpp
)
add_library(holoscan::op_inference ALIAS op_inference)

target_include_directories(op_inference
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../src/include
)

target_link_libraries(op_inference
    PUBLIC
        holoscan::core
        holoscan::infer
    PRIVATE
        CUDA::cudart
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

install(TARGETS op_inference
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT holoscan-inference
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/holoscan/operators/inference
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/operators
    COMPONENT holoscan-inference
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES ${CMAKE_SOURCE_DIR}/include/holoscan/utils/holoinfer_utils.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/utils
    COMPONENT holoscan-inference
)
