# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GenerateExportHeader)

# HolovizOp is the installed visualization boundary. Keep the HoloViz renderer private to this
# shared library so downstream applications need only the SDK package target.
add_library(holoscan_operators_holoviz SHARED src/holoviz_op.cpp)
add_library(holoscan::operators::holoviz ALIAS holoscan_operators_holoviz)

target_include_directories(holoscan_operators_holoviz PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<BUILD_INTERFACE:${holoscan_visualization_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${holoscan_visualization_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(holoscan_operators_holoviz
    PUBLIC holoscan::core
    PRIVATE holoviz::holoviz CUDA::cuda_driver
)

target_compile_features(holoscan_operators_holoviz PUBLIC cxx_std_20)
generate_export_header(holoscan_operators_holoviz
    BASE_NAME HOLOSCAN_OPERATOR_HOLOVIZ
    EXPORT_MACRO_NAME HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_OPERATOR_HOLOVIZ_NO_EXPORT
    INCLUDE_GUARD_NAME HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/operators/holoviz/holoscan_operator_holoviz_export.h"
)
set_target_properties(holoscan_operators_holoviz PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME operators::holoviz
    OUTPUT_NAME holoscan_operator_holoviz
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
holoscan_configure_linux_shared_link(holoscan_operators_holoviz)

if(holoscan_visualization_IS_TOP_LEVEL)
    install(TARGETS holoscan_operators_holoviz
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    )
else()
    list(APPEND HOLOSCAN_INSTALL_TARGETS holoscan_operators_holoviz)
    set(HOLOSCAN_INSTALL_TARGETS "${HOLOSCAN_INSTALL_TARGETS}" PARENT_SCOPE)
endif()

install(
    DIRECTORY include/holoscan
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/operators/holoviz/holoscan_operator_holoviz_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/operators/holoviz
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)

# HolovizOp's declaration uses HoloViz descriptor and backend types. Install the matching headers
# without exporting the private renderer target; its implementation is absorbed into the operator.
install(
    DIRECTORY "${holoscan_visualization_SOURCE_DIR}/include/holoviz"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES "${holoscan_visualization_BINARY_DIR}/include/holoviz/holoviz_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)
install(
    DIRECTORY "${holoscan_visualization_BINARY_DIR}/include/holoviz/imgui"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.h"
)
