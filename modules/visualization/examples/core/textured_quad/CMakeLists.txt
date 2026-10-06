# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project(holoviz_textured_quad LANGUAGES CXX)

add_executable(${PROJECT_NAME} Main.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        holoscan::visualization::holoviz
        stb::stb
)
holoviz_copy_runtime_dlls(${PROJECT_NAME})

# The example's shader is compiled to SPIR-V at build time (same generated-header path the
# library's built-in shaders use); only image assets ship as data files.
include(HolovizShaderHeader)
# shader_interface.hpp is included by both shader.slang and Main.cpp; the generated header asserts
# the shared PushConstants against the shader's reflected layout.
holoviz_generate_shader_header(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shader.slang.hpp"
    SLANG shader.slang
    NAMESPACE holoviz::examples::textured_quad
    PAIRS main=vertex_main:fragment_main
    PUSH_CONSTANTS_TYPE holoviz::examples::textured_quad::PushConstants
    PUSH_CONSTANTS_INCLUDE shader_interface.hpp
)
target_sources(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shader.slang.hpp")
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated"
    "${CMAKE_CURRENT_SOURCE_DIR}")

holoscan_configure_example_install(${PROJECT_NAME}
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${PROJECT_NAME}/cpp"
)

# The example resolves nv_logo.png against the resource path given on the command line,
# defaulting to the per-example build-tree asset directory next to the binary or its installed
# example directory.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}_assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/nv_logo.png
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}_assets
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${PROJECT_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)

install(FILES nv_logo.png
    DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${PROJECT_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)

# Smoke-test the default resource-path resolution the way an installed user runs the example:
# from an unrelated working directory, invoking the binary through PATH with a bare program
# name (argv[0] has no directory component).
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_test(NAME ${PROJECT_NAME}_path_smoke
        COMMAND ${CMAKE_COMMAND} -E env
            --modify PATH=path_list_prepend:$<TARGET_FILE_DIR:${PROJECT_NAME}>
            ${PROJECT_NAME} --print-resource-path
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(${PROJECT_NAME}_path_smoke
        PROPERTIES
            LABELS "examples;visualization"
            TIMEOUT 30
    )
endif()
