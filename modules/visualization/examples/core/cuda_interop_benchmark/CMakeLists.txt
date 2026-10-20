# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# the FFT Gaussian blur stage is CUDA code implemented with cuFFT
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(holoviz_cuda_interop_benchmark LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME} Main.cpp blur_cufft.cu)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        holoscan::visualization::holoviz
        CUDA::cufft
        stb::stb
)
holoviz_copy_runtime_dlls(${PROJECT_NAME})

# The example's shader is compiled to SPIR-V at build time (same generated-header path the
# library's built-in shaders use); only image assets ship as data files.
include(HolovizShaderHeader)
holoviz_generate_shader_header(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shader.slang.hpp"
    SLANG shader.slang
    NAMESPACE holoviz::examples::cuda_interop_benchmark
    PAIRS main=vertex_main:fragment_main
)
target_sources(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shader.slang.hpp")
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated")

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
