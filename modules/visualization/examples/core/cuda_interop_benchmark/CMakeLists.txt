# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(EXAMPLE_NAME holoviz_cuda_interop_benchmark)

# the FFT Gaussian blur stage is CUDA code implemented with cuFFT
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

add_executable(${EXAMPLE_NAME})

target_sources(${EXAMPLE_NAME}
    PRIVATE
        Main.cpp
        blur_cufft.cu
    )

target_include_directories(${EXAMPLE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    )

set_target_properties(${EXAMPLE_NAME}
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

target_link_libraries(${EXAMPLE_NAME}
    PRIVATE
        holoscan::visualization::holoviz
        CUDA::cufft
        stb::stb
    )
holoviz_copy_runtime_dlls(${EXAMPLE_NAME})

holoscan_configure_example_install(${EXAMPLE_NAME}
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
)

# The example resolves nv_logo.png and shader.slang against the resource path given on the
# command line, defaulting to the per-example build-tree asset directory next to the binary or
# its installed example directory. The directory is per-example because every core example ships its
# own shader.slang and they would otherwise overwrite each other in a shared install destination.
add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:${EXAMPLE_NAME}>/${EXAMPLE_NAME}_assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/nv_logo.png
        ${CMAKE_CURRENT_SOURCE_DIR}/shader.slang
        $<TARGET_FILE_DIR:${EXAMPLE_NAME}>/${EXAMPLE_NAME}_assets
)

install(TARGETS ${EXAMPLE_NAME}
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)

install(FILES nv_logo.png shader.slang
    DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT})

# Smoke-test the default resource-path resolution the way an installed user runs the example:
# from an unrelated working directory, invoking the binary through PATH with a bare program
# name (argv[0] has no directory component).
if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_test(NAME ${EXAMPLE_NAME}_path_smoke
        COMMAND ${CMAKE_COMMAND} -E env
            --modify PATH=path_list_prepend:$<TARGET_FILE_DIR:${EXAMPLE_NAME}>
            ${EXAMPLE_NAME} --print-resource-path
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    set_tests_properties(${EXAMPLE_NAME}_path_smoke
        PROPERTIES
            LABELS "examples;visualization"
            TIMEOUT 30
    )
endif()
