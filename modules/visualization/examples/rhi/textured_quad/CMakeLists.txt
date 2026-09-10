# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PROJECT_NAME holoviz_rhi_textured_quad)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        Main.cpp
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        holoviz::holoviz
        stb::stb
    )

holoviz_copy_runtime_dlls(${PROJECT_NAME})

# The example resolves nv_logo.png and shader.slang against the resource path given on the
# command line, defaulting to the per-example asset directory next to the binary. The
# directory is per-example because every RHI example ships its own shader.slang and they
# would otherwise overwrite each other in a shared install destination.
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}_assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/nv_logo.png
        ${CMAKE_CURRENT_SOURCE_DIR}/shader.slang
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/${PROJECT_NAME}_assets
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT holoscan-examples
)

install(FILES nv_logo.png shader.slang
    DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}_assets
    COMPONENT holoscan-examples)

# Smoke-test the default resource-path resolution the way an installed user runs the example:
# from an unrelated working directory, invoking the binary through PATH with a bare program
# name (argv[0] has no directory component).
add_test(NAME ${PROJECT_NAME}_path_smoke
    COMMAND ${CMAKE_COMMAND} -E env
        --modify PATH=path_list_prepend:$<TARGET_FILE_DIR:${PROJECT_NAME}>
        ${PROJECT_NAME} --print-resource-path
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
