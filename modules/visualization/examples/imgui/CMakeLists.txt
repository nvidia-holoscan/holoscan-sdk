# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project(holoviz_imgui LANGUAGES CXX)

add_executable(${PROJECT_NAME} Main.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        holoscan::visualization::holoviz
)

holoviz_copy_runtime_dlls(${PROJECT_NAME})

holoscan_configure_example_install(${PROJECT_NAME}
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${PROJECT_NAME}/cpp"
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${PROJECT_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)
