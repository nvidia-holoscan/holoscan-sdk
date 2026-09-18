# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(EXAMPLE_NAME holoviz_text_minimal)

add_executable(${EXAMPLE_NAME})

target_sources(${EXAMPLE_NAME}
    PRIVATE
        Main.cpp
    )

target_include_directories(${EXAMPLE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    )

target_link_libraries(${EXAMPLE_NAME}
    PRIVATE
        holoscan::visualization::holoviz
    )

holoviz_copy_runtime_dlls(${EXAMPLE_NAME})

holoscan_configure_example_install(${EXAMPLE_NAME}
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
)

install(TARGETS ${EXAMPLE_NAME}
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)
