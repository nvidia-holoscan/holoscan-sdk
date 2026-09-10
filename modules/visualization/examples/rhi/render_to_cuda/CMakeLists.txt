# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PROJECT_NAME holoviz_rhi_render_to_cuda)

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        Main.cpp
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        holoviz::holoviz
    )

holoviz_copy_runtime_dlls(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT holoscan-examples
)
