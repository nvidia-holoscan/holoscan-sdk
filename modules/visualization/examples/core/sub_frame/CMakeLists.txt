# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(EXAMPLE_NAME holoviz_sub_frame)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

add_executable(${EXAMPLE_NAME})

target_sources(${EXAMPLE_NAME}
    PRIVATE
        Main.cpp
        pattern.cu
    )

set_target_properties(${EXAMPLE_NAME}
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

target_link_libraries(${EXAMPLE_NAME}
    PRIVATE
        holoscan::visualization::holoviz
    )

# The shader is compiled to SPIR-V at build time (same generated-header path the library's
# built-in shaders use); the example ships no data files. shader_interface.hpp is included by
# both shader.slang and Main.cpp; the generated header asserts the shared PushConstants against
# the shader's reflected layout.
include(HolovizShaderHeader)
holoviz_generate_shader_header(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shader.slang.hpp"
    SLANG shader.slang
    NAMESPACE holoviz::examples::sub_frame
    PAIRS main=vertex_main:fragment_main
    PUSH_CONSTANTS_TYPE holoviz::examples::sub_frame::PushConstants
    PUSH_CONSTANTS_INCLUDE shader_interface.hpp
)
target_sources(${EXAMPLE_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shader.slang.hpp")
target_include_directories(${EXAMPLE_NAME} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated"
    "${CMAKE_CURRENT_SOURCE_DIR}")

holoviz_copy_runtime_dlls(${EXAMPLE_NAME})

holoscan_configure_example_install(${EXAMPLE_NAME}
    RUNTIME_DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
)


install(TARGETS ${EXAMPLE_NAME}
    RUNTIME DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT}
)
install(FILES shader.slang shader_interface.hpp
    DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/${EXAMPLE_NAME}/cpp"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT})
