# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# custom command to copy __init__.py file (only when needed)
set(INIT_FILE_IN ${CMAKE_CURRENT_LIST_DIR}/__init__.py)
file(RELATIVE_PATH relative_dest_path ${CMAKE_CURRENT_BINARY_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR})
set(INIT_FILE_OUT ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${relative_dest_path}/__init__.py)
add_custom_command(
    OUTPUT "${INIT_FILE_OUT}"
    COMMAND ${CMAKE_COMMAND} -E copy "${INIT_FILE_IN}" "${INIT_FILE_OUT}"
    DEPENDS "${INIT_FILE_IN}"
    COMMENT "Copying ping_rx __init__.py"
)
add_custom_target(ping_rx_python DEPENDS "${INIT_FILE_OUT}")
add_dependencies(holoscan-python ping_rx_python)
unset(INIT_FILE_IN)
unset(INIT_FILE_OUT)
