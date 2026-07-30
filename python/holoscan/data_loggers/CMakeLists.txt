# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# submodules for data loggers
add_subdirectory(async_console_logger)
add_subdirectory(basic_console_logger)

# same template approach as taken for the operators module
file(RELATIVE_PATH relative_dest_path ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR})
set(INIT_FILE_OUT ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${relative_dest_path}/__init__.py)
configure_file(__init__.py.in ${INIT_FILE_OUT})
unset(INIT_FILE_OUT)
