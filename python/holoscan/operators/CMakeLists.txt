# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# operator submodules
add_subdirectory(bayer_demosaic)
add_subdirectory(format_converter)
add_subdirectory(gxf_codelet)
add_subdirectory(holoviz)
add_subdirectory(raw_image_processor)
add_subdirectory(inference)
add_subdirectory(inference_processor)
add_subdirectory(ping_rx)
add_subdirectory(ping_tensor_rx)
add_subdirectory(ping_tensor_tx)
add_subdirectory(ping_tx)
add_subdirectory(segmentation_postprocessor)
add_subdirectory(test_ops)
add_subdirectory(v4l2_video_capture)
add_subdirectory(video_stream_recorder)
add_subdirectory(video_stream_replayer)

file(RELATIVE_PATH relative_dest_path ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR})
set(INIT_FILE_OUT ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${relative_dest_path}/__init__.py)
configure_file(__init__.py.in ${INIT_FILE_OUT})
unset(INIT_FILE_OUT)
