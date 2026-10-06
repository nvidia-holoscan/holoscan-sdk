# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Examples that use the module's Holoscan operator (HolovizOp) in a real Graph. They link
# holoscan::core and holoscan::visualization::op_holoviz like an application would, never the
# HoloViz backend directly (the samples under ../core and ../text cover that).
add_subdirectory(cuda_sink_example)
add_subdirectory(gray_scale_example)
add_subdirectory(holoviz_image_topic)
add_subdirectory(image_formats)
add_subdirectory(text_layer_example)
