/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <string>

#define RECORDER(visualizer)                                                                       \
  {                                                                                                \
    if (RECORD_OUTPUT) {                                                                           \
      std::shared_ptr<Operator> recorder_format_converter;                                         \
      recorder_format_converter = make_operator<ops::FormatConverterOp>(                           \
          "recorder_format_converter",                                                             \
          Arg("in_dtype", std::string("rgba8888")),                                                \
          Arg("out_dtype", std::string("rgb888")),                                                 \
          Arg("pool", make_resource<UnboundedAllocator>("pool")));                                 \
      auto recorder = make_operator<ops::VideoStreamRecorderOp>(                                   \
          "recorder",                                                                              \
          Arg("directory", std::string(RECORDING_DIR)),                                            \
          Arg("basename", std::string(SOURCE_VIDEO_BASENAME)));                                    \
      add_flow(visualizer, recorder_format_converter, {{"render_buffer_output", "source_video"}}); \
      add_flow(recorder_format_converter, recorder);                                               \
      visualizer->add_arg(Arg("enable_render_buffer_output", true));                               \
      visualizer->add_arg(Arg("allocator", make_resource<UnboundedAllocator>("allocator")));       \
    }                                                                                              \
  }
