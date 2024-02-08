/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
