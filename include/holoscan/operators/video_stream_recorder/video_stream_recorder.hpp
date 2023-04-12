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

#ifndef HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_RECORDER_HPP
#define HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_RECORDER_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/fragment.hpp"
#include "gxf/serialization/file_stream.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to record the video stream to a file.
 */
class VideoStreamRecorderOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoStreamRecorderOp)

  VideoStreamRecorderOp() = default;

  ~VideoStreamRecorderOp() override;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  // void deinitialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<holoscan::IOSpec*> receiver_;
  Parameter<std::shared_ptr<holoscan::Resource>> entity_serializer_;
  Parameter<std::string> directory_;
  Parameter<std::string> basename_;
  Parameter<bool> flush_on_tick_;

  // File stream for data index
  nvidia::gxf::FileStream index_file_stream_;
  // File stream for binary data
  nvidia::gxf::FileStream binary_file_stream_;
  // Offset into binary file
  size_t binary_file_offset_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_RECORDER_HPP */
