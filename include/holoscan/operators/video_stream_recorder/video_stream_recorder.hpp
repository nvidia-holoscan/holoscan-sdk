/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_RECORDER_HPP
#define HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_RECORDER_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gxf/serialization/file_stream.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_operator.hpp>

namespace holoscan::ops {

/**
 * @brief Operator class to record a video stream to a file.
 *
 * ==Named Inputs==
 *
 * - **input** : `nvidia::gxf::Tensor`
 *   - A message containing a video frame to serialize to disk. The input tensor can be on either
 *     the CPU or GPU. This data location will be recorded as part of the metadata serialized to
 *     disk and if the data is later read back in via `VideoStreamReplayerOp`, the tensor output of
 *     that operator will be on the same device (CPU or GPU).
 *
 * ==Parameters==
 *
 * - **directory**: Directory path for storing files.
 * - **basename**: User specified file name without extension.
 * - **flush_on_tick**: Flushes output buffer on every tick when `true`.
 *   Optional (default: `false`).
 */
class VideoStreamRecorderOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VideoStreamRecorderOp)

  VideoStreamRecorderOp() = default;

  ~VideoStreamRecorderOp() override;

  void setup(OperatorSpec& spec) override;

  void initialize() override;
  // void deinitialize() override;
  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

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
  size_t binary_file_offset_{0};
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_STREAM_PLAYBACK_VIDEO_STREAM_RECORDER_HPP */
