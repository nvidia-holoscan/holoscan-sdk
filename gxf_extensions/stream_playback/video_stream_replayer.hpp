/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_STREAM_PLAYBACK_VIDEO_STREAM_REPLAYER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_STREAM_PLAYBACK_VIDEO_STREAM_REPLAYER_HPP_

#include <string>

#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/serialization/file_stream.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {
namespace stream_playback {

/// @brief Replays entities by reading and deserializing from a file.
///
/// The file is processed sequentially and a single entity is published per tick.
/// This supports realtime, faster than realtime, or slower than realtime playback of prerecorded
/// data.
/// The input data can optionally be repeated to loop forever or only for a specified count.
class VideoStreamReplayer : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> transmitter_;
  gxf::Parameter<gxf::Handle<gxf::EntitySerializer>> entity_serializer_;
  gxf::Parameter<gxf::Handle<gxf::BooleanSchedulingTerm>> boolean_scheduling_term_;
  gxf::Parameter<std::string> directory_;
  gxf::Parameter<std::string> basename_;
  gxf::Parameter<size_t> batch_size_;
  gxf::Parameter<bool> ignore_corrupted_entities_;
  gxf::Parameter<float> frame_rate_;
  gxf::Parameter<bool> realtime_;
  gxf::Parameter<bool> repeat_;
  gxf::Parameter<uint64_t> count_;

  // File stream for entities
  gxf::FileStream entity_file_stream_;
  // File stream for index
  gxf::FileStream index_file_stream_;

  uint64_t playback_index_ = 0;
  uint64_t playback_count_ = 0;
  uint64_t index_start_timestamp_ = 0;
  uint64_t index_last_timestamp_ = 0;
  uint64_t index_timestamp_duration_ = 0;
  uint64_t index_frame_count_ = 1;
  uint64_t playback_start_timestamp_ = 0;
};

}  // namespace stream_playback
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_STREAM_PLAYBACK_VIDEO_STREAM_REPLAYER_HPP_
