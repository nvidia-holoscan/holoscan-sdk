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

#include "holoscan/operators/stream_playback/video_stream_replayer.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"

namespace holoscan::ops {

void VideoStreamReplayerOp::setup(OperatorSpec& spec) {
  auto& output = spec.output<::gxf::Entity>("output");

  spec.param(transmitter_,
             "transmitter",
             "Entity transmitter",
             "Transmitter channel for replaying entities",
             &output);

  spec.param(entity_serializer_,
             "entity_serializer",
             "Entity serializer",
             "Serializer for serializing entities");
  spec.param(
      boolean_scheduling_term_,
      "boolean_scheduling_term",
      "BooleanSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");
  spec.param(directory_, "directory", "Directory path", "Directory path for storing files");
  spec.param(basename_, "basename", "Base file name", "User specified file name without extension");
  spec.param(batch_size_,
             "batch_size",
             "Batch Size",
             "Number of entities to read and publish for one tick",
             1UL);
  spec.param(
      ignore_corrupted_entities_,
      "ignore_corrupted_entities",
      "Ignore Corrupted Entities",
      "If an entity could not be deserialized, it is ignored by default; otherwise a failure is "
      "generated.",
      true);
  spec.param(frame_rate_,
             "frame_rate",
             "Frame rate",
             "Frame rate to replay. If zero value is specified, it follows timings in timestamps.",
             0.f);
  spec.param(realtime_,
             "realtime",
             "Realtime playback",
             "Playback video in realtime, based on frame_rate or timestamps (default: true).",
             true);
  spec.param(repeat_, "repeat", "RepeatVideo", "Repeat video stream (default: false)", false);
  spec.param(count_,
             "count",
             "Number of frame counts to playback",
             "Number of frame counts to playback. If zero value is specified, it is ignored. If "
             "the count "
             "is less than the number of frames in the video, it would be finished early.",
             0UL);
}

void VideoStreamReplayerOp::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto entity_serializer =
      frag->make_resource<holoscan::VideoStreamSerializer>("entity_serializer");
  auto boolean_scheduling_term =
      frag->make_condition<holoscan::BooleanCondition>("boolean_scheduling_term");
  add_arg(Arg("entity_serializer") = entity_serializer);
  add_arg(Arg("boolean_scheduling_term") = boolean_scheduling_term);

  GXFOperator::initialize();
}

}  // namespace holoscan::ops
