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

#include "holoscan/operators/stream_playback/video_stream_recorder.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"

namespace holoscan::ops {

void VideoStreamRecorderOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<::gxf::Entity>("input");

  spec.param(receiver_, "receiver", "Entity receiver", "Receiver channel to log", &input);
  spec.param(entity_serializer_,
             "entity_serializer",
             "Entity serializer",
             "Serializer for serializing entities");
  spec.param(directory_, "directory", "Directory path", "Directory path for storing files");
  spec.param(basename_, "basename", "Base file name", "User specified file name without extension");
  spec.param(flush_on_tick_,
             "flush_on_tick",
             "Flush on tick",
             "Flushes output buffer on every tick when true",
             false);
}

void VideoStreamRecorderOp::initialize() {
  // Set up prerequisite parameters before calling GXFOperator::initialize()
  auto frag = fragment();
  auto entity_serializer =
      frag->make_resource<holoscan::VideoStreamSerializer>("entity_serializer");
  add_arg(Arg("entity_serializer") = entity_serializer);

  GXFOperator::initialize();
}

}  // namespace holoscan::ops
