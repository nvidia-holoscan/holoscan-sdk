/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/video_stream_replayer/video_stream_replayer.hpp"

#include <chrono>
#include <cinttypes>
#include <string>
#include <thread>
#include <utility>

#include "gxf/core/expected.hpp"
#include "gxf/serialization/entity_serializer.hpp"

#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/std_entity_serializer.hpp"

namespace holoscan::ops {

void VideoStreamReplayerOp::setup(OperatorSpec& spec) {
  auto& output = spec.output<gxf::Entity>("output");

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
      frag->make_resource<holoscan::StdEntitySerializer>("replayer__std_entity_serializer");
  if (graph_entity_) {
    entity_serializer->gxf_eid(graph_entity_->eid());
    entity_serializer->gxf_graph_entity(graph_entity_);
  }
  add_arg(Arg("entity_serializer") = entity_serializer);

  // Find if there is an argument for 'boolean_scheduling_term'
  auto has_boolean_scheduling_term =
      std::find_if(args().begin(), args().end(), [](const auto& arg) {
        return (arg.name() == "boolean_scheduling_term");
      });
  // Create the BooleanCondition if there is no argument provided.
  if (has_boolean_scheduling_term == args().end()) {
    boolean_scheduling_term_ =
        frag->make_condition<holoscan::BooleanCondition>("boolean_scheduling_term");
    add_arg(boolean_scheduling_term_.get());
  }

  // Operator::initialize must occur after all arguments have been added
  Operator::initialize();

  // Create path by appending component name to directory path if basename is not provided
  std::string path = directory_.get() + '/';

  // Note: basename was optional in the GXF operator, but not yet in the native operator,
  //       so in practice, this should always have a value.
  if (basename_.has_value()) {
    path += basename_.get();
  } else {
    path += name();
  }

  // Filenames for index and data
  const std::string index_filename = path + nvidia::gxf::FileStream::kIndexFileExtension;
  const std::string entity_filename = path + nvidia::gxf::FileStream::kBinaryFileExtension;

  // Open index file stream as read-only
  index_file_stream_ = nvidia::gxf::FileStream(index_filename, "");
  nvidia::gxf::Expected<void> result = index_file_stream_.open();
  if (!result) {
    HOLOSCAN_LOG_WARN("Could not open index file: {}", index_filename);
    auto code = nvidia::gxf::ToResultCode(result);
    throw std::runtime_error(fmt::format("File open failed with code: {}", code));
  }

  // Open entity file stream as read-only
  entity_file_stream_ = nvidia::gxf::FileStream(entity_filename, "");
  result = entity_file_stream_.open();
  if (!result) {
    HOLOSCAN_LOG_WARN("Could not open entity file: {}", entity_filename);
    auto code = nvidia::gxf::ToResultCode(result);
    throw std::runtime_error(fmt::format("File open failed with code: {}", code));
  }

  boolean_scheduling_term_->enable_tick();

  playback_index_ = 0;
  playback_count_ = 0;
  index_start_timestamp_ = 0;
  index_last_timestamp_ = 0;
  index_timestamp_duration_ = 0;
  index_frame_count_ = 1;
  playback_start_timestamp_ = 0;
}

VideoStreamReplayerOp::~VideoStreamReplayerOp() {
  // for the GXF codelet, this code is in a deinitialize() method

  // Close binary file stream
  nvidia::gxf::Expected<void> result = entity_file_stream_.close();
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(result);
    HOLOSCAN_LOG_ERROR("Failed to close entity_file_stream_ with code: {}", code);
  }

  // Close index file stream
  result = index_file_stream_.close();
  if (!result) {
    auto code = nvidia::gxf::ToResultCode(result);
    HOLOSCAN_LOG_ERROR("Failed to close index_file_stream_ with code: {}", code);
  }
}

void VideoStreamReplayerOp::compute(InputContext& op_input, OutputContext& op_output,
                                    ExecutionContext& context) {
  // avoid warning about unused variable
  (void)op_input;

  for (size_t i = 0; i < batch_size_; i++) {
    // Read entity index from index file
    // Break if index not found and clear stream errors
    nvidia::gxf::EntityIndex index;
    nvidia::gxf::Expected<size_t> size = index_file_stream_.readTrivialType(&index);
    if (!size) {
      if (repeat_) {
        // Rewind index stream
        index_file_stream_.clear();
        if (!index_file_stream_.setReadOffset(0)) {
          HOLOSCAN_LOG_ERROR("Could not rewind index file");
        }
        size = index_file_stream_.readTrivialType(&index);

        // Rewind entity stream
        entity_file_stream_.clear();
        if (!entity_file_stream_.setReadOffset(0)) {
          HOLOSCAN_LOG_ERROR("Could not rewind entity file");
        }

        // Initialize the frame index
        playback_index_ = 0;
      }
    }
    if ((!size && !repeat_) || (count_ > 0 && playback_count_ >= count_)) {
      HOLOSCAN_LOG_INFO("Reach end of file or playback count reaches to the limit. Stop ticking.");
      boolean_scheduling_term_->disable_tick();
      index_file_stream_.clear();
      break;
    }

    // dynamic cast from holoscan::Resource to holoscan::StdEntitySerializer
    auto vs_serializer =
        std::dynamic_pointer_cast<holoscan::StdEntitySerializer>(entity_serializer_.get());
    // get underlying GXF EntitySerializer
    auto entity_serializer = nvidia::gxf::Handle<nvidia::gxf::EntitySerializer>::Create(
        context.context(), vs_serializer->gxf_cid());
    // Read entity from binary file
    nvidia::gxf::Expected<nvidia::gxf::Entity> entity =
        entity_serializer.value()->deserializeEntity(context.context(), &entity_file_stream_);
    if (!entity) {
      if (ignore_corrupted_entities_) {
        continue;
      } else {
        auto code = nvidia::gxf::ToResultCode(entity);
        throw std::runtime_error(
            fmt::format("failed reading entity from entity_file_stream with code {}", code));
      }
    }

    int64_t time_to_delay = 0;

    if (playback_count_ == 0) {
      playback_start_timestamp_ = std::chrono::system_clock::now().time_since_epoch().count();
      index_start_timestamp_ = index.log_time;
    }
    // Update last timestamp
    if (index.log_time > index_last_timestamp_) {
      index_last_timestamp_ = index.log_time;
      index_frame_count_ = playback_count_ + 1;
      index_timestamp_duration_ = index_last_timestamp_ - index_start_timestamp_;
    }

    // Delay if realtime is specified
    if (realtime_) {
      // Calculate the delay time based on frame rate or timestamps.
      uint64_t current_timestamp = std::chrono::system_clock::now().time_since_epoch().count();
      int64_t time_delta = static_cast<int64_t>(current_timestamp - playback_start_timestamp_);
      if (frame_rate_ > 0.f) {
        time_to_delay =
            static_cast<int64_t>(1000000000 / frame_rate_) * playback_count_ - time_delta;
      } else {
        // Get timestamp from entity
        uint64_t timestamp = index.log_time;
        time_to_delay = static_cast<int64_t>((timestamp - index_start_timestamp_) +
                                             index_timestamp_duration_ *
                                                 (playback_count_ / index_frame_count_)) -
                        time_delta;
      }
      if (time_to_delay < 0 && (playback_count_ % index_frame_count_ != 0)) {
        HOLOSCAN_LOG_INFO(
            fmt::format("Playing video stream is lagging behind (count: {} , delay: {} ns)",
                        playback_count_,
                        time_to_delay));
      }
    }

    if (time_to_delay > 0) { std::this_thread::sleep_for(std::chrono::nanoseconds(time_to_delay)); }

    // emit the entity
    auto result = gxf::Entity(std::move(entity.value()));
    op_output.emit(result);

    // Increment frame counter and index
    ++playback_count_;
    ++playback_index_;
  }
}

}  // namespace holoscan::ops
