/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "video_stream_replayer.hpp"

#include <chrono>
#include <cinttypes>
#include <string>
#include <thread>

namespace nvidia {
namespace holoscan {
namespace stream_playback {

gxf_result_t VideoStreamReplayer::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(transmitter_, "transmitter", "Entity transmitter",
                                 "Transmitter channel for replaying entities");
  result &= registrar->parameter(entity_serializer_, "entity_serializer", "Entity serializer",
                                 "Serializer for serializing entities");
  result &= registrar->parameter(
      boolean_scheduling_term_, "boolean_scheduling_term", "BooleanSchedulingTerm",
      "BooleanSchedulingTerm to stop the codelet from ticking after all messages are published.");
  result &= registrar->parameter(directory_, "directory", "Directory path",
                                 "Directory path for storing files");
  result &= registrar->parameter(
      basename_, "basename", "Base file name", "User specified file name without extension",
      gxf::Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(batch_size_, "batch_size", "Batch Size",
                                 "Number of entities to read and publish for one tick", 1UL);
  result &= registrar->parameter(
      ignore_corrupted_entities_, "ignore_corrupted_entities", "Ignore Corrupted Entities",
      "If an entity could not be deserialized, it is ignored by default; otherwise a failure is "
      "generated.",
      true);
  result &= registrar->parameter(
      frame_rate_, "frame_rate", "Frame rate",
      "Frame rate to replay. If zero value is specified, it follows timings in timestamps.", 0.f);
  result &= registrar->parameter(
      realtime_, "realtime", "Realtime playback",
      "Playback video in realtime, based on frame_rate or timestamps (default: true).", true);
  result &= registrar->parameter(repeat_, "repeat", "RepeatVideo",
                                 "Repeat video stream (default: false)", false);
  result &= registrar->parameter(
      count_, "count", "Number of frame counts to playback",
      "Number of frame counts to playback. If zero value is specified, it is ignored. If the count "
      "is less than the number of frames in the video, it would be finished early.",
      0UL);

  return gxf::ToResultCode(result);
}

gxf_result_t VideoStreamReplayer::initialize() {
  // Create path by appending component name to directory path if basename is not provided
  std::string path = directory_.get() + '/';
  if (const auto& basename = basename_.try_get()) {
    path += basename.value();
  } else {
    path += name();
  }

  // Filenames for index and data
  const std::string index_filename = path + gxf::FileStream::kIndexFileExtension;
  const std::string entity_filename = path + gxf::FileStream::kBinaryFileExtension;

  // Open index file stream as read-only
  index_file_stream_ = gxf::FileStream(index_filename, "");
  gxf::Expected<void> result = index_file_stream_.open();
  if (!result) {
    GXF_LOG_WARNING("Could not open index file: %s", index_filename.c_str());
    return gxf::ToResultCode(result);
  }

  // Open entity file stream as read-only
  entity_file_stream_ = gxf::FileStream(entity_filename, "");
  result = entity_file_stream_.open();
  if (!result) {
    GXF_LOG_WARNING("Could not open entity file: %s", entity_filename.c_str());
    return gxf::ToResultCode(result);
  }

  boolean_scheduling_term_->enable_tick();

  playback_index_ = 0;
  playback_count_ = 0;
  index_start_timestamp_ = 0;
  index_last_timestamp_ = 0;
  index_timestamp_duration_ = 0;
  index_frame_count_ = 1;
  playback_start_timestamp_ = 0;

  return GXF_SUCCESS;
}

gxf_result_t VideoStreamReplayer::deinitialize() {
  // Close binary file stream
  gxf::Expected<void> result = entity_file_stream_.close();
  if (!result) { return gxf::ToResultCode(result); }

  // Close index file stream
  result = index_file_stream_.close();
  if (!result) { return gxf::ToResultCode(result); }

  return GXF_SUCCESS;
}

gxf_result_t VideoStreamReplayer::tick() {
  for (size_t i = 0; i < batch_size_; i++) {
    // Read entity index from index file
    // Break if index not found and clear stream errors
    gxf::EntityIndex index;
    gxf::Expected<size_t> size = index_file_stream_.readTrivialType(&index);
    if (!size) {
      if (repeat_) {
        // Rewind index stream
        index_file_stream_.clear();
        if (!index_file_stream_.setReadOffset(0)) { GXF_LOG_ERROR("Could not rewind index file"); }
        size = index_file_stream_.readTrivialType(&index);

        // Rewind entity stream
        entity_file_stream_.clear();
        entity_file_stream_.setReadOffset(0);
        if (!entity_file_stream_.setReadOffset(0)) {
          GXF_LOG_ERROR("Could not rewind entity file");
        }

        // Initialize the frame index
        playback_index_ = 0;
      }
    }
    if ((!size && !repeat_) || (count_ > 0 && playback_count_ >= count_)) {
      GXF_LOG_INFO("Reach end of file or playback count reaches to the limit. Stop ticking.");
      boolean_scheduling_term_->disable_tick();
      index_file_stream_.clear();
      break;
    }

    // Read entity from binary file
    gxf::Expected<gxf::Entity> entity =
        entity_serializer_->deserializeEntity(context(), &entity_file_stream_);
    if (!entity) {
      if (ignore_corrupted_entities_) {
        continue;
      } else {
        return gxf::ToResultCode(entity);
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
        GXF_LOG_INFO("Playing video stream is lagging behind (count: %" PRIu64 ", delay: %" PRId64
                     " ns)",
                     playback_count_, time_to_delay);
      }
    }

    if (time_to_delay > 0) { std::this_thread::sleep_for(std::chrono::nanoseconds(time_to_delay)); }

    // Publish entity
    gxf::Expected<void> result = transmitter_->publish(entity.value());

    // Increment frame counter and index
    ++playback_count_;
    ++playback_index_;

    if (!result) { return gxf::ToResultCode(result); }
  }

  return GXF_SUCCESS;
}

}  // namespace stream_playback
}  // namespace holoscan
}  // namespace nvidia
