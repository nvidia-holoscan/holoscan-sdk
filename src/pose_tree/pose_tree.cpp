/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/pose_tree/pose_tree.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace holoscan {

const char* PoseTree::error_to_str(Error error) {
  switch (error) {
    case Error::kInvalidArgument:
      return "Invalid Argument";
    case Error::kOutOfMemory:
      return "Out of Memory";
    case Error::kFrameNotFound:
      return "Frame not found";
    case Error::kAlreadyExists:
      return "Edge already exists";
    case Error::kCyclingDependency:
      return "Cycling dependency";
    case Error::kFramesNotLinked:
      return "Frames are not linked";
    case Error::kPoseOutOfOrder:
      return "Pose updated out of order";
    case Error::kLogicError:
      return "Logic Error";
  }
  return "Invalid Error Code";
}

PoseTree::expected_t<void> PoseTree::init(const int32_t number_frames, const int32_t number_edges,
                                          const int32_t history_length,
                                          const int32_t default_number_edges,
                                          const int32_t default_history_length,
                                          const int32_t edges_chunk_size,
                                          const int32_t history_chunk_size) {
  // Check the parameters are valid.
  if (number_frames <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (number_edges <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (history_length <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (default_number_edges <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (default_history_length <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (edges_chunk_size <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }
  if (history_chunk_size <= 0) {
    return unexpected_t(Error::kInvalidArgument);
  }

  std::unique_lock<std::shared_timed_mutex> lock(mutex_);

  init_params_ = {number_frames,
                  number_edges,
                  history_length,
                  default_number_edges,
                  default_history_length,
                  edges_chunk_size,
                  history_chunk_size};
  default_number_edges_ = default_number_edges;
  default_history_length_ = default_history_length;
  {  // Initialize the history map
    const auto result = histories_map_.initialize(number_edges);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the frame map
    const auto result = frame_map_.initialize(number_frames, number_frames * 2);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the history memory management
    const auto result = histories_management_.allocate(number_edges, edges_chunk_size);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the poses memory management
    const auto result = poses_management_.allocate(history_length, history_chunk_size);
    if (!result) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the callbacks
    // TODO(bbutin): We should use a parameter for the size of the callbacks
    const auto result1 = set_edge_callbacks_keys_.reserve(1024);
    const auto result2 = set_edge_callbacks_.initialize(1024);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the create frame callbacks
    // TODO(bbutin): We should use a parameter for the size of the callbacks
    const auto result1 = create_frame_callbacks_keys_.reserve(1024);
    const auto result2 = create_frame_callbacks_.initialize(1024);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the edges map
    const auto result1 = edges_map_.initialize(number_edges, 2 * number_edges);
    const auto result2 = edges_map_keys_.reserve(number_edges);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }
  {  // Initialize the edges map
    const auto result1 = name_to_uid_map_.initialize(number_frames, 2 * number_frames);
    const auto result2 = name_to_uid_map_keys_.reserve(number_frames);
    if (!result1 || !result2) {
      return unexpected_t(Error::kOutOfMemory);
    }
  }

  frames_stack_.reset(new (std::nothrow) frame_t[number_frames]);
  if (frames_stack_.get() == nullptr) {
    return unexpected_t(Error::kOutOfMemory);
  }

  hint_version_ = 0;
  version_ = 1;
  frame_cb_latest_uid_ = 0;
  edge_cb_latest_uid_ = 0;
  next_frame_id_ = 1;
  frame_id_increment_ = 1;

  return expected_t<void>{};
}

PoseTree::expected_t<void> PoseTree::set_multithreading_info(const frame_t start_frame_id,
                                                             const frame_t increment) {
  if (start_frame_id == 0) {
    HOLOSCAN_LOG_DEBUG(
        "Invalid start_frame_id: {}. Value must be greater than 0 since frame ID 0 "
        "is reserved for invalid frames",
        start_frame_id);
    return unexpected_t(Error::kInvalidArgument);
  }
  if (increment == 0 || std::numeric_limits<uint64_t>::max() / increment <
                            static_cast<uint64_t>(frame_map_.capacity())) {
    HOLOSCAN_LOG_DEBUG("Invalid increment: {}. Value must be between 1 and {}",
                       increment,
                       std::numeric_limits<uint64_t>::max() / frame_map_.capacity());
    return unexpected_t(Error::kInvalidArgument);
  }
  next_frame_id_ = start_frame_id;
  frame_id_increment_ = increment;
  return {};
}

void PoseTree::deinit() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  frames_stack_.reset(nullptr);
  histories_map_ = nvidia::UniqueIndexMap<PoseTreeEdgeHistory>();
  histories_management_ = FirstFitAllocator<history_t>();
  poses_management_ = FirstFitAllocator<PoseTreeEdgeHistory::TimedPose>();
  edges_map_keys_.clear();
  name_to_uid_map_keys_.clear();
  create_frame_callbacks_keys_.clear();
  set_edge_callbacks_keys_.clear();
}

PoseTree::version_t PoseTree::get_pose_tree_version() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return version_;
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_frame(std::string_view name) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return find_frame_impl(name);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_frame_impl(std::string_view name) const {
  const auto it = name_to_uid_map_.get(name);
  if (it) {
    return it.value();
  }
  return unexpected_t(Error::kFrameNotFound);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_or_create_frame(std::string_view name) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return find_or_create_frame_impl(name, default_number_edges_);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_or_create_frame(std::string_view name,
                                                                       const int32_t number_edges) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return find_or_create_frame_impl(name, number_edges);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::find_or_create_frame_impl(
    std::string_view name, const int32_t number_edges) {
  if (auto uid = find_frame_impl(name)) {
    return uid;
  }
  return create_frame_impl(name, number_edges, nullptr);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame_with_id(const frame_t id,
                                                                       std::string_view name) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_frame_impl(name, default_number_edges_, &id);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame(std::string_view name,
                                                               const int32_t number_edges) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_frame_impl(name, number_edges, nullptr);
}
PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame(std::string_view name) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_frame_impl(name, default_number_edges_, nullptr);
}
PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame(const int32_t number_edges) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_frame_impl({}, number_edges, nullptr);
}
PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_frame_impl({}, default_number_edges_, nullptr);
}

PoseTree::expected_t<PoseTree::frame_t> PoseTree::create_frame_impl(std::string_view name,
                                                                    const int32_t number_edges,
                                                                    const frame_t* id) {
  if (frame_map_.size() == frame_map_.capacity()) {
    return unexpected_t(Error::kOutOfMemory);
  }
  if (!name.empty()) {
    // Check the name does not start with the reserved character
    if (name[0] == kAutoGeneratedFrameNamePrefix[0] || name.size() > kFrameNameMaximumLength) {
      return unexpected_t(Error::kInvalidArgument);
    }
    // Checks it does not exist yet.
    if (find_frame_impl(name)) {
      HOLOSCAN_LOG_ERROR("Frame {} already exists.", name);
      return unexpected_t(Error::kAlreadyExists);
    }
  }
  frame_t uid{};
  if (id == nullptr) {
    for (int i = 0; i <= frame_map_.size(); i++) {
      uid = next_frame_id_;
      next_frame_id_ += frame_id_increment_;
      if (!frame_map_.has(uid)) {
        break;
      }
      if (i == frame_map_.size()) {
        // This can only happen if the increment * number_frames is bigger than 2^64
        return unexpected_t(Error::kLogicError);
      }
    }
  } else {
    if (*id != 0 && frame_map_.has(*id)) {
      HOLOSCAN_LOG_DEBUG("ID {} requested for '{}' conflicts with existing frame '{}'",
                         *id,
                         name,
                         frame_map_.get(*id).value().name);
      return unexpected_t(Error::kAlreadyExists);
    }
    uid = *id;
  }
  const auto history = histories_management_.acquire(number_edges);
  if (!history) {
    return unexpected_t(Error::kOutOfMemory);
  }
  const auto maybe_frame = frame_map_.insert(uid, {});
  if (!maybe_frame) {
    histories_management_.release(history.value().first);
    frame_map_.erase(uid);
    return unexpected_t(Error::kLogicError);
  }
  FrameInfo* frame = maybe_frame.value();
  frame->history = history.value().first;
  frame->maximum_number_edges = history.value().second;
  frame->number_edges = 0;
  frame->hint_version = version_;
  frame->distance_to_root = 0;
  frame->node_to_root = uid;
  frame->uid = uid;
  frame->root = frame->uid;
  if (!name.empty()) {
    for (size_t index = 0; index < name.size(); index++) {
      frame->name[index] = name[index];
    }
    frame->name_view = std::string_view(frame->name, name.size());
  } else {
    // Auto generate a name
    const int length_required =
        snprintf(nullptr, 0, "%s%zu", kAutoGeneratedFrameNamePrefix, frame->uid);
    if (length_required > kFrameNameMaximumLength) {
      histories_management_.release(history.value().first);
      frame_map_.erase(uid);
      return unexpected_t(Error::kOutOfMemory);
    }
    const int length_used = snprintf(
        frame->name, length_required + 1, "%s%zu", kAutoGeneratedFrameNamePrefix, frame->uid);
    if (length_required != length_used) {
      histories_management_.release(history.value().first);
      frame_map_.erase(uid);
      return unexpected_t(Error::kLogicError);
    }
    frame->name_view = std::string_view(frame->name, length_used);
  }
  name_to_uid_map_.insert(frame->name_view, frame->uid);
  name_to_uid_map_keys_.push_back(frame->name_view);
  {
    // Release the lock to avoid deadlock in the callback
    mutex_.unlock();
    // Call each registered CreateFrameCallback callback
    std::shared_lock<std::shared_timed_mutex> callbacks_lock(create_frame_callbacks_mutex_);
    for (const auto& cid : create_frame_callbacks_keys_) {
      if (!cid) {
        continue;
      }
      auto cb = create_frame_callbacks_.try_get(cid.value());
      if (cb) {
        (*cb.value())(frame->uid);
      }
    }
    // TODO(bbutin): Find a better way, we should not need to acquire the lock again as all the code
    // is safe below that point beside the lock_guard unlocking (requiring it to be locked).
    // Acquire the lock back to matain a valid state.
    mutex_.lock();
  }

  return frame->uid;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const frame_t lhs,
                                                                 const frame_t rhs,
                                                                 const int32_t maximum_length) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, maximum_length, PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const frame_t lhs,
                                                                 const frame_t rhs) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(
      lhs, rhs, default_history_length_, PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const frame_t lhs, const frame_t rhs, const int32_t maximum_length,
    PoseTreeEdgeHistory::AccessMethod method) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, maximum_length, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const frame_t lhs, const frame_t rhs, PoseTreeEdgeHistory::AccessMethod method) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return create_edges_impl(lhs, rhs, default_history_length_, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const std::string_view lhs,
                                                                 const std::string_view rhs,
                                                                 const int32_t maximum_length) {
  return create_edges(lhs, rhs, maximum_length, PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(const std::string_view lhs,
                                                                 const std::string_view rhs) {
  return create_edges(
      lhs, rhs, default_history_length_, PoseTreeEdgeHistory::AccessMethod::kDefault);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const std::string_view lhs, const std::string_view rhs,
    PoseTreeEdgeHistory::AccessMethod method) {
  return create_edges(lhs, rhs, default_history_length_, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges(
    const std::string_view lhs, const std::string_view rhs, const int32_t maximum_length,
    PoseTreeEdgeHistory::AccessMethod method) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto lhs_frame = find_frame_impl(lhs);
  if (!lhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto rhs_frame = find_frame_impl(rhs);
  if (!rhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  return create_edges_impl(lhs_frame.value(), rhs_frame.value(), maximum_length, method);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::create_edges_impl(
    frame_t lhs, frame_t rhs, const int32_t maximum_length,
    PoseTreeEdgeHistory::AccessMethod method) {
  if (lhs > rhs) {
    std::swap(lhs, rhs);
  }
  // Make sure the frames exists
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  // Make sure the edges do not exist yet
  if (edges_map_.has({lhs, rhs})) {
    return unexpected_t(Error::kAlreadyExists);
  }
  // Check both frame have room for one more edge
  FrameInfo* lhs_frame = lhs_it.value();
  FrameInfo* rhs_frame = rhs_it.value();
  if (lhs_frame->number_edges == lhs_frame->maximum_number_edges ||
      rhs_frame->number_edges == rhs_frame->maximum_number_edges) {
    return unexpected_t(Error::kOutOfMemory);
  }
  const auto lhs_edge = poses_management_.acquire(maximum_length);
  if (!lhs_edge) {
    return unexpected_t(Error::kOutOfMemory);
  }
  const auto uid =
      method == PoseTreeEdgeHistory::AccessMethod::kDefault
          ? histories_map_.emplace(lhs, rhs, lhs_edge.value().second, lhs_edge.value().first)
          : histories_map_.emplace(
                lhs, rhs, lhs_edge.value().second, method, lhs_edge.value().first);
  if (!uid) {
    poses_management_.release(lhs_edge.value().first);
    return unexpected_t(Error::kOutOfMemory);
  }
  auto res = edges_map_.insert({lhs, rhs}, uid.value());
  if (!res) {
    poses_management_.release(lhs_edge.value().first);
    return unexpected_t(Error::kOutOfMemory);
  }
  edges_map_keys_.push_back({lhs, rhs});
  lhs_frame->history[lhs_frame->number_edges++] = uid.value();
  rhs_frame->history[rhs_frame->number_edges++] = uid.value();
  return ++version_;
}

// TODO(ben): This function could be smarter, calling delete_edge_impl on each edge require multiple
// access to a map + rerooting many time. This operation is quite rare, so for now we can keep a
// simpler implementation but we should consider making it smarter.
PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_frame(const frame_t uid) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto uid_it = frame_map_.try_get(uid);
  if (!uid_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  FrameInfo* frame = uid_it.value();
  while (frame->number_edges) {
    const auto history = histories_map_.try_get(frame->history[frame->number_edges - 1]);
    if (!history) {
      // This should never happen, assert?
      return unexpected_t(Error::kLogicError);
    }
    delete_edge_impl(history.value()->lhs(), history.value()->rhs(), version_);
  }

  if (!histories_management_.release(frame->history)) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }

  name_to_uid_map_.erase(frame->name_view);
  for (size_t i = 0; i < name_to_uid_map_keys_.size(); i++) {
    const auto key = name_to_uid_map_keys_[i];
    if (!key) {
      return unexpected_t(Error::kLogicError);
    }
    if (key.value() == frame->name_view) {
      name_to_uid_map_keys_.erase(i);
      break;
    }
  }
  frame_map_.erase(uid);

  return ++version_;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_frame(const std::string_view name) {
  frame_t uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_uid = find_frame_impl(name);
    if (!maybe_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    uid = maybe_uid.value();
  }
  return delete_frame(uid);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_edge(const frame_t lhs,
                                                                const frame_t rhs) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  return delete_edge_impl(lhs, rhs, ++version_);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_edge(const std::string_view lhs,
                                                                const std::string_view rhs) {
  frame_t lhs_uid, rhs_uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_lhs_uid = find_frame_impl(lhs);
    if (!maybe_lhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    const auto maybe_rhs_uid = find_frame_impl(rhs);
    if (!maybe_rhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    lhs_uid = maybe_lhs_uid.value();
    rhs_uid = maybe_rhs_uid.value();
  }
  return delete_edge(lhs_uid, rhs_uid);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::delete_edge_impl(frame_t lhs, frame_t rhs,
                                                                     const version_t version) {
  if (lhs > rhs) {
    std::swap(lhs, rhs);
  }
  // Make sure both the frame and edge exists.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto lhs_rhs = edges_map_.get({lhs, rhs});
  if (!lhs_rhs) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  // Helper pointer to the frame infos.
  FrameInfo* lhs_frame = lhs_it.value();
  FrameInfo* rhs_frame = rhs_it.value();

  const auto history = histories_map_.try_get(lhs_rhs.value());
  if (!history) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  // Free the memory
  if (!poses_management_.release(history.value()->data())) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }

  // Move the last edge to the place where lhs/rhs is stored and then remove the last edge.
  for (int32_t id = --lhs_frame->number_edges;; id--) {
    if (id == -1) {
      return unexpected_t(Error::kLogicError);
    }
    if (lhs_frame->history[id] == lhs_rhs.value()) {
      lhs_frame->history[id] = lhs_frame->history[lhs_frame->number_edges];
      break;
    }
  }
  for (int32_t id = --rhs_frame->number_edges;; id--) {
    if (id == -1) {
      return unexpected_t(Error::kLogicError);
    }
    if (rhs_frame->history[id] == lhs_rhs.value()) {
      rhs_frame->history[id] = rhs_frame->history[rhs_frame->number_edges];
      break;
    }
  }

  edges_map_.erase({lhs, rhs});
  for (size_t i = 0; i < edges_map_keys_.size(); i++) {
    const auto key = edges_map_keys_[i];
    if (!key) {
      return unexpected_t(Error::kLogicError);
    }
    if (key.value().first == lhs && key.value().second == rhs) {
      edges_map_keys_.erase(i);
      break;
    }
  }

  // Recompute the root of the subtree which got disconnected.
  if (lhs_frame->node_to_root == rhs) {
    update_root(lhs);
  } else {
    update_root(rhs);
  }

  return version;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_edge(frame_t lhs, frame_t rhs,
                                                                    const double time) {
  if (lhs > rhs) {
    std::swap(lhs, rhs);
  }
  // Make sure both the frame and edge exists.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto lhs_rhs = edges_map_.get({lhs, rhs});
  if (!lhs_rhs) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  const auto history = histories_map_.try_get(lhs_rhs.value());
  if (!history) {
    // This should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  const auto result = history.value()->disconnect(time, version_ + 1);
  if (!result) {
    return unexpected_t(Error::kPoseOutOfOrder);
  }
  // Recompute the root of the subtree which got disconnected.
  if (lhs_it.value()->node_to_root == rhs) {
    update_root(lhs);
  } else {
    update_root(rhs);
  }
  return ++version_;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_edge(const std::string_view lhs,
                                                                    const std::string_view rhs,
                                                                    const double time) {
  frame_t lhs_uid, rhs_uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_lhs_uid = find_frame_impl(lhs);
    if (!maybe_lhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    const auto maybe_rhs_uid = find_frame_impl(rhs);
    if (!maybe_rhs_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    lhs_uid = maybe_lhs_uid.value();
    rhs_uid = maybe_rhs_uid.value();
  }
  return disconnect_edge(lhs_uid, rhs_uid, time);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_frame(const frame_t uid,
                                                                     const double time) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto uid_it = frame_map_.try_get(uid);
  if (!uid_it) {
    HOLOSCAN_LOG_WARN("No pose frame found with UID {}", uid);
    return unexpected_t(Error::kFrameNotFound);
  }
  FrameInfo* frame = uid_it.value();
  // Go through all the edges and attempt to delete them
  bool failed = false;
  for (int32_t edge = 0; edge < frame->number_edges; edge++) {
    const auto history = histories_map_.try_get(frame->history[edge]);
    if (!history) {
      // This should never happen, assert?
      return unexpected_t(Error::kLogicError);
    }
    if (!history.value()->disconnect(time, version_ + 1)) {
      failed = true;
      continue;
    }
    if (history.value()->rhs() == uid) {
      update_root(history.value()->lhs());
    } else {
      update_root(history.value()->rhs());
    }
  }
  update_root(uid);
  if (failed) {
    return unexpected_t(Error::kPoseOutOfOrder);
  }
  return ++version_;
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::disconnect_frame(const std::string_view name,
                                                                     const double time) {
  frame_t uid;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_uid = find_frame_impl(name);
    if (!maybe_uid) {
      return unexpected_t(Error::kFrameNotFound);
    }
    uid = maybe_uid.value();
  }
  return disconnect_frame(uid, time);
}

PoseTree::expected_t<std::string_view> PoseTree::get_frame_name(const frame_t uid) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto uid_it = frame_map_.try_get(uid);
  if (!uid_it) {
    HOLOSCAN_LOG_WARN("No pose frame found with UID {}", uid);
    return unexpected_t(Error::kFrameNotFound);
  }
  return uid_it.value()->name_view;
}

PoseTree::expected_t<PoseTree::InitParameters> PoseTree::get_init_parameters() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (frames_stack_ == nullptr) {
    return unexpected_t(Error::kLogicError);
  }
  return init_params_;
}

PoseTree::expected_t<void> PoseTree::update_root(const frame_t root) {
  int32_t size = 0;
  frame_t* stack = frames_stack_.get();
  auto it = frame_map_.try_get(root);
  if (!it || stack == nullptr) {
    return unexpected_t(Error::kLogicError);
  }
  it.value()->node_to_root = root;
  it.value()->root = root;
  it.value()->distance_to_root = 0;
  it.value()->hint_version = hint_version_;
  stack[size++] = root;
  bool valid = true;
  // DFS implementation
  while (size > 0) {
    auto* frame = frame_map_.try_get(stack[--size]).value();
    // Go through all the edges
    for (int edge = 0; edge < frame->number_edges; edge++) {
      const auto history = histories_map_.try_get(frame->history[edge]);
      if (!history) {
        continue;
      }
      // Ignore not connected edges
      if (!history.value()->connected()) {
        continue;
      }
      it = frame_map_.try_get(history.value()->rhs() == frame->uid ? history.value()->lhs()
                                                                   : history.value()->rhs());
      if (!it) {
        valid = false;
        continue;
      }
      // Make sure we haven't visited this edge before (it means it is where we come from).
      if (it.value()->hint_version == hint_version_) {
        continue;
      }
      // Update the information
      it.value()->root = root;
      it.value()->hint_version = hint_version_;
      it.value()->node_to_root = frame->uid;
      it.value()->distance_to_root = 1 + frame->distance_to_root;
      // Add to the stack.
      stack[size++] = it.value()->uid;
    }
  }
  hint_version_++;
  if (!valid) {
    // Should never happen, assert?
    return unexpected_t(Error::kLogicError);
  }
  return expected_t<void>{};
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::set(std::string_view lhs, std::string_view rhs,
                                                        const double time,
                                                        const Pose3d& lhs_T_rhs) {
  frame_t lhs_frame, rhs_frame;
  {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    const auto maybe_lhs_frame = find_frame_impl(lhs);
    if (!maybe_lhs_frame) {
      HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
      return unexpected_t(Error::kFrameNotFound);
    }
    lhs_frame = maybe_lhs_frame.value();
    const auto maybe_rhs_frame = find_frame_impl(rhs);
    if (!maybe_rhs_frame) {
      HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
      return unexpected_t(Error::kFrameNotFound);
    }
    rhs_frame = maybe_rhs_frame.value();
  }
  return set(lhs_frame, rhs_frame, time, lhs_T_rhs);
}

PoseTree::expected_t<PoseTree::version_t> PoseTree::set(const frame_t lhs, const frame_t rhs,
                                                        const double time,
                                                        const Pose3d& lhs_T_rhs) {
  if (lhs > rhs) {
    return set(rhs, lhs, time, lhs_T_rhs.inverse());
  }
  if (!lhs_T_rhs.translation.allFinite() ||
      !is_almost_one(lhs_T_rhs.rotation.quaternion().squaredNorm())) {
    HOLOSCAN_LOG_WARN("Invalid pose between: %zu and %zu", lhs, rhs);
    return unexpected_t(Error::kInvalidArgument);
  }
  // Will contain the version to return
  version_t version;
  {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    // Make sure both the frame and edge exists.
    const auto lhs_it = frame_map_.try_get(lhs);
    const auto rhs_it = frame_map_.try_get(rhs);
    if (!lhs_it || !rhs_it) {
      HOLOSCAN_LOG_WARN("Pose frame UID {} or UID {} not found", lhs, rhs);
      return unexpected_t(Error::kFrameNotFound);
    }
    // Helper pointer to the frame infos.
    FrameInfo* lhs_frame = lhs_it.value();
    FrameInfo* rhs_frame = rhs_it.value();

    auto lhs_rhs = edges_map_.get({lhs, rhs});
    // If the history does not exist yet, let's create it
    if (!lhs_rhs) {
      const auto result = create_edges_impl(
          lhs, rhs, default_history_length_, PoseTreeEdgeHistory::AccessMethod::kDefault);
      // It might fail due to a loop or memory issue.
      if (!result) {
        HOLOSCAN_LOG_DEBUG("Unable to set pose: Unable to create edges for '{}' ({}) -> '{}' ({})",
                           lhs_frame->name_view,
                           lhs,
                           rhs_frame->name_view,
                           rhs);
        return result;
      }
      lhs_rhs = edges_map_.get({lhs, rhs});
      if (!lhs_rhs) {
        HOLOSCAN_LOG_DEBUG(
            "Unable to set pose: Unable to get edge map item for '{}' ({}) -> '{}' ({})",
            lhs_frame->name_view,
            lhs,
            rhs_frame->name_view,
            rhs);
        // Safety check, but this should never happen.
        return unexpected_t(Error::kLogicError);
      }
    }
    // Check if we are creating a new connection, if so we will need to update the connected
    // component and we should detect loop.
    auto history = histories_map_.try_get(lhs_rhs.value());
    if (!history) {
      HOLOSCAN_LOG_DEBUG("Unable to set pose: Unable to get history for '{}' ({}) -> '{}' ({})",
                         lhs_frame->name_view,
                         lhs,
                         rhs_frame->name_view,
                         rhs);
      // Safety check, but this should never happen.
      return unexpected_t(Error::kLogicError);
    }
    const bool need_update = !history.value()->connected();
    if (need_update) {
      // The edge is disconnected at the moment, so we need to make sure there is no other
      // connection.
      if (lhs_frame->root == rhs_frame->root ||
          get_impl(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kNearest, version_)) {
        HOLOSCAN_LOG_DEBUG(
            "Unable to set pose: cycling dependency detected for '{}' ({}) -> '{}' ({})",
            lhs_frame->name_view,
            lhs,
            rhs_frame->name_view,
            rhs);
        return unexpected_t(Error::kCyclingDependency);
      }
    }
    // Store the pose in both histories
    const auto result = history.value()->set(time, lhs_T_rhs, version_ + 1);
    if (!result) {
      HOLOSCAN_LOG_DEBUG("Unable to set pose: out of order for '{}' ({}) -> '{}' ({})",
                         lhs_frame->name_view,
                         lhs,
                         rhs_frame->name_view,
                         rhs);
      return unexpected_t(Error::kPoseOutOfOrder);
    }
    // Connect the components if needed
    if (need_update) {
      update_root(lhs);
    }
    version = ++version_;
  }

  {
    // Call each registered SetEdgeCallback function
    std::shared_lock<std::shared_timed_mutex> callbacks_lock(set_edge_callbacks_mutex_);
    for (const auto& cid : set_edge_callbacks_keys_) {
      if (!cid) {
        continue;
      }
      auto cb = set_edge_callbacks_.try_get(cid.value());
      if (cb) {
        (*cb.value())(lhs, rhs, time, lhs_T_rhs);
      }
    }
  }

  return version;
}

PoseTree::expected_t<std::pair<Pose3d, double>> PoseTree::get_latest(std::string_view lhs,
                                                                     std::string_view rhs) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto lhs_frame = find_frame_impl(lhs);
  if (!lhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto rhs_frame = find_frame_impl(rhs);
  if (!rhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  return get_latest(lhs_frame.value(), rhs_frame.value());
}

PoseTree::expected_t<std::pair<Pose3d, double>> PoseTree::get_latest(frame_t lhs,
                                                                     frame_t rhs) const {
  if (lhs > rhs) {
    const auto result = get_latest(rhs, lhs);
    if (!result) {
      return result;
    }
    return std::make_pair(result.value().first.inverse(), result.value().second);
  }
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  // Make sure both the frame and edge exists.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    HOLOSCAN_LOG_WARN("Pose frame UID {} or UID {} not found", lhs, rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto lhs_rhs = edges_map_.get({lhs, rhs});
  // Check the edge exist
  if (!lhs_rhs) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  const auto history = histories_map_.try_get(lhs_rhs.value());
  if (!history) {
    // Safety check, but this should never happen.
    return unexpected_t(Error::kLogicError);
  }
  const auto timed_pose = history.value()->latest();
  if (!timed_pose) {
    return unexpected_t(Error::kInvalidArgument);
  }
  return std::make_pair(timed_pose.value().pose, timed_pose.value().time);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  const auto lhs_frame = find_frame_impl(lhs);
  if (!lhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", lhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  const auto rhs_frame = find_frame_impl(rhs);
  if (!rhs_frame) {
    HOLOSCAN_LOG_WARN("Pose frame {} not found", rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  return get_impl(lhs_frame.value(), rhs_frame.value(), time, method, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time, const version_t version) const {
  return get(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method) const {
  return get(lhs, rhs, time, method, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(std::string_view lhs, std::string_view rhs,
                                           const double time) const {
  return get(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const std::string_view lhs,
                                           const std::string_view rhs) const {
  return get(lhs,
             rhs,
             std::numeric_limits<double>::max(),
             PoseTreeEdgeHistory::AccessMethod::kPrevious,
             version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const std::string_view lhs, const std::string_view rhs,
                                           const version_t version) const {
  return get(lhs,
             rhs,
             std::numeric_limits<double>::max(),
             PoseTreeEdgeHistory::AccessMethod::kPrevious,
             version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs, const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, method, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs, const double time,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs, const double time,
                                           const PoseTreeEdgeHistory::AccessMethod method) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, method, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs,
                                           const double time) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs, rhs, time, PoseTreeEdgeHistory::AccessMethod::kDefault, version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs,
                  rhs,
                  std::numeric_limits<double>::max(),
                  PoseTreeEdgeHistory::AccessMethod::kPrevious,
                  version_);
}

PoseTree::expected_t<Pose3d> PoseTree::get(const frame_t lhs, const frame_t rhs,
                                           const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return get_impl(lhs,
                  rhs,
                  std::numeric_limits<double>::max(),
                  PoseTreeEdgeHistory::AccessMethod::kPrevious,
                  version);
}

PoseTree::expected_t<Pose3d> PoseTree::get_impl(const frame_t lhs, const frame_t rhs,
                                                const double time,
                                                const PoseTreeEdgeHistory::AccessMethod method,
                                                const version_t version) const {
  if (lhs == rhs) {
    return Pose3d::identity();
  }
  if (lhs > rhs) {
    const auto result = get_impl(rhs, lhs, time, method, version);
    if (!result) {
      return result;
    }
    return result.value().inverse();
  }
  // Make sure both the frame exist.
  const auto lhs_it = frame_map_.try_get(lhs);
  const auto rhs_it = frame_map_.try_get(rhs);
  if (!lhs_it || !rhs_it) {
    HOLOSCAN_LOG_WARN("Pose frame UID {} or UID {} not found", lhs, rhs);
    return unexpected_t(Error::kFrameNotFound);
  }
  // Helper pointer to the frame infos.
  const FrameInfo* lhs_frame = lhs_it.value();
  const FrameInfo* rhs_frame = rhs_it.value();
  // Check if the hint will succeed.
  if (lhs_frame->root != rhs_frame->root) {
    return get_dfs_impl(lhs, rhs, time, method, version);
  }
  Pose3d lhs_T_parent = Pose3d::identity();
  Pose3d rhs_T_parent = Pose3d::identity();

  const auto update_pose = [&](const FrameInfo** frame, Pose3d& pose) {
    const auto next = frame_map_.try_get((*frame)->node_to_root);
    if (!next) {
      return false;
    }
    bool inverse = false;
    std::pair<frame_t, frame_t> key;
    if ((*frame)->uid < next.value()->uid) {
      key = {(*frame)->uid, next.value()->uid};
    } else {
      key = {next.value()->uid, (*frame)->uid};
      inverse = true;
    }
    const auto history_it = edges_map_.get(key);
    if (!history_it) {
      return false;
    }
    const auto history = histories_map_.try_get(history_it.value());
    if (!history) {
      return false;
    }
    const auto current_T_next = history.value()->get(time, method, version);
    if (!current_T_next) {
      return false;
    }
    pose = pose * (inverse ? current_T_next.value().inverse() : current_T_next.value());
    *frame = next.value();
    return true;
  };

  // Move lhs to the same distance as rhs
  while (lhs_frame->distance_to_root > rhs_frame->distance_to_root) {
    if (!update_pose(&lhs_frame, lhs_T_parent)) {
      return get_dfs_impl(lhs, rhs, time, method, version);
    }
  }
  // Move rhs to the same distance as lhs
  while (lhs_frame->distance_to_root < rhs_frame->distance_to_root) {
    if (!update_pose(&rhs_frame, rhs_T_parent)) {
      return get_dfs_impl(lhs, rhs, time, method, version);
    }
  }
  // Move in parallel until lhs == rhs
  while (lhs_frame->uid != rhs_frame->uid) {
    if (!update_pose(&lhs_frame, lhs_T_parent) || !update_pose(&rhs_frame, rhs_T_parent)) {
      return get_dfs_impl(lhs, rhs, time, method, version);
    }
  }

  return lhs_T_parent * rhs_T_parent.inverse();
}

PoseTree::expected_t<PoseTree::uid_t> PoseTree::add_create_frame_callback(
    CreateFrameCallback callback) {
  std::unique_lock<std::shared_timed_mutex> lock(create_frame_callbacks_mutex_);
  const auto cid = create_frame_callbacks_.insert(callback);
  if (!cid) {
    return unexpected_t(Error::kOutOfMemory);
  }
  create_frame_callbacks_keys_.push_back(cid.value());
  return cid.value();
}

PoseTree::expected_t<void> PoseTree::remove_create_frame_callback(uid_t cid) {
  std::unique_lock<std::shared_timed_mutex> lock(create_frame_callbacks_mutex_);
  // If the cid does not have an associated callback function, return an error
  auto erase = create_frame_callbacks_.erase(cid);
  if (!erase) {
    return unexpected_t(Error::kInvalidArgument);
  }
  for (size_t i = 0; i < create_frame_callbacks_keys_.size(); i++) {
    auto cid = create_frame_callbacks_keys_.at(i);
    if (!cid) {
      return unexpected_t(Error::kLogicError);
    }
    if (cid.value() == cid) {
      auto last = create_frame_callbacks_keys_.back();
      if (!last) {
        return unexpected_t(Error::kLogicError);
      }
      cid.value() = last.value();
      create_frame_callbacks_keys_.pop_back();
      return expected_t<void>{};
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

PoseTree::expected_t<PoseTree::uid_t> PoseTree::add_set_edge_callback(SetEdgeCallback callback) {
  std::unique_lock<std::shared_timed_mutex> lock(set_edge_callbacks_mutex_);
  const auto cid = set_edge_callbacks_.insert(callback);
  if (!cid) {
    return unexpected_t(Error::kOutOfMemory);
  }
  set_edge_callbacks_keys_.push_back(cid.value());
  return cid.value();
}

PoseTree::expected_t<void> PoseTree::remove_set_edge_callback(uid_t cid) {
  std::unique_lock<std::shared_timed_mutex> lock(set_edge_callbacks_mutex_);
  // If the cid does not have an associated callback function, return an error
  auto erase = set_edge_callbacks_.erase(cid);
  if (!erase) {
    return unexpected_t(Error::kInvalidArgument);
  }
  for (size_t i = 0; i < set_edge_callbacks_keys_.size(); i++) {
    auto cid = set_edge_callbacks_keys_.at(i);
    if (!cid) {
      return unexpected_t(Error::kLogicError);
    }
    if (cid.value() == cid) {
      auto last = set_edge_callbacks_keys_.back();
      if (!last) {
        return unexpected_t(Error::kLogicError);
      }
      cid.value() = last.value();
      set_edge_callbacks_keys_.pop_back();
      return expected_t<void>{};
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

PoseTree::expected_t<Pose3d> PoseTree::get_dfs_impl(const frame_t lhs, const frame_t rhs,
                                                    const double time,
                                                    const PoseTreeEdgeHistory::AccessMethod method,
                                                    const version_t version) const {
  std::unique_lock<std::mutex> lock(dfs_mutex_);
  int32_t size = 0;
  frame_t* stack = frames_stack_.get();
  auto it = frame_map_.try_get(rhs);
  if (!it) {
    return unexpected_t(Error::kLogicError);
  }
  it.value()->hint_version = hint_version_;
  stack[size++] = rhs;
  // DFS implementation
  while (size > 0) {
    auto* frame = frame_map_.try_get(stack[--size]).value();
    if (frame->uid == lhs) {
      Pose3d pose = Pose3d::identity();
      while (frame->uid != rhs) {
        if (frame->uid < frame->dfs_link) {
          const auto history_it = edges_map_.get({frame->uid, frame->dfs_link});
          if (!history_it) {
            return unexpected_t(Error::kLogicError);
          }
          const auto history = histories_map_.try_get(history_it.value());
          if (!history) {
            return unexpected_t(Error::kLogicError);
          }
          pose = pose * history.value()->get(time, method, version).value();
        } else {
          const auto history_it = edges_map_.get({frame->dfs_link, frame->uid});
          if (!history_it) {
            return unexpected_t(Error::kLogicError);
          }
          const auto history = histories_map_.try_get(history_it.value());
          if (!history) {
            return unexpected_t(Error::kLogicError);
          }
          pose = pose * history.value()->get(time, method, version).value().inverse();
        }
        frame = frame_map_.try_get(frame->dfs_link).value();
      }
      hint_version_++;
      return pose;
    }
    // Go through all the edges
    for (int edge = 0; edge < frame->number_edges; edge++) {
      const auto history = histories_map_.try_get(frame->history[edge]);
      if (!history) {
        continue;
      }
      // Ignore not connected edges
      if (!history.value()->get(time, method, version)) {
        continue;
      }
      it = frame_map_.try_get(history.value()->rhs() == frame->uid ? history.value()->lhs()
                                                                   : history.value()->rhs());
      if (!it) {
        continue;
      }
      // Make sure we haven't visited this edge before (it means it is where we come from).
      if (it.value()->hint_version == hint_version_) {
        continue;
      }
      // Update the information
      it.value()->hint_version = hint_version_;
      it.value()->dfs_link = frame->uid;
      // Add to the stack.
      stack[size++] = it.value()->uid;
    }
  }
  hint_version_++;
  return unexpected_t(Error::kFramesNotLinked);
}

}  // namespace holoscan
