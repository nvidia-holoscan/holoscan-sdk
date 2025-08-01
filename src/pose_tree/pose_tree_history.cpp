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
#include "holoscan/pose_tree/pose_tree_history.hpp"

#include <mutex>

#include "holoscan/pose_tree/math/interpolation.hpp"

namespace holoscan {

PoseTreeEdgeHistory::PoseTreeEdgeHistory(const frame_t lhs, const frame_t rhs,
                                         const int32_t maximum_size, AccessMethod access_method,
                                         TimedPose* buffer)
    : edges_info_(buffer),
      lhs_(lhs),
      rhs_(rhs),
      maximum_size_(maximum_size),
      size_(0),
      pos_(0),
      default_access_method_(access_method) {}

PoseTreeEdgeHistory::PoseTreeEdgeHistory(const frame_t lhs, const frame_t rhs,
                                         const int32_t maximum_size, TimedPose* buffer)
    : edges_info_(buffer),
      lhs_(lhs),
      rhs_(rhs),
      maximum_size_(maximum_size),
      size_(0),
      pos_(0),
      default_access_method_(AccessMethod::kInterpolateLinearly) {}

PoseTreeEdgeHistory::expected_t<void> PoseTreeEdgeHistory::disconnect(const double time,
                                                                      const version_t version) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto maybe_idx = reserve_new_pose(time, version);
  if (!maybe_idx) {
    return unexpected_t(maybe_idx.error());
  }
  const int32_t idx = maybe_idx.value();
  edges_info_[idx].valid = false;
  return expected_t<void>{};
}

PoseTreeEdgeHistory::expected_t<void> PoseTreeEdgeHistory::set(const double time,
                                                               const Pose3d& pose,
                                                               const version_t version) {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  const auto maybe_idx = reserve_new_pose(time, version);
  if (!maybe_idx) {
    return unexpected_t(maybe_idx.error());
  }
  const int32_t idx = maybe_idx.value();
  edges_info_[idx].pose = pose;
  edges_info_[idx].valid = true;
  return expected_t<void>{};
}

PoseTreeEdgeHistory::expected_t<int32_t> PoseTreeEdgeHistory::reserve_new_pose(
    const double time, const version_t version) {
  if (maximum_size_ <= 0 || edges_info_ == nullptr) {
    return unexpected_t(Error::kOutOfRange);
  }
  const int32_t last_idx = (pos_ + size_ - 1) % maximum_size_;
  if (size_ != 0 &&
      (edges_info_[last_idx].time >= time || edges_info_[last_idx].version >= version)) {
    return unexpected_t{Error::kOutOfOrder};
  }
  int32_t idx;
  if (size_ == maximum_size_) {
    idx = pos_++;
    if (pos_ == maximum_size_) {
      pos_ = 0;
    }
  } else {
    idx = size_++;
  }
  edges_info_[idx].time = time;
  edges_info_[idx].version = version;
  return idx;
}

PoseTreeEdgeHistory::expected_t<PoseTreeEdgeHistory::TimedPose> PoseTreeEdgeHistory::latest()
    const {
  if (size_ == 0) {
    return unexpected_t(Error::kOutOfRange);
  }
  return edges_info_[(pos_ + size_ - 1) % maximum_size_];
}

PoseTreeEdgeHistory::expected_t<PoseTreeEdgeHistory::TimedPose> PoseTreeEdgeHistory::at(
    const int32_t index) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (index < 0 || index >= size_) {
    return unexpected_t(Error::kInvalidArgument);
  }
  const int32_t idx = (pos_ + index) % maximum_size_;
  return edges_info_[idx];
}

PoseTreeEdgeHistory::expected_t<Pose3d> PoseTreeEdgeHistory::get(const double time,
                                                                 AccessMethod method,
                                                                 const version_t version) const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  if (size_ == 0 || edges_info_[pos_].version > version || edges_info_[pos_].time > time) {
    return unexpected_t(Error::kFramesNotLinked);
  }

  // Perform a binary search in the array to find the last pose which time is lower than `time`.
  // If none exist, then index will be equal to pose_
  int32_t start = pos_;        // included
  int32_t end = pos_ + size_;  // excluded
  while (start + 1 < end) {
    const int32_t mid = (start + end) / 2;
    const auto& mid_info = edges_info_[mid >= maximum_size_ ? mid - maximum_size_ : mid];
    if (mid_info.time > time || mid_info.version > version) {
      end = mid;
    } else {
      start = mid;
    }
  }
  const int32_t index = start % maximum_size_;
  if (!edges_info_[index].valid) {
    return unexpected_t(Error::kFramesNotLinked);
  }
  const int32_t next_index = (start + 1) % maximum_size_;

  if (method == AccessMethod::kDefault) {
    method = default_access_method_;
  }

  // If the valid pose is the last one, we handle separately.
  if (next_index == (pos_ + size_) % maximum_size_ || edges_info_[next_index].version > version ||
      !edges_info_[next_index].valid) {
    switch (method) {
      case AccessMethod::kNearest:
      case AccessMethod::kPrevious:
      case AccessMethod::kInterpolateSlerp:
      case AccessMethod::kInterpolateLinearly: {
        return edges_info_[index].pose;
      }
      case AccessMethod::kExtrapolateSlerp:
      case AccessMethod::kExtrapolateLinearly: {
        if (pos_ == start) {
          return unexpected_t(Error::kOutOfRange);
        }
        const auto& pose1 = edges_info_[(index + maximum_size_ - 1) % maximum_size_];
        if (!pose1.valid) {
          return unexpected_t(Error::kOutOfRange);
        }
        const auto& pose2 = edges_info_[index];
        const double delta_time = time - pose1.time;
        const double total_time = pose2.time - pose1.time;
        if (method == AccessMethod::kExtrapolateSlerp) {
          return slerp_interpolate(delta_time / total_time, pose1.pose, pose2.pose);
        } else {
          return interpolate(delta_time / total_time, pose1.pose, pose2.pose);
        }
      }
      case AccessMethod::kDefault: {
        return unexpected_t(Error::kInvalidArgument);
      }
    }
    return unexpected_t(Error::kInvalidArgument);
  }

  switch (method) {
    case AccessMethod::kPrevious: {
      return edges_info_[index].pose;
    }
    case AccessMethod::kNearest: {
      if (edges_info_[next_index].time - time < time - edges_info_[index].time) {
        return edges_info_[next_index].pose;
      } else {
        return edges_info_[index].pose;
      }
    }
    case AccessMethod::kInterpolateLinearly:
    case AccessMethod::kExtrapolateLinearly: {
      const double delta_time = time - edges_info_[index].time;
      const double total_time = edges_info_[next_index].time - edges_info_[index].time;
      return interpolate(
          delta_time / total_time, edges_info_[index].pose, edges_info_[next_index].pose);
    }
    case AccessMethod::kInterpolateSlerp:
    case AccessMethod::kExtrapolateSlerp: {
      const double delta_time = time - edges_info_[index].time;
      const double total_time = edges_info_[next_index].time - edges_info_[index].time;
      return slerp_interpolate(
          delta_time / total_time, edges_info_[index].pose, edges_info_[next_index].pose);
    }
    case AccessMethod::kDefault: {
      return unexpected_t(Error::kInvalidArgument);
    }
  }
  return unexpected_t(Error::kInvalidArgument);
}

bool PoseTreeEdgeHistory::connected() const {
  std::shared_lock<std::shared_timed_mutex> lock(mutex_);
  return size_ > 0 && edges_info_[(pos_ + size_ - 1) % maximum_size_].valid;
}

void PoseTreeEdgeHistory::reset() {
  std::unique_lock<std::shared_timed_mutex> lock(mutex_);
  pos_ = 0;
  size_ = 0;
}

}  // namespace holoscan
