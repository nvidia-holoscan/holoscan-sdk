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
#ifndef HOLOSCAN_POSE_TREE_POSE_TREE_HISTORY_HPP
#define HOLOSCAN_POSE_TREE_POSE_TREE_HISTORY_HPP

#include <shared_mutex>

#include "holoscan/core/expected.hpp"
#include "holoscan/pose_tree/math/pose3.hpp"

namespace holoscan {

/**
 * @brief Class that stores a history of poses using pre-allocated memory and a cyclic buffer.
 *
 * Provides helper functions to add a new pose or query the pose at a given time. The history
 * uses a circular buffer to efficiently manage memory and provides various interpolation methods
 * for accessing poses at specific times.
 */
class PoseTreeEdgeHistory {
 public:
  /**
   * @brief Interpolation method for accessing poses in the pose tree.
   */
  enum class AccessMethod {
    /// Gets the value of the closest sample
    kNearest,
    /// Interpolates linearly between adjacent samples. If the query is outside the validity range,
    /// the closest pose will be returned.
    kInterpolateLinearly,
    /// Inter- or extrapolates linearly based on neighbouring samples. This require at least two
    /// valid poses or Error::kOutOfRange will be returned.
    kExtrapolateLinearly,
    /// Interpolates with slerp between adjacent samples. If the query is outside the validity
    /// range,
    /// the closest pose will be returned.
    kInterpolateSlerp,
    /// Inter- or extrapolates with slerp based on neighbouring samples. This require at least two
    /// valid poses or Error::kOutOfRange will be returned.
    kExtrapolateSlerp,
    /// Use the latest Pose before a given time.
    kPrevious,
    /// Fallback to the default interpolation
    kDefault,
  };

  /**
   * @brief Error codes used by this class.
   */
  enum class Error {
    /// kInvalidArgument is returned when a function is called with argument that does not make
    /// sense such as querying a pose outside the valid range or provide a wrong interpolation
    /// method.
    kInvalidArgument,
    /// kOutOfOrder is returned when a set/disconnected called is made with a version or time
    /// lower than the latest TimedPose. Both time and version must be strictly increasing.
    kOutOfOrder,
    /// kFramesNotLinked is returns if a get query is made and the tree is not connected at the
    /// given
    /// time and version of the edge.
    kFramesNotLinked,
    /// kOutOfRange is returned if not enough poses are available to do extrapolation or if the
    /// available buffer is too small to store a new pose.
    kOutOfRange,
  };

  /// Expected type used by this class.
  template <typename T>
  using expected_t = expected<T, Error>;
  /// Unexpected type used by this class.
  using unexpected_t = unexpected<Error>;

  /// Type used to uniquely identify a frame.
  using frame_t = uint64_t;
  /// Type used for versioning the edge.
  using version_t = uint64_t;

  /**
   * @brief Helper structure to store the pose at a given time on the edge.
   */
  struct TimedPose {
    /// 3D pose that transforms the lhs frame into the rhs frame.
    Pose3d pose;
    /// Time of the pose. Needs to be strictly increasing.
    double time;
    /// Version ID of the pose. Needs to be strictly increasing.
    version_t version;
    /// If false, then it marks the edge as being disconnected from this current time. The pose does
    /// not matter.
    bool valid;
  };

  /**
   * @brief Default constructor used to be able to pre-allocate memory.
   */
  PoseTreeEdgeHistory() = default;

  /**
   * @brief Constructor to create a usable PoseTreeEdgeHistory object.
   *
   * @param lhs Left hand side frame identifier.
   * @param rhs Right hand side frame identifier.
   * @param maximum_size Maximum number of poses this edge can store.
   * @param buffer Pre-allocated buffer that can hold `maximum_size` elements.
   */
  PoseTreeEdgeHistory(frame_t lhs, frame_t rhs, int32_t maximum_size, TimedPose* buffer);

  /**
   * @brief Constructor to create a usable PoseTreeEdgeHistory object with custom access method.
   *
   * @param lhs Left hand side frame identifier.
   * @param rhs Right hand side frame identifier.
   * @param maximum_size Maximum number of poses this edge can store.
   * @param access_method Default access method for interpolation.
   * @param buffer Pre-allocated buffer that can hold `maximum_size` elements.
   */
  PoseTreeEdgeHistory(frame_t lhs, frame_t rhs, int32_t maximum_size, AccessMethod access_method,
                      TimedPose* buffer);

  /**
   * @brief Set the pose at a given time.
   *
   * If the array is empty Error::kOutOfMemory will be returned, otherwise if a pose already
   * exists and `time` or `version` <= pose.time/version then Error::kOutOfOrder is returned.
   * Otherwise it will succeed, and if the history already contained maximum_size_ elements,
   * then the oldest pose will be forgotten.
   *
   * @param time Time at which to set the pose.
   * @param pose 3D pose transformation.
   * @param version Version ID for this pose update.
   * @return Success or error status.
   */
  expected_t<void> set(double time, const Pose3d& pose, version_t version);

  /**
   * @brief Get the TimedPose at a given position in the history.
   *
   * @param index Index of the pose to retrieve.
   * @return TimedPose on success, Error::kInvalidArgument if index is negative,
   *         Error::kOutOfRange if index >= size.
   */
  expected_t<TimedPose> at(int32_t index) const;

  /**
   * @brief Get the Pose3d at a given time using the given version of the PoseTree.
   *
   * If no pose existed at the given time, Error::kFramesNotLinked will be returned.
   * The desired method can be provided, for kExtrapolateLinearly, at least two poses are required.
   *
   * @param time Time at which to query the pose.
   * @param method Interpolation method to use.
   * @param version Version of the PoseTree to query.
   * @return Pose3d on success, error on failure.
   */
  expected_t<Pose3d> get(double time, AccessMethod method, version_t version) const;

  /**
   * @brief Disconnect a frame at a given time.
   *
   * @param time Time at which to disconnect the frames.
   * @param version Version ID for this disconnection.
   * @return Success or error status.
   */
  expected_t<void> disconnect(double time, version_t version);

  /**
   * @brief Check if the frames are currently connected.
   *
   * @return True if frames are connected, false otherwise.
   */
  bool connected() const;

  /**
   * @brief Reset the history, erasing all poses.
   */
  void reset();

  /**
   * @brief Get information about the latest pose.
   *
   * @return Latest TimedPose on success, error if no poses exist.
   */
  expected_t<TimedPose> latest() const;

  /**
   * @brief Get a pointer to the internal buffer.
   *
   * @return Const pointer to the TimedPose buffer.
   */
  const TimedPose* data() const { return edges_info_; }

  /**
   * @brief Get the current number of poses stored.
   *
   * @return Current size of the history.
   */
  int32_t size() const { return size_; }

  /**
   * @brief Get the maximum number of poses this edge can contain.
   *
   * @return Maximum size of the history buffer.
   */
  int32_t maximum_size() const { return maximum_size_; }

  /**
   * @brief Get the left hand side frame identifier.
   *
   * This edge represents the transformation from the rhs frame to the lhs frame.
   *
   * @return UID of the lhs frame.
   */
  frame_t lhs() const { return lhs_; }

  /**
   * @brief Get the right hand side frame identifier.
   *
   * This edge represents the transformation from the rhs frame to the lhs frame.
   *
   * @return UID of the rhs frame.
   */
  frame_t rhs() const { return rhs_; }

 private:
  /**
   * @brief Reserve a new pose for a given time/version and return its index.
   *
   * If there exists another pose with a later time/version, it returns kOutOfOrder.
   *
   * @param time Time for the new pose.
   * @param version Version for the new pose.
   * @return Index of the reserved pose on success, error on failure.
   */
  expected_t<int32_t> reserve_new_pose(double time, version_t version);

  /// Mutex to protect changes to this object.
  mutable std::shared_timed_mutex mutex_;
  /// Pointer to the buffer that contains the list of TimedPose (we use a circular buffer to access
  /// it).
  TimedPose* edges_info_ = nullptr;
  /// Name of the frame this edge connects to.
  frame_t lhs_, rhs_;
  /// Size of the buffer, aka maximum number of elements this edge can store at the same time.
  int32_t maximum_size_ = 0;
  /// Current number of poses on this edge.
  int32_t size_ = 0;
  /// Position of the first pose in the circular buffer.
  int32_t pos_ = 0;
  /// Default access method.
  AccessMethod default_access_method_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_POSE_TREE_HISTORY_HPP */
