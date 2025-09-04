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
#ifndef HOLOSCAN_POSE_TREE_POSE_TREE_HPP
#define HOLOSCAN_POSE_TREE_POSE_TREE_HPP

#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>  // NOLINT(build/include_order)
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/fixed_vector.hpp"
#include "common/unique_index_map.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/resources/gxf/first_fit_allocator.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/pose_tree/hash_map.hpp"
#include "holoscan/pose_tree/math/pose2.hpp"
#include "holoscan/pose_tree/math/pose3.hpp"
#include "holoscan/pose_tree/pose_tree_history.hpp"

namespace std {
/**
 * @brief Hash specialization for std::pair.
 *
 * @tparam A First type in the pair.
 * @tparam B Second type in the pair.
 */
template <class A, class B>
struct hash<pair<A, B>> {
  /**
   * @brief Compute hash value for a pair.
   *
   * @param p The pair to hash.
   * @return Hash value.
   */
  size_t operator()(const pair<A, B>& p) const {
    return std::hash<A>{}(p.first) ^ (std::hash<B>{}(p.second) << 1);
  }
};
}  // namespace std

namespace holoscan {

/**
 * @brief A temporal pose tree to store relative coordinate system transformations over time.
 *
 * This implementation does not support multiple paths between the same coordinate systems at a
 * given time. It does however allow to disconnect edge and create new connection using a different
 * path. It also allows for multiple "roots". In fact the transformation relationships form an
 * acylic, bi-directional, not necessarily fully-connected graph.
 * This PoseTree assigned a different version id to each operation that affects it, and this version
 * can be used to make a query ignore later changes made to the tree.
 */
class PoseTree {
 public:
  /**
   * @brief Error codes used by this class.
   */
  enum class Error {
    /// kInvalidArgument is returned when a function is called with argument that does not make
    /// sense
    /// such as negative number of frames.
    kInvalidArgument = 0,
    /// kOutOfMemory is returned if `initialize` failed to allocate the requested memory, or if an
    /// edge/frame can't be added because we run out of the pre-allocated memory.
    kOutOfMemory = 1,
    /// kFrameNotFound is returned if a query is made with a frame uid that does not match any
    /// existing frame.
    kFrameNotFound = 2,
    /// kAlreadyExists is returned if a frame or an edge that already exist is added.
    kAlreadyExists = 3,
    /// kCyclingDependency is returned if a pose is added that would create a cycle in the PoseTree
    /// structure.
    kCyclingDependency = 4,
    /// kFramesNotLinked is returned if a query is made between two not connected frame or if we
    /// attempt to disconnect/delete an edge that does not exist.
    kFramesNotLinked = 5,
    /// kPoseOutOfOrder is returned if a query is made to update the three in the past. For example
    /// if we try to disconnect or update a pose at a time older than the latest update on this
    /// edge.
    kPoseOutOfOrder = 6,
    /// kLogicError is used whenever an error that should not have happened happened. This should
    /// never happen and are here only to prevent crashes/assert,
    kLogicError = 7,
  };

  /**
   * @brief Parameters used to initialize the PoseTree.
   */
  struct InitParameters {
    /**
     * @brief Maximum number of frames to support.
     */
    int32_t number_frames;
    /**
     * @brief Maximum number of edges to support.
     */
    int32_t number_edges;
    /**
     * @brief Maximum history length.
     */
    int32_t history_length;
    /**
     * @brief Default number of edges per frame.
     */
    int32_t default_number_edges;
    /**
     * @brief Default history length per edge.
     */
    int32_t default_history_length;
    /**
     * @brief Chunk size for edge allocation.
     */
    int32_t edges_chunk_size;
    /**
     * @brief Chunk size for history allocation.
     */
    int32_t history_chunk_size;
  };

  /// The maximum size for the name of a frame. An additional '\0' is added at the end to make it
  /// 128 characters long.
  static constexpr int32_t kFrameNameMaximumLength = 127;

  /// Auto generated frame names will start with this prefix, followed by the uid of the frame.
  static constexpr char const* kAutoGeneratedFrameNamePrefix = "_frame_";

  /// Expected type used by this class.
  template <typename T>
  using expected_t = nvidia::Expected<T, Error>;
  /// Unexpected type used by this class.
  using unexpected_t = nvidia::Unexpected<Error>;

  /// Type used to uniquely identify a frame.
  using frame_t = uint64_t;
  /// Type used for versioning the PoseTree.
  using version_t = uint64_t;
  /// Type used as a key for the PoseTreeEdgeHistory map.
  using history_t = uint64_t;

  /// Type used as a key for the PoseTreeEdgeHistory map.
  using uid_t = uint64_t;

  /// Type for callback functions that are called every time a frame is created.
  using CreateFrameCallback = std::function<void(frame_t frame)>;
  /// Type for callback functions that are called every time an edge is set.
  using SetEdgeCallback =
      std::function<void(frame_t lhs, frame_t rhs, double time, const Pose3d& lhs_T_rhs)>;

  /**
   * @brief Allocate space for a given number of total frames and total number of edges.
   *
   * Total amount of memory required is approximately:
   *  number_frames * 128 + number_edges * 64 + history_length * 72.
   *
   * @param number_frames Maximum number of frames to support.
   * @param number_edges Maximum number of edges to support.
   * @param history_length Maximum history length.
   * @param default_number_edges Default number of edges per frame.
   * @param default_history_length Default history length per edge.
   * @param edges_chunk_size Chunk size for edge allocation.
   * @param history_chunk_size Chunk size for history allocation.
   * @return Success or error status.
   */
  expected_t<void> init(int32_t number_frames = 1024, int32_t number_edges = 16384,
                        int32_t history_length = 1048576, int32_t default_number_edges = 16,
                        int32_t default_history_length = 1024, int32_t edges_chunk_size = 4,
                        int32_t history_chunk_size = 64);

  /**
   * @brief Deinitialize the PoseTree and free all allocated resources.
   */
  void deinit();

  /**
   * @brief Set information needed to avoid overlap in frame id creating across PoseTree.

   * @param start_frame_id The first id to be assigned by this PoseTree (must be > 0(.
   * @param increment How much increment to leave between two frames id..
   */
  expected_t<void> set_multithreading_info(frame_t start_frame_id, frame_t increment);

  /**
   * @brief Get the current PoseTree version.
   *
   * @return Current version of the PoseTree.
   */
  version_t get_pose_tree_version() const;

  /**
   * @brief Create a new frame in the PoseTree.
   *
   * An optional name may be given to give a human-readable name to the frame. The name is a
   * null-terminated string with at most 63 characters. User defined name cannot start with "_",
   * which is reserved for auto generated names such as "_frame_i", where i is the uid of the frame.
   * A hint on the maximum number of edges this frame will be connected to can be provided.
   *
   * @param frame_id Frame id to be assigned.
   * @param name Human-readable name for the frame.
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> create_frame_with_id(frame_t frame_id, std::string_view name);

  /**
   * @brief Create a new frame in the PoseTree.
   *
   * An optional name may be given to give a human-readable name to the frame. The name is a
   * null-terminated string with at most 63 characters. User defined name cannot start with "_",
   * which is reserved for auto generated names such as "_frame_i", where i is the uid of the frame.
   * A hint on the maximum number of edges this frame will be connected to can be provided.
   *
   * @param name Human-readable name for the frame.
   * @param number_edges Hint for maximum number of edges this frame will have.
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> create_frame(std::string_view name, int32_t number_edges);

  /**
   * @brief Create a new frame in the PoseTree with a name.
   *
   * The maximum number of edges for this frame will be the default value.
   *
   * @param name Human-readable name for the frame.
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> create_frame(std::string_view name);

  /**
   * @brief Create a new frame in the PoseTree with edge count hint.
   *
   * A name will be generated automatically.
   *
   * @param number_edges Hint for maximum number of edges this frame will have.
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> create_frame(int32_t number_edges);

  /**
   * @brief Create a new frame in the PoseTree with default settings.
   *
   * A name will be generated automatically and the maximum number of edges will be the default
   * value.
   *
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> create_frame();

  /**
   * @brief Find a frame with the given name.
   *
   * @param name Name of the frame to find.
   * @return Frame id on success, Error::kFrameNotFound if no such frame exists.
   */
  expected_t<frame_t> find_frame(std::string_view name) const;

  /**
   * @brief Find a frame with the given name, or create it if it doesn't exist.
   *
   * @param name Name of the frame to find or create.
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> find_or_create_frame(std::string_view name);

  /**
   * @brief Find a frame with the given name, or create it with edge count hint if it doesn't exist.
   *
   * @param name Name of the frame to find or create.
   * @param number_edges Hint for maximum number of edges this frame will have.
   * @return Frame id on success, error on failure.
   */
  expected_t<frame_t> find_or_create_frame(std::string_view name, int32_t number_edges);

  /**
   * @brief Create an edge between two frames.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(frame_t lhs, frame_t rhs);

  /**
   * @brief Create an edge between two frames with maximum length hint.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param maximum_length Hint for maximum history length.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(frame_t lhs, frame_t rhs, int32_t maximum_length);

  /**
   * @brief Create an edge between two frames with access method.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param method Access method for the edge history.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(frame_t lhs, frame_t rhs,
                                     PoseTreeEdgeHistory::AccessMethod method);

  /**
   * @brief Create an edge between two frames with maximum length and access method.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param maximum_length Hint for maximum history length.
   * @param method Access method for the edge history.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(frame_t lhs, frame_t rhs, int32_t maximum_length,
                                     PoseTreeEdgeHistory::AccessMethod method);

  /**
   * @brief Create an edge between two frames using string names.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(std::string_view lhs, std::string_view rhs);

  /**
   * @brief Create an edge between two frames using string names with maximum length hint.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param maximum_length Hint for maximum history length.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(std::string_view lhs, std::string_view rhs,
                                     int32_t maximum_length);

  /**
   * @brief Create an edge between two frames using string names with access method.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param method Access method for the edge history.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(std::string_view lhs, std::string_view rhs,
                                     PoseTreeEdgeHistory::AccessMethod method);

  /**
   * @brief Create an edge between two frames using string names with maximum length and access
   * method.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param maximum_length Hint for maximum history length.
   * @param method Access method for the edge history.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> create_edges(std::string_view lhs, std::string_view rhs,
                                     int32_t maximum_length,
                                     PoseTreeEdgeHistory::AccessMethod method);

  /**
   * @brief Delete a frame in the PoseTree and all its relations to other frames.
   *
   * This action permanently erases the history information and frees its memory.
   * Upon success, it returns the version id of the change (however query made with a previous
   * version will also consider the frame as deleted).
   *
   * @param uid Frame id to delete.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> delete_frame(frame_t uid);

  /**
   * @brief Delete a frame in the PoseTree by name and all its relations to other frames.
   *
   * @param name Name of the frame to delete.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> delete_frame(std::string_view name);

  /**
   * @brief Delete an edge and free the memory.
   *
   * This action permanently erases the history. Upon success, it returns the version id of the
   * change (however query made with a previous version will also consider the edge as deleted).
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> delete_edge(frame_t lhs, frame_t rhs);

  /**
   * @brief Delete an edge by frame names and free the memory.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> delete_edge(std::string_view lhs, std::string_view rhs);

  /**
   * @brief Disconnect a frame from all the others starting at a given time.
   *
   * @param uid Frame id to disconnect.
   * @param time Time at which to start the disconnection.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> disconnect_frame(frame_t uid, double time);

  /**
   * @brief Disconnect a frame by name from all the others starting at a given time.
   *
   * @param name Name of the frame to disconnect.
   * @param time Time at which to start the disconnection.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> disconnect_frame(std::string_view name, double time);

  /**
   * @brief Disconnect an edge starting at a given time.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to start the disconnection.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> disconnect_edge(frame_t lhs, frame_t rhs, double time);

  /**
   * @brief Disconnect an edge by frame names starting at a given time.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to start the disconnection.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> disconnect_edge(std::string_view lhs, std::string_view rhs, double time);

  // Disable all the implicit cast (to make sure to catch a call with the wrong type for the time)
  template <class... Args>
  expected_t<Pose3d> disconnect_frame(Args&&... args) = delete;
  template <class... Args>
  expected_t<Pose3d> disconnect_edge(Args&&... args) = delete;

  /**
   * @brief Get the name of a frame.
   *
   * @param uid Frame id.
   * @return Frame name on success, error on failure.
   */
  expected_t<std::string_view> get_frame_name(frame_t uid) const;

  /**
   * @brief Retrieve the parameters used to initialize this PoseTree.
   *
   * @return Initialization parameters on success, or an error on failure.
   */
  expected_t<InitParameters> get_init_parameters() const;

  /**
   * @brief Get the latest pose between two frames as well as the time of that pose.
   *
   * The two poses needs to be directly linked.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @return Pair of pose and time on success, error on failure.
   */
  expected_t<std::pair<Pose3d, double>> get_latest(frame_t lhs, frame_t rhs) const;

  /**
   * @brief Get the latest pose between two frames by name as well as the time of that pose.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @return Pair of pose and time on success, error on failure.
   */
  expected_t<std::pair<Pose3d, double>> get_latest(std::string_view lhs,
                                                   std::string_view rhs) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames in the PoseTree at the given time.
   *
   * If the poses are not connected exactly at the given time, the indicated method is used to
   * interpolate the data.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to query the pose.
   * @param method Access method for interpolation.
   * @param version Version of the PoseTree to query.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs, double time,
                         PoseTreeEdgeHistory::AccessMethod method, version_t version) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames at the given time and version.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to query the pose.
   * @param version Version of the PoseTree to query.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs, double time, version_t version) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames at the given time with access method.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to query the pose.
   * @param method Access method for interpolation.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs, double time,
                         PoseTreeEdgeHistory::AccessMethod method) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames at the given time.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to query the pose.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs, double time) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames at a specific version.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param version Version of the PoseTree to query.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs, version_t version) const;

  /**
   * @brief Get the latest pose lhs_T_rhs between two frames.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames by name at the given time.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to query the pose.
   * @param method Access method for interpolation.
   * @param version Version of the PoseTree to query.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs, double time,
                         PoseTreeEdgeHistory::AccessMethod method, version_t version) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames by name at the given time and version.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to query the pose.
   * @param version Version of the PoseTree to query.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs, double time,
                         version_t version) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames by name at a specific version.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param version Version of the PoseTree to query.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs, version_t version) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames by name at the given time with access method.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to query the pose.
   * @param method Access method for interpolation.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs, double time,
                         PoseTreeEdgeHistory::AccessMethod method) const;

  /**
   * @brief Get the pose lhs_T_rhs between two frames by name at the given time.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to query the pose.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs, double time) const;

  /**
   * @brief Get the latest pose lhs_T_rhs between two frames by name.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @return Pose on success, error on failure.
   */
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs) const;

  // Disable all the implicit cast (to make sure to catch a call with the wrong type for the time)
  template <class... Args>
  expected_t<Pose3d> get(frame_t lhs, frame_t rhs, Args&&... args) const = delete;
  template <class... Args>
  expected_t<Pose3d> get(std::string_view lhs, std::string_view rhs, Args&&... args) const = delete;

  /**
   * @brief Helper function to get a Pose2d instead of Pose3d.
   *
   * @tparam Args Argument types to forward to the get method.
   * @param args Arguments to forward to the get method.
   * @return 2D pose on success, error on failure.
   */
  template <class... Args>
  expected_t<Pose2d> get_pose2_xy(Args&&... args) const {
    return get(std::forward<Args>(args)...).map([](const Pose3d& pose_3d) {
      return pose_3d.to_pose2_xy();
    });
  }

  /**
   * @brief Set the pose between two frames in the PoseTree.
   *
   * Note that poses can not be changed retrospectively. Thus for example once the pose at time
   * t=2.0 is set it is no longer allowed to set the pose for time t <= 2.0. It is not allowed to
   * form cycles. Frames are implicitly linked. If more than the maximum number of allowed poses
   * are set the oldest pose is deleted.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to set the pose.
   * @param lhs_T_rhs Pose transformation from lhs to rhs.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> set(frame_t lhs, frame_t rhs, double time, const Pose3d& lhs_T_rhs);

  /**
   * @brief Set the pose between two frames by name in the PoseTree.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to set the pose.
   * @param lhs_T_rhs Pose transformation from lhs to rhs.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> set(std::string_view lhs, std::string_view rhs, double time,
                            const Pose3d& lhs_T_rhs);

  /**
   * @brief Helper function to set a Pose2d instead of Pose3d.
   *
   * @param lhs Left hand side frame.
   * @param rhs Right hand side frame.
   * @param time Time at which to set the pose.
   * @param lhs_T_rhs 2D pose transformation from lhs to rhs.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> set(frame_t lhs, frame_t rhs, double time, const Pose2d& lhs_T_rhs) {
    return set(lhs, rhs, time, Pose3d::from_pose2_xy(lhs_T_rhs));
  }

  /**
   * @brief Helper function to set a Pose2d instead of Pose3d using frame names.
   *
   * @param lhs Name of left hand side frame.
   * @param rhs Name of right hand side frame.
   * @param time Time at which to set the pose.
   * @param lhs_T_rhs 2D pose transformation from lhs to rhs.
   * @return Version id of the change on success, error on failure.
   */
  expected_t<version_t> set(std::string_view lhs, std::string_view rhs, double time,
                            const Pose2d& lhs_T_rhs) {
    return set(lhs, rhs, time, Pose3d::from_pose2_xy(lhs_T_rhs));
  }
  // Then we disable all the calls not made with double.
  template <typename T, typename Pose>
  expected_t<version_t> set(frame_t lhs, frame_t rhs, T time, const Pose& lhs_T_rhs) = delete;
  template <typename T, typename Pose>
  expected_t<version_t> set(std::string_view lhs, std::string_view rhs, T time,
                            const Pose& lhs_T_rhs) = delete;

  /**
   * @brief Get list of edge UIDs.
   *
   * @tparam T Container type that supports clear(), capacity(), size(), and push_back().
   * @param container Container to fill with edge UIDs.
   * @return Success or error status.
   */
  template <typename T>
  expected_t<void> get_edge_uids(T& container) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    if (container.capacity() < static_cast<typename T::size_type>(edges_map_.size())) {
      return unexpected_t(Error::kOutOfMemory);
    }
    container.clear();
    for (const auto& key : edges_map_keys_) {
      if (!key) {
        return unexpected_t(Error::kLogicError);
      }
      const auto value = edges_map_.get(key.value());
      if (!value) {
        return unexpected_t(Error::kLogicError);
      }
      const auto history = histories_map_.try_get(value.value());
      if (!history) {
        return unexpected_t(Error::kLogicError);
      }
      if (!history.value()->connected()) {
        continue;
      }
      container.push_back(key.value());
    }
    return expected_t<void>{};
  }

  /**
   * @brief Get list of edge names.
   *
   * @tparam T Container type that supports clear(), capacity(), size(), and push_back().
   * @param container Container to fill with edge names as pairs.
   * @return Success or error status.
   */
  template <typename T>
  expected_t<void> get_edge_names(T& container) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    container.clear();
    if (container.capacity() < edges_map_.size()) {
      return unexpected_t(Error::kOutOfMemory);
    }
    for (const auto& key : edges_map_keys_) {
      if (!key) {
        return unexpected_t(Error::kLogicError);
      }
      const auto value = edges_map_.get(key.value());
      if (!value) {
        return unexpected_t(Error::kLogicError);
      }
      const auto history = histories_map_.try_get(value.value());
      if (!history) {
        return unexpected_t(Error::kLogicError);
      }
      if (!history.value()->connected()) {
        continue;
      }
      auto lhs_name = get_frame_name(key.value().first);
      auto rhs_name = get_frame_name(key.value().second);
      if (!lhs_name || !rhs_name) {
        continue;
      }
      container.push_back({lhs_name.value(), rhs_name.value()});
    }
    return expected_t<void>{};
  }

  /**
   * @brief Get list of frame UIDs.
   *
   * @tparam T Container type that supports clear(), capacity(), size(), and push_back().
   * @param container Container to fill with frame UIDs.
   * @return Success or error status.
   */
  template <typename T>
  expected_t<void> get_frame_uids(T& container) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    container.clear();
    if (container.capacity() < name_to_uid_map_keys_.size()) {
      return unexpected_t(Error::kOutOfMemory);
    }
    for (const auto& key : name_to_uid_map_keys_) {
      if (!key) {
        return unexpected_t(Error::kLogicError);
      }
      const auto value = name_to_uid_map_.get(key.value());
      if (!value) {
        return unexpected_t(Error::kLogicError);
      }
      container.push_back(value.value());
    }
    return expected_t<void>{};
  }

  /**
   * @brief Get list of frame names.
   *
   * @tparam T Container type that supports clear(), capacity(), size(), and push_back().
   * @param container Container to fill with frame names.
   * @return Success or error status.
   */
  template <typename T>
  expected_t<void> get_frame_names(T& container) const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    container.clear();
    if (container.capacity() < name_to_uid_map_keys_.size()) {
      return unexpected_t(Error::kOutOfMemory);
    }
    for (const auto& key : name_to_uid_map_keys_) {
      if (!key) {
        return unexpected_t(Error::kLogicError);
      }
      container.push_back(key.value());
    }
    return expected_t<void>{};
  }

  /**
   * @brief Register a callback function for every time a frame is created.
   *
   * @param callback Callback function to register.
   * @return Unique ID for the callback on success, error on failure.
   */
  expected_t<uid_t> add_create_frame_callback(CreateFrameCallback callback);

  /**
   * @brief Deregister a callback function for frame creation.
   *
   * @param cid Component ID of the callback to remove.
   * @return Success or error status.
   */
  expected_t<void> remove_create_frame_callback(uid_t cid);

  /**
   * @brief Register a callback function for every time an edge is set.
   *
   * @param callback Callback function to register.
   * @return Unique ID for the callback on success, error on failure.
   */
  expected_t<uid_t> add_set_edge_callback(SetEdgeCallback callback);

  /**
   * @brief Deregister a callback function for edge setting.
   *
   * @param cid Component ID of the callback to remove.
   * @return Success or error status.
   */
  expected_t<void> remove_set_edge_callback(uid_t cid);

  /**
   * @brief Convert an error code to a human readable error string.
   *
   * @param error Error code to convert.
   * @return Human-readable error string.
   */
  static const char* error_to_str(Error error);

 private:
  /**
   * @brief Helper structure that stores the information about a frame.
   */
  struct FrameInfo {
    /// Array containing the list of edges.
    history_t* history;
    /// Current number of edges
    int32_t number_edges;
    /// Maximum number of edges allowed
    int32_t maximum_number_edges;
    /// Name of the frame. It has to be null terminated, so it can hold at most 63 characters.
    char name[kFrameNameMaximumLength + 1];
    std::string_view name_view;
    /// Hint to quickly find a path:
    /// Store the distance from the node to the root (== 0 if this frame is the root)
    int32_t distance_to_root;
    /// Frame to follow to reach the root
    frame_t node_to_root;
    /// Name of the root
    frame_t root;
    /// Some helper id to computer the path between two nodes
    mutable version_t hint_version;
    /// Some helper to memorize the path we took during the dfs
    mutable frame_t dfs_link;
    /// Name of the frame.
    frame_t uid;
  };

  // Implementation of find_or_create_frame
  expected_t<frame_t> find_or_create_frame_impl(std::string_view name, int32_t number_edges);
  // Implementation of find_frame
  expected_t<frame_t> find_frame_impl(std::string_view name) const;
  // Implementation of create_frame
  expected_t<frame_t> create_frame_impl(std::string_view name, int32_t number_edges,
                                        const frame_t* id);
  // Implementation of create_edges
  expected_t<version_t> create_edges_impl(frame_t lhs, frame_t rhs, int32_t maximum_length,
                                          PoseTreeEdgeHistory::AccessMethod method);
  // Implementation of delete_edge
  expected_t<version_t> delete_edge_impl(frame_t lhs, frame_t rhs, version_t version);
  // Update the path to the root for a given connected component starting from the given node.
  expected_t<void> update_root(frame_t root);
  // Implementation of get using the pre-computed path to the root as a hint. If it fails, it falls
  // back to get_dfs_impl.
  expected_t<Pose3d> get_impl(frame_t lhs, frame_t rhs, double time,
                              PoseTreeEdgeHistory::AccessMethod method, version_t version) const;
  // Implementation of get that do a dfs to see if a path exists at a given time.
  expected_t<Pose3d> get_dfs_impl(frame_t lhs, frame_t rhs, double time,
                                  PoseTreeEdgeHistory::AccessMethod method,
                                  version_t version) const;

  /// Lock to protect access to the parameter below.
  mutable std::shared_timed_mutex mutex_;
  /// Lock to protect access to get_dfs_impl. This function is rarely called, but can be called
  /// while mutex_ is lock in read access, and get_dfs_impl is modifying the dfs_link of some frames
  /// as well as using the frames_stack_. We need a special protection for this function while not
  /// blocking all the concurrent read which most likely won't call it.
  mutable std::mutex dfs_mutex_;

  /// Lock to protect create_frame_callbacks_
  mutable std::shared_timed_mutex create_frame_callbacks_mutex_;
  /// Callback functions for the create frame operation.
  nvidia::UniqueIndexMap<CreateFrameCallback> create_frame_callbacks_;
  nvidia::FixedVector<uid_t> create_frame_callbacks_keys_;
  /// Lock to protect set_edge_callbacks_
  mutable std::shared_timed_mutex set_edge_callbacks_mutex_;
  /// Callback functions for the set edge operation.
  nvidia::UniqueIndexMap<SetEdgeCallback> set_edge_callbacks_;
  nvidia::FixedVector<uid_t> set_edge_callbacks_keys_;

  /// Mapping from a frame to its index.
  // TODO(ben): We need to get rid of std::map
  pose_tree::HashMap<std::pair<frame_t, frame_t>, history_t> edges_map_;
  nvidia::FixedVector<std::pair<frame_t, frame_t>> edges_map_keys_;

  /// TODO(ben): We need to get rid of std::map, but for now UniqueIndexMap does not support
  /// iterating through all the elements.
  pose_tree::HashMap<std::string_view, frame_t> name_to_uid_map_;
  nvidia::FixedVector<std::string_view> name_to_uid_map_keys_;

  /// Store the list of the current frame of the PoseTree.
  pose_tree::HashMap<frame_t, FrameInfo> frame_map_;
  frame_t next_frame_id_{};
  frame_t frame_id_increment_{};

  /// Used to implement a dfs.
  std::unique_ptr<frame_t[]> frames_stack_;

  /// Store the list of PoseTreeEdgeHistory used by the frames. Each PoseTreeEdgeHistory correspond
  /// to a bi-directional edge.
  // TODO(ben): We need to get rid of std::map
  nvidia::UniqueIndexMap<PoseTreeEdgeHistory> histories_map_;

  /// Helper to `allocate` an array of PoseTreeEdgeHistory (storing only the uid).
  FirstFitAllocator<history_t> histories_management_;

  /// Helper to `allocate` an array of TimedPose.
  FirstFitAllocator<PoseTreeEdgeHistory::TimedPose> poses_management_;

  /// Current version of the PoseTree.
  version_t version_{};
  /// Version of the hint. Mostly used to know if a node in the stack has been processed already.
  mutable version_t hint_version_{};

  /// The initialization parameters.
  InitParameters init_params_{};

  /// Default maximum number of edges a given frame can have
  int32_t default_number_edges_{};
  /// Default length of the history used by an edge.
  int32_t default_history_length_{};

  uid_t frame_cb_latest_uid_{};
  uid_t edge_cb_latest_uid_{};
};

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_POSE_TREE_HPP */
