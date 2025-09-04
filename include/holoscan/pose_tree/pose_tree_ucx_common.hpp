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

#ifndef HOLOSCAN_POSE_TREE_UCX_COMMON_HPP
#define HOLOSCAN_POSE_TREE_UCX_COMMON_HPP

#include <algorithm>
#include <cstddef>  // For offsetof
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>  // Add this include for std::numeric_limits
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/core/expected.hpp"
#include "holoscan/pose_tree/pose_tree.hpp"

namespace holoscan {

// Message type identifiers for Active Message (AM) communication
enum MessageType : uint16_t {
  MSG_SUBSCRIBE = 1,           // Client -> Server: Request to join and optionally get a snapshot
  MSG_DELTA = 2,               // Both ways: Incremental PoseTree update (frame/edge)
  MSG_SNAPSHOT_DATA = 3,       // Server -> Client: Full PoseTree state snapshot
  MSG_CLOSE = 4,               // Both ways: Close connection
  MSG_DISTRIBUTED_CONFIG = 5,  // Server -> Client config information
};

// Ensure packed layout for all serialized structs
#pragma pack(push, 1)

// Payload for the MSG_SUBSCRIBE message
struct SubscribeMessage {
  uint8_t request_snapshot;  // Boolean flag
};

// Type of change in a DELTA message
enum DeltaType : uint8_t {
  DELTA_FRAME_CREATED = 1,
  DELTA_EDGE_SET = 2,
};

// Data for a new frame
struct FrameData {
  uint64_t frame_id;  // Using fixed-width type directly instead of frame_t
  char name[PoseTree::kFrameNameMaximumLength + 1];  // Match PoseTree's frame name size (128)
};

// Data for a new or updated edge (pose)
struct EdgeData {
  uint64_t lhs_frame;  // Using fixed-width type directly
  uint64_t rhs_frame;  // Using fixed-width type directly
  double time;
  double qw, qx, qy, qz;  // Quaternion for rotation
  double tx, ty, tz;      // Vector for translation
};

// Data to configure the distributed frame id assignment
struct DistributedConfig {
  uint64_t start_frame_id;
  uint64_t increment;
};

// The DELTA message structure, containing the type of change and the data
struct DeltaMessage {
  DeltaType delta_type;
  union {
    FrameData frame_data;
    EdgeData edge_data;
  } data;
};

// Information about a single frame for snapshot serialization
struct FrameInfo {
  uint64_t frame_id;                                 // Using fixed-width type directly
  char name[PoseTree::kFrameNameMaximumLength + 1];  // Match PoseTree's frame name size (128)
};

#pragma pack(pop)

// Static assertions to verify expected sizes
static_assert(sizeof(SubscribeMessage) == sizeof(uint8_t), "Unexpected SubscribeMessage size");
static_assert(sizeof(FrameData) ==
                  sizeof(uint64_t) + sizeof(char[PoseTree::kFrameNameMaximumLength + 1]),
              "Unexpected FrameData size");
static_assert(sizeof(EdgeData) == 2 * sizeof(uint64_t) + 8 * sizeof(double),
              "Unexpected EdgeData size");
static_assert(offsetof(DeltaMessage, data) == sizeof(DeltaType), "Unexpected padding before union");
static_assert(sizeof(DeltaMessage) ==
                  sizeof(DeltaType) + std::max(sizeof(FrameData), sizeof(EdgeData)),
              "Unexpected DeltaMessage size");
static_assert(sizeof(FrameInfo) ==
                  sizeof(uint64_t) + sizeof(char[PoseTree::kFrameNameMaximumLength + 1]),
              "Unexpected FrameInfo size");

// Helper to serialize a Pose3d object into an EdgeData struct.
// Note: This function intentionally ignores the 'time' field in EdgeData;
// callers must set the 'time' field explicitly if needed.
inline void serialize_pose3d(const holoscan::Pose3d& pose, EdgeData& edge_data) {
  const auto& q = pose.rotation.quaternion();
  edge_data.qw = q.w();
  edge_data.qx = q.x();
  edge_data.qy = q.y();
  edge_data.qz = q.z();
  const auto& t = pose.translation;
  edge_data.tx = t.x();
  edge_data.ty = t.y();
  edge_data.tz = t.z();
}

// Helper to deserialize an EdgeData struct back into a Pose3d object.
// Note: This function intentionally ignores the 'time' field in EdgeData;
// callers must set the 'time' field explicitly if needed.
inline holoscan::Pose3d deserialize_pose3d(const EdgeData& edge_data) {
  holoscan::Quaterniond q(edge_data.qw, edge_data.qx, edge_data.qy, edge_data.qz);
  holoscan::Vector3d t(edge_data.tx, edge_data.ty, edge_data.tz);
  return holoscan::Pose3d(holoscan::SO3d::from_normalized_quaternion(q), t);
}

// Serialize a full snapshot of frames and edges into a byte vector
inline std::vector<char> serialize_snapshot(const std::vector<FrameInfo>& frames,
                                            const std::vector<EdgeData>& edges) {
  // Helper lambda to safely multiply size_t values and check for overflow
  auto safe_multiply = [](size_t count, size_t elem_sz) -> size_t {
    if (count > std::numeric_limits<size_t>::max() / elem_sz) {
      throw std::overflow_error("Snapshot too large to serialize safely");
    }
    return count * elem_sz;
  };

  // Calculate sizes with overflow protection
  size_t frames_size = sizeof(uint64_t) + safe_multiply(frames.size(), sizeof(FrameInfo));
  size_t edges_size = sizeof(uint64_t) + safe_multiply(edges.size(), sizeof(EdgeData));

  // Prevent overflow of the final allocation size
  if (frames_size > std::numeric_limits<size_t>::max() - edges_size) {
    throw std::overflow_error("Snapshot too large to serialize safely (size addition overflow)");
  }
  std::vector<char> buffer(frames_size + edges_size);

  char* ptr = buffer.data();
  uint64_t num_frames = frames.size();
  std::memcpy(ptr, &num_frames, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  std::memcpy(ptr, frames.data(), safe_multiply(frames.size(), sizeof(FrameInfo)));
  ptr += frames.size() * sizeof(FrameInfo);

  uint64_t num_edges = edges.size();
  std::memcpy(ptr, &num_edges, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  std::memcpy(ptr, edges.data(), safe_multiply(edges.size(), sizeof(EdgeData)));

  return buffer;
}

// Deserialize a byte array back into lists of frames and edges
inline void deserialize_snapshot(const uint8_t* data, size_t size, std::vector<FrameInfo>& frames,
                                 std::vector<EdgeData>& edges) {
  if (size < 2 * sizeof(uint64_t)) {
    throw std::runtime_error("Snapshot data is too small to be valid.");
  }

  // Helper lambdas for safe arithmetic operations
  auto safe_multiply = [](size_t count, size_t elem_sz) -> size_t {
    if (count > std::numeric_limits<size_t>::max() / elem_sz) {
      throw std::overflow_error("Size multiplication overflow while deserializing snapshot");
    }
    return count * elem_sz;
  };

  auto safe_add = [](size_t a, size_t b) -> size_t {
    if (a > std::numeric_limits<size_t>::max() - b) {
      throw std::overflow_error("Size addition overflow while deserializing snapshot");
    }
    return a + b;
  };

  const char* ptr = reinterpret_cast<const char*>(data);

  uint64_t num_frames;
  std::memcpy(&num_frames, ptr, sizeof(uint64_t));
  ptr += sizeof(uint64_t);

  size_t expected_size = safe_add(sizeof(uint64_t), safe_multiply(num_frames, sizeof(FrameInfo)));
  if (size < expected_size) {
    throw std::runtime_error("Snapshot data is incomplete for frames.");
  }
  frames.resize(num_frames);
  std::memcpy(frames.data(), ptr, safe_multiply(num_frames, sizeof(FrameInfo)));
  ptr += num_frames * sizeof(FrameInfo);

  uint64_t num_edges;
  std::memcpy(&num_edges, ptr, sizeof(uint64_t));
  ptr += sizeof(uint64_t);

  expected_size = safe_add(expected_size,
                           safe_add(sizeof(uint64_t), safe_multiply(num_edges, sizeof(EdgeData))));
  if (size < expected_size) {
    throw std::runtime_error("Snapshot data is incomplete for edges.");
  }
  edges.resize(num_edges);
  std::memcpy(edges.data(), ptr, safe_multiply(num_edges, sizeof(EdgeData)));

  // Check for trailing bytes to ensure strict format validation
  if (size != expected_size) {
    throw std::runtime_error("Trailing bytes detected in snapshot payload");
  }
}

// Safe initialization helper for DeltaMessage
inline DeltaMessage create_pose_tree_frame_delta(uint64_t frame_id, const char* name) {
  // Check if the frame name is null
  if (name == nullptr) {
    throw std::invalid_argument("Frame name cannot be null");
  }

  // Check if the frame name exceeds the maximum allowed length
  if (std::strlen(name) > PoseTree::kFrameNameMaximumLength) {
    throw std::invalid_argument("Frame name length exceeds maximum allowed length of " +
                                std::to_string(PoseTree::kFrameNameMaximumLength) + " characters");
  }

  DeltaMessage msg{};  // Zero-initialize
  msg.delta_type = DELTA_FRAME_CREATED;
  msg.data.frame_data.frame_id = frame_id;
  std::snprintf(msg.data.frame_data.name, sizeof(msg.data.frame_data.name), "%s", name);
  return msg;
}

inline DeltaMessage create_pose_tree_edge_delta(const EdgeData& edge_data) {
  DeltaMessage msg{};  // Zero-initialize
  msg.delta_type = DELTA_EDGE_SET;
  msg.data.edge_data = edge_data;
  return msg;
}

}  // namespace holoscan

#endif /* HOLOSCAN_POSE_TREE_UCX_COMMON_HPP */
