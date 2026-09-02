/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/expected.hpp>
#include <holoscan/pose_tree/pose_tree.hpp>

namespace holoscan {

// Message type identifiers for Active Message (AM) communication
enum MessageType : uint16_t {
  MSG_SUBSCRIBE = 1,           // Client -> Server: Request to join and optionally get a snapshot
  MSG_DELTA = 2,               // Both ways: Incremental PoseTree update (frame/edge)
  MSG_SNAPSHOT_DATA = 3,       // Server -> Client: Full PoseTree state snapshot
  MSG_CLOSE = 4,               // Both ways: Close connection
  MSG_DISTRIBUTED_CONFIG = 5,  // Server -> Client config information
  MSG_SNAPSHOT_ACK = 6,        // Client -> Server: Acknowledge snapshot applied
};

// Ensure packed layout for all serialized structs
#pragma pack(push, 1)

// Payload for the MSG_SUBSCRIBE message
struct SubscribeMessage {
  uint8_t request_snapshot;  // Boolean flag
};

// Payload for the MSG_SNAPSHOT_ACK message
struct SnapshotAckMessage {
  uint8_t snapshot_applied;  // Boolean flag
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
static_assert(sizeof(SnapshotAckMessage) == sizeof(uint8_t), "Unexpected SnapshotAckMessage size");
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

// Decode a fixed-width frame name received from the wire without reading past the field.
inline std::string_view deserialize_frame_name(
    const char (&name)[PoseTree::kFrameNameMaximumLength + 1]) {
  const auto* terminator = static_cast<const char*>(std::memchr(name, '\0', sizeof(name)));
  if (terminator == nullptr) {
    throw std::runtime_error("Invalid PoseTree frame name: missing NUL terminator");
  }
  return {name, static_cast<size_t>(terminator - name)};
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
  if (data == nullptr && size != 0) {
    throw std::runtime_error("Invalid snapshot: null data pointer");
  }

  size_t offset = 0;
  auto require_bytes = [&](size_t byte_count, const char* description) {
    if (offset > size || byte_count > size - offset) {
      throw std::runtime_error(std::string("Invalid snapshot: incomplete ") + description);
    }
  };

  auto read_count = [&](const char* description) -> uint64_t {
    require_bytes(sizeof(uint64_t), description);
    uint64_t count = 0;
    std::memcpy(&count, data + offset, sizeof(count));
    offset += sizeof(count);
    return count;
  };

  auto checked_body_size =
      [&](uint64_t count, size_t element_size, const char* description) -> size_t {
    if (count > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error(std::string("Invalid snapshot: ") + description +
                               " count is too large");
    }
    const size_t converted_count = static_cast<size_t>(count);
    // Comparing before multiplying prevents both arithmetic overflow and out-of-bounds access.
    if (offset > size || converted_count > (size - offset) / element_size) {
      throw std::runtime_error(std::string("Invalid snapshot: incomplete ") + description);
    }
    return converted_count * element_size;
  };

  std::vector<FrameInfo> parsed_frames;
  const uint64_t num_frames = read_count("frame count");
  const size_t frames_size = checked_body_size(num_frames, sizeof(FrameInfo), "frame body");
  if (num_frames > parsed_frames.max_size()) {
    throw std::runtime_error("Invalid snapshot: frame count exceeds container capacity");
  }
  parsed_frames.resize(static_cast<size_t>(num_frames));
  if (frames_size != 0) {
    std::memcpy(parsed_frames.data(), data + offset, frames_size);
  }
  offset += frames_size;

  std::vector<EdgeData> parsed_edges;
  const uint64_t num_edges = read_count("edge count");
  const size_t edges_size = checked_body_size(num_edges, sizeof(EdgeData), "edge body");
  if (num_edges > parsed_edges.max_size()) {
    throw std::runtime_error("Invalid snapshot: edge count exceeds container capacity");
  }
  parsed_edges.resize(static_cast<size_t>(num_edges));
  if (edges_size != 0) {
    std::memcpy(parsed_edges.data(), data + offset, edges_size);
  }
  offset += edges_size;

  // Check for trailing bytes to ensure strict format validation
  if (offset != size) {
    throw std::runtime_error("Invalid snapshot: trailing bytes");
  }

  for (const auto& frame : parsed_frames) {
    (void)deserialize_frame_name(frame.name);
  }

  // Do not partially replace caller-owned state when parsing fails.
  frames = std::move(parsed_frames);
  edges = std::move(parsed_edges);
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
  // snprintf cannot fail: name length is validated above and buffer is sized to fit
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
  (void)std::snprintf(msg.data.frame_data.name, sizeof(msg.data.frame_data.name), "%s", name);
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
