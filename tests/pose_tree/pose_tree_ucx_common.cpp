/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pose_tree/pose_tree_ucx_common.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan {
namespace {

template <typename T>
void append_value(std::vector<uint8_t>& bytes, const T& value) {
  const size_t old_size = bytes.size();
  bytes.resize(old_size + sizeof(value));
  std::memcpy(bytes.data() + old_size, &value, sizeof(value));
}

void expect_invalid_snapshot(const std::vector<uint8_t>& bytes) {
  std::vector<FrameInfo> frames;
  std::vector<EdgeData> edges;
  EXPECT_THROW(deserialize_snapshot(bytes.data(), bytes.size(), frames, edges), std::runtime_error);
}

TEST(PoseTreeUCXCommon, DeserializeFrameNameStopsAtEarlyNul) {
  FrameInfo frame{};
  std::memset(frame.name, 'x', sizeof(frame.name));
  std::memcpy(frame.name, "abc", 3);
  frame.name[3] = '\0';

  EXPECT_EQ(deserialize_frame_name(frame.name), "abc");
}

TEST(PoseTreeUCXCommon, DeserializeFrameNameAcceptsMaximumLength) {
  FrameInfo frame{};
  std::memset(frame.name, 'x', PoseTree::kFrameNameMaximumLength);
  frame.name[PoseTree::kFrameNameMaximumLength] = '\0';

  const auto name = deserialize_frame_name(frame.name);
  EXPECT_EQ(name.size(), PoseTree::kFrameNameMaximumLength);
  EXPECT_EQ(name, std::string(PoseTree::kFrameNameMaximumLength, 'x'));
}

TEST(PoseTreeUCXCommon, DeserializeFrameNameRejectsMissingNul) {
  FrameInfo frame{};
  std::memset(frame.name, 'x', sizeof(frame.name));

  EXPECT_THROW(deserialize_frame_name(frame.name), std::runtime_error);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotAcceptsEmptySnapshot) {
  const auto serialized = serialize_snapshot({}, {});
  std::vector<FrameInfo> frames;
  std::vector<EdgeData> edges;

  deserialize_snapshot(
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size(), frames, edges);

  EXPECT_TRUE(frames.empty());
  EXPECT_TRUE(edges.empty());
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotAcceptsNonemptySnapshot) {
  FrameInfo frame{};
  frame.frame_id = 42;
  std::memcpy(frame.name, "camera", sizeof("camera"));

  EdgeData edge{};
  edge.lhs_frame = 42;
  edge.rhs_frame = 84;
  edge.time = 1.5;
  edge.qw = 1.0;
  edge.tx = 2.0;
  edge.ty = 3.0;
  edge.tz = 4.0;

  const auto serialized = serialize_snapshot({frame}, {edge});
  std::vector<FrameInfo> frames;
  std::vector<EdgeData> edges;
  deserialize_snapshot(
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size(), frames, edges);

  ASSERT_EQ(frames.size(), 1);
  EXPECT_EQ(frames[0].frame_id, frame.frame_id);
  EXPECT_EQ(deserialize_frame_name(frames[0].name), "camera");
  ASSERT_EQ(edges.size(), 1);
  EXPECT_EQ(std::memcmp(&edges[0], &edge, sizeof(edge)), 0);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsTruncationBeforeEdgeCount) {
  FrameInfo frame{};
  std::memcpy(frame.name, "frame", sizeof("frame"));
  const auto serialized = serialize_snapshot({frame}, {});
  std::vector<uint8_t> truncated(serialized.begin(), serialized.end() - sizeof(uint64_t));

  expect_invalid_snapshot(truncated);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsPartialFrameCount) {
  std::vector<uint8_t> bytes;
  append_value(bytes, uint32_t{0});

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsPartialEdgeCount) {
  std::vector<uint8_t> bytes;
  append_value(bytes, uint64_t{0});
  append_value(bytes, uint32_t{0});

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsIncompleteFrameBody) {
  std::vector<uint8_t> bytes;
  append_value(bytes, uint64_t{1});
  bytes.resize(bytes.size() + sizeof(FrameInfo) - 1);

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsIncompleteEdgeBody) {
  std::vector<uint8_t> bytes;
  append_value(bytes, uint64_t{0});
  append_value(bytes, uint64_t{1});
  bytes.resize(bytes.size() + sizeof(EdgeData) - 1);

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsHugeFrameCount) {
  std::vector<uint8_t> bytes;
  append_value(bytes, std::numeric_limits<uint64_t>::max());

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsHugeEdgeCount) {
  std::vector<uint8_t> bytes;
  append_value(bytes, uint64_t{0});
  append_value(bytes, std::numeric_limits<uint64_t>::max());

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsTrailingBytes) {
  const auto serialized = serialize_snapshot({}, {});
  std::vector<uint8_t> bytes(serialized.begin(), serialized.end());
  bytes.push_back(0);

  expect_invalid_snapshot(bytes);
}

TEST(PoseTreeUCXCommon, DeserializeSnapshotRejectsFrameNameWithoutNul) {
  FrameInfo frame{};
  std::memset(frame.name, 'x', sizeof(frame.name));
  const auto serialized = serialize_snapshot({frame}, {});
  std::vector<uint8_t> bytes(serialized.begin(), serialized.end());

  expect_invalid_snapshot(bytes);
}

}  // namespace
}  // namespace holoscan
