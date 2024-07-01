/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <map>
#include <string>
#include <vector>

#include "../config.hpp"
#include "../utils.hpp"
#include "common/assert.hpp"
#include "holoscan/core/dataflow_tracker.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

class MockDataFlowTracker : public DataFlowTracker {
 public:
  using DataFlowTracker::update_latency;
  using DataFlowTracker::update_source_messages_number;
};

// Test case to check set_skip_starting_messages
TEST(DataFlowTracker, SetSkipStartingMessages) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  int num_start_messages_to_skip_ = 5;
  std::string pathname = "test";
  tracker.set_skip_starting_messages(num_start_messages_to_skip_);

  for (int i = 0; i < num_start_messages_to_skip_; i++) {
    tracker.update_latency(pathname, i + 1);

    // No message should be updated
    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  }

  for (int i = 0; i < 5; i++) {
    tracker.update_latency(pathname, i + 1);

    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), i + 1);
  }
}

// Test case to check if skipping latencies
TEST(DataFlowTracker, SetSkipLatencies1) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  int is_file_logging_enabled_ = 5;
  std::string pathname = "test";
  tracker.set_skip_latencies(is_file_logging_enabled_);

  tracker.update_latency(pathname, 4);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  tracker.update_latency(pathname, 6);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 1);
  tracker.update_latency(pathname, 5);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 2);
  tracker.update_latency(pathname, 3);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 2);
  tracker.update_latency(pathname, 7);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 3);
}

// Another test case for skip latencies but with a different threshold
TEST(DataFlowTracker, SetSkipLatencies2) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  int is_file_logging_enabled_ = 2;
  std::string pathname = "test";
  tracker.set_skip_latencies(is_file_logging_enabled_);

  tracker.update_latency(pathname, 4);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 1);

  tracker.update_latency(pathname, 6);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 2);

  tracker.update_latency(pathname, 5);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 3);

  tracker.update_latency(pathname, 3);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 4);

  tracker.update_latency(pathname, 7);
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 5);
}

// Test case to check if discarding last messages
TEST(DataFlowTracker, SetDiscardLastMessages) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  int num_last_messages_to_discard_ = 5;
  tracker.set_discard_last_messages(num_last_messages_to_discard_);

  std::string pathname = "test";

  for (int i = 0; i < num_last_messages_to_discard_; i++) {
    tracker.update_latency(pathname, i + 1);

    // No message should be updated
    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  }

  for (int i = 0; i < 5; i++) {
    tracker.update_latency(pathname, i + 1);

    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kMaxE2ELatency), i + 1);
  }
}

TEST(DataFlowTracker, SkipMessagesAndDiscardLastMessages) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  int num_start_messages_to_skip_ = 5;
  int num_last_messages_to_discard_ = 10;

  tracker.set_skip_starting_messages(num_start_messages_to_skip_);
  tracker.set_discard_last_messages(num_last_messages_to_discard_);

  std::string pathname = "test";

  for (int i = 0; i < num_start_messages_to_skip_; i++) {
    tracker.update_latency(pathname, i + 1);

    // No message should be updated
    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  }

  for (int i = 0; i < num_last_messages_to_discard_; i++) {
    tracker.update_latency(pathname, i + 1);

    // Still no message should be updated
    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  }

  tracker.update_latency(pathname, 5);

  // Finally one message should be updated
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 1);
}

TEST(DataFlowTracker, SkipStartingMessagesAndSkipLatencies) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  int num_start_messages_to_skip_ = 5;
  tracker.set_skip_starting_messages(num_start_messages_to_skip_);

  int threshold = 10;
  tracker.set_skip_latencies(threshold);

  std::string pathname = "test";

  // Skip latencies for less than threshold latencies
  for (int i = 0; i < num_start_messages_to_skip_; i++) {
    tracker.update_latency(pathname, i + 1);

    // No message should be updated
    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  }

  // Skip messages at the beginning for valid latencies
  for (int i = 0; i < num_start_messages_to_skip_; i++) {
    tracker.update_latency(pathname, 12);

    // No message should be updated
    ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);
  }

  tracker.update_latency(pathname, 8);
  // Still no message should be updated
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 0);

  tracker.update_latency(pathname, 10);
  // Finally, a message should be updated
  ASSERT_EQ(tracker.get_metric(pathname, DataFlowMetric::kNumDstMessages), 1);
}

TEST(DataFlowTracker, UpdateSourceMessagesNumber) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  std::string source_output_pair = "operator->transmitter";

  for (int i = 1; i <= 10; i++) {
    tracker.update_source_messages_number(source_output_pair, i);

    auto src_message_num = tracker.get_metric(DataFlowMetric::kNumSrcMessages);

    ASSERT_EQ(src_message_num.size(), 1);
    ASSERT_EQ(src_message_num[source_output_pair], i);
  }
}

TEST(DataFlowTracker, GetNumPaths) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  std::string pathname = "test";

  for (int i = 1; i <= 10; i++) {
    std::string new_pathname = pathname + std::to_string(i);
    tracker.update_latency(new_pathname, 5);
    ASSERT_EQ(tracker.get_num_paths(), i);
  }

  for (int i = 1; i <= 10; i++) {
    std::string new_pathname = pathname + std::to_string(i);
    tracker.update_latency(new_pathname, 10);
    ASSERT_EQ(tracker.get_num_paths(), 10);
  }
}

TEST(DataFlowTracker, GetPathStrings) {
  Fragment F;
  auto& tracker = (MockDataFlowTracker&)F.track(0, 0, 0);

  std::string pathname = "test";

  std::vector<std::string> path_strings;
  for (int i = 1; i <= 10; i++) {
    std::string new_pathname = pathname + std::to_string(i);
    path_strings.push_back(new_pathname);
    tracker.update_latency(new_pathname, 5);
  }

  auto paths = tracker.get_path_strings();

  for (const auto& path : paths) {
    ASSERT_TRUE(std::find(path_strings.begin(), path_strings.end(), path) != path_strings.end());
  }
}

}  // namespace holoscan
