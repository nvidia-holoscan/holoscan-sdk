/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unistd.h>

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/core/application.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/conditions/gxf/periodic.hpp"

#include "sample_test_graphs.hpp"

namespace holoscan {

///////////////////////////////////////////////////////////////////////////////
// Test Applications for add_probe_operator
///////////////////////////////////////////////////////////////////////////////

/* LinearChainWithProbe
 *
 * root--->middle--->leaf
 *        (probe)
 */
class LinearChainWithProbe : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto root =
        make_operator<OneOutOp>("root", make_condition<CountCondition>("count-condition", 10));
    auto middle = make_operator<OneInOneOutOp>("middle");
    auto leaf = make_operator<OneInOp>("leaf");

    add_flow(root, middle);
    add_flow(middle, leaf);
  }
};

/* LinearChainMultipleProbes
 *
 * root--->middle1--->middle2--->leaf
 *        (probe)     (probe)
 */
class LinearChainMultipleProbes : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto root =
        make_operator<OneOutOp>("root", make_condition<CountCondition>("count-condition", 10));
    auto middle1 = make_operator<OneInOneOutOp>("middle1");
    auto middle2 = make_operator<OneInOneOutOp>("middle2");
    auto leaf = make_operator<OneInOp>("leaf");

    add_flow(root, middle1);
    add_flow(middle1, middle2);
    add_flow(middle2, leaf);
  }
};

/* DAGWithProbe
 *
 * root--->middle1--->middle3--->leaf
 *    |              (probe)
 *    +---->middle2----+
 */
class DAGWithProbe : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto root =
        make_operator<OneOutOp>("root", make_condition<CountCondition>("count-condition", 10));
    auto middle1 = make_operator<OneInOneOutOp>("middle1");
    auto middle2 = make_operator<OneInOneOutOp>("middle2");
    auto middle3 = make_operator<TwoInOneOutNoConditionOp>("middle3");
    auto leaf = make_operator<OneInOp>("leaf");

    add_flow(root, middle1);
    add_flow(root, middle2);
    add_flow(middle1, middle3, {{"out", "in0"}});
    add_flow(middle2, middle3, {{"out", "in1"}});
    add_flow(middle3, leaf);
  }
};

/* DAGWithMultipleProbes
 *
 * root--->middle1--->middle3--->leaf
 *    |   (probe)    (probe)
 *    +---->middle2----+
 *         (probe)
 */
class DAGWithMultipleProbes : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto root =
        make_operator<OneOutOp>("root", make_condition<CountCondition>("count-condition", 10));
    auto middle1 = make_operator<OneInOneOutOp>("middle1");
    auto middle2 = make_operator<OneInOneOutOp>("middle2");
    auto middle3 = make_operator<TwoInOneOutNoConditionOp>("middle3");
    auto leaf = make_operator<OneInOp>("leaf");

    add_flow(root, middle1);
    add_flow(root, middle2);
    add_flow(middle1, middle3, {{"out", "in0"}});
    add_flow(middle2, middle3, {{"out", "in1"}});
    add_flow(middle3, leaf);
  }
};

/* CycleWithProbe
 *
 * root--->middle--->leaf
 *   ^    (probe)     |
 *   |                |
 *   +----------------+
 */
class CycleWithProbe : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto root = make_operator<OneOptionalInOneOutOp>(
        "root",
        make_condition<CountCondition>("count-condition", 10),
        make_condition<PeriodicCondition>("periodic-condition", std::chrono::milliseconds(5)));
    auto middle = make_operator<OneInOneOutOp>("middle");
    auto leaf = make_operator<OneInOneOutOp>("leaf");

    add_flow(root, middle);
    add_flow(middle, leaf);
    add_flow(leaf, root);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Tests for Linear Chains
///////////////////////////////////////////////////////////////////////////////

TEST(AddProbeOperator, LinearChainSingleProbe) {
  auto app = make_application<LinearChainWithProbe>();
  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("middle");

  testing::internal::CaptureStdout();
  app->run();
  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Should have exactly 2 paths: root->middle (to probe), root->middle->leaf (to leaf)
  EXPECT_EQ(tracker.get_num_paths(), 2);

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 2);

  // Check for exact paths
  bool has_root_to_middle = false;
  bool has_root_to_leaf = false;

  for (const auto& path : paths) {
    if (path == "root,middle") {
      has_root_to_middle = true;
    } else if (path == "root,middle,leaf") {
      has_root_to_leaf = true;
    }
  }

  EXPECT_TRUE(has_root_to_middle) << "Missing exact path: root,middle\n=== LOG ===\n"
                                  << log_output << "\n===========\n";
  EXPECT_TRUE(has_root_to_leaf) << "Missing exact path: root,middle,leaf\n=== LOG ===\n"
                                << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that root operator published messages
  bool has_root_source = false;
  bool has_middle_probe = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "root->out") {
      has_root_source = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from root, got " << count;
    } else if (source == "middle->out") {
      has_middle_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle, got " << count;
    }
  }
  EXPECT_TRUE(has_root_source) << "Root operator source messages not found";
  EXPECT_TRUE(has_middle_probe) << "Middle operator source messages not found";

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("root->out"), std::string::npos)
      << "Printed output does not contain root->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle->out"), std::string::npos)
      << "Printed output does not contain middle->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(AddProbeOperator, LinearChainMultipleProbes) {
  auto app = make_application<LinearChainMultipleProbes>();
  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("middle1");
  tracker.add_probe_operator("middle2");

  testing::internal::CaptureStdout();
  app->run();
  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Should have exactly 3 paths: root->middle1, root->middle1->middle2,
  // root->middle1->middle2->leaf
  EXPECT_EQ(tracker.get_num_paths(), 3);

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 3);

  // Check for exact paths
  bool has_path1 = false;
  bool has_path2 = false;
  bool has_path3 = false;

  for (const auto& path : paths) {
    if (path == "root,middle1") {
      has_path1 = true;
    } else if (path == "root,middle1,middle2") {
      has_path2 = true;
    } else if (path == "root,middle1,middle2,leaf") {
      has_path3 = true;
    }
  }

  EXPECT_TRUE(has_path1) << "Missing exact path: root,middle1\n=== LOG ===\n"
                         << log_output << "\n===========\n";
  EXPECT_TRUE(has_path2) << "Missing exact path: root,middle1,middle2\n=== LOG ===\n"
                         << log_output << "\n===========\n";
  EXPECT_TRUE(has_path3) << "Missing exact path: root,middle1,middle2,leaf\n=== LOG ===\n"
                         << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that root operator published messages
  bool has_root_source = false;
  bool has_middle1_probe = false;
  bool has_middle2_probe = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "root->out") {
      has_root_source = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from root, got " << count;
    } else if (source == "middle1->out") {
      has_middle1_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle1, got " << count;
    } else if (source == "middle2->out") {
      has_middle2_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle2, got " << count;
    }
  }
  EXPECT_TRUE(has_root_source) << "Root operator source messages not found";
  EXPECT_TRUE(has_middle1_probe) << "Middle1 operator source messages not found";
  EXPECT_TRUE(has_middle2_probe) << "Middle2 operator source messages not found";

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("root->out"), std::string::npos)
      << "Printed output does not contain root->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle1->out"), std::string::npos)
      << "Printed output does not contain middle1->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle2->out"), std::string::npos)
      << "Printed output does not contain middle2->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(AddProbeOperator, LinearChainNonExistentProbe) {
  auto app = make_application<LinearChainWithProbe>();
  auto& tracker = app->track(0, 0, 0);

  // Add a probe for a non-existent operator - finalize_probe() throws with invalid names
  tracker.add_probe_operator("nonexistent");

  try {
    app->run();
    FAIL() << "Expected std::runtime_error for invalid probe operator";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("Invalid probe operators found:"), std::string::npos) << "Message: " << msg;
    EXPECT_NE(msg.find("nonexistent"), std::string::npos) << "Message: " << msg;
  }
}

TEST(AddProbeOperator, LinearChainMultipleNonExistentProbes) {
  auto app = make_application<LinearChainWithProbe>();
  auto& tracker = app->track(0, 0, 0);

  tracker.add_probe_operator("nonexistent1");
  tracker.add_probe_operator("nonexistent2");

  try {
    app->run();
    FAIL() << "Expected std::runtime_error for invalid probe operators";
  } catch (const std::runtime_error& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("Invalid probe operators found:"), std::string::npos) << "Message: " << msg;
    EXPECT_NE(msg.find("nonexistent1"), std::string::npos) << "Message: " << msg;
    EXPECT_NE(msg.find("nonexistent2"), std::string::npos) << "Message: " << msg;
  }
}

TEST(AddProbeOperator, LinearChainDuplicateProbe) {
  auto app = make_application<LinearChainWithProbe>();
  auto& tracker = app->track(0, 0, 0);

  // Add probe twice - second one should be ignored with warning
  tracker.add_probe_operator("middle");
  tracker.add_probe_operator("middle");  // Duplicate

  app->run();

  // Should still have 2 paths (duplicate probe doesn't add more)
  EXPECT_EQ(tracker.get_num_paths(), 2);

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 2);
}

///////////////////////////////////////////////////////////////////////////////
// DAGs and Cycles (Greedy vs Event-Based Scheduler)
///////////////////////////////////////////////////////////////////////////////

class AddProbeOperatorSchedulerParameterized : public ::testing::TestWithParam<std::string> {};

static void ConfigureScheduler(holoscan::Application* app, const std::string& scheduler_type) {
  if (scheduler_type == "event_based") {
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based-scheduler",
        holoscan::Arg{"worker_thread_number", static_cast<int64_t>(2)},
        holoscan::Arg{"stop_on_deadlock_timeout", static_cast<int64_t>(100)}));
  }
  // greedy: default scheduler, no explicit setting
}

INSTANTIATE_TEST_SUITE_P(
    SchedulerType, AddProbeOperatorSchedulerParameterized,
    ::testing::Values(std::string("greedy"), std::string("event_based")),
    [](const ::testing::TestParamInfo<AddProbeOperatorSchedulerParameterized::ParamType>& info) {
      if (info.param == "greedy")
        return std::string("Greedy");
      if (info.param == "event_based")
        return std::string("EventBased");
      return std::string("UnknownScheduler");
    });

///////////////////////////////////////////////////////////////////////////////
// DAG with probe(s) - parameterized by scheduler
///////////////////////////////////////////////////////////////////////////////

TEST_P(AddProbeOperatorSchedulerParameterized, DAGWithProbe) {
  const std::string scheduler_type = GetParam();
  ASSERT_TRUE(scheduler_type == "greedy" || scheduler_type == "event_based")
      << "Unsupported scheduler type: " << scheduler_type;

  auto app = make_application<DAGWithProbe>();
  ConfigureScheduler(app.get(), scheduler_type);

  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("middle3");

  testing::internal::CaptureStdout();
  app->run();
  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // DAG has 2 branches to middle3, so expect paths through both branches
  // root->middle1->middle3, root->middle2->middle3, root->middle1->middle3->leaf,
  // root->middle2->middle3->leaf
  EXPECT_EQ(tracker.get_num_paths(), 4);

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 4);

  // Check for expected paths
  bool has_path_m1_m3 = false;
  bool has_path_m2_m3 = false;
  bool has_path_m1_m3_leaf = false;
  bool has_path_m2_m3_leaf = false;

  for (const auto& path : paths) {
    if (path == "root,middle1,middle3") {
      has_path_m1_m3 = true;
    } else if (path == "root,middle2,middle3") {
      has_path_m2_m3 = true;
    } else if (path == "root,middle1,middle3,leaf") {
      has_path_m1_m3_leaf = true;
    } else if (path == "root,middle2,middle3,leaf") {
      has_path_m2_m3_leaf = true;
    }
  }

  EXPECT_TRUE(has_path_m1_m3) << "Missing path: root,middle1,middle3\n=== LOG ===\n"
                              << log_output << "\n===========\n";
  EXPECT_TRUE(has_path_m2_m3) << "Missing path: root,middle2,middle3\n=== LOG ===\n"
                              << log_output << "\n===========\n";
  EXPECT_TRUE(has_path_m1_m3_leaf) << "Missing path: root,middle1,middle3,leaf\n=== LOG ===\n"
                                   << log_output << "\n===========\n";
  EXPECT_TRUE(has_path_m2_m3_leaf) << "Missing path: root,middle2,middle3,leaf\n=== LOG ===\n"
                                   << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that root operator published messages
  bool has_root_source = false;
  bool has_middle3_probe = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "root->out") {
      has_root_source = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from root, got " << count;
    } else if (source == "middle3->out") {
      has_middle3_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle3, got " << count;
    }
  }
  EXPECT_TRUE(has_root_source) << "Root operator source messages not found";
  EXPECT_TRUE(has_middle3_probe) << "Middle3 operator source messages not found";

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("root->out"), std::string::npos)
      << "Printed output does not contain root->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle3->out"), std::string::npos)
      << "Printed output does not contain middle3->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST_P(AddProbeOperatorSchedulerParameterized, DAGWithMultipleProbes) {
  const std::string scheduler_type = GetParam();
  ASSERT_TRUE(scheduler_type == "greedy" || scheduler_type == "event_based")
      << "Unsupported scheduler type: " << scheduler_type;

  auto app = make_application<DAGWithMultipleProbes>();
  ConfigureScheduler(app.get(), scheduler_type);

  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("middle1");
  tracker.add_probe_operator("middle2");
  tracker.add_probe_operator("middle3");

  testing::internal::CaptureStdout();
  app->run();
  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Should have exactly 6 paths:
  // root->middle1, root->middle2 (to first level probes)
  // root->middle1->middle3, root->middle2->middle3 (to middle3 probe through both branches)
  // root->middle1->middle3->leaf, root->middle2->middle3->leaf (to leaf through both branches)
  EXPECT_EQ(tracker.get_num_paths(), 6);

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 6);

  // Check for exact paths
  int found_count = 0;
  for (const auto& path : paths) {
    if (path == "root,middle1" || path == "root,middle2" || path == "root,middle1,middle3" ||
        path == "root,middle2,middle3" || path == "root,middle1,middle3,leaf" ||
        path == "root,middle2,middle3,leaf") {
      found_count++;
    }
  }

  EXPECT_EQ(found_count, 6) << "Not all expected paths found\n=== LOG ===\n"
                            << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that root operator published messages
  bool has_root_source = false;
  bool has_middle1_probe = false;
  bool has_middle2_probe = false;
  bool has_middle3_probe = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "root->out") {
      has_root_source = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from root, got " << count;
    } else if (source == "middle1->out") {
      has_middle1_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle1, got " << count;
    } else if (source == "middle2->out") {
      has_middle2_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle2, got " << count;
    } else if (source == "middle3->out") {
      has_middle3_probe = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from middle3, got " << count;
    }
  }
  EXPECT_TRUE(has_root_source) << "Root operator source messages not found";
  EXPECT_TRUE(has_middle1_probe) << "Middle1 operator source messages not found";
  EXPECT_TRUE(has_middle2_probe) << "Middle2 operator source messages not found";
  EXPECT_TRUE(has_middle3_probe) << "Middle3 operator source messages not found";

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("root->out"), std::string::npos)
      << "Printed output does not contain root->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle1->out"), std::string::npos)
      << "Printed output does not contain middle1->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle2->out"), std::string::npos)
      << "Printed output does not contain middle2->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle3->out"), std::string::npos)
      << "Printed output does not contain middle3->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

///////////////////////////////////////////////////////////////////////////////
// Cycle with probe(s) - parameterized by scheduler
///////////////////////////////////////////////////////////////////////////////

TEST_P(AddProbeOperatorSchedulerParameterized, CycleWithProbe) {
  const std::string scheduler_type = GetParam();
  ASSERT_TRUE(scheduler_type == "greedy" || scheduler_type == "event_based")
      << "Unsupported scheduler type: " << scheduler_type;

  auto app = make_application<CycleWithProbe>();
  ConfigureScheduler(app.get(), scheduler_type);

  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("middle");

  testing::internal::CaptureStdout();
  app->run();
  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  // Cycle: root->middle->leaf->root
  // With probe on middle, should have exactly 2 paths:
  // 1. root->middle (to probe)
  // 2. root->middle->leaf->root (full cycle)
  EXPECT_EQ(tracker.get_num_paths(), 2);

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 2);

  // Check for exact paths
  bool has_to_probe = false;
  bool has_cycle = false;

  for (const auto& path : paths) {
    if (path == "root,middle") {
      has_to_probe = true;
    } else if (path == "root,middle,leaf,root") {
      has_cycle = true;
    }
  }

  EXPECT_TRUE(has_to_probe) << "Missing path: root,middle\n=== LOG ===\n"
                            << log_output << "\n===========\n";
  EXPECT_TRUE(has_cycle) << "Missing cyclic path: root,middle,leaf,root\n=== LOG ===\n"
                         << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that root operator published messages (10 initial + cyclic messages)
  bool has_root_source = false;
  bool has_middle_probe = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "root->out") {
      has_root_source = true;
      EXPECT_GE(count, 10) << "Expected at least 10 messages from root (due to cycle), got "
                           << count;
    } else if (source == "middle->out") {
      has_middle_probe = true;
      EXPECT_GE(count, 10) << "Expected at least 10 messages from middle, got " << count;
    }
  }
  EXPECT_TRUE(has_root_source) << "Root operator source messages not found";
  EXPECT_TRUE(has_middle_probe) << "Middle operator source messages not found";

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("root->out"), std::string::npos)
      << "Printed output does not contain root->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("middle->out"), std::string::npos)
      << "Printed output does not contain middle->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST_P(AddProbeOperatorSchedulerParameterized, CycleWithMiddleProbe) {
  const std::string scheduler_type = GetParam();
  ASSERT_TRUE(scheduler_type == "greedy" || scheduler_type == "event_based")
      << "Unsupported scheduler type: " << scheduler_type;

  auto app = make_application<CycleWithSourceApp>();
  ConfigureScheduler(app.get(), scheduler_type);

  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("TwoInOneOut");

  testing::internal::CaptureStdout();
  app->run();
  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  // Should have 3 paths:
  // 1. OneOut->TwoInOneOut (to probe)
  // 2. OneOut->TwoInOneOut->OneInOneOut->TwoInOneOut (from root to end of cycle)
  // 3. TwoInOneOut->OneInOneOut->TwoInOneOut (the cycle itself)
  EXPECT_EQ(tracker.get_num_paths(), 3)
      << "Expected 3 paths, but tracker output is the following: " << log_output;

  auto paths = tracker.get_path_strings();
  EXPECT_EQ(paths.size(), 3) << "Expected 3 paths, but tracker output is the following: "
                             << log_output;

  // Check for exact paths
  int total_paths = 3;

  for (const auto& path : paths) {
    if (path == "OneOut,TwoInOneOut") {
      total_paths--;
    } else if (path == "OneOut,TwoInOneOut,OneInOneOut,TwoInOneOut") {
      total_paths--;
    } else if (path == "TwoInOneOut,OneInOneOut,TwoInOneOut") {
      total_paths--;
    }
  }

  EXPECT_EQ(total_paths, 0) << "Not all expected paths found\n=== LOG ===\n"
                            << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that OneOut published 1 message and TwoInOneOut published messages
  bool has_oneout_source = false;
  bool has_twoinoneout_source = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "OneOut->out") {
      has_oneout_source = true;
      EXPECT_EQ(count, 1) << "Expected 1 message from OneOut, got " << count;
    } else if (source == "TwoInOneOut->out") {
      has_twoinoneout_source = true;
      EXPECT_GE(count, 10) << "Expected at least 10 messages from TwoInOneOut, got " << count;
    }
  }
  EXPECT_TRUE(has_oneout_source) << "OneOut operator source messages not found";
  EXPECT_TRUE(has_twoinoneout_source) << "TwoInOneOut operator source messages not found";

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("OneOut->out"), std::string::npos)
      << "Printed output does not contain OneOut->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("TwoInOneOut->out"), std::string::npos)
      << "Printed output does not contain TwoInOneOut->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

/* CycleWithSourceVariantApp
 *
 * OneOut--->TwoInOneOut (probe)---->OneInOneOut--->OneInOneOut2 (probe)
 *             ^                                        |
 *             |                                        |
 *             +----------------------------------------+
 */

class CycleWithSourceVariantApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto one_out =
        make_operator<OneOutOp>("OneOut", make_condition<CountCondition>("count-condition", 1));
    auto two_in_one_out = make_operator<TwoInOneOutOp>(
        "TwoInOneOut",
        make_condition<CountCondition>(10),
        make_condition<PeriodicCondition>("periodic-condition", std::chrono::milliseconds(5)));
    auto one_in_one_out = make_operator<OneInOneOutOp>("OneInOneOut");
    auto one_in_one_out2 = make_operator<OneInOneOutOp>("OneInOneOut2");

    add_flow(one_out, two_in_one_out, {{"out", "in0"}});
    add_flow(two_in_one_out, one_in_one_out, {{"out", "in"}});
    add_flow(one_in_one_out, one_in_one_out2, {{"out", "in"}});
    add_flow(one_in_one_out2, two_in_one_out, {{"out", "in1"}});
  }
};

TEST(AddProbeOperator, CycleWithMiddleProbeEventBasedScheduler) {
  auto app = make_application<CycleWithSourceVariantApp>();
  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("TwoInOneOut");
  tracker.add_probe_operator("OneInOneOut2");

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler",
      holoscan::Arg{"worker_thread_number", static_cast<int64_t>(2)},
      holoscan::Arg{"stop_on_deadlock_timeout", static_cast<int64_t>(100)}));

  testing::internal::CaptureStdout();
  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  int total_paths = 5;
  // Paths:
  // 1. OneOut->TwoInOneOut (from root to probe1)
  // 2. OneOut->TwoInOneOut->OneInOneOut->OneInOneOut2 (from root to probe2)
  // 3. OneOut->TwoInOneOut->OneInOneOut->OneInOneOut2->TwoInOneOut (from root to the end of cycle)
  // 4. TwoInOneOut->OneInOneOut->OneInOneOut2->TwoInOneOut (the cycle itself)
  // 5. TwoInOneOut->OneInOneOut->OneInOneOut2 - TwoInOneOut will run without any input message from
  // OneOut, and such messages will be tracked until the probe2 operator
  EXPECT_EQ(tracker.get_num_paths(), total_paths) << log_output;

  auto paths = tracker.get_path_strings();

  for (const auto& path : paths) {
    if (path == "OneOut,TwoInOneOut" || path == "OneOut,TwoInOneOut,OneInOneOut,OneInOneOut2" ||
        path == "OneOut,TwoInOneOut,OneInOneOut,OneInOneOut2,TwoInOneOut" ||
        path == "TwoInOneOut,OneInOneOut,OneInOneOut2,TwoInOneOut" ||
        path == "TwoInOneOut,OneInOneOut,OneInOneOut2") {
      total_paths--;
    }
  }

  EXPECT_EQ(total_paths, 0) << "Not all expected paths found\n=== LOG ===\n"
                            << log_output << "\n===========\n";

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0) << "No source messages tracked";

  // Check that OneOut published 1 message and TwoInOneOut, OneInOneOut2 published
  // messages
  /*bool has_oneout_source = false;
  bool has_twoinoneout_source = false;
  bool has_oneinoneout2_source = false;
  for (const auto& [source, count] : src_messages) {
    if (source == "OneOut->out") {
      has_oneout_source = true;
      EXPECT_EQ(count, 1) << "Expected 1 message from OneOut, got " << count;
    } else if (source == "TwoInOneOut->out") {
      has_twoinoneout_source = true;
      EXPECT_EQ(count, 10) << "Expected 10 messages from TwoInOneOut, got " << count;
    } else if (source == "OneInOneOut2->out") {
      has_oneinoneout2_source = true;
      // Expect more than 1 message, no specific number because it depends on the scheduler
      EXPECT_GT(count, 1) << "Expected more than 1 message from OneInOneOut2, got " << count;
    }
  }
  EXPECT_TRUE(has_oneout_source) << "OneOut operator source messages not found";
  EXPECT_TRUE(has_twoinoneout_source) << "TwoInOneOut operator source messages not found";
  EXPECT_TRUE(has_oneinoneout2_source) << "OneInOneOut2 operator source messages not found";*/

  // Check that the printed output contains source message information in operator->transmitter
  // format
  EXPECT_NE(log_output.find("OneOut->out"), std::string::npos)
      << "Printed output does not contain OneOut->transmitter_name source message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("TwoInOneOut->out"), std::string::npos)
      << "Printed output does not contain TwoInOneOut->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_NE(log_output.find("OneInOneOut2->out"), std::string::npos)
      << "Printed output does not contain OneInOneOut2->transmitter_name (probed operator) source "
         "message info\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}
///////////////////////////////////////////////////////////////////////////////
// Tests for Metrics Verification
///////////////////////////////////////////////////////////////////////////////

TEST(AddProbeOperator, VerifyMetricsWithProbe) {
  auto app = make_application<LinearChainWithProbe>();
  auto& tracker = app->track(0, 0, 0);
  tracker.add_probe_operator("middle");

  app->run();

  auto paths = tracker.get_path_strings();

  // Verify that we can get metrics for each path
  for (const auto& path : paths) {
    auto max_latency = tracker.get_metric(path, DataFlowMetric::kMaxE2ELatency);
    auto avg_latency = tracker.get_metric(path, DataFlowMetric::kAvgE2ELatency);
    auto min_latency = tracker.get_metric(path, DataFlowMetric::kMinE2ELatency);

    // Latencies should be non-negative and reasonable
    EXPECT_GE(max_latency, 0.0) << "Path: " << path;
    EXPECT_GE(avg_latency, 0.0) << "Path: " << path;
    EXPECT_GE(min_latency, 0.0) << "Path: " << path;
    EXPECT_GE(max_latency, min_latency) << "Path: " << path;
  }

  // Verify source message counts
  auto src_messages = tracker.get_metric(DataFlowMetric::kNumSrcMessages);
  EXPECT_GT(src_messages.size(), 0);
}

}  // namespace holoscan
