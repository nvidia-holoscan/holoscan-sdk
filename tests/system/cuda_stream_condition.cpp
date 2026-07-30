/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp>
#include "../config.hpp"

namespace holoscan {

namespace ops {

// ============================================================================
// Operators for CudaStreamCondition tests
// ============================================================================

/**
 * @brief Simple receiver operator with a single input port for CudaStreamCondition testing.
 *
 * This operator receives entities and logs when compute is called. The CudaStreamCondition
 * should ensure that compute() is only called after GPU work on the input stream has completed.
 */
class CudaStreamCondSingleRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaStreamCondSingleRxOp)

  CudaStreamCondSingleRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Input port - CudaStreamCondition will be added externally
    spec.input<gxf::Entity>("in").condition(ConditionType::kNone);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_message = op_input.receive<gxf::Entity>("in");
    if (!maybe_message) {
      HOLOSCAN_LOG_ERROR("{}: Failed to receive message", name());
      return;
    }
    count_++;
    HOLOSCAN_LOG_INFO("{}: compute() called, message count = {}", name(), count_);
  }

 private:
  size_t count_ = 0;
};

/**
 * @brief Receiver operator with two input ports for CudaStreamCondition testing.
 *
 * This tests CudaStreamCondition monitoring multiple regular input ports.
 */
class CudaStreamCondDualRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaStreamCondDualRxOp)

  CudaStreamCondDualRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Two input ports - CudaStreamCondition will monitor both
    spec.input<gxf::Entity>("in1").condition(ConditionType::kNone);
    spec.input<gxf::Entity>("in2").condition(ConditionType::kNone);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_msg1 = op_input.receive<gxf::Entity>("in1");
    auto maybe_msg2 = op_input.receive<gxf::Entity>("in2");

    bool has_msg1 = maybe_msg1.has_value();
    bool has_msg2 = maybe_msg2.has_value();

    count_++;
    HOLOSCAN_LOG_INFO("{}: compute() called, count = {}, in1 = {}, in2 = {}",
                      name(),
                      count_,
                      has_msg1 ? "received" : "empty",
                      has_msg2 ? "received" : "empty");
  }

 private:
  size_t count_ = 0;
};

/**
 * @brief Receiver operator with kAnySize multi-receiver port for CudaStreamCondition testing.
 *
 * This tests CudaStreamCondition with multi-receiver ports where the actual port names
 * are "receivers:0", "receivers:1", etc.
 */
class CudaStreamCondMultiRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaStreamCondMultiRxOp)

  CudaStreamCondMultiRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Multi-receiver port - CudaStreamCondition will monitor all receivers:N ports
    spec.input<gxf::Entity>("receivers", IOSpec::kAnySize).condition(ConditionType::kNone);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_messages = op_input.receive<std::vector<gxf::Entity>>("receivers");
    if (!maybe_messages) {
      HOLOSCAN_LOG_ERROR("{}: Failed to receive messages", name());
      return;
    }
    auto& messages = maybe_messages.value();
    count_++;
    HOLOSCAN_LOG_INFO(
        "{}: compute() called, count = {}, received {} messages", name(), count_, messages.size());
  }

 private:
  size_t count_ = 0;
};

/**
 * @brief Receiver operator using the LEGACY Parameter<std::vector<IOSpec*>> pattern.
 *
 * This tests CudaStreamCondition with the legacy parameter-based multi-receiver,
 * which exercises the else branch in update_connector_arguments() where the port name
 * is not found in spec_->inputs() but the indexed ports are.
 */
class CudaStreamCondLegacyRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaStreamCondLegacyRxOp)

  CudaStreamCondLegacyRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Legacy pattern: use spec.param() with Parameter<std::vector<IOSpec*>>
    // This means "receivers" is a parameter name, NOT an input port name.
    // The indexed ports (receivers:0, receivers:1, etc.) are created dynamically.
    spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_messages = op_input.receive<std::vector<gxf::Entity>>("receivers");
    if (!maybe_messages) {
      HOLOSCAN_LOG_ERROR("{}: Failed to receive messages", name());
      return;
    }
    auto& messages = maybe_messages.value();
    count_++;
    HOLOSCAN_LOG_INFO(
        "{}: compute() called, count = {}, received {} messages", name(), count_, messages.size());
  }

 private:
  // Legacy member variable required for Parameter<std::vector<IOSpec*>> pattern
  Parameter<std::vector<IOSpec*>> receivers_;
  size_t count_ = 0;
};

/**
 * @brief Receiver operator with BOTH a kAnySize port AND a regular port.
 *
 * This tests CudaStreamCondition monitoring both types of ports simultaneously.
 */
class CudaStreamCondMixedRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaStreamCondMixedRxOp)

  CudaStreamCondMixedRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Regular input port
    spec.input<gxf::Entity>("regular_in").condition(ConditionType::kNone);
    // Multi-receiver port
    spec.input<gxf::Entity>("multi_in", IOSpec::kAnySize).condition(ConditionType::kNone);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_regular = op_input.receive<gxf::Entity>("regular_in");
    auto maybe_multi = op_input.receive<std::vector<gxf::Entity>>("multi_in");

    bool has_regular = maybe_regular.has_value();
    size_t multi_count = maybe_multi.has_value() ? maybe_multi.value().size() : 0;

    count_++;
    HOLOSCAN_LOG_INFO("{}: compute() called, count = {}, regular_in = {}, multi_in count = {}",
                      name(),
                      count_,
                      has_regular ? "received" : "empty",
                      multi_count);
  }

 private:
  size_t count_ = 0;
};

}  // namespace ops

// ============================================================================
// CudaStreamCondition "receiver" Backwards Compatibility Test Applications
// ============================================================================

/**
 * @brief Test application for CudaStreamCondition(s) with a single input port.
 *
 * When use_legacy=false (default): Uses CudaStreamCondition.
 * When use_legacy=true: Uses the legacy CudaStreamCondition.
 *
 * Tests that the condition correctly waits for GPU work to complete
 * before allowing the downstream operator to execute.
 */
class CudaStreamConditionSingleApp : public holoscan::Application {
 public:
  void use_legacy(bool value) { use_legacy_ = value; }

  void compose() override {
    const int32_t width = 64;
    const int32_t height = 64;

    // PingTensorTxOp will use its default internal CudaStreamPool
    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("async_device_allocation", true),
    });

    auto tx = make_operator<ops::PingTensorTxOp>("tx", make_condition<CountCondition>(5), tx_args);

    std::shared_ptr<ops::CudaStreamCondSingleRxOp> rx;

    if (use_legacy_) {
      // Legacy CudaStreamCondition
      auto stream_cond =
          make_condition<CudaStreamCondition>("stream_cond", Arg("receiver", std::string("in")));
      rx = make_operator<ops::CudaStreamCondSingleRxOp>("rx", stream_cond);
    } else {
      // CudaStreamCondition
      auto stream_cond =
          make_condition<CudaStreamCondition>("stream_cond", Arg("receivers", std::string("in")));
      rx = make_operator<ops::CudaStreamCondSingleRxOp>("rx", stream_cond);
    }

    add_flow(tx, rx, {{"out", "in"}});
  }

 private:
  bool use_legacy_ = false;
};

/**
 * @brief Test application for CudaStreamCondition(s) with two input ports.
 *
 * When use_legacy=false (default): Uses a single CudaStreamCondition to monitor both ports.
 * When use_legacy=true: Uses TWO separate CudaStreamConditions (one per port).
 *
 * This demonstrates the advantage of CudaStreamCondition over the legacy approach.
 */
class CudaStreamConditionDualApp : public holoscan::Application {
 public:
  void use_legacy(bool value) { use_legacy_ = value; }

  void compose() override {
    const int32_t width = 64;
    const int32_t height = 64;

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("async_device_allocation", true),
    });

    auto tx1 =
        make_operator<ops::PingTensorTxOp>("tx1", make_condition<CountCondition>(5), tx_args);
    auto tx2 =
        make_operator<ops::PingTensorTxOp>("tx2", make_condition<CountCondition>(5), tx_args);

    std::shared_ptr<ops::CudaStreamCondDualRxOp> rx;

    if (use_legacy_) {
      // Legacy approach: need TWO separate CudaStreamConditions, one for each port
      auto stream_cond1 =
          make_condition<CudaStreamCondition>("stream_cond1", Arg("receiver", std::string("in1")));
      auto stream_cond2 =
          make_condition<CudaStreamCondition>("stream_cond2", Arg("receiver", std::string("in2")));
      rx = make_operator<ops::CudaStreamCondDualRxOp>("rx", stream_cond1, stream_cond2);
    } else {
      //  approach: single condition monitors both ports
      auto stream_cond = make_condition<CudaStreamCondition>(
          "stream_cond", Arg("receivers", std::vector<std::string>{"in1", "in2"}));
      rx = make_operator<ops::CudaStreamCondDualRxOp>("rx", stream_cond);
    }

    add_flow(tx1, rx, {{"out", "in1"}});
    add_flow(tx2, rx, {{"out", "in2"}});
  }

 private:
  bool use_legacy_ = false;
};

/**
 * @brief Test application for CudaStreamCondition with kAnySize multi-receiver port.
 *
 * Tests that CudaStreamCondition correctly discovers and monitors all ports matching
 * the "receivers:N" pattern when given the base name "receivers".
 */
class CudaStreamConditionMultiRxApp : public holoscan::Application {
 public:
  void compose() override {
    const int32_t width = 64;
    const int32_t height = 64;

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("async_device_allocation", true),
    });

    // Create 3 source operators
    auto tx1 =
        make_operator<ops::PingTensorTxOp>("tx1", make_condition<CountCondition>(5), tx_args);
    auto tx2 =
        make_operator<ops::PingTensorTxOp>("tx2", make_condition<CountCondition>(5), tx_args);
    auto tx3 =
        make_operator<ops::PingTensorTxOp>("tx3", make_condition<CountCondition>(5), tx_args);

    // Create CudaStreamCondition monitoring "receivers" base name
    // This should automatically find receivers:0, receivers:1, receivers:2
    auto stream_cond = make_condition<CudaStreamCondition>(
        "stream_cond", Arg("receivers", std::string("receivers")));

    auto rx = make_operator<ops::CudaStreamCondMultiRxOp>("rx", stream_cond);

    // Connect all sources to the multi-receiver port
    add_flow(tx1, rx, {{"out", "receivers"}});
    add_flow(tx2, rx, {{"out", "receivers"}});
    add_flow(tx3, rx, {{"out", "receivers"}});
  }
};

/**
 * @brief Test application for CudaStreamCondition with LEGACY parameter-based multi-receiver.
 *
 * Tests that CudaStreamCondition works with the deprecated Parameter<std::vector<IOSpec*>>
 * pattern. This exercises the else branch in update_connector_arguments() where the base
 * name is not found in spec_->inputs().
 */
class CudaStreamConditionLegacyApp : public holoscan::Application {
 public:
  void compose() override {
    const int32_t width = 64;
    const int32_t height = 64;

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("async_device_allocation", true),
    });

    // Create 2 source operators
    auto tx1 =
        make_operator<ops::PingTensorTxOp>("tx1", make_condition<CountCondition>(5), tx_args);
    auto tx2 =
        make_operator<ops::PingTensorTxOp>("tx2", make_condition<CountCondition>(5), tx_args);

    // Create CudaStreamCondition monitoring "receivers" base name
    // For the legacy pattern, "receivers" is a parameter, not an input port.
    // The indexed ports (receivers:0, receivers:1) exist but the base doesn't.
    auto stream_cond = make_condition<CudaStreamCondition>(
        "stream_cond", Arg("receivers", std::string("receivers")));

    auto rx = make_operator<ops::CudaStreamCondLegacyRxOp>("rx", stream_cond);

    // Connect sources to the legacy multi-receiver (creates receivers:0, receivers:1)
    add_flow(tx1, rx, {{"out", "receivers"}});
    add_flow(tx2, rx, {{"out", "receivers"}});
  }
};

/**
 * @brief Test application for CudaStreamCondition with BOTH kAnySize and regular ports.
 *
 * Tests that CudaStreamCondition correctly handles a mix of regular ports and
 * kAnySize multi-receiver ports.
 */
class CudaStreamConditionMixedApp : public holoscan::Application {
 public:
  void compose() override {
    const int32_t width = 64;
    const int32_t height = 64;

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("async_device_allocation", true),
    });

    // One source for regular port
    auto tx_regular = make_operator<ops::PingTensorTxOp>(
        "tx_regular", make_condition<CountCondition>(5), tx_args);

    // Two sources for multi-receiver port
    auto tx_multi1 =
        make_operator<ops::PingTensorTxOp>("tx_multi1", make_condition<CountCondition>(5), tx_args);
    auto tx_multi2 =
        make_operator<ops::PingTensorTxOp>("tx_multi2", make_condition<CountCondition>(5), tx_args);

    // Create CudaStreamCondition monitoring both "regular_in" and "multi_in" (base name)
    auto stream_cond = make_condition<CudaStreamCondition>(
        "stream_cond", Arg("receivers", std::vector<std::string>{"regular_in", "multi_in"}));

    auto rx = make_operator<ops::CudaStreamCondMixedRxOp>("rx", stream_cond);

    // Connect regular port
    add_flow(tx_regular, rx, {{"out", "regular_in"}});
    // Connect multi-receiver port (creates multi_in:0, multi_in:1)
    add_flow(tx_multi1, rx, {{"out", "multi_in"}});
    add_flow(tx_multi2, rx, {{"out", "multi_in"}});
  }
};

/**
 * @brief Test application for CudaStreamCondition with check_all_messages=false.
 *
 * Tests that when check_all_messages is false, only the first message in the queue
 * is checked for CUDA streams (matching the behavior of the original CudaStreamCondition).
 */
class CudaStreamConditionFirstOnlyApp : public holoscan::Application {
 public:
  void compose() override {
    const int32_t width = 64;
    const int32_t height = 64;

    auto tx_args = ArgList({
        Arg("rows", height),
        Arg("columns", width),
        Arg("channels", 4),
        Arg("storage_type", std::string("device")),
        Arg("async_device_allocation", true),
    });

    auto tx = make_operator<ops::PingTensorTxOp>("tx", make_condition<CountCondition>(5), tx_args);

    // Create CudaStreamCondition with check_all_messages=false
    auto stream_cond = make_condition<CudaStreamCondition>(
        "stream_cond", Arg("receivers", std::string("in")), Arg("check_all_messages", false));

    auto rx = make_operator<ops::CudaStreamCondSingleRxOp>("rx", stream_cond);

    add_flow(tx, rx, {{"out", "in"}});
  }
};

}  // namespace holoscan

// ============================================================================
// CudaStreamCondition Tests
// ============================================================================

// Parameterized test for CudaStreamConditionSingleApp
// Tests both legacy CudaStreamCondition and CudaStreamCondition
class CudaStreamConditionSingleAppTest : public ::testing::TestWithParam<bool> {};

TEST_P(CudaStreamConditionSingleAppTest, TestSingleInputPort) {
  using namespace holoscan;

  bool use_legacy = GetParam();
  auto app = make_application<CudaStreamConditionSingleApp>();
  app->use_legacy(use_legacy);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app->run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // If an exception was thrown, fail with log output for debugging
  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify that compute was called the expected number of times (5)
  std::string compute_msg = "rx: compute() called, message count = 5";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected 5 compute() calls:\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify no errors were logged
  EXPECT_TRUE(log_output.find("error") == std::string::npos)
      << "Unexpected error in log output:\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

INSTANTIATE_TEST_SUITE_P(CudaStreamConditionApps, CudaStreamConditionSingleAppTest,
                         ::testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "LegacyAPI" : "NewAPI";
                         });

// Parameterized test for CudaStreamConditionDualApp
// Tests both legacy ("receiver" parameter) and new API ("receivers" parameter)
class CudaStreamConditionDualAppTest : public ::testing::TestWithParam<bool> {};

TEST_P(CudaStreamConditionDualAppTest, TestDualInputPorts) {
  using namespace holoscan;

  bool use_legacy = GetParam();
  auto app = make_application<CudaStreamConditionDualApp>();
  app->use_legacy(use_legacy);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app->run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // If an exception was thrown, fail with log output for debugging
  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify that compute was called and both inputs were received
  std::string compute_msg = "rx: compute() called, count = 5, in1 = received, in2 = received";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected 5 compute() calls with both inputs received:\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify no errors were logged
  EXPECT_TRUE(log_output.find("error") == std::string::npos)
      << "Unexpected error in log output:\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

INSTANTIATE_TEST_SUITE_P(CudaStreamConditionApps, CudaStreamConditionDualAppTest,
                         ::testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "LegacyAPI" : "NewAPI";
                         });

TEST(CudaStreamConditionApps, TestMultiReceiverPort) {
  using namespace holoscan;

  auto app = make_application<CudaStreamConditionMultiRxApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app->run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // If an exception was thrown, fail with log output for debugging
  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify that compute was called and received 3 messages (from 3 sources)
  std::string compute_msg = "rx: compute() called, count = 5, received 3 messages";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected 5 compute() calls with 3 messages each:\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify no errors were logged
  EXPECT_TRUE(log_output.find("error") == std::string::npos)
      << "Unexpected error in log output:\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(CudaStreamConditionApps, TestMixedPorts) {
  using namespace holoscan;

  auto app = make_application<CudaStreamConditionMixedApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app->run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // If an exception was thrown, fail with log output for debugging
  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify that compute was called with both regular and multi inputs
  // regular_in should be received, multi_in should have 2 messages (from 2 sources)
  std::string compute_msg =
      "rx: compute() called, count = 5, regular_in = received, multi_in count = 2";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected 5 compute() calls with regular_in received and 2 multi_in messages:\n=== LOG "
         "===\n"
      << log_output << "\n===========\n";

  // Verify no errors were logged
  EXPECT_TRUE(log_output.find("error") == std::string::npos)
      << "Unexpected error in log output:\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(CudaStreamConditionApps, TestLegacyParameterMultiReceiver) {
  using namespace holoscan;

  auto app = make_application<CudaStreamConditionLegacyApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app->run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // If an exception was thrown, fail with log output for debugging
  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify that compute was called and received 2 messages (from 2 sources)
  // This tests the legacy Parameter<std::vector<IOSpec*>> pattern
  std::string compute_msg = "rx: compute() called, count = 5, received 2 messages";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected 5 compute() calls with 2 messages each:\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify no errors were logged
  EXPECT_TRUE(log_output.find("error") == std::string::npos)
      << "Unexpected error in log output:\n=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(CudaStreamConditionApps, TestCheckFirstMessageOnly) {
  using namespace holoscan;

  auto app = make_application<CudaStreamConditionFirstOnlyApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  std::string exception_message;
  bool exception_thrown = false;

  try {
    app->run();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // If an exception was thrown, fail with log output for debugging
  if (exception_thrown) {
    FAIL() << "Application threw an exception: " << exception_message << "\n=== LOG OUTPUT ===\n"
           << log_output << "\n=================\n";
  }

  // Verify that compute was called the expected number of times (5)
  std::string compute_msg = "rx: compute() called, message count = 5";
  EXPECT_TRUE(log_output.find(compute_msg) != std::string::npos)
      << "Expected 5 compute() calls:\n=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify no errors were logged
  EXPECT_TRUE(log_output.find("error") == std::string::npos)
      << "Unexpected error in log output:\n=== LOG ===\n"
      << log_output << "\n===========\n";
}
