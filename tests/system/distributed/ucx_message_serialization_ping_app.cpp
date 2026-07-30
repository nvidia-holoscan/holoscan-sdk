/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../env_wrapper.hpp"
#include "common/assert.hpp"

#include "ping_message_rx_op.hpp"
#include "ping_message_tx_op.hpp"
#include "utils.hpp"

using namespace std::string_literals;

namespace holoscan {

// Non-UCX variant (single-fragment)
class MessageSerializationApp : public holoscan::Application {
 public:
  explicit MessageSerializationApp(std::vector<MessageType> types) : types_(std::move(types)) {}

  void compose() override {
    using namespace holoscan;
    auto tx =
        make_operator<ops::PingMessageTxOp>("tx", make_condition<CountCondition>(types_.size()));
    tx->set_message_types(types_);
    auto rx =
        make_operator<ops::PingMessageRxOp>("rx", make_condition<CountCondition>(types_.size()));
    rx->set_message_types(types_);

    add_flow(tx, rx, {{"out", "in"}});
  }

 private:
  std::vector<MessageType> types_ = {MessageType::FLOAT};
};

TEST(MessageSerializationTests, TestSingleFragmentMessageSerialization) {
  // Test all message types in a single application run
  std::vector<MessageType> all_types = {MessageType::BOOL,
                                        MessageType::INT32,
                                        MessageType::UINT32,
                                        MessageType::FLOAT,
                                        MessageType::STRING,
                                        MessageType::VEC_BOOL,
                                        MessageType::VEC_FLOAT,
                                        MessageType::VEC_STRING,
                                        MessageType::SHARED_VEC_STRING,
                                        MessageType::VEC_VEC_BOOL,
                                        MessageType::VEC_VEC_FLOAT,
                                        MessageType::VEC_VEC_STRING,
                                        MessageType::VEC_INPUTSPEC,
                                        MessageType::CAMERA_POSE};

  HOLOSCAN_LOG_INFO("Creating MessageSerializationApp with {} message types", all_types.size());

  auto app = make_application<MessageSerializationApp>(all_types);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();

  // Count successful test cases
  size_t success_count = 0;
  size_t pos = 0;
  while ((pos = log_output.find("Found expected value in deserialized message for test case:",
                                pos)) != std::string::npos) {
    success_count++;
    pos++;
  }

  // If count doesn't match, print detailed information
  if (success_count != all_types.size()) {
    std::cerr << "\n=== Test case results ===\n";
    std::istringstream iss(log_output);
    std::string line;
    while (std::getline(iss, line)) {
      if (line.find("test case:") != std::string::npos) {
        std::cerr << line << "\n";
      }
    }
    std::cerr << "Expected " << all_types.size() << " test cases, got " << success_count << "\n";
  }

  EXPECT_EQ(success_count, all_types.size())
      << "Expected " << all_types.size() << " successful type tests, got " << success_count
      << "\nCheck stderr for details on which test cases passed/failed.";
}

// Multi-fragment UCX variant

class TxFragment : public holoscan::Fragment {
 public:
  explicit TxFragment(std::vector<MessageType> types) : types_(std::move(types)) {}

  void compose() override {
    using namespace holoscan;
    auto tx =
        make_operator<ops::PingMessageTxOp>("tx", make_condition<CountCondition>(types_.size()));
    tx->set_message_types(types_);

    add_operator(tx);
  }

 private:
  std::vector<MessageType> types_ = {MessageType::FLOAT};
};

class RxFragment : public holoscan::Fragment {
 public:
  explicit RxFragment(std::vector<MessageType> types) : types_(std::move(types)) {}

  void compose() override {
    using namespace holoscan;
    auto rx =
        make_operator<ops::PingMessageRxOp>("rx", make_condition<CountCondition>(types_.size()));
    rx->set_message_types(types_);
    add_operator(rx);
  }

 private:
  std::vector<MessageType> types_ = {MessageType::FLOAT};
};

class UcxMessageSerializationApp : public holoscan::Application {
 public:
  explicit UcxMessageSerializationApp(std::vector<MessageType> types) : types_(std::move(types)) {}

  void compose() override {
    using namespace holoscan;

    auto tx_fragment = make_fragment<TxFragment>("tx_fragment", types_);
    tx_fragment->is_metadata_enabled(true);
    auto rx_fragment = make_fragment<RxFragment>("rx_fragment", types_);
    rx_fragment->is_metadata_enabled(true);

    add_flow(tx_fragment, rx_fragment, {{"tx", "rx"}});
  }

 private:
  std::vector<MessageType> types_ = {MessageType::FLOAT};
};

TEST(UcxMessageSerializationTests, TestDistributedMessageSerialization) {
  // Test all message types in a single application run to reduce connection overhead
  std::vector<MessageType> all_types = {MessageType::BOOL,
                                        MessageType::INT32,
                                        MessageType::UINT32,
                                        MessageType::FLOAT,
                                        MessageType::STRING,
                                        MessageType::VEC_BOOL,
                                        MessageType::VEC_FLOAT,
                                        MessageType::VEC_STRING,
                                        MessageType::SHARED_VEC_STRING,
                                        MessageType::VEC_VEC_BOOL,
                                        MessageType::VEC_VEC_FLOAT,
                                        MessageType::VEC_VEC_STRING,
                                        MessageType::VEC_INPUTSPEC,
                                        MessageType::VEC_DOUBLE_LARGE,
                                        MessageType::CAMERA_POSE};

  // Set buffer size large enough to hold VEC_DOUBLE_LARGE message.
  // EnvVarWrapper saves the original value and restores it when it goes out of scope.
  EnvVarWrapper env_wrapper("HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE",
                            std::to_string(10 * 1024 * 1024));

  HOLOSCAN_LOG_INFO("Creating UcxMessageSerializationApp with {} message types", all_types.size());

  auto app = make_application<UcxMessageSerializationApp>(all_types);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  // check for the string that gets printed if receive value validation succeeded
  std::string log_output = testing::internal::GetCapturedStderr();

  // Count successful test cases
  size_t success_count = 0;
  size_t pos = 0;
  while ((pos = log_output.find("Found expected value in deserialized message for test case:",
                                pos)) != std::string::npos) {
    success_count++;
    pos++;
  }

  // If count doesn't match, print detailed information
  if (success_count != all_types.size()) {
    std::cerr << "\n=== Test case results ===\n";
    std::istringstream iss(log_output);
    std::string line;
    while (std::getline(iss, line)) {
      if (line.find("test case:") != std::string::npos) {
        std::cerr << line << "\n";
      }
    }
    std::cerr << "Expected " << all_types.size() << " test cases, got " << success_count << "\n";
  }

  EXPECT_EQ(success_count, all_types.size())
      << "Expected " << all_types.size() << " successful type tests, got " << success_count
      << "\nCheck stderr for details on which test cases passed/failed.";

  EXPECT_TRUE(remove_ignored_errors(log_output).find("error") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
