/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdlib.h>

#include <iostream>
#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "common/assert.hpp"

#include "ping_message_rx_op.hpp"
#include "ping_message_tx_op.hpp"
#include "utils.hpp"

using namespace std::string_literals;

namespace holoscan {

class MessageTypeParmeterizedTestFixture : public ::testing::TestWithParam<MessageType> {};

class UcxMessageTypeParmeterizedTestFixture : public ::testing::TestWithParam<MessageType> {};

// Non-UCX variant
class MessageSerializationApp : public holoscan::Application {
 public:
  explicit MessageSerializationApp(MessageType type) : type_(type) {}

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMessageTxOp>("tx", make_condition<CountCondition>(1));
    tx->set_message_type(type_);
    auto rx = make_operator<ops::PingMessageRxOp>("rx");
    rx->set_message_type(type_);

    add_flow(tx, rx, {{"out", "in"}});
  }

 private:
  MessageType type_ = MessageType::FLOAT;
};

TEST_P(MessageTypeParmeterizedTestFixture, TestMessageSerializationApp) {
  MessageType message_type = GetParam();

  std::cout << "Creating MessageSerializationApp for type: "
            << message_type_name_map.at(message_type) << std::endl;

  auto app = make_application<MessageSerializationApp>(message_type);
  app->is_metadata_enabled(true);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Found expected value in deserialized message.") !=
              std::string::npos);
}

INSTANTIATE_TEST_CASE_P(MessageSerializationAppTests, MessageTypeParmeterizedTestFixture,
                        ::testing::Values(MessageType::BOOL, MessageType::INT32,
                                          MessageType::UINT32, MessageType::FLOAT,
                                          MessageType::STRING, MessageType::VEC_BOOL,
                                          MessageType::VEC_FLOAT, MessageType::VEC_STRING,
                                          MessageType::SHARED_VEC_STRING, MessageType::VEC_VEC_BOOL,
                                          MessageType::VEC_VEC_FLOAT, MessageType::VEC_VEC_STRING,
                                          MessageType::VEC_INPUTSPEC, MessageType::CAMERA_POSE));

// Multi-fragment UCX variant

class TxFragment : public holoscan::Fragment {
 public:
  explicit TxFragment(MessageType type) : type_(type) {}

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMessageTxOp>("tx", make_condition<CountCondition>(1));
    tx->set_message_type(type_);

    add_operator(tx);
  }

 private:
  MessageType type_ = MessageType::FLOAT;
};

class RxFragment : public holoscan::Fragment {
 public:
  explicit RxFragment(MessageType type) : type_(type) {}

  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<ops::PingMessageRxOp>("rx");
    rx->set_message_type(type_);
    add_operator(rx);
  }

 private:
  MessageType type_ = MessageType::FLOAT;
};

class UcxMessageSerializationApp : public holoscan::Application {
 public:
  // Inherit the constructor
  using Application::Application;

  explicit UcxMessageSerializationApp(MessageType type) : type_(type) {}

  void compose() override {
    using namespace holoscan;

    auto tx_fragment = make_fragment<TxFragment>("tx_fragment", type_);
    tx_fragment->is_metadata_enabled(true);
    auto rx_fragment = make_fragment<RxFragment>("rx_fragment", type_);
    rx_fragment->is_metadata_enabled(true);

    add_flow(tx_fragment, rx_fragment, {{"tx", "rx"}});
  }

 private:
  MessageType type_ = MessageType::FLOAT;
};

TEST_P(UcxMessageTypeParmeterizedTestFixture, TestUcxMessageSerializationApp) {
  MessageType message_type = GetParam();

  const char* env_orig = std::getenv("HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE");

  if (message_type == MessageType::VEC_DOUBLE_LARGE) {
    // message is larger than kDefaultUcxSerializationBufferSize
    // set HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE to a value large enough to hold the data
    std::string buffer_size = std::to_string(10 * 1024 * 1024);
    setenv("HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE", buffer_size.c_str(), 1);
  }

  HOLOSCAN_LOG_INFO("Creating UcxMessageSerializationApp for type: {}",
                    message_type_name_map.at(message_type));

  auto app = make_application<UcxMessageSerializationApp>(message_type);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  // check for the string that gets printed if receive value validation succeeded
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Found expected value in deserialized message.") !=
              std::string::npos);

  EXPECT_TRUE(remove_ignored_errors(log_output).find("error") == std::string::npos);

  // restore the original log level
  if (message_type == MessageType::VEC_DOUBLE_LARGE) {
    if (env_orig) {
      setenv("HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE", env_orig, 1);
    } else {
      unsetenv("HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE");
    }
  }
}

INSTANTIATE_TEST_CASE_P(UcxMessageSerializationAppTests, UcxMessageTypeParmeterizedTestFixture,
                        ::testing::Values(MessageType::BOOL, MessageType::INT32,
                                          MessageType::UINT32, MessageType::FLOAT,
                                          MessageType::STRING, MessageType::VEC_BOOL,
                                          MessageType::VEC_FLOAT, MessageType::VEC_STRING,
                                          MessageType::SHARED_VEC_STRING, MessageType::VEC_VEC_BOOL,
                                          MessageType::VEC_VEC_FLOAT, MessageType::VEC_VEC_STRING,
                                          MessageType::VEC_INPUTSPEC, MessageType::VEC_DOUBLE_LARGE,
                                          MessageType::CAMERA_POSE));

}  // namespace holoscan
