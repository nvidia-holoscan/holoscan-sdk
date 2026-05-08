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

/**
 * @file control_channel.hpp
 *
 * Control channel (ControlReader / ControlWriter) for ACQUIRED/RELEASED and ACK/NACK messages.
 */

#ifndef HOLOSCAN_IPC_DETAIL_CONTROLCHANNEL_HPP
#define HOLOSCAN_IPC_DETAIL_CONTROLCHANNEL_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/rtps/common/InstanceHandle.hpp>

#include <holoscan/ipc/detail/key.hpp>
#include <holoscan/ipc/transport/fastdds/gen/control_messagePubSubTypes.hpp>

namespace holoscan {
namespace ipc {
namespace transport {
namespace fastdds {

/**
 * @brief Reads control messages (DataReader side).
 *
 * Used by publishers to receive ACQUIRED/RELEASED messages from subscribers.
 * Single inheritance: DataReaderListener only.
 */
class ControlReader : public eprosima::fastdds::dds::DataReaderListener {
 public:
  using MessageCallback =
      std::function<void(const gen::ControlMessage& msg,
                         const eprosima::fastdds::rtps::InstanceHandle_t& publication_handle)>;

  using DisconnectionCallback = std::function<void(
      const std::vector<eprosima::fastdds::rtps::InstanceHandle_t>& currently_matched)>;

  ControlReader(eprosima::fastdds::dds::DomainParticipant* participant,
                const std::string& type_name, const std::string& topic_name,
                MessageCallback message_callback, DisconnectionCallback disconnection_callback);

  ~ControlReader() override;

  ControlReader(const ControlReader&) = delete;
  ControlReader& operator=(const ControlReader&) = delete;

  const std::string& get_topic_name() const;

 private:
  void on_data_available(eprosima::fastdds::dds::DataReader* reader) override;

  void on_subscription_matched(
      eprosima::fastdds::dds::DataReader* reader,
      const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;

  void on_liveliness_changed(
      eprosima::fastdds::dds::DataReader* reader,
      const eprosima::fastdds::dds::LivelinessChangedStatus& status) override;

  std::shared_ptr<eprosima::fastdds::dds::Topic> topic_;
  MessageCallback message_callback_;
  DisconnectionCallback disconnection_callback_;
  std::shared_ptr<eprosima::fastdds::dds::Subscriber> subscriber_;
  std::shared_ptr<eprosima::fastdds::dds::DataReader> reader_;
};

/**
 * @brief Writes control messages (DataWriter side).
 *
 * Used by subscribers to send ACQUIRED/RELEASED to publishers; by publishers to send ACK to
 * subscribers.
 */
class ControlWriter {
 public:
  /** Topic is obtained via make_topic (find first, create if not found). */
  ControlWriter(eprosima::fastdds::dds::DomainParticipant* participant,
                const std::string& type_name, const std::string& topic_name);

  ~ControlWriter();

  ControlWriter(const ControlWriter&) = delete;
  ControlWriter& operator=(const ControlWriter&) = delete;
  ControlWriter(ControlWriter&&) = default;
  ControlWriter& operator=(ControlWriter&&) = default;

  const std::string& get_topic_name() const;

  /** Send a control message. Set reply_to_topic_name (e.g. sender inbox for ACQUIRED/RELEASED;
   * publisher inbox for ACK/NACK). */
  bool send_message(const Key& key, gen::ControlMessageType message_type,
                    const std::string& reply_to_topic_name);

 private:
  std::shared_ptr<eprosima::fastdds::dds::Topic> topic_;
  std::shared_ptr<eprosima::fastdds::dds::Publisher> publisher_;
  std::shared_ptr<eprosima::fastdds::dds::DataWriter> writer_;
};

}  // namespace fastdds
}  // namespace transport
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_DETAIL_CONTROLCHANNEL_HPP
