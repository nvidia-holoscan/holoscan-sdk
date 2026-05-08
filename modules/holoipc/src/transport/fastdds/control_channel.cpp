/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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
 * @file control_channel.cpp
 *
 * Implementation of ControlReader and ControlWriter for the control channel.
 */

#include <holoscan/ipc/transport/fastdds/control_channel.hpp>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fastdds/dds/core/Time_t.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastdds/dds/subscriber/qos/SubscriberQos.hpp>

#include <holoscan/ipc/detail/control_message.hpp>
#include <holoscan/ipc/log.hpp>

namespace holoscan {
namespace ipc {
namespace transport {
namespace fastdds {

namespace {

// Control-channel QoS timings. Centralized for tuning; may be made configurable if load tests show
// liveliness or heartbeat as too aggressive under GPU/CPU pressure.
constexpr int kControlChannelLivelinessLeaseSec = 0;
constexpr uint32_t kControlChannelLivelinessLeaseNanosec = 500000000;  // 500 ms

constexpr uint32_t kControlChannelReaderHeartbeatResponseDelayNanosec = 10000000;  // 10 ms

constexpr uint32_t kControlChannelWriterLivelinessAnnouncementNanosec = 100000000;  // 100 ms
constexpr uint32_t kControlChannelWriterHeartbeatPeriodNanosec = 50000000;          // 50 ms
constexpr uint32_t kControlChannelWriterNackResponseDelayNanosec = 5000000;         // 5 ms

using namespace eprosima::fastdds::dds;

/** Builds a ControlMessage with the given key, type, and reply topic name. */
gen::ControlMessage make_control_message(const Key& key, gen::ControlMessageType message_type,
                                         const std::string& reply_to_topic_name) {
  gen::ControlMessage msg;
  msg.version(kIpcProtocolVersion);
  msg.key(key.to_bytes());
  msg.message_type(message_type);
  msg.reply_to_topic_name(reply_to_topic_name);
  return msg;
}

/** Throws std::invalid_argument if \a f is empty; returns \a f (moved). */
template <typename F>
F validate_function(F f, const char* msg = "[ControlChannel] Invalid function") {
  if (!f) {
    throw std::invalid_argument(msg);
  }
  return f;
}

/**
 * Returns a shared_ptr to a Topic with a custom deleter (participant->delete_topic).
 *
 * If a topic with \a topic_name is already registered on \a participant, `find_topic`
 * returns a **new Topic proxy** bound to that existing registration (the underlying
 * topic is shared; the returned object is still a distinct handle that must be
 * `delete_topic`'d when done — same as after `create_topic`). If none exists, we
 * `create_topic` so we have a proxy to attach a reader or writer (e.g. subscriber
 * sending to the publisher inbox, or publisher to a subscriber inbox, possibly on the
 * same participant as an existing ControlReader inbox topic).
 *
 * Always called from ControlReader / ControlWriter construction; throws if \a participant
 * is null.
 */
std::shared_ptr<Topic> make_topic(DomainParticipant* participant, const std::string& topic_name,
                                  const std::string& type_name) {
  if (participant == nullptr) {
    throw std::invalid_argument("[ControlChannel] Cannot create topic with null participant");
  }
  Topic* raw = participant->find_topic(topic_name, Duration_t(0, 0));
  if (raw == nullptr) {
    raw = participant->create_topic(topic_name, type_name, participant->get_default_topic_qos());
  }
  auto topic = std::shared_ptr<Topic>(raw, [participant](Topic* t) {
    if (t != nullptr) {
      participant->delete_topic(t);
    }
  });
  if (!topic) {
    throw std::runtime_error("[ControlChannel] Failed to create topic");
  }
  return topic;
}

/**
 * Returns a shared_ptr to a Subscriber with a custom deleter
 * (participant->delete_subscriber).
 */
std::shared_ptr<Subscriber> make_subscriber(DomainParticipant* participant) {
  auto sub = std::shared_ptr<Subscriber>(
      participant->create_subscriber(
          participant->get_default_subscriber_qos(), nullptr, StatusMask::none()),
      [participant](Subscriber* s) {
        if (s != nullptr) {
          participant->delete_subscriber(s);
        }
      });
  if (!sub) {
    throw std::runtime_error("[ControlChannel] Failed to create subscriber");
  }
  return sub;
}

/**
 * Returns DataReaderQos configured for the control channel (reliable, keep-all,
 * transient-local, liveliness).
 */
DataReaderQos control_channel_reader_qos() {
  DataReaderQos qos = DATAREADER_QOS_DEFAULT;
  qos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  qos.history().kind = KEEP_ALL_HISTORY_QOS;
  qos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
  qos.liveliness().kind = MANUAL_BY_PARTICIPANT_LIVELINESS_QOS;
  qos.liveliness().lease_duration.seconds = kControlChannelLivelinessLeaseSec;
  qos.liveliness().lease_duration.nanosec = kControlChannelLivelinessLeaseNanosec;
  qos.reliable_reader_qos().times.heartbeat_response_delay.seconds = 0;
  qos.reliable_reader_qos().times.heartbeat_response_delay.nanosec =
      kControlChannelReaderHeartbeatResponseDelayNanosec;
  return qos;
}

/**
 * Returns a shared_ptr to a DataReader with control-channel QoS and a custom deleter
 * (subscriber->delete_datareader).
 */
std::shared_ptr<DataReader> make_reader(const std::shared_ptr<Subscriber>& subscriber, Topic* topic,
                                        DataReaderListener* listener) {
  StatusMask listener_mask = StatusMask::data_available();
  listener_mask |= StatusMask::subscription_matched();
  listener_mask |= StatusMask::liveliness_changed();

  auto reader = std::shared_ptr<DataReader>(
      subscriber->create_datareader(topic, control_channel_reader_qos(), listener, listener_mask),
      [subscriber](DataReader* r) {
        if (r != nullptr) {
          subscriber->delete_datareader(r);
        }
      });

  if (!reader) {
    throw std::runtime_error("[ControlChannel] Failed to create DataReader");
  }
  return reader;
}

/**
 * Returns a shared_ptr to a Publisher with a custom deleter
 * (participant->delete_publisher).
 */
std::shared_ptr<Publisher> make_publisher(DomainParticipant* participant) {
  auto pub = std::shared_ptr<Publisher>(
      participant->create_publisher(
          participant->get_default_publisher_qos(), nullptr, StatusMask::none()),
      [participant](Publisher* p) {
        if (p != nullptr) {
          participant->delete_publisher(p);
        }
      });
  if (!pub) {
    throw std::runtime_error("[ControlChannel] Failed to create publisher");
  }
  return pub;
}

/**
 * Returns DataWriterQos configured for the control channel (reliable, keep-all,
 * transient-local, liveliness).
 */
DataWriterQos control_channel_writer_qos() {
  DataWriterQos qos = DATAWRITER_QOS_DEFAULT;
  qos.reliability().kind = RELIABLE_RELIABILITY_QOS;
  qos.history().kind = KEEP_ALL_HISTORY_QOS;
  qos.durability().kind = TRANSIENT_LOCAL_DURABILITY_QOS;
  qos.liveliness().kind = MANUAL_BY_PARTICIPANT_LIVELINESS_QOS;
  qos.liveliness().lease_duration.seconds = kControlChannelLivelinessLeaseSec;
  qos.liveliness().lease_duration.nanosec = kControlChannelLivelinessLeaseNanosec;
  qos.liveliness().announcement_period.seconds = 0;
  qos.liveliness().announcement_period.nanosec = kControlChannelWriterLivelinessAnnouncementNanosec;
  qos.reliable_writer_qos().times.heartbeat_period.seconds = 0;
  qos.reliable_writer_qos().times.heartbeat_period.nanosec =
      kControlChannelWriterHeartbeatPeriodNanosec;
  qos.reliable_writer_qos().times.nack_response_delay.seconds = 0;
  qos.reliable_writer_qos().times.nack_response_delay.nanosec =
      kControlChannelWriterNackResponseDelayNanosec;
  return qos;
}

/**
 * Returns a shared_ptr to a DataWriter with control-channel QoS and a custom deleter
 * (publisher->delete_datawriter).
 */
std::shared_ptr<DataWriter> make_writer(const std::shared_ptr<Publisher>& publisher, Topic* topic) {
  auto writer = std::shared_ptr<DataWriter>(
      publisher->create_datawriter(
          topic, control_channel_writer_qos(), nullptr, StatusMask::none()),
      [publisher](DataWriter* w) {
        if (w != nullptr) {
          publisher->delete_datawriter(w);
        }
      });
  if (!writer) {
    throw std::runtime_error("[ControlChannel] Failed to create DataWriter");
  }
  return writer;
}

/**
 * Gets matched publications, logs on error or "Currently matched publications", then
 * invokes \a callback.
 */
void notify_disconnection(DataReader* reader, const ControlReader::DisconnectionCallback& callback,
                          const std::string& log_suffix = "") {
  std::vector<InstanceHandle_t> currently_matched;
  ReturnCode_t return_code = reader->get_matched_publications(currently_matched);
  if (return_code != RETCODE_OK) {
    HOLOSCAN_IPC_LOG_ERROR("{}: Failed to get matched publications (code {})",
                           __FUNCTION__,
                           static_cast<int>(return_code));
    return;
  }
  HOLOSCAN_IPC_LOG_DEBUG("{}: Currently matched publications: {}{}",
                         __FUNCTION__,
                         currently_matched.size(),
                         log_suffix);
  callback(currently_matched);
}

}  // namespace

// ========== ControlReader ==========

ControlReader::ControlReader(eprosima::fastdds::dds::DomainParticipant* participant,
                             const std::string& type_name, const std::string& topic_name,
                             MessageCallback message_callback,
                             DisconnectionCallback disconnection_callback)
    : topic_(make_topic(participant, topic_name, type_name)),
      message_callback_(validate_function(std::move(message_callback),
                                          "[ControlReader] message_callback must not be null")),
      disconnection_callback_(
          validate_function(std::move(disconnection_callback),
                            "[ControlReader] disconnection_callback must not be null")),
      subscriber_(make_subscriber(participant)),
      reader_(make_reader(subscriber_, topic_.get(), this)) {
  HOLOSCAN_IPC_LOG_INFO("{}: Reader initialized on topic: {}", __FUNCTION__, topic_->get_name());
}

ControlReader::~ControlReader() = default;

const std::string& ControlReader::get_topic_name() const {
  return topic_->get_name();
}

void ControlReader::on_data_available(eprosima::fastdds::dds::DataReader* reader) {
  using namespace eprosima::fastdds::dds;

  gen::ControlMessage msg;
  SampleInfo info;

  while (reader->take_next_sample(&msg, &info) == RETCODE_OK) {
    if (!info.valid_data) {
      continue;
    }

    HOLOSCAN_IPC_LOG_DEBUG("{}: Dispatching control message (message_type={})",
                           __FUNCTION__,
                           static_cast<int>(msg.message_type()));
    message_callback_(msg, info.publication_handle);
  }
}

void ControlReader::on_subscription_matched(
    eprosima::fastdds::dds::DataReader* reader,
    const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) {
  using namespace eprosima::fastdds::dds;

  // No change in matched count: nothing to do.
  if (info.current_count_change == 0) {
    return;
  }

  // Remote DataWriter(s) matched (publications we read from): log and return.
  if (info.current_count_change > 0) {
    std::ostringstream oss;
    oss << info.last_publication_handle;
    HOLOSCAN_IPC_LOG_DEBUG(
        "{}: Remote DataWriter matched (publication): {}", __FUNCTION__, oss.str());
    return;
  }

  // Remote writer(s) unmatched: notify callback with currently matched set.
  HOLOSCAN_IPC_LOG_DEBUG(
      "{}: {} remote DataWriter(s) unmatched", __FUNCTION__, (-info.current_count_change));
  notify_disconnection(reader, disconnection_callback_);
}

void ControlReader::on_liveliness_changed(
    eprosima::fastdds::dds::DataReader* reader,
    const eprosima::fastdds::dds::LivelinessChangedStatus& status) {
  using namespace eprosima::fastdds::dds;

  // No change in liveliness: nothing to do.
  if (status.not_alive_count_change == 0 && status.alive_count_change == 0) {
    return;
  }

  // Remote writer(s) became alive (liveliness): log and return.
  if (status.alive_count_change > 0) {
    HOLOSCAN_IPC_LOG_DEBUG("{}: {} remote DataWriter(s) became alive (liveliness)",
                           __FUNCTION__,
                           status.alive_count_change);
    return;
  }

  // Remote writer(s) lost liveliness: notify callback with currently matched set.
  HOLOSCAN_IPC_LOG_DEBUG(
      "{}: {} remote DataWriter(s) lost liveliness (total alive: {}, not alive: {})",
      __FUNCTION__,
      status.not_alive_count_change,
      status.alive_count,
      status.not_alive_count);
  notify_disconnection(
      reader, disconnection_callback_, " (alive: " + std::to_string(status.alive_count) + ")");
}

// ========== ControlWriter ==========

ControlWriter::ControlWriter(eprosima::fastdds::dds::DomainParticipant* participant,
                             const std::string& type_name, const std::string& topic_name)
    : topic_(make_topic(participant, topic_name, type_name)),
      publisher_(make_publisher(participant)),
      writer_(make_writer(publisher_, topic_.get())) {
  HOLOSCAN_IPC_LOG_INFO("{}: Writer initialized on topic: {}", __FUNCTION__, topic_->get_name());
}

ControlWriter::~ControlWriter() = default;

const std::string& ControlWriter::get_topic_name() const {
  return topic_->get_name();
}

bool ControlWriter::send_message(const Key& key, gen::ControlMessageType message_type,
                                 const std::string& reply_to_topic_name) {
  gen::ControlMessage msg = make_control_message(key, message_type, reply_to_topic_name);

  eprosima::fastdds::dds::ReturnCode_t ret = writer_->write(&msg);
  if (ret != eprosima::fastdds::dds::RETCODE_OK) {
    HOLOSCAN_IPC_LOG_ERROR(
        "{}: write() failed with return code {}", __FUNCTION__, static_cast<int>(ret));
    return false;
  }

  HOLOSCAN_IPC_LOG_DEBUG(
      "{}: wrote control message (reply_to={})", __FUNCTION__, reply_to_topic_name);
  return true;
}

}  // namespace fastdds
}  // namespace transport
}  // namespace ipc
}  // namespace holoscan
