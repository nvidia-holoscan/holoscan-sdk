/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/ipc/transport/fastdds/fast_dds_transport.hpp>

#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fastdds/rtps/common/Guid.hpp>
#include <fastdds/rtps/common/InstanceHandle.hpp>

#include <holoscan/ipc/log.hpp>

#include <holoscan/ipc/detail/control_message.hpp>
#include <holoscan/ipc/detail/key.hpp>
#include <holoscan/ipc/detail/pointer_descriptor.hpp>
#include <holoscan/ipc/transport/fastdds/control_channel.hpp>
#include <holoscan/ipc/transport/fastdds/gen/control_messagePubSubTypes.hpp>
#include <holoscan/ipc/transport/fastdds/gen/pointer_descriptor.hpp>

namespace holoscan {
namespace ipc {
namespace transport {
namespace fastdds {

namespace {

// IDL-generated enums must stay ordinally aligned with ipc::* types; enum conversions below use
// static_cast<int32_t> and rely on these checks.
static_assert(static_cast<int32_t>(gen::ControlMessageType::ACQUIRED) ==
              static_cast<int32_t>(ControlMessageType::ACQUIRED));
static_assert(static_cast<int32_t>(gen::ControlMessageType::RELEASED) ==
              static_cast<int32_t>(ControlMessageType::RELEASED));
static_assert(static_cast<int32_t>(gen::ControlMessageType::ACK) ==
              static_cast<int32_t>(ControlMessageType::ACK));
static_assert(static_cast<int32_t>(gen::ControlMessageType::NACK) ==
              static_cast<int32_t>(ControlMessageType::NACK));

static_assert(static_cast<int32_t>(gen::HandleType::CUDA_IPC) ==
              static_cast<int32_t>(holoscan::ipc::HandleType::CUDA_IPC));

// Returns callback if non-null; otherwise logs and throws std::runtime_error.
holoscan::ipc::TransportInterface::Callback* validate_callback(
    holoscan::ipc::TransportInterface::Callback* callback) {
  if (callback == nullptr) {
    HOLOSCAN_IPC_LOG_ERROR("[FastDdsTransport] {}: Callback is null", __FUNCTION__);
    throw std::runtime_error("[FastDdsTransport] Callback is null");
  }
  return callback;
}

// Returns participant if non-null; otherwise logs and throws std::runtime_error.
::eprosima::fastdds::dds::DomainParticipant* validate_participant(
    ::eprosima::fastdds::dds::DomainParticipant* participant) {
  if (participant == nullptr) {
    HOLOSCAN_IPC_LOG_ERROR("[FastDdsTransport] {}: DomainParticipant is null", __FUNCTION__);
    throw std::runtime_error("[FastDdsTransport] DomainParticipant is null");
  }
  return participant;
}

// Creates and registers ControlMessage type with participant. Throws on failure.
::eprosima::fastdds::dds::TypeSupport make_type_support(
    ::eprosima::fastdds::dds::DomainParticipant* participant) {
  ::eprosima::fastdds::dds::TypeSupport type_support(new gen::ControlMessagePubSubType());
  if (type_support.register_type(participant) != ::eprosima::fastdds::dds::RETCODE_OK) {
    throw std::runtime_error("[FastDdsTransport] Failed to register message type");
  }
  return type_support;
}

std::string make_inbox_topic_name(::eprosima::fastdds::dds::DomainParticipant* participant) {
  const auto& g = participant->guid();
  std::stringstream ss;
  ss << "holoscan_ipc_fastdds_";
  ss << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < eprosima::fastdds::rtps::GuidPrefix_t::size; ++i) {
    ss << std::setw(2) << static_cast<int>(static_cast<unsigned char>(g.guidPrefix.value[i]));
  }
  ss << "_";
  for (unsigned int i = 0; i < eprosima::fastdds::rtps::EntityId_t::size; ++i) {
    ss << std::setw(2) << static_cast<int>(static_cast<unsigned char>(g.entityId.value[i]));
  }
  return ss.str();
}

holoscan::ipc::ControlMessage to_control_message(const gen::ControlMessage& dds_msg) {
  holoscan::ipc::ControlMessage data;
  data.version = dds_msg.version();
  data.key = dds_msg.key();
  data.message_type = static_cast<ControlMessageType>(static_cast<int32_t>(dds_msg.message_type()));
  data.reply_to_topic_name = dds_msg.reply_to_topic_name();
  return data;
}

holoscan::ipc::PublicationHandle to_publication_handle(
    const eprosima::fastdds::rtps::InstanceHandle_t& h) {
  holoscan::ipc::PublicationHandle out;
  out.resize(sizeof(eprosima::fastdds::rtps::InstanceHandle_t));
  std::memcpy(out.data(), &h, sizeof(eprosima::fastdds::rtps::InstanceHandle_t));
  return out;
}

std::vector<holoscan::ipc::PublicationHandle> to_publication_handles(
    const std::vector<eprosima::fastdds::rtps::InstanceHandle_t>& matched) {
  std::vector<holoscan::ipc::PublicationHandle> out;
  out.reserve(matched.size());
  for (const auto& h : matched) {
    out.push_back(to_publication_handle(h));
  }
  return out;
}

// Adapter: implements TransportInterface::ControlWriter and forwards to DDS ControlWriter.
class FastDdsControlWriterAdapter : public holoscan::ipc::TransportInterface::ControlWriter {
 public:
  FastDdsControlWriterAdapter(::eprosima::fastdds::dds::DomainParticipant* participant,
                              const std::string& type_name, const std::string& topic_name)
      : impl_(participant, type_name, topic_name) {}

  const std::string& get_topic_name() const override { return impl_.get_topic_name(); }

  bool send_message(const holoscan::ipc::Key& key, holoscan::ipc::ControlMessageType message_type,
                    const std::string& reply_to_topic_name) override {
    gen::ControlMessageType dds_type =
        static_cast<gen::ControlMessageType>(static_cast<int32_t>(message_type));
    return impl_.send_message(key, dds_type, reply_to_topic_name);
  }

 private:
  fastdds::ControlWriter impl_;
};

}  // unnamed namespace

FastDdsTransport::FastDdsTransport(::eprosima::fastdds::dds::DomainParticipant* participant)
    : participant_(validate_participant(participant)),
      type_support_(make_type_support(participant)),
      inbox_topic_name_(make_inbox_topic_name(participant)) {}

void FastDdsTransport::start(TransportInterface::Callback* callback) {
  if (control_reader_) {
    throw std::runtime_error("[FastDdsTransport] start() called twice");
  }
  callback_ = validate_callback(callback);
  control_reader_ = std::make_unique<ControlReader>(
      participant_,
      type_support_.get_type_name(),
      inbox_topic_name_,
      [this](const gen::ControlMessage& msg,
             const eprosima::fastdds::rtps::InstanceHandle_t& publication_handle) {
        callback_->on_control_message(to_control_message(msg),
                                      to_publication_handle(publication_handle));
      },
      [this](const std::vector<eprosima::fastdds::rtps::InstanceHandle_t>& currently_matched) {
        callback_->on_control_disconnections(to_publication_handles(currently_matched));
      });
}

void FastDdsTransport::stop() {
  control_reader_.reset();
  callback_ = nullptr;
}

const std::string& FastDdsTransport::get_inbox_topic_name() const {
  return inbox_topic_name_;
}

std::shared_ptr<TransportInterface::ControlWriter> FastDdsTransport::create_control_writer(
    const std::string& topic_name) {
  return std::make_shared<FastDdsControlWriterAdapter>(
      participant_, type_support_.get_type_name(), topic_name);
}

PointerDescriptor FastDdsTransport::from_wire_descriptor(const std::any& wire_descriptor) {
  const auto& g = std::any_cast<const gen::PointerDescriptor&>(wire_descriptor);
  PointerDescriptor p;
  p.version = g.version();
  p.key = g.key();
  p.handle_type = static_cast<holoscan::ipc::HandleType>(static_cast<int32_t>(g.handle_type()));
  p.handle = g.handle();
  p.reply_to_topic_name = g.reply_to_topic_name();
  return p;
}

std::any FastDdsTransport::to_wire_descriptor(const PointerDescriptor& descriptor) {
  gen::PointerDescriptor g;
  g.version(descriptor.version);
  g.key(descriptor.key);
  g.handle_type(static_cast<gen::HandleType>(static_cast<int32_t>(descriptor.handle_type)));
  g.handle(descriptor.handle);
  g.reply_to_topic_name(descriptor.reply_to_topic_name);
  return std::any(g);
}

}  // namespace fastdds
}  // namespace transport
}  // namespace ipc
}  // namespace holoscan
