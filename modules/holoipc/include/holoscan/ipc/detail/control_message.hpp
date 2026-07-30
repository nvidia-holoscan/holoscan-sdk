/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_IPC_DETAIL_CONTROL_MESSAGE_DATA_HPP
#define HOLOSCAN_IPC_DETAIL_CONTROL_MESSAGE_DATA_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan {
namespace ipc {

/** Opaque publication/sender handle for transport-agnostic callbacks. */
using PublicationHandle = std::vector<uint8_t>;

/**
 * Holoscan IPC protocol version string.
 * Use for PointerDescriptor.version and ControlMessage.version (same value on the wire).
 */
inline constexpr const char* kIpcProtocolVersion = "1.0";

/** Control message type; values match the DDS/IDL ControlMessageType for easy conversion. */
enum class ControlMessageType : int32_t { ACQUIRED, RELEASED, ACK, NACK };

/**
 * Transport-agnostic control message for the transport callback interface.
 * Mirrors the logical fields of the DDS gen::ControlMessage (version, key, message_type,
 * reply_to_topic_name).
 */
struct ControlMessage {
  std::string version;
  std::vector<uint8_t> key;
  ControlMessageType message_type = ControlMessageType::ACQUIRED;
  std::string reply_to_topic_name;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_DETAIL_CONTROL_MESSAGE_DATA_HPP
