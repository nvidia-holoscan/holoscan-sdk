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
