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

#ifndef PUBSUB_COMMON_INCLUDE_PUBSUB_HOLOSCAN_COMPONENT_PAYLOADS_HPP
#define PUBSUB_COMMON_INCLUDE_PUBSUB_HOLOSCAN_COMPONENT_PAYLOADS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/message.hpp>
#include <holoscan/core/messagelabel.hpp>
#include <holoscan/core/metadata.hpp>

namespace holoscan {

/// Encoded Message: codec name resolved via CodecRegistry + serialized payload bytes.
/// The caller is responsible for framing these fields into their backend wire format.
struct EncodedMessagePayload {
  std::string codec_name;
  std::vector<uint8_t> payload;
};

/// Encode a Message value via CodecRegistry lookup.
/// Returns the resolved codec name and the codec-serialized payload bytes.
expected<EncodedMessagePayload, RuntimeError> encode_message_payload(const Message& message);

/// Decode a Message value via CodecRegistry using the given codec name.
expected<Message, RuntimeError> decode_message_payload(const std::string& codec_name,
                                                       const std::vector<uint8_t>& payload);

/// Encode a MetadataDictionary into a self-contained byte blob.
///
/// Wire format:
///   [entry_count: u32]
///   per entry:
///     [key_size: u32] [codec_name_size: u32] [payload_size: u32]
///     [key bytes] [codec_name bytes] [payload bytes]
expected<std::vector<uint8_t>, RuntimeError> encode_metadata_dictionary_payload(
    const MetadataDictionary& metadata);

/// Decode a MetadataDictionary from a byte blob produced by encode_metadata_dictionary_payload.
expected<MetadataDictionary, RuntimeError> decode_metadata_dictionary_payload(
    const std::vector<uint8_t>& payload);

/// Encode a MessageLabel into a self-contained byte blob.
///
/// Wire format:
///   [path_count: u32]
///   per path:
///     [op_count: u32]
///     per op: [name_size: u32] [name bytes] [rec_timestamp: i64] [pub_timestamp: i64]
///   [frame_entry_count: u32]
///   per frame: [key_size: u32] [key bytes] [frame_value: u64]
expected<std::vector<uint8_t>, RuntimeError> encode_message_label_payload(
    const MessageLabel& label);

/// Decode a MessageLabel from a byte blob produced by encode_message_label_payload.
expected<MessageLabel, RuntimeError> decode_message_label_payload(
    const std::vector<uint8_t>& payload);

}  // namespace holoscan

#endif /* PUBSUB_COMMON_INCLUDE_PUBSUB_HOLOSCAN_COMPONENT_PAYLOADS_HPP */
