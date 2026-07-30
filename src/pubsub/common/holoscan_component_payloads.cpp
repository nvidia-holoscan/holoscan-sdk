/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/common/holoscan_component_payloads.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include <holoscan/core/codec_registry.hpp>
#include <holoscan/core/endpoint.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/message.hpp>
#include <holoscan/core/messagelabel.hpp>
#include <holoscan/core/metadata.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {

// Safety limits enforced on both encode and decode paths so that a corrupted or
// adversarial payload cannot cause unbounded memory allocation.
constexpr uint32_t kMaxEntryCount = 4096;
constexpr uint32_t kMaxNameSize = 4096;
constexpr uint32_t kMaxCodecNameSize = 4096;
constexpr uint32_t kMaxPayloadSize = 64 * 1024 * 1024;  // 64 MiB safety cap
constexpr uint32_t kMaxPaths = 4096;
constexpr uint32_t kMaxOpsPerPath = 4096;
constexpr uint32_t kMaxFrameEntries = 4096;

/// Entry header for metadata dictionary wire format.
struct MetadataEntryHeader {
  uint32_t key_size = 0;
  uint32_t codec_name_size = 0;
  uint32_t payload_size = 0;
};

/// Simple byte-buffer Endpoint for in-memory serialization/deserialization.
class VectorEndpoint : public Endpoint {
 public:
  explicit VectorEndpoint(std::vector<uint8_t>* output) : output_(output) {}
  explicit VectorEndpoint(const std::vector<uint8_t>* input) : input_(input) {}

  bool is_write_available() override { return output_ != nullptr; }
  bool is_read_available() override { return input_ != nullptr && pos_ < input_->size(); }

  expected<size_t, RuntimeError> write(const void* data, size_t size) override {
    if (!output_) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Endpoint not writable"));
    }
    if (size == 0)
      return 0UL;
    const auto* p = static_cast<const uint8_t*>(data);
    output_->insert(output_->end(), p, p + size);
    return size;
  }

  expected<size_t, RuntimeError> read(void* data, size_t size) override {
    if (!input_) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Endpoint not readable"));
    }
    if (pos_ + size > input_->size()) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Insufficient data in payload"));
    }
    if (size > 0)
      std::memcpy(data, input_->data() + pos_, size);
    pos_ += size;
    return size;
  }

  expected<void, RuntimeError> write_ptr(const void* pointer, size_t size,
                                         MemoryStorageType /*type*/) override {
    auto r = write(pointer, size);
    if (!r)
      return make_unexpected<RuntimeError>(std::move(r.error()));
    return {};
  }

 private:
  std::vector<uint8_t>* output_ = nullptr;
  const std::vector<uint8_t>* input_ = nullptr;
  size_t pos_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

expected<EncodedMessagePayload, RuntimeError> encode_message_payload(const Message& message) {
  auto& registry = CodecRegistry::get_instance();

  auto maybe_name = registry.index_to_name(std::type_index(message.value().type()));
  if (!maybe_name) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "No codec registered for message value type"));
  }

  std::vector<uint8_t> payload;
  VectorEndpoint ep(&payload);
  auto serialize_func = registry.get_serializer(maybe_name.value());
  auto result = serialize_func(message, &ep);
  if (!result) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Codec serialize failed for message payload"));
  }

  return EncodedMessagePayload{maybe_name.value(), std::move(payload)};
}

expected<Message, RuntimeError> decode_message_payload(const std::string& codec_name,
                                                       const std::vector<uint8_t>& payload) {
  auto& registry = CodecRegistry::get_instance();

  auto codec_index = registry.name_to_index(codec_name);
  if (!codec_index) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Unknown codec name: " + codec_name));
  }

  const std::vector<uint8_t>* view = &payload;
  VectorEndpoint ep(view);
  const auto& deserialize_func = registry.get_deserializer(codec_name);
  auto maybe_msg = deserialize_func(&ep);
  if (!maybe_msg) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Codec deserialize failed for: " + codec_name));
  }
  return maybe_msg.value();
}

// ---------------------------------------------------------------------------
// MetadataDictionary
// ---------------------------------------------------------------------------

expected<std::vector<uint8_t>, RuntimeError> encode_metadata_dictionary_payload(
    const MetadataDictionary& metadata) {
  std::vector<uint8_t> buf;
  VectorEndpoint ep(&buf);

  uint32_t entry_count = static_cast<uint32_t>(metadata.size());
  if (entry_count > kMaxEntryCount) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Metadata entry count exceeds limit"));
  }
  auto count_r = ep.write_trivial_type(&entry_count);
  if (!count_r)
    return forward_error(count_r);

  for (const auto& [key, value] : metadata) {
    if (!value) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Null metadata value for key: " + key));
    }

    auto maybe_encoded = encode_message_payload(*value);
    if (!maybe_encoded)
      return forward_error(maybe_encoded);

    MetadataEntryHeader hdr;
    hdr.key_size = static_cast<uint32_t>(key.size());
    hdr.codec_name_size = static_cast<uint32_t>(maybe_encoded.value().codec_name.size());
    hdr.payload_size = static_cast<uint32_t>(maybe_encoded.value().payload.size());

    if (hdr.key_size > kMaxNameSize || hdr.codec_name_size > kMaxCodecNameSize ||
        hdr.payload_size > kMaxPayloadSize) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Metadata entry exceeds size limits"));
    }

    auto hdr_r = ep.write_trivial_type(&hdr);
    if (!hdr_r)
      return forward_error(hdr_r);
    if (hdr.key_size > 0) {
      auto r = ep.write(key.data(), key.size());
      if (!r)
        return forward_error(r);
    }
    if (hdr.codec_name_size > 0) {
      auto r = ep.write(maybe_encoded.value().codec_name.data(),
                        maybe_encoded.value().codec_name.size());
      if (!r)
        return forward_error(r);
    }
    if (hdr.payload_size > 0) {
      auto r = ep.write(maybe_encoded.value().payload.data(), maybe_encoded.value().payload.size());
      if (!r)
        return forward_error(r);
    }
  }
  return buf;
}

expected<MetadataDictionary, RuntimeError> decode_metadata_dictionary_payload(
    const std::vector<uint8_t>& payload) {
  const std::vector<uint8_t>* view = &payload;
  VectorEndpoint ep(view);

  uint32_t entry_count = 0;
  auto count_r = ep.read_trivial_type(&entry_count);
  if (!count_r)
    return forward_error(count_r);
  if (entry_count > kMaxEntryCount) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Metadata entry count exceeds limit"));
  }

  MetadataDictionary metadata(MetadataPolicy::kUpdate);
  for (uint32_t i = 0; i < entry_count; ++i) {
    MetadataEntryHeader hdr;
    auto hdr_r = ep.read_trivial_type(&hdr);
    if (!hdr_r)
      return forward_error(hdr_r);
    if (hdr.key_size > kMaxNameSize || hdr.codec_name_size > kMaxCodecNameSize ||
        hdr.payload_size > kMaxPayloadSize) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Metadata entry exceeds size limits"));
    }

    std::string key(hdr.key_size, '\0');
    std::string codec_name(hdr.codec_name_size, '\0');
    std::vector<uint8_t> value_payload(hdr.payload_size);

    if (hdr.key_size > 0) {
      auto r = ep.read(key.data(), hdr.key_size);
      if (!r)
        return forward_error(r);
    }
    if (hdr.codec_name_size > 0) {
      auto r = ep.read(codec_name.data(), hdr.codec_name_size);
      if (!r)
        return forward_error(r);
    }
    if (hdr.payload_size > 0) {
      auto r = ep.read(value_payload.data(), hdr.payload_size);
      if (!r)
        return forward_error(r);
    }
    if (key.empty() || codec_name.empty()) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Invalid metadata entry: empty key or codec"));
    }

    auto maybe_msg = decode_message_payload(codec_name, value_payload);
    if (!maybe_msg)
      return forward_error(maybe_msg);
    metadata.set(key, std::make_shared<Message>(maybe_msg.value()));
  }
  return metadata;
}

// ---------------------------------------------------------------------------
// MessageLabel
// ---------------------------------------------------------------------------

expected<std::vector<uint8_t>, RuntimeError> encode_message_label_payload(
    const MessageLabel& label) {
  std::vector<uint8_t> buf;
  VectorEndpoint ep(&buf);

  uint32_t path_count = static_cast<uint32_t>(label.paths().size());
  if (path_count > kMaxPaths) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Too many message-label paths"));
  }
  auto pc_r = ep.write_trivial_type(&path_count);
  if (!pc_r)
    return forward_error(pc_r);

  for (const auto& path : label.paths()) {
    uint32_t op_count = static_cast<uint32_t>(path.size());
    if (op_count > kMaxOpsPerPath) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Too many operators in message-label path"));
    }
    auto oc_r = ep.write_trivial_type(&op_count);
    if (!oc_r)
      return forward_error(oc_r);

    for (const auto& op : path) {
      uint32_t name_size = static_cast<uint32_t>(op.operator_name.size());
      if (name_size > kMaxNameSize) {
        return make_unexpected<RuntimeError>(
            RuntimeError(ErrorCode::kCodecError, "Operator name too large"));
      }
      auto ns_r = ep.write_trivial_type(&name_size);
      if (!ns_r)
        return forward_error(ns_r);
      if (name_size > 0) {
        auto n_r = ep.write(op.operator_name.data(), name_size);
        if (!n_r)
          return forward_error(n_r);
      }
      auto rec_r = ep.write_trivial_type(&op.rec_timestamp);
      if (!rec_r)
        return forward_error(rec_r);
      auto pub_r = ep.write_trivial_type(&op.pub_timestamp);
      if (!pub_r)
        return forward_error(pub_r);
    }
  }

  // Frame numbers
  const auto frame_numbers = label.get_frame_numbers();
  uint32_t frame_entry_count = static_cast<uint32_t>(frame_numbers.size());
  if (frame_entry_count > kMaxFrameEntries) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Too many message-label frame entries"));
  }
  auto fc_r = ep.write_trivial_type(&frame_entry_count);
  if (!fc_r)
    return forward_error(fc_r);
  for (const auto& [frame_key, frame_value] : frame_numbers) {
    uint32_t key_size = static_cast<uint32_t>(frame_key.size());
    if (key_size > kMaxNameSize) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Frame number key too large"));
    }
    auto ks_r = ep.write_trivial_type(&key_size);
    if (!ks_r)
      return forward_error(ks_r);
    if (key_size > 0) {
      auto k_r = ep.write(frame_key.data(), key_size);
      if (!k_r)
        return forward_error(k_r);
    }
    auto v_r = ep.write_trivial_type(&frame_value);
    if (!v_r)
      return forward_error(v_r);
  }

  return buf;
}

expected<MessageLabel, RuntimeError> decode_message_label_payload(
    const std::vector<uint8_t>& payload) {
  const std::vector<uint8_t>* view = &payload;
  VectorEndpoint ep(view);

  uint32_t path_count = 0;
  auto pc_r = ep.read_trivial_type(&path_count);
  if (!pc_r)
    return forward_error(pc_r);
  if (path_count > kMaxPaths) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Too many message-label paths"));
  }

  MessageLabel label;
  for (uint32_t p = 0; p < path_count; ++p) {
    uint32_t op_count = 0;
    auto oc_r = ep.read_trivial_type(&op_count);
    if (!oc_r)
      return forward_error(oc_r);
    if (op_count > kMaxOpsPerPath) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Too many operators in message-label path"));
    }

    MessageLabel::TimestampedPath path;
    path.reserve(op_count);
    for (uint32_t o = 0; o < op_count; ++o) {
      uint32_t name_size = 0;
      auto ns_r = ep.read_trivial_type(&name_size);
      if (!ns_r)
        return forward_error(ns_r);
      if (name_size > kMaxNameSize) {
        return make_unexpected<RuntimeError>(
            RuntimeError(ErrorCode::kCodecError, "Operator name too large"));
      }

      std::string name(name_size, '\0');
      if (name_size > 0) {
        auto n_r = ep.read(name.data(), name_size);
        if (!n_r)
          return forward_error(n_r);
      }

      int64_t rec_ts = 0, pub_ts = 0;
      auto rec_r = ep.read_trivial_type(&rec_ts);
      if (!rec_r)
        return forward_error(rec_r);
      auto pub_r = ep.read_trivial_type(&pub_ts);
      if (!pub_r)
        return forward_error(pub_r);

      path.emplace_back(name, rec_ts, pub_ts);
    }
    label.add_new_path(std::move(path));
  }

  // Frame numbers
  uint32_t frame_entry_count = 0;
  auto fc_r = ep.read_trivial_type(&frame_entry_count);
  if (!fc_r)
    return forward_error(fc_r);
  if (frame_entry_count > kMaxFrameEntries) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "Too many message-label frame entries"));
  }

  for (uint32_t i = 0; i < frame_entry_count; ++i) {
    uint32_t key_size = 0;
    auto ks_r = ep.read_trivial_type(&key_size);
    if (!ks_r)
      return forward_error(ks_r);
    if (key_size > kMaxNameSize) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Frame number key too large"));
    }

    std::string frame_key(key_size, '\0');
    if (key_size > 0) {
      auto k_r = ep.read(frame_key.data(), key_size);
      if (!k_r)
        return forward_error(k_r);
    }
    uint64_t frame_value = 0;
    auto v_r = ep.read_trivial_type(&frame_value);
    if (!v_r)
      return forward_error(v_r);

    if (frame_key.empty()) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Invalid empty frame number key"));
    }
    const auto delim = frame_key.rfind('-');
    if (delim == std::string::npos || delim == 0 || delim + 1 >= frame_key.size()) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "Invalid frame number key format"));
    }
    label.set_frame_number(frame_key.substr(0, delim), frame_key.substr(delim + 1), frame_value);
  }

  return label;
}

}  // namespace holoscan
