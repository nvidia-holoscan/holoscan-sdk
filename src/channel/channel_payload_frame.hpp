/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "channel_common.hpp"

#include <holoscan/core/sample.hpp>

#include <holomq/byte_slice.hpp>
#include <holomq/pubsub_options.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

// Channel payload frames carried in HoloMQ byte slices. A frame is a profile's wire header,
// padding up to the profile's data offset, and then the header's payload_size bytes. Each layout
// keeps the header fields and data offset of the fixed-size payload buckets it replaced.
namespace holoscan::channel_internal {

/// Frame layouts; each value is also the HoloMQ byte-slice profile id of services carrying it.
enum class PayloadFrameProfile : std::uint32_t {
  /// `inline-v1`: operation, sample metadata, and payload size.
  kInline = 1U,
  /// `tensor-schema-v1`: `inline-v1` plus catalog producer generation and health report sequence.
  kTensorSchema = 2U,
};

/// Wire header of an `inline-v1` frame.
struct InlinePayloadFrameHeader {
  InlinePayloadOperation operation{InlinePayloadOperation::kData};
  SampleMetadata metadata{};
  std::uint32_t payload_size{};
};

/// Wire header of a `tensor-schema-v1` frame.
struct TensorSchemaPayloadFrameHeader {
  InlinePayloadOperation operation{InlinePayloadOperation::kData};
  SampleMetadata metadata{};
  std::uint64_t producer_generation{};
  std::uint64_t health_report_sequence{};
  std::uint32_t payload_size{};
};

// Parsing and writing address header fields with offsetof.
static_assert(std::is_standard_layout_v<InlinePayloadFrameHeader>);
static_assert(std::is_standard_layout_v<TensorSchemaPayloadFrameHeader>);
static_assert(std::is_trivially_copyable_v<InlinePayloadFrameHeader>);
static_assert(std::is_trivially_copyable_v<TensorSchemaPayloadFrameHeader>);

/// Return \p size rounded up to the frame alignment.
[[nodiscard]] constexpr std::size_t align_frame_offset(const std::size_t size) noexcept {
  return (size + holomq::kByteSliceAlignment - 1U) / holomq::kByteSliceAlignment *
         holomq::kByteSliceAlignment;
}

/// Offset of the payload bytes in an `inline-v1` frame.
inline constexpr std::size_t kInlinePayloadFrameDataOffset =
    align_frame_offset(sizeof(InlinePayloadFrameHeader));

/// Offset of the payload bytes in a `tensor-schema-v1` frame.
inline constexpr std::size_t kTensorSchemaPayloadFrameDataOffset =
    align_frame_offset(sizeof(TensorSchemaPayloadFrameHeader));

// The data offsets are part of the frame profiles; changing either requires a new profile id.
static_assert(kInlinePayloadFrameDataOffset == 96U, "inline-v1 data offset changed");
static_assert(kTensorSchemaPayloadFrameDataOffset == 112U, "tensor-schema-v1 data offset changed");

/// @brief Return the offset of the payload bytes in a frame of \p profile.
[[nodiscard]] constexpr std::size_t payload_frame_data_offset(
    const PayloadFrameProfile profile) noexcept {
  return profile == PayloadFrameProfile::kTensorSchema ? kTensorSchemaPayloadFrameDataOffset
                                                       : kInlinePayloadFrameDataOffset;
}

/// @brief Return the byte-slice capacity of a service carrying \p profile frames.
/// @param profile Frame layout of the service.
/// @param payload_capacity Largest payload the service carries.
[[nodiscard]] constexpr std::size_t payload_frame_capacity(
    const PayloadFrameProfile profile, const std::size_t payload_capacity) noexcept {
  return payload_frame_data_offset(profile) + payload_capacity;
}

/// @brief Return the HoloMQ contract of a data or terminal service carrying \p profile frames.
/// @param service_options Queue, fanout, history, and wake contract of the service.
/// @param profile Frame layout of the service.
/// @param payload_capacity Largest payload the service carries.
[[nodiscard]] inline holomq::PubSubServiceOptions payload_frame_service_options(
    holomq::PubSubServiceOptions service_options, const PayloadFrameProfile profile,
    const std::size_t payload_capacity) noexcept {
  service_options.byte_slice_capacity = payload_frame_capacity(profile, payload_capacity);
  service_options.byte_slice_profile = static_cast<std::uint32_t>(profile);
  return service_options;
}

/// Header fields of a frame of either profile. Catalog attribution is zero for `inline-v1`.
struct PayloadFrameFields {
  InlinePayloadOperation operation{InlinePayloadOperation::kData};
  SampleMetadata metadata{};
  std::uint64_t producer_generation{};
  std::uint64_t health_report_sequence{};
  std::uint32_t payload_size{};
};

/// A received frame, parsed once when it is taken.
struct ParsedPayloadFrame {
  PayloadFrameFields header{};
  /// Exactly `header.payload_size` bytes when \ref well_formed.
  std::span<const std::byte> payload{};
  /// Whether the frame holds a complete header and every payload byte its header declares.
  bool well_formed{false};
};

/// @brief Parse a received frame into \p parsed without trusting its length fields.
///
/// Header fields are copied straight from the frame into \p parsed, so the receive path copies each
/// header byte once and builds no temporaries.
/// @param profile Frame layout of the service that delivered \p frame.
/// @param frame Complete received frame.
/// @param parsed Destination; not \c well_formed when the frame is shorter than its header or than
/// the payload size the header declares. A frame shorter than its header leaves a default header.
inline void parse_payload_frame_into(const PayloadFrameProfile profile,
                                     const std::span<const std::byte> frame,
                                     ParsedPayloadFrame& parsed) noexcept {
  const std::size_t data_offset = payload_frame_data_offset(profile);
  parsed.well_formed = false;
  parsed.payload = {};
  PayloadFrameFields& header = parsed.header;
  if (frame.size() < data_offset) {
    header = PayloadFrameFields{};
    return;
  }
  const std::byte* const bytes = frame.data();
  if (profile == PayloadFrameProfile::kTensorSchema) {
    using Wire = TensorSchemaPayloadFrameHeader;
    std::memcpy(&header.operation, bytes + offsetof(Wire, operation), sizeof(header.operation));
    std::memcpy(&header.metadata, bytes + offsetof(Wire, metadata), sizeof(header.metadata));
    std::memcpy(&header.producer_generation,
                bytes + offsetof(Wire, producer_generation),
                sizeof(header.producer_generation));
    std::memcpy(&header.health_report_sequence,
                bytes + offsetof(Wire, health_report_sequence),
                sizeof(header.health_report_sequence));
    std::memcpy(
        &header.payload_size, bytes + offsetof(Wire, payload_size), sizeof(header.payload_size));
  } else {
    using Wire = InlinePayloadFrameHeader;
    std::memcpy(&header.operation, bytes + offsetof(Wire, operation), sizeof(header.operation));
    std::memcpy(&header.metadata, bytes + offsetof(Wire, metadata), sizeof(header.metadata));
    std::memcpy(
        &header.payload_size, bytes + offsetof(Wire, payload_size), sizeof(header.payload_size));
    header.producer_generation = 0U;
    header.health_report_sequence = 0U;
  }
  if (header.payload_size > frame.size() - data_offset) {
    return;
  }
  parsed.payload = frame.subspan(data_offset, header.payload_size);
  parsed.well_formed = true;
}

/// @brief Parse a received frame without trusting its length fields.
/// @param profile Frame layout of the service that delivered \p frame.
/// @param frame Complete received frame.
/// @return The parsed frame, as \ref parse_payload_frame_into describes.
[[nodiscard]] inline ParsedPayloadFrame parse_payload_frame(
    const PayloadFrameProfile profile, const std::span<const std::byte> frame) noexcept {
  ParsedPayloadFrame parsed;
  parse_payload_frame_into(profile, frame, parsed);
  return parsed;
}

/// @brief Copy of one received frame that outlives its transport sample.
///
/// The storage is sized once, at setup, to the service's frame capacity, so receive paths that keep
/// a frame (for example a protected terminal record) never allocate.
class PayloadFrameCopy {
 public:
  /// @param profile Frame layout of the service whose frames are copied.
  /// @param frame_capacity Byte-slice capacity of that service.
  PayloadFrameCopy(const PayloadFrameProfile profile, const std::size_t frame_capacity)
      : storage_(frame_capacity), profile_(profile) {}

  [[nodiscard]] bool has_value() const noexcept { return present_; }
  [[nodiscard]] explicit operator bool() const noexcept { return present_; }

  /// Copy and parse \p frame. A frame larger than the setup capacity cannot come from a peer of
  /// the service; it is kept as a malformed frame, so the receive path reports a protocol error.
  void emplace(const std::span<const std::byte> frame) noexcept {
    if (frame.size() > storage_.size()) {
      frame_ = ParsedPayloadFrame{};
    } else {
      if (!frame.empty()) {
        std::memcpy(storage_.data(), frame.data(), frame.size());
      }
      parse_payload_frame_into(
          profile_, std::span<const std::byte>{storage_}.first(frame.size()), frame_);
    }
    present_ = true;
  }

  void reset() noexcept { present_ = false; }

  // Callers use these accessors only after has_value().
  [[nodiscard]] const ParsedPayloadFrame& operator*() const noexcept { return frame_; }
  [[nodiscard]] const ParsedPayloadFrame* operator->() const noexcept { return &frame_; }

 private:
  std::vector<std::byte> storage_;
  ParsedPayloadFrame frame_{};
  PayloadFrameProfile profile_;
  bool present_{};
};

/// @brief Write a frame header at the start of a loaned frame.
/// @param profile Frame layout of the service.
/// @param frame Loaned frame of at least the profile's data offset.
/// @param header Header to write; `inline-v1` drops the catalog attribution fields. The payload
/// bytes are written separately.
inline void write_payload_frame_header(const PayloadFrameProfile profile,
                                       const std::span<std::byte> frame,
                                       const PayloadFrameFields& header) noexcept {
  std::byte* const bytes = frame.data();
  if (profile == PayloadFrameProfile::kTensorSchema) {
    using Wire = TensorSchemaPayloadFrameHeader;
    std::memcpy(bytes + offsetof(Wire, operation), &header.operation, sizeof(header.operation));
    std::memcpy(bytes + offsetof(Wire, metadata), &header.metadata, sizeof(header.metadata));
    std::memcpy(bytes + offsetof(Wire, producer_generation),
                &header.producer_generation,
                sizeof(header.producer_generation));
    std::memcpy(bytes + offsetof(Wire, health_report_sequence),
                &header.health_report_sequence,
                sizeof(header.health_report_sequence));
    std::memcpy(
        bytes + offsetof(Wire, payload_size), &header.payload_size, sizeof(header.payload_size));
  } else {
    using Wire = InlinePayloadFrameHeader;
    std::memcpy(bytes + offsetof(Wire, operation), &header.operation, sizeof(header.operation));
    std::memcpy(bytes + offsetof(Wire, metadata), &header.metadata, sizeof(header.metadata));
    std::memcpy(
        bytes + offsetof(Wire, payload_size), &header.payload_size, sizeof(header.payload_size));
  }
}

/// @brief Write a complete frame loaned for exactly \p payload.
/// @param profile Frame layout of the service.
/// @param frame Loaned frame of the profile's data offset plus `payload.size()` bytes.
/// @param header Header to write; its payload size is replaced by `payload.size()`.
/// @param payload Payload bytes copied after the header.
inline void write_payload_frame(const PayloadFrameProfile profile, const std::span<std::byte> frame,
                                PayloadFrameFields header,
                                const std::span<const std::byte> payload) noexcept {
  header.payload_size = static_cast<std::uint32_t>(payload.size());
  write_payload_frame_header(profile, frame, header);
  if (!payload.empty()) {
    std::memcpy(frame.data() + payload_frame_data_offset(profile), payload.data(), payload.size());
  }
}

}  // namespace holoscan::channel_internal
