/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

#include <holomq/buffer.hpp>
#include <holomq/buffer_handle.hpp>
#include <holomq/const_buffer.hpp>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "channel_common.hpp"

namespace holoscan::channel_internal {

/// Source-private access to the bytes of the public transport-opaque reference envelope.
struct OpaquePayloadReferenceAccess {
  [[nodiscard]] static BufferHandlePayloadReference from_bytes(
      const std::span<const std::byte> bytes) noexcept {
    BufferHandlePayloadReference reference;
    if (bytes.empty() || bytes.size() > BufferHandlePayloadReference::kPayloadCapacity) {
      return reference;
    }
    reference.payload_size_ = static_cast<std::uint16_t>(bytes.size());
    std::memcpy(reference.payload_.data(), bytes.data(), bytes.size());
    return reference;
  }

  [[nodiscard]] static std::span<const std::byte> bytes(
      const BufferHandlePayloadReference& reference) noexcept {
    if (!reference) {
      return {};
    }
    return {reference.payload_.data(), reference.payload_size_};
  }
};

inline constexpr std::uint32_t kHoloMQPayloadReferenceCarrierMagic = 0x484D5152U;  // "HMQR"
inline constexpr std::uint16_t kHoloMQPayloadReferenceCarrierAbiVersion = 1U;

/// Source-private carrier for one HoloMQ buffer handle inside a Holoscan opaque reference.
struct HoloMQPayloadReferenceCarrier {
  std::uint32_t magic{kHoloMQPayloadReferenceCarrierMagic};
  std::uint16_t abi_version{kHoloMQPayloadReferenceCarrierAbiVersion};
  std::uint16_t header_size{sizeof(HoloMQPayloadReferenceCarrier)};
  holomq::BufferHandle handle{};

  [[nodiscard]] bool has_supported_abi() const noexcept {
    return magic == kHoloMQPayloadReferenceCarrierMagic &&
           abi_version == kHoloMQPayloadReferenceCarrierAbiVersion &&
           header_size == sizeof(HoloMQPayloadReferenceCarrier) && static_cast<bool>(handle);
  }
};

static_assert(std::is_standard_layout_v<HoloMQPayloadReferenceCarrier>);
static_assert(std::is_trivially_copyable_v<HoloMQPayloadReferenceCarrier>);
static_assert(sizeof(HoloMQPayloadReferenceCarrier) <=
              BufferHandlePayloadReference::kPayloadCapacity);

[[nodiscard]] inline BufferHandlePayloadReference make_holomq_payload_reference(
    const holomq::BufferHandle& handle) noexcept {
  if (!handle) {
    return {};
  }
  const HoloMQPayloadReferenceCarrier carrier{.handle = handle};
  return OpaquePayloadReferenceAccess::from_bytes(
      std::as_bytes(std::span<const HoloMQPayloadReferenceCarrier, 1U>(&carrier, 1U)));
}

[[nodiscard]] inline expected<holomq::BufferHandle, Error>
holomq_buffer_handle_from_payload_reference(
    const BufferHandlePayloadReference& reference) noexcept {
  const auto bytes = OpaquePayloadReferenceAccess::bytes(reference);
  if (bytes.size() != sizeof(HoloMQPayloadReferenceCarrier)) {
    return make_unexpected_nothrow<holomq::BufferHandle>(
        ErrorCode::kProtocolError, "opaque payload reference has an unsupported carrier size");
  }

  HoloMQPayloadReferenceCarrier carrier;
  std::memcpy(&carrier, bytes.data(), sizeof(carrier));
  if (!carrier.has_supported_abi()) {
    return make_unexpected_nothrow<holomq::BufferHandle>(
        ErrorCode::kProtocolError, "opaque payload reference has an unsupported HoloMQ carrier");
  }
  return carrier.handle;
}

inline constexpr std::uint32_t kHoloMQTensorDescriptorMagic = 0x48545042U;  // "HTPB"
inline constexpr std::uint16_t kHoloMQTensorDescriptorAbiVersion = 1U;
inline constexpr std::size_t kHoloMQTensorDescriptorMaxRank = 8U;

enum class HoloMQTensorDescriptorMemoryKind : std::uint8_t {
  kNone = 0U,
  kDevice = 1U,
  kHostPageable = 2U,
  kHostPinned = 3U,
};

enum class HoloMQTensorDescriptorFlags : std::uint32_t {
  kHasByteStrides = 1U << 0U,
  kReadOnly = 1U << 1U,
};

/// Fixed-size Channel-layer metadata needed to wrap a HoloMQ buffer handle as a Tensor.
///
/// This descriptor is intentionally separate from ChannelSampleDescriptor: it carries Tensor view
/// metadata and the HoloMQ buffer handle required for cross-process import, while
/// ChannelSampleDescriptor remains the 72-byte same-process sidecar descriptor.
struct HoloMQTensorDescriptor {
  std::uint32_t magic{kHoloMQTensorDescriptorMagic};
  std::uint16_t abi_version{kHoloMQTensorDescriptorAbiVersion};
  std::uint16_t header_size{sizeof(HoloMQTensorDescriptor)};
  std::uint32_t flags{};
  std::uint32_t rank{};
  DLDataType dtype{};
  HoloMQTensorDescriptorMemoryKind memory_kind{HoloMQTensorDescriptorMemoryKind::kNone};
  std::int32_t cuda_device_ordinal{-1};
  std::uint64_t byte_size{};
  std::array<std::int64_t, kHoloMQTensorDescriptorMaxRank> shape{};
  std::array<std::int64_t, kHoloMQTensorDescriptorMaxRank> byte_strides{};
  holomq::BufferHandle buffer_handle{};

  [[nodiscard]] constexpr bool has_supported_abi() const noexcept {
    return magic == kHoloMQTensorDescriptorMagic &&
           abi_version == kHoloMQTensorDescriptorAbiVersion &&
           header_size == sizeof(HoloMQTensorDescriptor);
  }
};

static_assert(std::is_standard_layout_v<HoloMQTensorDescriptor>);
static_assert(std::is_trivially_copyable_v<HoloMQTensorDescriptor>);
static_assert(sizeof(HoloMQTensorDescriptor) == 368U);
static_assert(alignof(HoloMQTensorDescriptor) == 8U);
static_assert(offsetof(HoloMQTensorDescriptor, magic) == 0U);
static_assert(offsetof(HoloMQTensorDescriptor, abi_version) == 4U);
static_assert(offsetof(HoloMQTensorDescriptor, header_size) == 6U);
static_assert(offsetof(HoloMQTensorDescriptor, flags) == 8U);
static_assert(offsetof(HoloMQTensorDescriptor, rank) == 12U);
static_assert(offsetof(HoloMQTensorDescriptor, dtype) == 16U);
static_assert(offsetof(HoloMQTensorDescriptor, memory_kind) == 20U);
static_assert(offsetof(HoloMQTensorDescriptor, cuda_device_ordinal) == 24U);
static_assert(offsetof(HoloMQTensorDescriptor, byte_size) == 32U);
static_assert(offsetof(HoloMQTensorDescriptor, shape) == 40U);
static_assert(offsetof(HoloMQTensorDescriptor, byte_strides) == 104U);
static_assert(offsetof(HoloMQTensorDescriptor, buffer_handle) == 168U);

/// Source-private Tensor backing contract for storage that can produce one HoloMQ handle.
///
/// This is deliberately absent from installed headers. It lets endpoint-owned pools and imported
/// HoloMQ buffers share the same opaque descriptor lowering without exposing BufferHandle through
/// the Holoscan API.
class HoloMQExportableTensorMemoryReference : public TensorMemoryReference {
 public:
  [[nodiscard]] virtual expected<holomq::BufferHandle, Error> export_handle() noexcept = 0;
};

/// Resolve the source-private exportable backing once for a Tensor direct-export transaction.
[[nodiscard]] HOLOSCAN_CORE_EXPORT HoloMQExportableTensorMemoryReference*
holomq_exportable_tensor_memory_reference(const Tensor& tensor) noexcept;

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<HoloMQTensorDescriptor, Error>
make_holomq_tensor_descriptor(const Tensor& tensor, const holomq::BufferHandle& buffer_handle);

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<HoloMQTensorDescriptor, Error>
export_holomq_tensor_descriptor(const Tensor& tensor);

/// Export one descriptor using a backing reference already resolved for the supplied Tensor.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<HoloMQTensorDescriptor, Error>
export_holomq_tensor_descriptor(const Tensor& tensor,
                                HoloMQExportableTensorMemoryReference& memory);

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<HoloMQTensorDescriptor, Error>
make_holomq_tensor_descriptor(TensorShapeView shape, DLDataType dtype,
                              TensorByteStridesView byte_strides, std::uint64_t byte_size,
                              const holomq::BufferHandle& buffer_handle, bool read_only,
                              std::int32_t cuda_device_ordinal = -1);

[[nodiscard]] HOLOSCAN_CORE_EXPORT PayloadDescriptor
payload_descriptor_from_holomq_tensor_descriptor(const HoloMQTensorDescriptor& descriptor) noexcept;

/// Wrap a mutable HoloMQ pool lease as a Tensor.
///
/// The returned Tensor owns \p buffer through its TensorMemoryReference. Dropping the last Tensor
/// handle releases the HoloMQ lease; calling Tensor::set_deallocation_stream() first defers that
/// release until the supplied CUDA stream reaches prior work.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor_from_holomq_buffer(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    holomq::Buffer buffer);

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor_from_holomq_buffer(
    const HoloMQTensorDescriptor& descriptor, holomq::Buffer buffer);

/// Wrap a read-only imported HoloMQ buffer view as a Tensor.
///
/// The returned Tensor owns \p buffer through its TensorMemoryReference and is marked read-only
/// using the DLPack READ_ONLY flag. Dropping the last Tensor handle releases the imported HoloMQ
/// lease; calling Tensor::set_deallocation_stream() first defers release until the supplied CUDA
/// stream reaches prior work.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor_from_holomq_const_buffer(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    holomq::ConstBuffer buffer);

[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor_from_holomq_const_buffer(
    const HoloMQTensorDescriptor& descriptor, holomq::ConstBuffer buffer);

}  // namespace holoscan::channel_internal
