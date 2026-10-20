/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace holomq {

/// @brief ABI version for @ref LocalBufferDescriptor.
inline constexpr std::uint32_t kLocalBufferDescriptorAbiVersion = 1U;

/// @brief Same-process buffer descriptor that can be embedded in a typed pub/sub payload.
///
/// This descriptor is an opaque control-plane token. HoloMQ transports it as bytes but does not
/// resolve it into memory ownership. The runtime that creates the descriptor owns the sidecar
/// store, validates the store identity and generation, and maps the token back to the real Tensor
/// or buffer lifetime. Cross-process buffer transfer uses @ref BufferHandle instead.
struct LocalBufferDescriptor {
  /// Descriptor ABI version. Receivers should reject versions they do not understand.
  std::uint32_t abi_version{kLocalBufferDescriptorAbiVersion};
  /// Runtime-defined flags. HoloMQ does not interpret these bits.
  std::uint32_t flags{};
  /// Runtime-owned sidecar store identity. Zero is reserved for an empty descriptor.
  std::uint64_t store_id{};
  /// Runtime-owned record index within the sidecar store.
  std::uint64_t record_index{};
  /// ABA-protection generation for the sidecar record.
  std::uint64_t generation{};
  /// Runtime-defined token used to validate or release the sidecar claim.
  std::uint64_t token{};

  /// @brief Return whether the descriptor uses the ABI compiled into this HoloMQ build.
  /// @return `true` when @ref abi_version equals @ref kLocalBufferDescriptorAbiVersion.
  [[nodiscard]] constexpr bool has_supported_abi() const noexcept {
    return abi_version == kLocalBufferDescriptorAbiVersion;
  }

  /// @brief Return whether the descriptor is non-empty and ABI-compatible.
  /// @return `true` when the descriptor has a supported ABI and nonzero sidecar store id.
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return has_supported_abi() && store_id != 0U;
  }
};

static_assert(std::is_standard_layout_v<LocalBufferDescriptor>);
static_assert(std::is_trivially_copyable_v<LocalBufferDescriptor>);

}  // namespace holomq
