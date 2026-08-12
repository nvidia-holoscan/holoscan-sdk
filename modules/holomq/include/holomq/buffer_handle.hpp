/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/buffer_memory_type.hpp>
#include <holomq/detail/buffer_pool/device/handle_data.hpp>
#include <holomq/detail/buffer_pool/host/handle_data.hpp>

#include <cstdint>
#include <type_traits>

namespace holomq {

static_assert(detail::device::BufferHandleData::fd_pass_socket_path_max ==
              detail::host::BufferHandleData::fd_pass_socket_path_max);

class BufferHandle;

namespace detail {

struct BufferHandleWireAccess;

/// Builds a device handle after validating wire layout and \p slot_index pool geometry.
[[nodiscard]] BufferHandle make_device_buffer_handle(const device::BufferHandleData& handle_data,
                                                     std::uint32_t slot_index,
                                                     std::uint64_t ready_generation = 0U) noexcept;

/// Builds a host handle after validating wire layout and \p slot_index payload/generation pool
/// geometry (devicemap interleaved or separate sync-pool VMM).
[[nodiscard]] BufferHandle make_host_buffer_handle(const host::BufferHandleData& handle_data,
                                                   std::uint32_t slot_index,
                                                   std::uint64_t ready_generation = 0U) noexcept;

/// Validates device wire fields (layout only; \p slot_index is not checked here).
[[nodiscard]] bool is_valid_device_buffer_wire(
    const device::BufferHandleData& handle_data) noexcept;

/// Validates host wire fields (layout only; \p slot_index is not checked here).
[[nodiscard]] bool is_valid_host_buffer_wire(const host::BufferHandleData& handle_data) noexcept;

/// Validates that \p slot_index occupies a complete payload slot and, for split layouts, a
/// complete generation slot within the respective claimed pool sizes.
[[nodiscard]] bool is_valid_host_buffer_slot_geometry(const host::BufferHandleData& handle_data,
                                                      std::uint32_t slot_index) noexcept;

}  // namespace detail

/// Serializable buffer handle for publishing in an iceoryx2 payload (standard-layout, trivially
/// copyable).
class BufferHandle {
 public:
  BufferHandle() noexcept = default;

  [[nodiscard]] HOLOMQ_EXPORT explicit operator bool() const noexcept;

  [[nodiscard]] HOLOMQ_EXPORT BufferMemoryType memory_type() const noexcept;

 private:
  friend struct detail::BufferHandleWireAccess;
  friend BufferHandle detail::make_device_buffer_handle(
      const detail::device::BufferHandleData& handle_data, std::uint32_t slot_index,
      std::uint64_t ready_generation) noexcept;
  friend BufferHandle detail::make_host_buffer_handle(
      const detail::host::BufferHandleData& handle_data, std::uint32_t slot_index,
      std::uint64_t ready_generation) noexcept;

  union HandleData {
    detail::device::BufferHandleData device;
    detail::host::BufferHandleData host;
  };

  BufferHandle(BufferMemoryType memory_type, HandleData data, std::uint32_t slot_index,
               std::uint64_t ready_generation) noexcept;

  BufferMemoryType memory_type_{BufferMemoryType::None};
  std::uint32_t slot_index_{};
  std::uint64_t ready_generation_{};
  HandleData data_{};

  static_assert(sizeof(HandleData) >= sizeof(detail::device::BufferHandleData));
  static_assert(sizeof(HandleData) >= sizeof(detail::host::BufferHandleData));
};

static_assert(std::is_standard_layout_v<BufferHandle>);
static_assert(std::is_trivially_copyable_v<BufferHandle>);

namespace detail {

/// Library-internal read access to pool-type wire fields, slot index, and export fence token.
struct BufferHandleWireAccess {
  [[nodiscard]] static const detail::device::BufferHandleData& device_wire(
      const BufferHandle& handle) noexcept;
  [[nodiscard]] static const detail::host::BufferHandleData& host_wire(
      const BufferHandle& handle) noexcept;
  [[nodiscard]] static std::uint32_t slot_index(const BufferHandle& handle) noexcept;
  /// Per-export GPU-ready token from \ref Buffer::export_handle; \c 0 means no cross-process fence
  /// at import.
  [[nodiscard]] static std::uint64_t ready_generation(const BufferHandle& handle) noexcept;
};

}  // namespace detail

}  // namespace holomq
