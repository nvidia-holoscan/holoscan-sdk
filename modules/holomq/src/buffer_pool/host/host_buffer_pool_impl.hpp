/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/host_buffer_allocator.hpp>
#include <holomq/host_buffer_pool_memory.hpp>
#include "buffer_pool/buffer_pool_impl.hpp"
#include "buffer_pool/device/vmm_exported_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace holomq {

class ContextImpl;

/// Host \ref BufferPool implementation: POSIX SHM payload backing plus optional split-generation
/// VMM pool.
///
/// Selects devicemap-interleaved layout when pinned with \c CU_MEMHOSTREGISTER_DEVICEMAP; otherwise
/// uses payload-only SHM with generation words in a separate \ref VmmExportedPool (requires fd-pass
/// export).
class HostBufferPoolImpl final : public BufferPool::Impl {
 public:
  [[nodiscard]] static auto try_create(std::size_t buffer_bytes, std::uint32_t buffer_count,
                                       bool payload_pinned, unsigned int cuda_host_register_flags,
                                       HostBufferAllocator allocator,
                                       std::string fd_pass_socket_path,
                                       const std::shared_ptr<ContextImpl>& session) noexcept
      -> expected<std::unique_ptr<HostBufferPoolImpl>, Error>;

  HostBufferPoolImpl(const HostBufferPoolImpl&) = delete;
  HostBufferPoolImpl& operator=(const HostBufferPoolImpl&) = delete;
  HostBufferPoolImpl(HostBufferPoolImpl&&) noexcept = delete;
  HostBufferPoolImpl& operator=(HostBufferPoolImpl&&) noexcept = delete;
  ~HostBufferPoolImpl() override = default;

  [[nodiscard]] BufferMemoryType memory_type() const noexcept override;
  [[nodiscard]] std::optional<int32_t> cuda_device_ordinal() const noexcept override;
  [[nodiscard]] void* base_ptr() const noexcept override;
  [[nodiscard]] BufferHandle handle(std::uint32_t index) const noexcept override;
  [[nodiscard]] Buffer buffer_at_index(std::uint32_t index) noexcept override;

  [[nodiscard]] bool payload_pinned() const noexcept { return payload_pinned_; }

  [[nodiscard]] unsigned int cuda_host_register_flags() const noexcept {
    return cuda_host_register_flags_;
  }

  [[nodiscard]] bool uses_split_generation_vmm() const noexcept {
    return uses_split_generation_vmm_;
  }

  [[nodiscard]] void* generation_base_ptr() const noexcept { return generation_vmm_pool_.ptr(); }

  [[nodiscard]] std::size_t generation_pool_total_bytes() const noexcept {
    return generation_vmm_pool_.size();
  }

  [[nodiscard]] std::size_t generation_slot_stride_bytes() const noexcept {
    return generation_slot_stride_bytes_;
  }

  [[nodiscard]] std::uint64_t gen_pool_cookie() const noexcept {
    return generation_vmm_pool_.cookie();
  }

 private:
  struct PrivateTag {};

  HostBufferPoolImpl(PrivateTag, ShmRefCountMappingPtr ref_count_mapping,
                     std::unique_ptr<HostBufferPoolMemory> payload_backing,
                     VmmExportedPool generation_vmm_pool, std::string fd_pass_socket_path,
                     std::size_t buffer_bytes, std::size_t slot_stride_bytes,
                     std::uint32_t buffer_count, bool payload_pinned,
                     unsigned int cuda_host_register_flags, bool uses_split_generation_vmm,
                     std::size_t generation_slot_stride_bytes) noexcept;

  bool payload_pinned_{false};
  unsigned int cuda_host_register_flags_{0U};
  bool uses_split_generation_vmm_{false};
  std::size_t generation_slot_stride_bytes_{0U};
  std::string fd_pass_socket_path_;
  VmmExportedPool generation_vmm_pool_;
  std::unique_ptr<HostBufferPoolMemory> payload_backing_;
};

}  // namespace holomq
