/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/host/host_buffer_pool_impl.hpp"

#include <holomq/detail/buffer_pool/host/handle_data.hpp>
#include <holomq/device_buffer_allocator.hpp>
#include <holomq/logger.hpp>
#include "buffer_pool/host/host_buffer_impl.hpp"
#include "buffer_pool/pool_layout.hpp"
#include "buffer_pool/slot_region.hpp"
#include "context_impl.hpp"
#include "fd_pass/sockaddr.hpp"

#include <cuda.h>

#include <algorithm>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include <unistd.h>

namespace holomq {

namespace {

[[nodiscard]] std::size_t page_size() noexcept {
  const long pagesize = ::sysconf(_SC_PAGESIZE);
  return pagesize > 0 ? static_cast<std::size_t>(pagesize) : 4096U;
}

[[nodiscard]] std::optional<std::size_t> aligned_region_bytes(
    const std::size_t unit_bytes, const std::uint32_t count, const std::size_t alignment) noexcept {
  const auto product_or = checked_product(unit_bytes, static_cast<std::size_t>(count));
  if (!product_or.has_value()) {
    return std::nullopt;
  }
  return round_up(product_or.value(), alignment);
}

}  // namespace

HostBufferPoolImpl::HostBufferPoolImpl(
    const PrivateTag, ShmRefCountMappingPtr ref_count_mapping,
    std::shared_ptr<HostBufferPoolMemory> payload_backing, VmmExportedPool generation_vmm_pool,
    std::string fd_pass_socket_path, const std::size_t buffer_bytes,
    const std::size_t slot_stride_bytes, const std::uint32_t buffer_count,
    const bool payload_pinned, const unsigned int cuda_host_register_flags,
    const bool uses_split_generation_vmm, const std::size_t generation_slot_stride_bytes) noexcept
    : BufferPool::Impl(std::move(ref_count_mapping), buffer_bytes, slot_stride_bytes, buffer_count),
      payload_pinned_(payload_pinned),
      cuda_host_register_flags_(cuda_host_register_flags),
      uses_split_generation_vmm_(uses_split_generation_vmm),
      generation_slot_stride_bytes_(generation_slot_stride_bytes),
      fd_pass_socket_path_(std::move(fd_pass_socket_path)),
      generation_vmm_pool_(std::move(generation_vmm_pool)),
      payload_backing_(std::move(payload_backing)) {}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
auto HostBufferPoolImpl::try_create(const std::size_t buffer_bytes,
                                    const std::uint32_t buffer_count, const bool payload_pinned,
                                    const unsigned int cuda_host_register_flags,
                                    HostBufferAllocator allocator, std::string fd_pass_socket_path,
                                    const std::shared_ptr<ContextImpl>& session) noexcept
    -> expected<std::unique_ptr<HostBufferPoolImpl>, Error> {
  if (session == nullptr) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: null context session");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (buffer_bytes == 0U) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: zero buffer bytes");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (buffer_count == 0U) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: zero buffer count");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!allocator) {
    allocator = default_host_buffer_allocator();
  }
  if (!allocator) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  // DEVICEMAP lets a pinned pool keep generation words in its payload SHM, which needs no fd-pass.
  // Use device-resident generation words whenever fd-pass export is available anyway: a consumer
  // waits for each ready generation on its stream, and a wait on host memory polls system memory
  // while holding a GPU hardware queue that unrelated streams, such as the ones carrying buffer
  // release host functions, may share.
  const bool devicemap_interleaved_possible =
      payload_pinned && (cuda_host_register_flags & CU_MEMHOSTREGISTER_DEVICEMAP) != 0U;
  const bool uses_split_generation_vmm =
      !devicemap_interleaved_possible || session->vmm_fd_export_enabled();

  const std::size_t page = page_size();
  std::size_t payload_region_bytes = 0U;
  std::size_t payload_slot_stride = 0U;
  std::size_t generation_slot_stride_bytes = 0U;
  VmmExportedPool generation_vmm_pool;

  if (uses_split_generation_vmm) {
    if (!session->vmm_fd_export_enabled()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: VMM fd-pass export required");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (fd_pass_socket_path.size() > fd_pass::endpoint_max_length()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: fd_pass socket path too long");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    payload_slot_stride = buffer_bytes;
    const auto payload_region_or = aligned_region_bytes(payload_slot_stride, buffer_count, page);
    if (!payload_region_or.has_value()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    payload_region_bytes = payload_region_or.value();

    const std::size_t granularity = recommended_device_vmm_granularity();
    if (granularity == 0U) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }

    generation_slot_stride_bytes = k_slot_gpu_generation_bytes;
    const auto generation_pool_bytes_or =
        aligned_region_bytes(generation_slot_stride_bytes, buffer_count, granularity);
    if (!generation_pool_bytes_or.has_value()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }

    auto generation_vmm_pool_or = VmmExportedPool::create(default_device_buffer_allocator(),
                                                          generation_pool_bytes_or.value());
    if (!generation_vmm_pool_or.has_value()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: split generation VMM pool create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    generation_vmm_pool = std::move(generation_vmm_pool_or).value();
    generation_vmm_pool.register_fd_export(session);
  } else {
    payload_slot_stride = k_slot_gpu_generation_bytes + buffer_bytes;
    const auto payload_region_or = aligned_region_bytes(payload_slot_stride, buffer_count, page);
    if (!payload_region_or.has_value()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    payload_region_bytes = payload_region_or.value();
  }

  auto payload_backing = allocator(
      payload_region_bytes, payload_pinned, payload_pinned ? cuda_host_register_flags : 0U);
  if (!payload_backing || payload_backing->size() < payload_region_bytes) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: host payload backing create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  if (!uses_split_generation_vmm) {
    if (!make_host_devicemap_interleaved_slot_regions(
             payload_backing->ptr(), payload_slot_stride, buffer_bytes)
             .has_value()) {
      HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
  }

  auto ref_count_mapping_or = ShmRefCountMapping::create(buffer_count);
  if (!ref_count_mapping_or.has_value()) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: ref-count SHM create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  std::shared_ptr<HostBufferPoolMemory> payload_owner;
  try {
    payload_owner = std::move(payload_backing);
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  auto pool = std::unique_ptr<HostBufferPoolImpl>(
      new (std::nothrow) HostBufferPoolImpl(PrivateTag{},
                                            std::move(ref_count_mapping_or).value(),
                                            std::move(payload_owner),
                                            std::move(generation_vmm_pool),
                                            std::move(fd_pass_socket_path),
                                            buffer_bytes,
                                            payload_slot_stride,
                                            buffer_count,
                                            payload_pinned,
                                            cuda_host_register_flags,
                                            uses_split_generation_vmm,
                                            generation_slot_stride_bytes));
  if (!pool) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: host pool allocation failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return pool;
}

BufferMemoryType HostBufferPoolImpl::memory_type() const noexcept {
  return payload_pinned_ ? BufferMemoryType::HostPinned : BufferMemoryType::HostPageable;
}

std::optional<int32_t> HostBufferPoolImpl::cuda_device_ordinal() const noexcept {
  return std::nullopt;
}

void* HostBufferPoolImpl::base_ptr() const noexcept {
  return payload_backing_ != nullptr ? payload_backing_->ptr() : nullptr;
}

Buffer HostBufferPoolImpl::buffer_at_index(const std::uint32_t index) noexcept {
  if (index >= buffer_count()) {
    return {};
  }
  return Buffer(std::make_unique<HostBufferImpl>(*this, index));
}

BufferHandle HostBufferPoolImpl::handle(const std::uint32_t index) const noexcept {
  detail::host::BufferHandleData data{};
  data.owner_pid = static_cast<std::uint32_t>(ref_count_mapping_->owner_pid());
  data.shm_pool_id = payload_backing_->shm_pool_id();
  data.shm_ref_count_id = ref_count_mapping_->shm_ref_count_id();
  data.slot_stride_bytes = slot_stride_bytes();
  data.buffer_bytes = buffer_bytes_;
  data.pool_total_bytes = payload_backing_->size();
  data.payload_pinned =
      payload_pinned_ ? static_cast<std::uint8_t>(1U) : static_cast<std::uint8_t>(0U);
  data.host_register_flags = payload_pinned_ ? cuda_host_register_flags_ : 0U;

  if (uses_split_generation_vmm_) {
    data.gen_pool_cookie = generation_vmm_pool_.cookie();
    data.gen_slot_stride_bytes = generation_slot_stride_bytes_;
    data.gen_pool_total_bytes = generation_vmm_pool_.size();
    const auto copy_len = std::min(fd_pass_socket_path_.size(), fd_pass::endpoint_max_length());
    std::copy_n(fd_pass_socket_path_.data(), copy_len, data.fd_pass_socket_path);
    data.fd_pass_socket_path[copy_len] = '\0';
  }

  return detail::make_host_buffer_handle(data, index, 0U);
}

}  // namespace holomq
