/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/device/device_buffer_pool_impl.hpp"

#include <holomq/detail/buffer_pool/device/handle_data.hpp>
#include <holomq/device_buffer_allocator.hpp>
#include <holomq/logger.hpp>
#include "buffer_pool/device/device_buffer_impl.hpp"
#include "buffer_pool/pool_layout.hpp"
#include "buffer_pool/slot_region.hpp"
#include "context_impl.hpp"
#include "cuda/current_device.hpp"
#include "cuda/stream.hpp"
#include "fd_pass/sockaddr.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace holomq {

namespace {

/// Zero the interleaved per-slot GPU generation words before publishing the pool.
///
/// The CPU generation counters start at zero and the first export publishes generation one.
/// Subscribers wait with a greater-than-or-equal comparison, so an undefined initial GPU value
/// could satisfy that wait before the publisher writes generation one. Initializing the words
/// prevents a subscriber from observing the payload before the publisher's export-time GPU work
/// completes.
///
/// Each generation word is separated from the next by user payload and stride padding. Clear only
/// the generation metadata, using a private stream so initialization does not synchronize unrelated
/// application work. Synchronizing that stream ensures the zeros are visible before the VMM mapping
/// is registered for export.
///
/// @return \c true if every generation word was initialized and the stream completed successfully.
/// @see https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html
[[nodiscard]] bool initialize_generation_words(void* const base_ptr,
                                               const std::size_t slot_stride_bytes,
                                               const std::uint32_t buffer_count) noexcept {
  if (base_ptr == nullptr || slot_stride_bytes < k_slot_gpu_generation_bytes ||
      buffer_count == 0U) {
    return false;
  }

  const cuda::Stream stream(CU_STREAM_NON_BLOCKING);
  if (!stream) {
    return false;
  }

  const CUdeviceptr base = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(base_ptr));
  const auto cuda_stream = reinterpret_cast<CUstream>(stream.get());
  for (std::uint32_t index = 0U; index < buffer_count; ++index) {
    const CUdeviceptr generation_word = base + static_cast<std::size_t>(index) * slot_stride_bytes;
    if (cuMemsetD8Async(generation_word, 0U, k_slot_gpu_generation_bytes, cuda_stream) !=
        CUDA_SUCCESS) {
      return false;
    }
  }
  return cuStreamSynchronize(cuda_stream) == CUDA_SUCCESS;
}

}  // namespace

DeviceBufferPoolImpl::DeviceBufferPoolImpl(
    const PrivateTag, ShmRefCountMappingPtr ref_count_mapping, VmmExportedPool vmm_pool,
    std::string fd_pass_socket_path, const std::size_t buffer_bytes,
    const std::size_t slot_stride_bytes, const std::uint32_t buffer_count,
    const int32_t cuda_device_ordinal) noexcept
    : BufferPool::Impl(std::move(ref_count_mapping), buffer_bytes, slot_stride_bytes, buffer_count),
      fd_pass_socket_path_(std::move(fd_pass_socket_path)),
      vmm_pool_(std::move(vmm_pool)),
      cuda_device_ordinal_(cuda_device_ordinal) {}

auto DeviceBufferPoolImpl::try_create(const std::size_t buffer_bytes,
                                      const std::uint32_t buffer_count,
                                      std::string fd_pass_socket_path,
                                      const std::shared_ptr<ContextImpl>& session) noexcept
    -> expected<std::unique_ptr<DeviceBufferPoolImpl>, Error> {
  return try_create(
      buffer_bytes, buffer_count, DeviceBufferAllocator{}, std::move(fd_pass_socket_path), session);
}

auto DeviceBufferPoolImpl::try_create(const std::size_t buffer_bytes,
                                      const std::uint32_t buffer_count,
                                      DeviceBufferAllocator allocator,
                                      std::string fd_pass_socket_path,
                                      const std::shared_ptr<ContextImpl>& session) noexcept
    -> expected<std::unique_ptr<DeviceBufferPoolImpl>, Error> {
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
  if (fd_pass_socket_path.size() > fd_pass::endpoint_max_length()) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: fd_pass socket path too long");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!allocator) {
    allocator = default_device_buffer_allocator();
  }

  const std::size_t granularity = recommended_device_vmm_granularity();
  if (granularity == 0U) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: device pool create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  const std::size_t aligned_slot_bytes =
      round_up(k_slot_gpu_generation_bytes + buffer_bytes, granularity);
  const auto region_size_or =
      checked_product(aligned_slot_bytes, static_cast<std::size_t>(buffer_count));
  if (!region_size_or.has_value()) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: device pool create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  auto vmm_pool_or = VmmExportedPool::create(allocator, region_size_or.value());
  if (!vmm_pool_or.has_value()) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: device pool create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  auto vmm_pool = std::move(vmm_pool_or).value();
  if (!initialize_generation_words(vmm_pool.ptr(), aligned_slot_bytes, buffer_count)) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: device generation initialization failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  vmm_pool.register_fd_export(session);

  auto ref_count_mapping_or = ShmRefCountMapping::create(buffer_count);
  if (!ref_count_mapping_or.has_value()) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: ref-count SHM create failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }

  const auto cuda_device_ordinal_or = cuda::current_device_ordinal();
  if (!cuda_device_ordinal_or.has_value()) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: no current CUDA device context");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  auto pool = std::unique_ptr<DeviceBufferPoolImpl>(
      new (std::nothrow) DeviceBufferPoolImpl(PrivateTag{},
                                              std::move(ref_count_mapping_or).value(),
                                              std::move(vmm_pool),
                                              std::move(fd_pass_socket_path),
                                              buffer_bytes,
                                              aligned_slot_bytes,
                                              buffer_count,
                                              cuda_device_ordinal_or.value()));
  if (!pool) {
    HOLOMQ_LOG_ERROR("BufferPool::create failed: device pool allocation failed");
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return pool;
}

BufferMemoryType DeviceBufferPoolImpl::memory_type() const noexcept {
  return BufferMemoryType::Device;
}

std::optional<int32_t> DeviceBufferPoolImpl::cuda_device_ordinal() const noexcept {
  return cuda_device_ordinal_;
}

void* DeviceBufferPoolImpl::base_ptr() const noexcept {
  return vmm_pool_.ptr();
}

Buffer DeviceBufferPoolImpl::buffer_at_index(const std::uint32_t index) noexcept {
  if (index >= buffer_count()) {
    return {};
  }
  return Buffer(std::make_unique<DeviceBufferImpl>(*this, index));
}

BufferHandle DeviceBufferPoolImpl::handle(const std::uint32_t index) const noexcept {
  detail::device::BufferHandleData d{};
  d.pool_cookie = pool_cookie();
  d.slot_stride_bytes = slot_stride_bytes();
  d.buffer_bytes = buffer_bytes_;
  d.pool_total_bytes = vmm_pool_.size();
  d.cuda_device_ordinal = cuda_device_ordinal_;
  d.owner_pid = static_cast<std::uint32_t>(ref_count_mapping_->owner_pid());
  d.shm_ref_count_id = ref_count_mapping_->shm_ref_count_id();
  const auto copy_len = std::min(fd_pass_socket_path_.size(), fd_pass::endpoint_max_length());
  std::copy_n(fd_pass_socket_path_.data(), copy_len, d.fd_pass_socket_path);
  d.fd_pass_socket_path[copy_len] = '\0';
  return detail::make_device_buffer_handle(d, index, 0U);
}

std::uint64_t DeviceBufferPoolImpl::pool_cookie() const noexcept {
  return vmm_pool_.cookie();
}

}  // namespace holomq
