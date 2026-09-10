/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_tensor_resources.hpp"

#include "runtime_tensor_output_plan.hpp"
#include "runtime_tensor_output_pool.hpp"
#include "runtime_tensor_resource_materialization.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <holomq/buffer.hpp>
#include <holomq/buffer_pool.hpp>
#include <holomq/context_builder.hpp>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include "../channel/channel_common.hpp"
#include "../channel/holomq_tensor.hpp"
#include "../compiler/execution_plan_contracts.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/domain/tensor_access.hpp"
#include "../core/domain/tensor_layout_validation.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/plan_record_contracts.hpp"
#include "../core/tensor_allocation_record_validation.hpp"
#include "../core/tensor_output_loan_internal.hpp"
#include "graph_exec.hpp"
#include "runtime_callback_context.hpp"
#include "runtime_cuda_execution_resources.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {

namespace {

[[nodiscard]] std::optional<std::uint64_t> contiguous_axis_byte_stride(
    const TensorShapeView shape, const std::uint64_t item_bytes, const std::size_t axis) noexcept {
  if (axis >= shape.size()) {
    return std::nullopt;
  }
  std::uint64_t stride = item_bytes;
  for (std::size_t reverse = shape.size(); reverse > axis + 1U; --reverse) {
    const auto extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(shape[reverse - 1U]), 1U);
    if (stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return std::nullopt;
    }
    stride *= extent;
  }
  return stride;
}

[[nodiscard]] std::optional<std::uint64_t> request_axis_byte_stride(
    const TensorLoanRequest& request, const std::uint64_t item_bytes,
    const std::size_t axis) noexcept {
  if (request.byte_strides) {
    const auto strides = request.byte_strides.value();
    if (strides.size() != request.shape.size() || axis >= strides.size() || strides[axis] < 0) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(strides[axis]);
  }
  return contiguous_axis_byte_stride(request.shape, item_bytes, axis);
}

[[nodiscard]] std::optional<ValidatedTensorLayout> validate_request_against_allocation(
    const TensorAllocationPlanRecord& plan, const TensorLoanRequest& request) noexcept {
  if (!valid_tensor_dtype(request.dtype) || request.shape.size() > plan.bounds.max_rank ||
      (plan.layout && plan.layout->memory_kind && *plan.layout->memory_kind != plan.memory_kind) ||
      (plan.layout && plan.layout->rank && request.shape.size() != *plan.layout->rank) ||
      (plan.layout && plan.layout->dtype &&
       !dl_data_type_exactly_equal(request.dtype, *plan.layout->dtype)) ||
      plan.axes.size() > request.shape.size()) {
    return std::nullopt;
  }
  const std::optional<std::span<const std::int64_t>> strides =
      request.byte_strides
          ? std::optional<std::span<const std::int64_t>>{request.byte_strides.value()}
          : std::nullopt;
  const auto validated =
      validate_tensor_layout(request.shape, request.dtype, strides, TensorStrideUnit::kBytes);
  if (!validated || validated->byte_span > plan.bounds.max_byte_span) {
    return std::nullopt;
  }
  if (plan.layout && plan.layout->strides == TensorStridePolicy::kContiguousOnly &&
      request.byte_strides) {
    for (std::size_t axis = 0U; axis < request.shape.size(); ++axis) {
      const auto canonical =
          contiguous_axis_byte_stride(request.shape, validated->item_bytes, axis);
      if (!canonical || request.byte_strides.value()[axis] < 0 ||
          static_cast<std::uint64_t>(request.byte_strides.value()[axis]) != *canonical) {
        return std::nullopt;
      }
    }
  }
  for (std::size_t axis = 0U; axis < plan.axes.size(); ++axis) {
    if (request.shape[axis] < 0 ||
        static_cast<std::uint64_t>(request.shape[axis]) > plan.axes[axis].max_extent) {
      return std::nullopt;
    }
    const auto stride = request_axis_byte_stride(request, validated->item_bytes, axis);
    if (!stride || (plan.axes[axis].max_abs_stride_bytes != 0U &&
                    *stride > plan.axes[axis].max_abs_stride_bytes)) {
      return std::nullopt;
    }
  }
  return *validated;
}

class ScopedCudaDevice final {
 public:
  explicit ScopedCudaDevice(DeviceId requested) noexcept {
    if (!requested.valid() || cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return;
      }
      changed_ = true;
    }
    active_ = true;
  }

  ~ScopedCudaDevice() {
    if (changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_{};
  bool active_{};
  bool changed_{};
};

class TensorBacking final {
 public:
  TensorBacking(const TensorBacking&) = delete;
  TensorBacking& operator=(const TensorBacking&) = delete;
  ~TensorBacking() {
    if (exportable_) {
      return;
    }
    if (pointer_ == nullptr) {
      return;
    }
    switch (kind_) {
      case MemoryKind::kHost:
        delete[] static_cast<std::byte*>(pointer_);
        break;
      case MemoryKind::kPinnedHost:
        static_cast<void>(cudaFreeHost(pointer_));
        break;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged: {
        const ScopedCudaDevice selected{device_};
        if (selected.active()) {
          static_cast<void>(cudaFree(pointer_));
        }
        break;
      }
      case MemoryKind::kUnknown:
        break;
    }
  }

  [[nodiscard]] static expected<std::shared_ptr<TensorBacking>, Error> create(
      MemoryKind kind, DeviceId device, std::size_t bytes, std::string_view pool_id) {
    if (kind == MemoryKind::kUnknown || bytes == 0U ||
        ((kind == MemoryKind::kCudaDevice || kind == MemoryKind::kCudaManaged) &&
         !device.valid())) {
      return make_unexpected(tensor_resource_error("invalid backing request", "pool", pool_id));
    }
    auto result = std::shared_ptr<TensorBacking>{new TensorBacking{kind, device, bytes}};
    switch (kind) {
      case MemoryKind::kHost:
        result->pointer_ = new (std::nothrow) std::byte[bytes];
        break;
      case MemoryKind::kPinnedHost:
        if (cudaMallocHost(&result->pointer_, bytes) != cudaSuccess) {
          return make_unexpected(
              tensor_resource_error("pinned-host allocation failed", "pool", pool_id));
        }
        break;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged: {
        const ScopedCudaDevice selected{device};
        if (!selected.active()) {
          return make_unexpected(
              tensor_resource_error("CUDA device selection failed", "pool", pool_id));
        }
        const cudaError_t status =
            kind == MemoryKind::kCudaDevice
                ? cudaMalloc(&result->pointer_, bytes)
                : cudaMallocManaged(&result->pointer_, bytes, cudaMemAttachGlobal);
        if (status != cudaSuccess) {
          return make_unexpected(
              tensor_resource_error("CUDA backing allocation failed", "pool", pool_id));
        }
        break;
      }
      case MemoryKind::kUnknown:
        break;
    }
    if (result->pointer_ == nullptr) {
      return make_unexpected(
          tensor_resource_error("backing allocation returned null", "pool", pool_id));
    }
    return result;
  }

  [[nodiscard]] static expected<std::shared_ptr<TensorBacking>, Error> create_exportable(
      MemoryKind kind, DeviceId device, std::size_t bytes,
      std::shared_ptr<holomq::BufferPool> pool_owner, std::string_view pool_id) {
    if ((kind != MemoryKind::kHost && kind != MemoryKind::kPinnedHost &&
         kind != MemoryKind::kCudaDevice) ||
        bytes == 0U ||
        ((kind == MemoryKind::kPinnedHost || kind == MemoryKind::kCudaDevice) && !device.valid()) ||
        (kind == MemoryKind::kHost && device.valid()) || pool_owner == nullptr) {
      return make_unexpected(
          tensor_resource_error("invalid exportable backing request", "pool", pool_id));
    }
    auto result = std::shared_ptr<TensorBacking>{new TensorBacking{kind, device, bytes}};
    result->pool_owner_ = std::move(pool_owner);
    result->exportable_ = true;
    return result;
  }

  [[nodiscard]] expected<void, Error> bind(holomq::Buffer buffer,
                                           std::string_view pool_id) noexcept {
    const holomq::BufferMemoryType expected =
        kind_ == MemoryKind::kCudaDevice
            ? holomq::BufferMemoryType::Device
            : (kind_ == MemoryKind::kPinnedHost ? holomq::BufferMemoryType::HostPinned
                                                : holomq::BufferMemoryType::HostPageable);
    if (!exportable_ || buffer_ || !buffer || buffer.memory_type() != expected ||
        buffer.size_bytes() < bytes_ ||
        (kind_ == MemoryKind::kCudaDevice &&
         buffer.cuda_device_ordinal() != std::optional<std::int32_t>{device_.value})) {
      return make_unexpected(
          tensor_resource_error("exportable buffer lease mismatch", "pool", pool_id));
    }
    pointer_ = buffer.ptr();
    buffer_.emplace(std::move(buffer));
    return {};
  }

  [[nodiscard]] bool has_buffer() const noexcept { return buffer_.has_value(); }

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle(cudaStream_t stream) noexcept {
    if (!exportable_ || !buffer_) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
          ErrorCode::kInvalidArgument, "direct Tensor output has no exportable buffer lease");
    }
    auto handle = stream == nullptr ? buffer_->export_handle_unsynchronized()
                                    : buffer_->export_handle(stream);
    if (!handle) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(handle).error(), "failed to export direct Tensor output buffer"));
    }
    return std::move(handle).value();
  }

  [[nodiscard]] bool prepare_for_reuse() noexcept {
    return !exportable_ || (buffer_ && buffer_->prepare_for_reuse());
  }

  [[nodiscard]] void* pointer() const noexcept { return pointer_; }

  [[nodiscard]] MemoryKind memory_kind() const noexcept { return kind_; }

 private:
  TensorBacking(MemoryKind kind, DeviceId device, std::size_t bytes) noexcept
      : kind_(kind), device_(device), bytes_(bytes) {}

  MemoryKind kind_{MemoryKind::kUnknown};
  DeviceId device_{};
  std::size_t bytes_{};
  void* pointer_{};
  // HoloMQ Buffer implementations reference their owning pool. Member order also releases the
  // optional Buffer before its final shared pool claim.
  std::shared_ptr<holomq::BufferPool> pool_owner_;
  std::optional<holomq::Buffer> buffer_;
  bool exportable_{};
};

class SlotLifetime final {
 public:
  SlotLifetime(const SlotLifetime&) = delete;
  SlotLifetime& operator=(const SlotLifetime&) = delete;
  ~SlotLifetime() {
    if (readiness_event_ == nullptr && release_events_.empty()) {
      return;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return;
    }
    for (cudaEvent_t event : release_events_) {
      if (event != nullptr && cudaEventDestroy(event) == cudaSuccess && test_control_ != nullptr) {
        test_control_->record_cuda_event_destroyed(test_pool_index_);
      }
    }
    if (readiness_event_ != nullptr) {
      if (cudaEventDestroy(readiness_event_) == cudaSuccess && test_control_ != nullptr) {
        test_control_->record_cuda_event_destroyed(test_pool_index_);
      }
    }
  }

  [[nodiscard]] static expected<std::shared_ptr<SlotLifetime>, Error> create(
      std::shared_ptr<TensorBacking> backing, DeviceId device, bool cuda_ordered,
      bool elide_same_stream_readiness, std::uint32_t release_stream_capacity,
      std::string_view pool_id, std::shared_ptr<RuntimeTensorOutputTestControl> test_control = {},
      std::size_t test_pool_index = 0U) {
    auto result = std::shared_ptr<SlotLifetime>{new SlotLifetime{std::move(backing),
                                                                 device,
                                                                 elide_same_stream_readiness,
                                                                 std::move(test_control),
                                                                 test_pool_index}};
    if (!cuda_ordered) {
      return result;
    }
    if (result->test_control_ != nullptr &&
        result->test_control_->consume_fault(
            RuntimeTensorOutputTestFault::kEventMaterializationFailure, result->test_pool_index_)) {
      return make_unexpected(
          tensor_resource_error("readiness-event materialization failed", "pool", pool_id));
    }
    const ScopedCudaDevice selected{device};
    if (!selected.active() || cudaEventCreateWithFlags(&result->readiness_event_,
                                                       cudaEventDisableTiming) != cudaSuccess) {
      return make_unexpected(
          tensor_resource_error("readiness-event materialization failed", "pool", pool_id));
    }
    if (result->test_control_ != nullptr) {
      result->test_control_->record_cuda_events_created(result->test_pool_index_, 1U);
    }
    result->release_streams_.reserve(release_stream_capacity);
    result->release_events_.resize(release_stream_capacity, nullptr);
    for (cudaEvent_t& event : result->release_events_) {
      if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
        return make_unexpected(
            tensor_resource_error("release-event materialization failed", "pool", pool_id));
      }
      if (result->test_control_ != nullptr) {
        result->test_control_->record_cuda_events_created(result->test_pool_index_, 1U);
      }
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> bind_exportable_buffer(holomq::Buffer buffer,
                                                             std::string_view pool_id) noexcept {
    return backing_ != nullptr ? backing_->bind(std::move(buffer), pool_id)
                               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }

  [[nodiscard]] bool has_exportable_buffer() const noexcept {
    return backing_ != nullptr && backing_->has_buffer();
  }

  [[nodiscard]] void* data() const noexcept {
    return backing_ != nullptr ? backing_->pointer() : nullptr;
  }

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept {
    cudaStream_t stream = nullptr;
    {
      const std::lock_guard lock(mutex_);
      stream = producer_stream_;
    }
    return backing_ != nullptr
               ? backing_->export_handle(stream)
               : channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
                     ErrorCode::kFailure, "direct Tensor output backing is missing");
  }

  [[nodiscard]] bool prepare_exportable_buffer_for_reuse() noexcept {
    return backing_ != nullptr && backing_->prepare_for_reuse();
  }

  [[nodiscard]] bool record_readiness(cudaStream_t stream) noexcept {
    if (readiness_event_ == nullptr || stream == nullptr) {
      return false;
    }
    const std::lock_guard lock(mutex_);
    producer_stream_ = stream;
    if (elide_same_stream_readiness_) {
      readiness_recorded_ = true;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_readiness_elision(test_pool_index_);
      }
      return true;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active() || cudaEventRecord(readiness_event_, stream) != cudaSuccess) {
      producer_stream_ = nullptr;
      return false;
    }
    readiness_recorded_ = true;
    readiness_event_recorded_ = true;
    if (test_control_ != nullptr) {
      test_control_->record_cuda_readiness(test_pool_index_);
    }
    return true;
  }

  [[nodiscard]] bool record_release() noexcept {
    const std::lock_guard lock(mutex_);
    if (release_streams_.empty()) {
      // A canceled allocation that never entered a CUDA write has no work to retire. A completed
      // device write is still protected by readiness_event_ in retirement_complete().
      return true;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return false;
    }
    release_record_count_ = 0U;
    for (std::size_t index = 0U; index < release_streams_.size(); ++index) {
      if (cudaEventRecord(release_events_[index], release_streams_[index]) != cudaSuccess) {
        return false;
      }
      ++release_record_count_;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_release(test_pool_index_);
      }
    }
    return true;
  }

  [[nodiscard]] bool retirement_complete() const noexcept {
    const std::lock_guard lock(mutex_);
    if (release_record_count_ == 0U && !readiness_event_recorded_) {
      return true;
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return false;
    }
    bool complete = true;
    if (release_record_count_ != 0U) {
      for (std::size_t index = 0U; index < release_record_count_; ++index) {
        complete = complete && cudaEventQuery(release_events_[index]) == cudaSuccess;
      }
    } else if (readiness_event_recorded_) {
      complete = cudaEventQuery(readiness_event_) == cudaSuccess;
    }
    if (test_control_ != nullptr) {
      test_control_->record_cuda_retirement_query(test_pool_index_, complete);
    }
    return complete;
  }

  [[nodiscard]] bool has_release_streams() const noexcept {
    const std::lock_guard lock(mutex_);
    return !release_streams_.empty();
  }

  [[nodiscard]] bool retirement_requires_query() const noexcept {
    const std::lock_guard lock(mutex_);
    return release_record_count_ != 0U || readiness_event_recorded_;
  }

  void reset_generation() noexcept {
    const std::lock_guard lock(mutex_);
    readiness_recorded_ = false;
    readiness_event_recorded_ = false;
    producer_stream_ = nullptr;
    release_record_count_ = 0U;
    release_streams_.clear();
  }

  [[nodiscard]] bool set_release_stream(cudaStream_t stream) noexcept {
    if (stream == nullptr) {
      return false;
    }
    const std::lock_guard lock(mutex_);
    if (release_events_.empty()) {
      return false;
    }
    if (elide_same_stream_readiness_ && producer_stream_ == stream) {
      // Reuse work is enqueued on this same fixed stream after logical handle retirement.
      if (test_control_ != nullptr) {
        test_control_->record_cuda_release_stream_elision(test_pool_index_);
      }
      return true;
    }
    if (std::find(release_streams_.begin(), release_streams_.end(), stream) !=
        release_streams_.end()) {
      return true;
    }
    if (release_streams_.size() == release_streams_.capacity()) {
      return false;
    }
    release_streams_.push_back(stream);
    if (test_control_ != nullptr) {
      test_control_->record_cuda_release_stream(test_pool_index_);
    }
    return true;
  }

  [[nodiscard]] bool has_readiness() const noexcept { return readiness_event_ != nullptr; }

  [[nodiscard]] cudaStream_t producer_stream() const noexcept {
    const std::lock_guard lock(mutex_);
    return readiness_recorded_ ? producer_stream_ : nullptr;
  }

  [[nodiscard]] expected<void, Error> synchronize(cudaStream_t consumer) const noexcept {
    const std::lock_guard lock(mutex_);
    if (committed_host_write_ready_locked()) {
      return {};
    }
    if (consumer == nullptr || readiness_event_ == nullptr || !readiness_recorded_ ||
        producer_stream_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (consumer == producer_stream_) {
      if (test_control_ != nullptr) {
        test_control_->record_cuda_stream_wait_elision(test_pool_index_);
      }
      return {};
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (!readiness_event_recorded_) {
      if (cudaEventRecord(readiness_event_, producer_stream_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      readiness_event_recorded_ = true;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_readiness(test_pool_index_);
      }
    }
    if (cudaStreamWaitEvent(consumer, readiness_event_, 0U) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (test_control_ != nullptr) {
      test_control_->record_cuda_stream_wait(test_pool_index_);
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> synchronize_host() const noexcept {
    const std::lock_guard lock(mutex_);
    if (committed_host_write_ready_locked()) {
      return {};
    }
    const ScopedCudaDevice selected{device_};
    if (!selected.active() || readiness_event_ == nullptr || !readiness_recorded_ ||
        producer_stream_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (!readiness_event_recorded_) {
      if (cudaEventRecord(readiness_event_, producer_stream_) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      readiness_event_recorded_ = true;
      if (test_control_ != nullptr) {
        test_control_->record_cuda_readiness(test_pool_index_);
      }
    }
    if (cudaEventSynchronize(readiness_event_) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (test_control_ != nullptr) {
      test_control_->record_cuda_host_wait(test_pool_index_);
    }
    return {};
  }

 private:
  [[nodiscard]] bool committed_host_write_ready_locked() const noexcept {
    if (readiness_recorded_ || backing_ == nullptr) {
      return false;
    }
    // write_host() completes before commit for every CPU-addressable placement, so there is no
    // asynchronous producer work for either a CUDA consumer or the host to fence.
    const MemoryKind kind = backing_->memory_kind();
    return kind == MemoryKind::kHost || kind == MemoryKind::kPinnedHost ||
           kind == MemoryKind::kCudaManaged;
  }

  SlotLifetime(std::shared_ptr<TensorBacking> backing, DeviceId device,
               bool elide_same_stream_readiness,
               std::shared_ptr<RuntimeTensorOutputTestControl> test_control,
               std::size_t test_pool_index) noexcept
      : backing_(std::move(backing)),
        device_(device),
        elide_same_stream_readiness_(elide_same_stream_readiness),
        test_control_(std::move(test_control)),
        test_pool_index_(test_pool_index) {}

  std::shared_ptr<TensorBacking> backing_;
  DeviceId device_{};
  bool elide_same_stream_readiness_{};
  std::shared_ptr<RuntimeTensorOutputTestControl> test_control_;
  std::size_t test_pool_index_{};
  cudaEvent_t readiness_event_{};
  std::vector<cudaEvent_t> release_events_;
  std::vector<cudaStream_t> release_streams_;
  mutable std::mutex mutex_;
  cudaStream_t producer_stream_{};
  std::size_t release_record_count_{};
  bool readiness_recorded_{};
  mutable bool readiness_event_recorded_{};
};

enum class SlotPhase : std::uint8_t {
  kFree,
  kLoaned,
  kWriterActive,
  kReady,
  kSubmitPending,
  kRetained,
  kPublicationTerminal,
  kReleaseRecording,
  kRetiring,
  kQuarantined,
};

enum class TensorOutputOperation : std::uint8_t {
  kAllocate,
  kBeginWrite,
  kCommitWrite,
  kDropWrite,
  kEmit,
  kCancel,
  kRetire,
  kRestart,
};

class OutputPool;

class OutputPoolReferenceControl final {
 public:
  explicit OutputPoolReferenceControl(OutputPool* pool) noexcept : pool_(pool) {}

  [[nodiscard]] OutputPool* acquire() noexcept {
    if (detaching_.load(std::memory_order_acquire)) {
      return nullptr;
    }
    in_flight_.fetch_add(1U, std::memory_order_acq_rel);
    if (detaching_.load(std::memory_order_acquire)) {
      release();
      return nullptr;
    }
    auto* const result = pool_.load(std::memory_order_acquire);
    if (result == nullptr) {
      release();
    }
    return result;
  }

  void release() noexcept {
    if (in_flight_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      in_flight_.notify_all();
    }
  }

  void detach() noexcept {
    detaching_.store(true, std::memory_order_release);
    pool_.store(nullptr, std::memory_order_release);
    std::uint32_t current = in_flight_.load(std::memory_order_acquire);
    while (current != 0U) {
      in_flight_.wait(current, std::memory_order_acquire);
      current = in_flight_.load(std::memory_order_acquire);
    }
  }

 private:
  std::atomic<OutputPool*> pool_{};
  std::atomic<std::uint32_t> in_flight_{};
  std::atomic_bool detaching_{};
};

struct OutputSlot {
  OutputPool* pool{};
  std::shared_ptr<OutputPoolReferenceControl> pool_control;
  std::weak_ptr<OutputSlot> self;
  std::size_t index{};
  std::uint64_t generation{};
  PlanGenerationId loan_plan_generation{};
  RunGenerationId loan_run_generation{};
  std::uint64_t loan_pool_generation{};
  std::uint64_t loan_granted_ns{};
  Tag activation{};
  SlotPhase phase{SlotPhase::kFree};
  std::atomic<std::uint32_t> tensor_handles{};
  std::atomic_bool retirement_requested{};
  std::atomic_bool quarantine_requested{};
  std::atomic<OutputSlot*> deferred_next{};
  std::atomic_bool deferred_queued{};
  TensorSlotClaimLedger claims;
  bool on_free_list{};
  bool on_retirement_queue{};
  bool loan_live{};
  bool retry_cancellation_demand{};
  bool cancel_pending{};
  bool writer_cuda{};
  cudaStream_t writer_stream{};
  std::size_t byte_size{};
  DLDataType dtype{};
  DeviceId device;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> byte_strides;
  bool explicit_strides{};
  std::shared_ptr<SlotLifetime> lifetime;
  std::shared_ptr<DLManagedTensorContext> context;
};

class OutputPoolCallbackClaim final {
 public:
  explicit OutputPoolCallbackClaim(const OutputSlot& slot) noexcept
      : control_(slot.pool_control), pool_(control_ != nullptr ? control_->acquire() : nullptr) {}
  OutputPoolCallbackClaim(const OutputPoolCallbackClaim&) = delete;
  OutputPoolCallbackClaim& operator=(const OutputPoolCallbackClaim&) = delete;
  ~OutputPoolCallbackClaim() {
    if (pool_ != nullptr) {
      control_->release();
    }
  }

  [[nodiscard]] explicit operator bool() const noexcept { return pool_ != nullptr; }
  [[nodiscard]] OutputPool* pool() const noexcept { return pool_; }

 private:
  std::shared_ptr<OutputPoolReferenceControl> control_;
  OutputPool* pool_{};
};

template <typename Base>
class SlotMemoryReference final : public Base {
 public:
  SlotMemoryReference(std::shared_ptr<OutputPoolReferenceControl> control, std::size_t slot,
                      std::shared_ptr<SlotLifetime> lifetime) noexcept
      : control_(std::move(control)), slot_(slot), lifetime_(std::move(lifetime)) {}

  void retain_tensor_handle() noexcept override;
  void release_tensor_handle() noexcept override;

  bool set_deallocation_stream(cudaStream_t stream) override {
    return lifetime_ != nullptr && lifetime_->set_release_stream(stream);
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    return lifetime_ != nullptr && lifetime_->record_readiness(stream);
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return lifetime_ != nullptr && lifetime_->has_readiness();
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return lifetime_ != nullptr ? lifetime_->producer_stream() : nullptr;
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    return lifetime_ != nullptr
               ? lifetime_->synchronize(consumer_stream)
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kNotSupported})};
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    return lifetime_ != nullptr
               ? lifetime_->synchronize_host()
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kNotSupported})};
  }

  // This is virtual only for the HoloMQ-exportable Base specialization.
  // NOLINTNEXTLINE(clang-diagnostic-inconsistent-missing-override)
  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept {
    return lifetime_ != nullptr
               ? lifetime_->export_handle()
               : channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
                     ErrorCode::kFailure, "direct Tensor output lifetime is missing");
  }

 private:
  std::shared_ptr<OutputPoolReferenceControl> control_;
  std::size_t slot_{};
  std::shared_ptr<SlotLifetime> lifetime_;
};

using NativeSlotMemoryReference = SlotMemoryReference<TensorMemoryReference>;
using ExportableSlotMemoryReference =
    SlotMemoryReference<channel_internal::HoloMQExportableTensorMemoryReference>;

using DeviceResource = RuntimeTensorDeviceResource;

[[nodiscard]] expected<bool, Error> output_requires_holomq_backing(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
    const std::optional<std::uint32_t> tensor_index = std::nullopt) {
  auto required = false;
  for (const auto& service : plan.cross_process_channel_services) {
    if (service.source != allocation.output) {
      continue;
    }
    const auto view = tensor_service_contract_view(service);
    const bool direct_tensor = view.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor_schema = descriptor_tensor_schema_service(view);
    bool storage_matches = view.tensor_max_byte_span == allocation.bounds.max_byte_span &&
                           view.tensor_memory_kind == allocation.memory_kind &&
                           view.tensor_device == allocation.device;
    if (tensor_index) {
      storage_matches =
          descriptor_tensor_schema && *tensor_index < view.tensor_fields.size() &&
          view.tensor_fields[*tensor_index].tensor_index == *tensor_index &&
          view.tensor_fields[*tensor_index].max_byte_span == allocation.bounds.max_byte_span &&
          view.tensor_fields[*tensor_index].memory_kind == allocation.memory_kind &&
          view.tensor_fields[*tensor_index].device == allocation.device;
    }
    if (!view.tensor_payload || (!direct_tensor && !descriptor_tensor_schema) ||
        (direct_tensor && view.tensor_pool_id != allocation.pool_id) || !storage_matches) {
      return make_unexpected(tensor_resource_error(
          "cross-process export contract mismatch", "output", allocation.output.value()));
    }
    required = true;
  }
  if (required && allocation.memory_kind != MemoryKind::kHost &&
      allocation.memory_kind != MemoryKind::kPinnedHost &&
      allocation.memory_kind != MemoryKind::kCudaDevice) {
    return make_unexpected(tensor_resource_error(
        "cross-process output placement is not exportable", "output", allocation.output.value()));
  }
  return expected<bool, Error>{required};
}

[[nodiscard]] expected<void, Error> validate_lifecycle_candidate_retained_contract(
    const ExecutionPlanState& plan, const ProviderAllocationPlanRecord& provider,
    const TensorAllocationPlanRecord& allocation) {
  if (provider.geometry_mode != AllocationGeometryMode::kLifecycleCandidate) {
    return {};
  }
  if (provider.allocation_kind != ProviderAllocationKind::kTensorOutput || !provider.output ||
      provider.tensor_field_id || provider.tensor_index || provider.interface_topic ||
      !provider.tensor_template || *provider.output != allocation.output ||
      provider.pool_id != allocation.pool_id || allocation.memory_kind != MemoryKind::kHost ||
      allocation.device.valid() || provider.tensor_template->memory_kind != MemoryKind::kHost ||
      provider.tensor_template->device.valid() || !provider.tensor_template->layout ||
      !provider.tensor_template->layout->memory_kind ||
      *provider.tensor_template->layout->memory_kind != MemoryKind::kHost) {
    return make_unexpected(
        tensor_resource_error("lifecycle candidate Tensor provider is not direct host geometry",
                              "output",
                              allocation.output.value()));
  }
  auto exportable = output_requires_holomq_backing(plan, allocation);
  if (!exportable) {
    return make_unexpected(std::move(exportable.error()));
  }
  auto found_service = false;
  for (const auto& service : plan.process_local_channel_services) {
    if (service.source != allocation.output) {
      continue;
    }
    found_service = true;
    if (service.delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
        service.cross_process_transport || service.queue_policy != QueuePolicy::kBlocking) {
      return make_unexpected(tensor_resource_error(
          "lifecycle candidate Tensor provider requires direct retained delivery",
          "output",
          allocation.output.value()));
    }
  }
  if (*exportable || !found_service) {
    return make_unexpected(
        tensor_resource_error("lifecycle candidate Tensor provider cannot use transport backing",
                              "output",
                              allocation.output.value()));
  }
  return {};
}

[[nodiscard]] expected<holomq::BufferPool, Error> create_holomq_output_pool(
    const TensorAllocationPlanRecord& allocation,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    std::size_t test_pool_index) {
  if (allocation.bounds.max_byte_span > std::numeric_limits<std::size_t>::max()) {
    return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
        ErrorCode::kResourceExhausted, "direct Tensor output byte span exceeds address space");
  }
  const auto create_pool = [&]() -> expected<holomq::BufferPool, Error> {
    if (cudaFree(nullptr) != cudaSuccess) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
          ErrorCode::kNotSupported, "direct Tensor output requires CUDA VMM support");
    }
    if (test_control != nullptr) {
      CUdevice current_device = -1;
      if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS) {
        return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
            ErrorCode::kNotSupported, "direct Tensor output could not observe its CUDA VMM device");
      }
      test_control->record_holomq_generation_vmm_device(test_pool_index, current_device);
    }
    auto context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
    if (!context) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(context).error(), "failed to create direct Tensor export context"));
    }
    auto pool =
        allocation.memory_kind == MemoryKind::kCudaDevice
            ? context->create_device_buffer_pool(
                  static_cast<std::size_t>(allocation.bounds.max_byte_span), allocation.slot_count)
        : allocation.memory_kind == MemoryKind::kPinnedHost
            ? context->create_host_pinned_buffer_pool(
                  static_cast<std::size_t>(allocation.bounds.max_byte_span),
                  allocation.slot_count,
                  CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP)
            : context->create_host_buffer_pool(
                  static_cast<std::size_t>(allocation.bounds.max_byte_span), allocation.slot_count);
    if (!pool) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(pool).error(), "failed to create direct Tensor export pool"));
    }
    return std::move(pool).value();
  };
  if (allocation.memory_kind == MemoryKind::kHost ||
      allocation.memory_kind == MemoryKind::kCudaDevice ||
      allocation.memory_kind == MemoryKind::kPinnedHost) {
    // Host payloads carry the DLPack-mandated device id 0 but have no allocation device. Their
    // split generation words are still CUDA VMM, so create that private synchronization pool on
    // the runtime's fixed host synchronization device instead of the caller's ambient device.
    const DeviceId pool_device =
        allocation.memory_kind == MemoryKind::kHost ? DeviceId{0} : allocation.device;
    const ScopedCudaDevice selected{pool_device};
    if (!selected.active()) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferPool>(
          ErrorCode::kNotSupported, "direct Tensor output device is unavailable");
    }
    return create_pool();
  }
  return create_pool();
}

class OutputPool final {
 public:
  OutputPool(const OutputPool&) = delete;
  OutputPool& operator=(const OutputPool&) = delete;
  ~OutputPool() {
    detach();
    if (reference_control_ != nullptr) {
      reference_control_->detach();
    }
    if (materialization_recorded_ && test_control_ != nullptr) {
      test_control_->release_materialization(test_pool_index_,
                                             static_cast<std::uint32_t>(slots_.size()),
                                             plan_.aggregate_byte_capacity);
    }
    record_test_snapshot(true);
  }

  // Pool construction authenticates all frozen geometry and materializes every slot as one cold
  // transaction; splitting the validation would duplicate partially constructed cleanup paths.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] static expected<std::unique_ptr<OutputPool>, Error> create(
      const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
      OutputPlanContract contract, RuntimeOperatorInstance& instance, void* declaration,
      const RuntimePortTypeOps* type_ops, cudaStream_t stream, bool elide_same_stream_readiness,
      RunGenerationId run_generation,
      std::shared_ptr<RuntimeTensorOutputTestControl> test_control = {},
      std::size_t test_pool_index = 0U, bool bind = true, std::uint64_t pool_generation = 1U,
      std::uint64_t allocation_generation = 1U,
      std::optional<std::uint32_t> descriptor_tensor_index = std::nullopt,
      std::optional<std::uint32_t> active_slot_count = std::nullopt,
      RuntimeTensorPoolAllocationAction allocation_action =
          RuntimeTensorPoolAllocationAction::kFresh) {
    const std::uint32_t selected_active_slot_count =
        active_slot_count.value_or(allocation.slot_count);
    if (!validate_tensor_allocation_record(allocation, plan.plan_generation) ||
        run_generation == 0U || pool_generation == 0U || allocation_generation == 0U ||
        selected_active_slot_count == 0U || selected_active_slot_count > allocation.slot_count ||
        declaration == nullptr || type_ops == nullptr || type_ops->publish_tensor_loan == nullptr ||
        type_ops->bind_tensor_output == nullptr || type_ops->unbind_tensor_output == nullptr ||
        ((allocation.memory_kind == MemoryKind::kCudaDevice ||
          allocation.memory_kind == MemoryKind::kCudaManaged) &&
         (!allocation.device.valid() || stream == nullptr)) ||
        (allocation.memory_kind == MemoryKind::kPinnedHost && allocation.device.valid() &&
         stream == nullptr)) {
      return make_unexpected(tensor_resource_error(
          "invalid output-pool contract", "output", allocation.output.value()));
    }
    const auto max_release_streams = release_stream_capacity(plan, allocation);
    if (!max_release_streams) {
      return make_unexpected(std::move(max_release_streams.error()));
    }

    auto exportable = output_requires_holomq_backing(plan, allocation, descriptor_tensor_index);
    if (!exportable) {
      return make_unexpected(std::move(exportable.error()));
    }
    std::shared_ptr<holomq::BufferPool> export_pool;
    if (*exportable) {
      auto pool = create_holomq_output_pool(allocation, test_control, test_pool_index);
      if (!pool) {
        return make_unexpected(std::move(pool.error()));
      }
      export_pool = std::make_shared<holomq::BufferPool>(std::move(*pool));
    }

    auto result = std::unique_ptr<OutputPool>{new OutputPool{allocation,
                                                             std::move(contract),
                                                             instance.vertex,
                                                             instance.operator_id,
                                                             declaration,
                                                             type_ops,
                                                             stream,
                                                             elide_same_stream_readiness,
                                                             plan.plan_generation,
                                                             run_generation,
                                                             pool_generation,
                                                             allocation_generation,
                                                             selected_active_slot_count,
                                                             allocation_action,
                                                             std::move(export_pool),
                                                             std::move(test_control),
                                                             test_pool_index,
                                                             descriptor_tensor_index}};
    result->slots_.reserve(allocation.slot_count);
    result->references_.reserve(allocation.slot_count);
    result->free_slots_.reserve(allocation.slot_count);
    result->retiring_slots_.reserve(allocation.retirement_capacity);
    result->diagnostic_storage_.resize(allocation.control_capacity);
    result->diagnostics_.reset(result->diagnostic_storage_);
    result->reference_control_ = std::make_shared<OutputPoolReferenceControl>(result.get());
    for (std::uint32_t index = 0U; index < allocation.slot_count; ++index) {
      const bool active_slot = index < selected_active_slot_count;
      auto slot = std::make_shared<OutputSlot>();
      slot->pool = result.get();
      slot->pool_control = result->reference_control_;
      slot->self = slot;
      slot->index = index;
      slot->generation = 1U;
      slot->on_free_list = active_slot;
      slot->shape.reserve(allocation.bounds.max_rank);
      slot->byte_strides.reserve(allocation.bounds.max_rank);
      // Native and exportable slots own their physical allocation for the pool lifetime. A
      // cold-bound exportable lease advances only after its remote import ref-count returns to the
      // owner-only state, so hot draws need no Buffer implementation allocation.
      auto backing = result->export_pool_
                         ? TensorBacking::create_exportable(allocation.memory_kind,
                                                            allocation.device,
                                                            allocation.bounds.max_byte_span,
                                                            result->export_pool_,
                                                            allocation.pool_id)
                         : TensorBacking::create(allocation.memory_kind,
                                                 allocation.device,
                                                 allocation.bounds.max_byte_span,
                                                 allocation.pool_id);
      if (!backing) {
        return make_unexpected(std::move(backing.error()));
      }
      auto lifetime = SlotLifetime::create(std::move(*backing),
                                           allocation.device,
                                           stream != nullptr,
                                           elide_same_stream_readiness,
                                           *max_release_streams,
                                           allocation.pool_id,
                                           result->test_control_,
                                           result->test_pool_index_);
      if (!lifetime) {
        return make_unexpected(std::move(lifetime.error()));
      }
      slot->lifetime = std::move(*lifetime);
      if (result->export_pool_) {
        auto acquired = result->export_pool_->acquire();
        if (!acquired) {
          return make_unexpected(tensor_resource_error(
              "exportable buffer cold binding failed", "pool", allocation.pool_id));
        }
        auto bound =
            slot->lifetime->bind_exportable_buffer(std::move(*acquired), allocation.pool_id);
        if (!bound) {
          return make_unexpected(std::move(bound).error());
        }
      }
      std::shared_ptr<TensorMemoryReference> reference;
      if (result->export_pool_) {
        reference = std::make_shared<ExportableSlotMemoryReference>(
            result->reference_control_, index, slot->lifetime);
      } else {
        reference = std::make_shared<NativeSlotMemoryReference>(
            result->reference_control_, index, slot->lifetime);
      }
      slot->context = std::make_shared<DLManagedTensorContext>();
      slot->context->dl_shape.reserve(allocation.bounds.max_rank);
      slot->context->dl_strides.reserve(allocation.bounds.max_rank);
      slot->context->memory_reference = reference;
      slot->context->tensor.dl_tensor.data = slot->lifetime->data();
      slot->context->tensor.dl_tensor.device =
          runtime_tensor_dl_device(allocation.memory_kind, allocation.device);
      slot->device = allocation.device;
      slot->memory_kind = allocation.memory_kind;
      result->references_.push_back(std::move(reference));
      result->slots_.push_back(std::move(slot));
      if (active_slot) {
        result->free_slots_.push_back(selected_active_slot_count - index - 1U);
      }
    }
    if (result->test_control_ != nullptr) {
      result->test_control_->record_materialization(result->test_pool_index_,
                                                    result->allocation_generation_,
                                                    allocation.slot_count,
                                                    result->active_slot_count_,
                                                    allocation.aggregate_byte_capacity);
      result->materialization_recorded_ = true;
    }
    if (bind) {
      result->bind();
      result->record_test_snapshot();
    }
    return result;
  }

  [[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> recreate_unbound(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance) const {
    return recreate_unbound(plan, instance, plan_);
  }

  [[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> recreate_unbound(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance,
      const TensorAllocationPlanRecord& allocation,
      std::optional<std::uint32_t> active_slot_count = std::nullopt) const {
    if (pool_generation_ == std::numeric_limits<std::uint64_t>::max() ||
        allocation_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(
          tensor_resource_error("Tensor allocation generation overflow", "pool", plan_.pool_id));
    }
    return create(plan,
                  allocation,
                  contract_,
                  instance,
                  declaration_,
                  type_ops_,
                  stream_,
                  elide_same_stream_readiness_,
                  run_generation_,
                  test_control_,
                  test_pool_index_,
                  false,
                  pool_generation_ + 1U,
                  allocation_generation_ + 1U,
                  descriptor_tensor_index_,
                  active_slot_count,
                  RuntimeTensorPoolAllocationAction::kReplaced);
  }

  [[nodiscard]] OperatorInstanceId owner() const noexcept { return owner_; }
  [[nodiscard]] PortId output() const noexcept { return plan_.output; }
  [[nodiscard]] const TensorAllocationPlanRecord& allocation() const noexcept { return plan_; }
  [[nodiscard]] std::uint64_t allocation_generation() const noexcept {
    return allocation_generation_;
  }
  [[nodiscard]] std::uint64_t storage_binding_generation() const noexcept {
    return pool_generation_;
  }
  [[nodiscard]] std::uint32_t physical_slot_count() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
  }
  [[nodiscard]] std::uint32_t active_slot_count() const noexcept { return active_slot_count_; }

  [[nodiscard]] expected<void, Error> configure_startup_active_slots(
      const std::uint32_t active_slot_count) noexcept {
    auto preflight = preflight_startup_active_slots(active_slot_count);
    if (!preflight) {
      return preflight;
    }
    publish_startup_active_slots(active_slot_count);
    return {};
  }

  [[nodiscard]] expected<void, Error> preflight_startup_active_slots(
      const std::uint32_t active_slot_count) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    reclaim_completed_retirements(lock);
    if (active_slot_count == 0U || active_slot_count > slots_.size() || closed_ ||
        restart_fenced_ || lifecycle_transition_active_ ||
        (active_slot_count != active_slot_count_ && !quiescent_free_inventory_locked())) {
      return make_unexpected(tensor_resource_error(
          "Tensor active-slot configuration rejected", "output", plan_.output.value()));
    }
    return {};
  }

  void publish_startup_active_slots(const std::uint32_t active_slot_count) noexcept {
    const std::lock_guard lock(mutex_);
    if (active_slot_count == active_slot_count_) {
      record_test_snapshot();
      return;
    }
    active_slot_count_ = active_slot_count;
    rebuild_active_free_list_locked();
    record_test_snapshot();
  }

  [[nodiscard]] expected<bool, Error> can_reuse_retained(
      const TensorAllocationPlanRecord& candidate, const std::uint32_t active_slot_count) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    reclaim_completed_retirements(lock);
    if (std::ranges::any_of(
            slots_, [](const auto& slot) { return slot->phase == SlotPhase::kQuarantined; })) {
      return make_unexpected(tensor_resource_error(
          "Tensor retained allocation is quarantined", "output", plan_.output.value()));
    }
    return restart_fenced_ && closed_ && !lifecycle_transition_active_ && active_slot_count != 0U &&
           active_slot_count <= slots_.size() && retained_geometry_compatible(plan_, candidate) &&
           quiescent_free_inventory_locked() &&
           !consume_test_fault(RuntimeTensorOutputTestFault::kRateReuseCompatibilityFailure);
  }

  [[nodiscard]] bool restart_reuse_publication_ready(
      const std::uint32_t active_slot_count) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    reclaim_completed_retirements(lock);
    return pool_generation_ != std::numeric_limits<std::uint64_t>::max() &&
           active_slot_count != 0U && active_slot_count <= slots_.size() && restart_fenced_ &&
           closed_ && !lifecycle_transition_active_ && quiescent_free_inventory_locked() &&
           !consume_test_fault(RuntimeTensorOutputTestFault::kRestartPublicationPreflightFailure);
  }

  void publish_rate_reuse(const std::uint32_t active_slot_count,
                          const RuntimeTensorPoolAllocationAction action,
                          const bool bind_output = true) noexcept {
    const std::lock_guard lock(mutex_);
    ++pool_generation_;
    active_slot_count_ = active_slot_count;
    allocation_action_ = action;
    rebuild_active_free_list_locked();
    restart_fenced_ = false;
    closed_ = false;
    if (bind_output) {
      bind();
    }
    record_test_snapshot();
  }

  void mark_bundle_binding_published() noexcept {
    const std::lock_guard lock(mutex_);
    ever_bound_ = true;
    record_test_snapshot();
  }

  [[nodiscard]] expected<void, Error> begin_restart() noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (restart_fenced_ || closed_ || lifecycle_transition_active_) {
      return make_unexpected(
          tensor_resource_error("restart fence rejected", "output", plan_.output.value()));
    }
    restart_fenced_ = true;
    closed_ = true;
    lifecycle_transition_active_ = true;
    for (auto& slot : slots_) {
      if (slot->phase == SlotPhase::kWriterActive) {
        slot->cancel_pending = true;
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
      } else if (slot->phase == SlotPhase::kLoaned || slot->phase == SlotPhase::kReady) {
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
        begin_retirement(*slot, lock);
      }
    }
    lifecycle_transition_active_ = false;
    return {};
  }

  [[nodiscard]] expected<void, Error> reopen(const bool bind_output = true) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (!restart_fenced_ || lifecycle_transition_active_) {
      return make_unexpected(
          tensor_resource_error("restart reopen rejected", "output", plan_.output.value()));
    }
    restart_fenced_ = false;
    closed_ = false;
    if (bind_output) {
      bind();
    }
    record_test_snapshot();
    return {};
  }

  [[nodiscard]] bool restart_publication_ready() noexcept {
    const std::lock_guard lock(mutex_);
    return restart_fenced_ && closed_ && !lifecycle_transition_active_ &&
           !consume_test_fault(RuntimeTensorOutputTestFault::kRestartPublicationPreflightFailure);
  }

  // The controller calls this only after preflighting every affected pool. No other controller
  // mutation may interleave at a lifecycle safe point, so publication contains no remaining
  // fallible step after another restart participant has committed.
  void publish_reopen(const bool bind_output = true) noexcept {
    const std::lock_guard lock(mutex_);
    restart_fenced_ = false;
    closed_ = false;
    if (bind_output) {
      bind();
    }
    record_test_snapshot();
  }

  void bind() noexcept {
    if (!bound_) {
      type_ops_->bind_tensor_output(declaration_,
                                    TensorOutputBinding{
                                        .state = this,
                                        .ops = &kBindingOps,
                                        .tensor_dtype = exact_bundle_dtype(),
                                    });
      bound_ = true;
      ever_bound_ = true;
    }
  }

  void close() noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_) {
      return;
    }
    closed_ = true;
    lifecycle_transition_active_ = true;
    for (auto& slot : slots_) {
      if (slot->phase == SlotPhase::kWriterActive) {
        slot->cancel_pending = true;
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
      } else if (slot->phase == SlotPhase::kLoaned || slot->phase == SlotPhase::kReady) {
        consume_loan(*slot);
        add_claim(*slot, TensorSlotClaimReason::kCancellation);
        begin_retirement(*slot, lock);
      }
    }
    lifecycle_transition_active_ = false;
    record_test_snapshot();
  }

  [[nodiscard]] expected<bool, Error> advance_terminal_retirement() noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    reclaim_completed_retirements(lock);
    if (std::ranges::any_of(
            slots_, [](const auto& slot) { return slot->phase == SlotPhase::kQuarantined; })) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return retirement_records_in_flight_ == 0U && live_loans_ == 0U &&
           deferred_head_.load(std::memory_order_acquire) == nullptr &&
           std::ranges::all_of(slots_, [this](const auto& slot) {
             return slot->phase == SlotPhase::kFree &&
                    slot->on_free_list == (slot->index < active_slot_count_) && !slot->loan_live &&
                    !slot->on_retirement_queue &&
                    slot->tensor_handles.load(std::memory_order_acquire) == 0U &&
                    !slot->claims.any();
           });
  }

  [[nodiscard]] expected<bool, Error> preflight_terminal_close() noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (std::ranges::any_of(
            slots_, [](const auto& slot) { return slot->phase == SlotPhase::kQuarantined; })) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return expected<bool, Error>{live_loans_ == 0U};
  }

  void detach() noexcept {
    if (bound_) {
      type_ops_->unbind_tensor_output(declaration_);
      bound_ = false;
    }
  }

  void retain_tensor(std::size_t index) noexcept {
    if (index >= slots_.size()) {
      return;
    }
    auto& slot = *slots_[index];
    std::uint32_t current = slot.tensor_handles.load(std::memory_order_acquire);
    while (current != std::numeric_limits<std::uint32_t>::max() &&
           !slot.tensor_handles.compare_exchange_weak(
               current, current + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    if (current == std::numeric_limits<std::uint32_t>::max()) {
      // A no-throw Tensor copy cannot report saturation. Quarantine the backing permanently rather
      // than wrap the counter and make a still-referenced slot reusable.
      slot.quarantine_requested.store(true, std::memory_order_release);
      enqueue_deferred(slot);
    }
  }

  void release_tensor(std::size_t index) noexcept {
    if (index >= slots_.size()) {
      return;
    }
    auto& slot = *slots_[index];
    std::uint32_t current = slot.tensor_handles.load(std::memory_order_acquire);
    while (current != 0U &&
           !slot.tensor_handles.compare_exchange_weak(
               current, current - 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    if (current == 0U) {
      slot.quarantine_requested.store(true, std::memory_order_release);
      enqueue_deferred(slot);
      return;
    }
    if (current == 1U) {
      slot.retirement_requested.store(true, std::memory_order_release);
      enqueue_deferred(slot);
    }
  }

  struct PreparedBundleField {
    OutputSlot* slot{};
    Tensor tensor;
  };

  [[nodiscard]] bool accepts_bundle_request(const TensorLoanRequest& request) const noexcept {
    return validate_request_against_allocation(plan_, request).has_value();
  }

  [[nodiscard]] std::optional<DLDataType> exact_bundle_dtype() const noexcept {
    if (plan_.layout && plan_.layout->dtype) {
      return plan_.layout->dtype;
    }
    return std::nullopt;
  }

  [[nodiscard]] expected<TensorOutputLoan, Error> allocate_bundle_field(
      TensorLoanRequest request) noexcept {
    return allocate(request);
  }

  [[nodiscard]] bool preflight_bundle_field(const TensorOutputLoan& loan) noexcept {
    auto* const selected =
        static_cast<OutputSlot*>(TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps));
    if (selected == nullptr) {
      return false;
    }
    const OutputPoolCallbackClaim selected_pool{*selected};
    if (!selected_pool || selected_pool.pool() != this || selected->index >= slots_.size() ||
        slots_[selected->index].get() != selected) {
      return false;
    }
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    return !closed_ && plan_identity_current(*selected) && callback_activation_current(*selected) &&
           selected->loan_live && selected->phase == SlotPhase::kReady && identity_valid(*selected);
  }

  [[nodiscard]] expected<PreparedBundleField, Error> prepare_bundle_field(
      const TensorOutputLoan& loan) noexcept {
    auto* const selected =
        static_cast<OutputSlot*>(TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps));
    if (selected == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto context = prepare_emit(*selected, false);
    if (!context) {
      return make_unexpected(std::move(context.error()));
    }
    return PreparedBundleField{
        .slot = selected,
        .tensor = TensorAccess::from_context(std::move(*context), TensorAuthority::kLocalWritable),
    };
  }

  [[nodiscard]] expected<void, Error> finish_bundle_field(
      OutputSlot* slot, const expected<void, Error>& publication) noexcept {
    if (slot == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (publication) {
      return finish_publication(*slot, {});
    }
    return finish_publication(*slot,
                              make_unexpected(Error{.code = publication.error().code,
                                                    .message = publication.error().message}));
  }

 private:
  [[nodiscard]] static bool retained_geometry_compatible(
      const TensorAllocationPlanRecord& retained,
      const TensorAllocationPlanRecord& candidate) noexcept {
    return retained.output == candidate.output &&
           retained.plan_generation == candidate.plan_generation &&
           retained.run_generation == candidate.run_generation &&
           retained.pool_id == candidate.pool_id &&
           retained.allocator_root_id == candidate.allocator_root_id &&
           retained.allocator_process_group == candidate.allocator_process_group &&
           retained.bounds.max_rank == candidate.bounds.max_rank &&
           retained.bounds.max_byte_span == candidate.bounds.max_byte_span &&
           retained.axes == candidate.axes &&
           tensor_port_layout_exactly_equal(retained.layout, candidate.layout) &&
           retained.memory_kind == candidate.memory_kind && retained.device == candidate.device &&
           retained.max_outstanding_loans == candidate.max_outstanding_loans &&
           retained.max_emits_per_compute == candidate.max_emits_per_compute &&
           retained.materialize_before == candidate.materialize_before &&
           retained.effective_profile == candidate.effective_profile &&
           retained.disposition == candidate.disposition &&
           retained.fallback_rejected == candidate.fallback_rejected;
  }

  [[nodiscard]] bool quiescent_free_inventory_locked() const noexcept {
    return live_loans_ == 0U && retirement_records_in_flight_ == 0U && retiring_slots_.empty() &&
           deferred_head_.load(std::memory_order_acquire) == nullptr &&
           free_slots_.size() == active_slot_count_ &&
           std::ranges::all_of(slots_, [this](const auto& slot) {
             return slot->phase == SlotPhase::kFree &&
                    slot->on_free_list == (slot->index < active_slot_count_) && !slot->loan_live &&
                    !slot->on_retirement_queue &&
                    !slot->deferred_queued.load(std::memory_order_acquire) &&
                    slot->tensor_handles.load(std::memory_order_acquire) == 0U &&
                    !slot->claims.any();
           });
  }

  void rebuild_active_free_list_locked() noexcept {
    free_slots_.clear();
    for (std::uint32_t index = 0U; index < slots_.size(); ++index) {
      slots_[index]->on_free_list = index < active_slot_count_;
      if (index < active_slot_count_) {
        free_slots_.push_back(active_slot_count_ - index - 1U);
      }
    }
  }

  OutputPool(TensorAllocationPlanRecord plan, OutputPlanContract contract, VertexId vertex,
             OperatorInstanceId owner, void* declaration, const RuntimePortTypeOps* type_ops,
             cudaStream_t stream, bool elide_same_stream_readiness,
             PlanGenerationId plan_generation, RunGenerationId run_generation,
             std::uint64_t pool_generation, std::uint64_t allocation_generation,
             std::uint32_t active_slot_count, RuntimeTensorPoolAllocationAction allocation_action,
             std::shared_ptr<holomq::BufferPool> export_pool,
             std::shared_ptr<RuntimeTensorOutputTestControl> test_control,
             std::size_t test_pool_index,
             std::optional<std::uint32_t> descriptor_tensor_index) noexcept
      : plan_(std::move(plan)),
        contract_(std::move(contract)),
        vertex_(vertex),
        owner_(owner),
        declaration_(declaration),
        type_ops_(type_ops),
        stream_(stream),
        elide_same_stream_readiness_(elide_same_stream_readiness),
        plan_generation_(plan_generation),
        run_generation_(run_generation),
        pool_generation_(pool_generation),
        allocation_generation_(allocation_generation),
        active_slot_count_(active_slot_count),
        allocation_action_(allocation_action),
        export_pool_(std::move(export_pool)),
        test_control_(std::move(test_control)),
        test_pool_index_(test_pool_index),
        descriptor_tensor_index_(descriptor_tensor_index) {}

  [[nodiscard]] bool pool_plan_identity_current() const noexcept {
    return vertex_ && plan_.plan_generation == plan_generation_ && plan_.run_generation == 0U &&
           run_generation_ != 0U;
  }

  [[nodiscard]] bool plan_identity_current(const OutputSlot& slot) const noexcept {
    return pool_plan_identity_current() && slot.loan_plan_generation == plan_generation_ &&
           slot.loan_run_generation == run_generation_;
  }

  [[nodiscard]] bool callback_activation_current(const OutputSlot& slot) const noexcept {
    const RuntimeCallbackIdentity identity = runtime_callback_identity();
    return identity.context == RuntimeCallbackContext::kCompute && identity.operator_id == owner_ &&
           identity.tag.valid() && identity.tag == slot.activation;
  }

  [[nodiscard]] bool current_activation(const OutputSlot& slot) const noexcept {
    return plan_identity_current(slot) && callback_activation_current(slot);
  }

  [[nodiscard]] bool callback_can_allocate(Tag& activation) const noexcept {
    const RuntimeCallbackIdentity identity = runtime_callback_identity();
    if (!vertex_ || plan_.plan_generation != plan_generation_ || plan_.run_generation != 0U ||
        run_generation_ == 0U || identity.context != RuntimeCallbackContext::kCompute ||
        identity.operator_id != owner_ || !identity.tag.valid()) {
      return false;
    }
    activation = identity.tag;
    return true;
  }

  [[nodiscard]] bool claim_admitted(TensorSlotClaimReason reason) const noexcept {
    return contract_.admitted_claims[tensor_slot_claim_index(reason)];
  }

  [[nodiscard]] std::uint32_t admitted_claim_mask() const noexcept {
    std::uint32_t result = 0U;
    for (std::size_t index = 0U; index < contract_.admitted_claims.size(); ++index) {
      result |= static_cast<std::uint32_t>(contract_.admitted_claims[index]) << index;
    }
    return result;
  }

  [[nodiscard]] std::uint64_t metric_time_ns() const noexcept {
    if (test_control_ != nullptr) {
      const auto injected = test_control_->metric_time_ns();
      if (injected != std::numeric_limits<std::uint64_t>::max()) {
        return injected;
      }
    }
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
  }

  [[nodiscard]] static std::uint64_t elapsed_ns(std::uint64_t start, std::uint64_t end) noexcept {
    return end >= start ? end - start : 0U;
  }

  [[nodiscard]] static std::uint64_t saturating_add(std::uint64_t left,
                                                    std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
  }

  void update_primitive_metrics() noexcept {
    const auto active_free_depth = static_cast<std::uint32_t>(free_slots_.size());
    const auto inactive_free_depth =
        static_cast<std::uint32_t>(slots_.size()) -
        std::min(active_slot_count_, static_cast<std::uint32_t>(slots_.size()));
    const std::uint32_t physical_free_depth = inactive_free_depth + active_free_depth;
    const std::uint32_t publisher_loan_credit =
        live_loans_ < plan_.max_outstanding_loans ? plan_.max_outstanding_loans - live_loans_ : 0U;
    const std::uint32_t effective_free_depth = std::min(active_free_depth, publisher_loan_credit);
    metrics_.physical_free_slot_depth = physical_free_depth;
    metrics_.publisher_loan_credit = publisher_loan_credit;
    metrics_.effective_free_slot_depth = closed_ ? 0U : effective_free_depth;
    if (!metrics_initialized_) {
      metrics_.minimum_physical_free_slot_depth = physical_free_depth;
      metrics_.minimum_publisher_loan_credit = publisher_loan_credit;
      metrics_.minimum_effective_free_slot_depth = metrics_.effective_free_slot_depth;
      metrics_initialized_ = true;
    } else {
      metrics_.minimum_physical_free_slot_depth =
          std::min(metrics_.minimum_physical_free_slot_depth, physical_free_depth);
      metrics_.minimum_publisher_loan_credit =
          std::min(metrics_.minimum_publisher_loan_credit, publisher_loan_credit);
      metrics_.minimum_effective_free_slot_depth =
          std::min(metrics_.minimum_effective_free_slot_depth, metrics_.effective_free_slot_depth);
    }

    const std::uint64_t now = metric_time_ns();
    if (metrics_.effective_free_slot_depth == 0U) {
      if (!zero_depth_active_) {
        zero_depth_active_ = true;
        zero_depth_started_ = now;
        metrics_.zero_depth_since_ns = now;
        ++metrics_.zero_depth_interval_count;
      }
      const std::uint64_t active = elapsed_ns(zero_depth_started_, now);
      metrics_.zero_depth_total_ns = saturating_add(zero_depth_accumulated_ns_, active);
      metrics_.zero_depth_longest_ns = std::max(metrics_.zero_depth_longest_ns, active);
    } else if (zero_depth_active_) {
      const std::uint64_t completed = elapsed_ns(zero_depth_started_, now);
      zero_depth_accumulated_ns_ = saturating_add(zero_depth_accumulated_ns_, completed);
      metrics_.zero_depth_total_ns = zero_depth_accumulated_ns_;
      metrics_.zero_depth_longest_ns = std::max(metrics_.zero_depth_longest_ns, completed);
      metrics_.zero_depth_since_ns = std::numeric_limits<std::uint64_t>::max();
      zero_depth_active_ = false;
    }

    auto oldest_grant = std::numeric_limits<std::uint64_t>::max();
    metrics_.oldest_retained_slot = std::numeric_limits<std::uint32_t>::max();
    metrics_.oldest_retained_phase = 0U;
    metrics_.oldest_retained_claim_mask = 0U;
    metrics_.oldest_retained_age_ns = 0U;
    for (const auto& slot : slots_) {
      if (slot->phase != SlotPhase::kFree &&
          (slot->loan_granted_ns < oldest_grant ||
           (slot->loan_granted_ns == oldest_grant &&
            static_cast<std::uint32_t>(slot->index) < metrics_.oldest_retained_slot))) {
        oldest_grant = slot->loan_granted_ns;
        metrics_.oldest_retained_slot = static_cast<std::uint32_t>(slot->index);
        metrics_.oldest_retained_phase = static_cast<std::uint8_t>(slot->phase);
        metrics_.oldest_retained_claim_mask = slot->claims.mask();
        metrics_.oldest_retained_age_ns = elapsed_ns(slot->loan_granted_ns, now);
      }
    }
  }

  void record_test_snapshot(bool terminal = false, bool invariant_failure = false) noexcept {
    if (test_control_ == nullptr) {
      return;
    }
    update_primitive_metrics();
    auto active_claims = std::uint32_t{};
    auto active_slots = std::size_t{};
    for (const auto& slot : slots_) {
      const std::uint32_t mask = slot->claims.mask();
      active_claims |= mask;
      active_slots += static_cast<std::size_t>(mask != 0U);
    }
    test_control_->record(test_pool_index_,
                          admitted_claim_mask(),
                          active_claims,
                          active_slots,
                          pool_generation_,
                          allocation_generation_,
                          static_cast<std::uint32_t>(slots_.size()),
                          active_slot_count_,
                          allocation_action_,
                          ever_bound_,
                          metrics_,
                          invariant_failure,
                          terminal);
  }

  void record_failed_loan(bool active_exhausted, bool credit_exhausted) noexcept {
    update_primitive_metrics();
    ++metrics_.failed_loan_count;
    metrics_.failed_loan_physical_free_slot_depth = metrics_.physical_free_slot_depth;
    metrics_.failed_loan_publisher_credit =
        live_loans_ < plan_.max_outstanding_loans ? plan_.max_outstanding_loans - live_loans_ : 0U;
    metrics_.failed_loan_effective_free_slot_depth = metrics_.effective_free_slot_depth;
    metrics_.failed_loan_physical_limited =
        active_exhausted && metrics_.failed_loan_physical_free_slot_depth == 0U;
    metrics_.failed_loan_active_limited =
        active_exhausted && metrics_.failed_loan_physical_free_slot_depth != 0U;
    metrics_.failed_loan_credit_limited = credit_exhausted;
    metrics_.failed_loan_blocking_claim_mask = 0U;
    for (const auto& slot : slots_) {
      metrics_.failed_loan_blocking_claim_mask |= slot->claims.mask();
    }
    metrics_.failed_loan_oldest_retained_slot = metrics_.oldest_retained_slot;
    metrics_.failed_loan_oldest_retained_phase = metrics_.oldest_retained_phase;
    metrics_.failed_loan_oldest_retained_claim_mask = metrics_.oldest_retained_claim_mask;
    metrics_.failed_loan_oldest_retained_age_ns = metrics_.oldest_retained_age_ns;
    record_test_snapshot();
  }

  void record_retry_cancellation(OutputSlot& slot) noexcept {
    if (slot.retry_cancellation_demand) {
      return;
    }
    slot.retry_cancellation_demand = true;
    ++metrics_.retry_cancellation_total_started;
    ++metrics_.current_retry_cancellation_demand;
    metrics_.peak_retry_cancellation_demand = std::max(metrics_.peak_retry_cancellation_demand,
                                                       metrics_.current_retry_cancellation_demand);
    record_test_snapshot();
  }

  [[nodiscard]] bool consume_test_fault(RuntimeTensorOutputTestFault fault) noexcept {
    return test_control_ != nullptr && test_control_->consume_fault(fault, test_pool_index_);
  }

  [[nodiscard]] RuntimeTensorOutputTestCorruption consume_test_corruption() noexcept {
    return test_control_ != nullptr ? test_control_->consume_corruption()
                                    : RuntimeTensorOutputTestCorruption::kNone;
  }

  void inject_test_corruption(OutputSlot& slot) noexcept {
    switch (consume_test_corruption()) {
      case RuntimeTensorOutputTestCorruption::kPlanGeneration:
        slot.loan_plan_generation = plan_generation_ + 1U;
        break;
      case RuntimeTensorOutputTestCorruption::kPoolGeneration:
        slot.loan_pool_generation = pool_generation_ + 1U;
        break;
      case RuntimeTensorOutputTestCorruption::kSlotGeneration:
        slot.generation = 0U;
        break;
      case RuntimeTensorOutputTestCorruption::kNone:
      default:
        break;
    }
  }

  [[nodiscard]] TensorOutputDiagnosticReason identity_failure_reason(
      const OutputSlot& slot) const noexcept {
    return slot.loan_pool_generation != pool_generation_
               ? TensorOutputDiagnosticReason::kLoanPoolGenerationMismatch
               : TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch;
  }

  [[nodiscard]] bool identity_valid(const OutputSlot& slot) const noexcept {
    return slot.pool == this && slot.index < slots_.size() && slots_[slot.index].get() == &slot &&
           pool_generation_ != 0U && slot.loan_pool_generation == pool_generation_ &&
           slot.generation != 0U && slot.generation != std::numeric_limits<std::uint64_t>::max();
  }

  void fail_identity(OutputSlot& slot, TensorOutputOperation operation) noexcept {
    record_diagnostic(&slot, operation, ErrorCode::kFailure, identity_failure_reason(slot));
    if (slot.phase == SlotPhase::kWriterActive) {
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
    }
    consume_loan(slot);
    quarantine(slot);
    record_test_snapshot(false, true);
  }

  void record_diagnostic(const OutputSlot* slot, TensorOutputOperation operation, ErrorCode error,
                         TensorOutputDiagnosticReason reason) noexcept {
    const TensorOutputDiagnosticRecord record{
        .output = plan_.output,
        .plan_generation = plan_generation_,
        .run_generation = run_generation_,
        .activation = slot != nullptr ? slot->activation : Tag{},
        .pool_generation = pool_generation_,
        .slot_generation = slot != nullptr ? slot->generation : 0U,
        .slot = slot != nullptr ? static_cast<std::uint32_t>(slot->index) : 0U,
        .from_state = slot != nullptr ? static_cast<std::uint8_t>(slot->phase) : std::uint8_t{0U},
        .attempted_operation = static_cast<std::uint8_t>(operation),
        .writer_count = slot != nullptr && slot->phase == SlotPhase::kWriterActive
                            ? std::uint8_t{1U}
                            : std::uint8_t{0U},
        .error = error,
        .reason = reason,
    };
    diagnostics_.append(record);
    if (test_control_ != nullptr) {
      test_control_->record_diagnostic(test_pool_index_, record);
    }
  }

  void add_claim(OutputSlot& slot, TensorSlotClaimReason reason) noexcept {
    if (!claim_admitted(reason)) {
      slot.phase = SlotPhase::kQuarantined;
      if (claim_admitted(TensorSlotClaimReason::kQuarantine)) {
        static_cast<void>(
            slot.claims.add(TensorSlotClaimReason::kQuarantine, contract_.admitted_claims));
      }
      record_test_snapshot();
      return;
    }
    if (!slot.claims.add(reason, contract_.admitted_claims)) {
      slot.phase = SlotPhase::kQuarantined;
    }
    record_test_snapshot();
  }

  void remove_claim(OutputSlot& slot, TensorSlotClaimReason reason) noexcept {
    slot.claims.remove(reason);
    record_test_snapshot();
  }

  void quarantine(OutputSlot& slot) noexcept {
    slot.phase = SlotPhase::kQuarantined;
    add_claim(slot, TensorSlotClaimReason::kQuarantine);
  }

  void enqueue_deferred(OutputSlot& slot) noexcept {
    if (slot.deferred_queued.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    auto* head = deferred_head_.load(std::memory_order_acquire);
    do {
      slot.deferred_next.store(head, std::memory_order_relaxed);
    } while (!deferred_head_.compare_exchange_weak(
        head, &slot, std::memory_order_release, std::memory_order_acquire));
  }

  void reconcile_deferred_releases(std::unique_lock<std::mutex>& lock) noexcept {
    auto* slot = deferred_head_.exchange(nullptr, std::memory_order_acquire);
    while (slot != nullptr) {
      OutputSlot* const next = slot->deferred_next.exchange(nullptr, std::memory_order_relaxed);
      slot->deferred_queued.store(false, std::memory_order_release);
      if (slot->quarantine_requested.exchange(false, std::memory_order_acq_rel)) {
        quarantine(*slot);
      }
      if (slot->retirement_requested.load(std::memory_order_acquire) &&
          slot->tensor_handles.load(std::memory_order_acquire) == 0U) {
        if (slot->phase == SlotPhase::kRetained) {
          slot->retirement_requested.store(false, std::memory_order_release);
          begin_retirement(*slot, lock);
        } else if (slot->phase == SlotPhase::kPublicationTerminal) {
          slot->retirement_requested.store(false, std::memory_order_release);
          quarantine(*slot);
        }
      }
      slot = next;
    }
  }

  void erase_retiring_slot(std::size_t position) noexcept {
    retiring_slots_[position] = retiring_slots_.back();
    retiring_slots_.pop_back();
  }

  void quarantine_retiring_slot(OutputSlot& slot, std::size_t position) noexcept {
    slot.on_retirement_queue = false;
    quarantine(slot);
    erase_retiring_slot(position);
  }

  void recycle_retired_slot(OutputSlot& slot, std::size_t position) noexcept {
    slot.phase = SlotPhase::kFree;
    slot.claims.clear();
    slot.loan_plan_generation = 0U;
    slot.loan_run_generation = 0U;
    slot.loan_pool_generation = 0U;
    slot.loan_granted_ns = 0U;
    if (slot.retry_cancellation_demand) {
      slot.retry_cancellation_demand = false;
      if (metrics_.current_retry_cancellation_demand != 0U) {
        --metrics_.current_retry_cancellation_demand;
      }
    }
    slot.cancel_pending = false;
    slot.writer_cuda = false;
    slot.writer_stream = nullptr;
    slot.byte_size = 0U;
    slot.activation = {};
    slot.retirement_requested.store(false, std::memory_order_release);
    slot.quarantine_requested.store(false, std::memory_order_release);
    slot.on_free_list = slot.index < active_slot_count_;
    if (slot.on_free_list) {
      free_slots_.push_back(static_cast<std::uint32_t>(slot.index));
    }
    if (test_control_ != nullptr) {
      test_control_->record_retired_slot(test_pool_index_, static_cast<std::uint32_t>(slot.index));
    }
    record_test_snapshot();
    erase_retiring_slot(position);
  }

  void reclaim_completed_retirements(std::unique_lock<std::mutex>& lock) noexcept {
    if (retirement_query_active_) {
      // Allocation remains nonblocking while another controller queries the active inventory.
      return;
    }
    retirement_query_active_ = true;
    std::size_t position = 0U;
    while (position < retiring_slots_.size()) {
      const auto index = retiring_slots_[position];
      if (index >= slots_.size()) {
        erase_retiring_slot(position);
        continue;
      }
      auto& slot = *slots_[index];
      if (!slot.on_retirement_queue || slot.phase != SlotPhase::kRetiring) {
        quarantine_retiring_slot(slot, position);
        continue;
      }
      const std::uint64_t generation = slot.generation;
      bool complete = true;
      if (slot.lifetime->retirement_requires_query()) {
        lock.unlock();
        complete = slot.lifetime->retirement_complete();
        lock.lock();
      }
      if (slot.generation != generation || !slot.on_retirement_queue ||
          slot.phase != SlotPhase::kRetiring) {
        quarantine_retiring_slot(slot, position);
        continue;
      }
      if (slot.tensor_handles.load(std::memory_order_acquire) != 0U || !complete) {
        ++position;
        continue;
      }
      if (export_pool_ && !slot.lifetime->prepare_exportable_buffer_for_reuse()) {
        ++position;
        continue;
      }
      slot.on_retirement_queue = false;
      if (!advance_tensor_slot_generation(slot.generation)) {
        record_diagnostic(&slot,
                          TensorOutputOperation::kRetire,
                          ErrorCode::kFailure,
                          TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch);
        quarantine(slot);
        continue;
      }
      recycle_retired_slot(slot, position);
    }
    retirement_query_active_ = false;
  }

  void consume_loan(OutputSlot& slot) noexcept {
    if (slot.loan_live) {
      slot.loan_live = false;
      if (live_loans_ != 0U) {
        --live_loans_;
      } else {
        quarantine(slot);
      }
      remove_claim(slot, TensorSlotClaimReason::kProducerLoan);
    }
  }

  void begin_retirement(OutputSlot& slot, std::unique_lock<std::mutex>& lock) noexcept {
    if (slot.tensor_handles.load(std::memory_order_acquire) != 0U ||
        slot.phase == SlotPhase::kWriterActive || slot.phase == SlotPhase::kSubmitPending ||
        slot.phase == SlotPhase::kPublicationTerminal ||
        slot.phase == SlotPhase::kReleaseRecording || slot.phase == SlotPhase::kQuarantined ||
        slot.phase == SlotPhase::kRetiring || slot.on_retirement_queue) {
      return;
    }
    add_claim(slot, TensorSlotClaimReason::kRetirementService);
    if (slot.phase == SlotPhase::kQuarantined) {
      return;
    }
    if (retiring_slots_.size() + retirement_records_in_flight_ >= plan_.retirement_capacity) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kRetire,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanRetirementStalled);
      quarantine(slot);
      return;
    }
    if (!slot.lifetime->has_release_streams()) {
      slot.phase = SlotPhase::kRetiring;
      slot.on_retirement_queue = true;
      retiring_slots_.push_back(static_cast<std::uint32_t>(slot.index));
      reclaim_completed_retirements(lock);
      return;
    }
    const std::uint64_t generation = slot.generation;
    // Publish an in-progress phase before dropping the pool lock. Other controllers may continue,
    // but this slot cannot enter the retirement inventory or free list until recording completes.
    slot.phase = SlotPhase::kReleaseRecording;
    ++retirement_records_in_flight_;
    lock.unlock();
    const bool release_recorded = slot.lifetime->record_release();
    lock.lock();
    --retirement_records_in_flight_;
    if (slot.generation != generation || slot.phase != SlotPhase::kReleaseRecording) {
      quarantine(slot);
      return;
    }
    if (!release_recorded) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kRetire,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanReleaseEvidenceMissing);
      quarantine(slot);
      return;
    }
    slot.phase = SlotPhase::kRetiring;
    slot.on_retirement_queue = true;
    retiring_slots_.push_back(static_cast<std::uint32_t>(slot.index));
    reclaim_completed_retirements(lock);
  }

  // Keep allocation admission, diagnostics, and the bounded slot transition in one transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate(TensorLoanRequest request) noexcept {
    Tag activation{};
    if (!callback_can_allocate(activation)) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kAllocationCapabilityMissing);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto validated = validate_request_against_allocation(plan_, request);
    if (!validated) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanRequestLayoutMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_) {
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kClosed,
                        TensorOutputDiagnosticReason::kAllocationCapabilityMissing);
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    reclaim_completed_retirements(lock);
    const bool credit_exhausted = live_loans_ >= plan_.max_outstanding_loans;
    const auto active_exhausted = free_slots_.empty();
    if (credit_exhausted || active_exhausted) {
      record_diagnostic(nullptr,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kResourceExhausted,
                        TensorOutputDiagnosticReason::kAllocationReadinessRecordExhausted);
      if (test_control_ != nullptr) {
        record_failed_loan(active_exhausted, credit_exhausted);
      }
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const auto index = free_slots_.back();
    free_slots_.pop_back();
    auto& slot = *slots_[index];
    if (slot.pool != this || slot.index != index || slot.generation == 0U ||
        slot.generation == std::numeric_limits<std::uint64_t>::max() ||
        slot.phase != SlotPhase::kFree || !slot.on_free_list || slot.loan_live ||
        slot.on_retirement_queue || slot.deferred_queued.load(std::memory_order_acquire) ||
        slot.tensor_handles.load(std::memory_order_acquire) != 0U || slot.claims.any()) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kAllocate,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanSlotGenerationMismatch);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (export_pool_ &&
        (!slot.lifetime->has_exportable_buffer() || slot.lifetime->data() == nullptr)) {
      slot.on_free_list = false;
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    slot.on_free_list = false;
    slot.activation = activation;
    slot.loan_plan_generation = plan_generation_;
    slot.loan_run_generation = run_generation_;
    slot.loan_pool_generation = pool_generation_;
    slot.loan_granted_ns = test_control_ != nullptr ? metric_time_ns() : 0U;
    slot.phase = SlotPhase::kLoaned;
    slot.loan_live = true;
    slot.cancel_pending = false;
    slot.writer_cuda = false;
    slot.writer_stream = nullptr;
    slot.byte_size = static_cast<std::size_t>(validated->byte_span);
    slot.dtype = request.dtype;
    slot.shape.assign(request.shape.begin(), request.shape.end());
    slot.byte_strides.clear();
    slot.explicit_strides = request.byte_strides.has_value();
    slot.retirement_requested.store(false, std::memory_order_release);
    slot.quarantine_requested.store(false, std::memory_order_release);
    ++live_loans_;
    add_claim(slot, TensorSlotClaimReason::kProducerLoan);
    add_claim(slot, TensorSlotClaimReason::kProducerReadiness);
    if (slot.phase == SlotPhase::kQuarantined) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (request.byte_strides) {
      slot.byte_strides.assign(request.byte_strides.begin(), request.byte_strides.end());
    }
    slot.lifetime->reset_generation();

    auto& context = *slot.context;
    context.dl_shape.assign(slot.shape.begin(), slot.shape.end());
    context.dl_strides.clear();
    const std::uint64_t item_bytes = dlpack_dtype_item_size_bytes(request.dtype);
    if (slot.explicit_strides) {
      for (const std::int64_t stride : slot.byte_strides) {
        context.dl_strides.push_back(stride / static_cast<std::int64_t>(item_bytes));
      }
    }
    context.tensor.flags = 0U;
    context.tensor.dl_tensor.ndim = static_cast<std::int32_t>(context.dl_shape.size());
    context.tensor.dl_tensor.dtype = request.dtype;
    context.tensor.dl_tensor.shape = context.dl_shape.empty() ? nullptr : context.dl_shape.data();
    context.tensor.dl_tensor.strides =
        context.dl_strides.empty() ? nullptr : context.dl_strides.data();
    context.tensor.dl_tensor.byte_offset = 0U;
    record_test_snapshot();
    return TensorOutputLoanAccess::loan(&slot, &kLoanOps, slots_[index]);
  }

  [[nodiscard]] expected<TensorOutputWriteGuard, Error> begin_write(OutputSlot& slot,
                                                                    cudaStream_t stream,
                                                                    bool cuda_write) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kClosed,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (!current_activation(slot)) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    inject_test_corruption(slot);
    if (!identity_valid(slot)) {
      fail_identity(slot, TensorOutputOperation::kBeginWrite);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (slot.phase != SlotPhase::kLoaned || !slot.loan_live) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kInvalidArgument,
                        slot.phase == SlotPhase::kWriterActive
                            ? TensorOutputDiagnosticReason::kLoanWriterActive
                            : TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (cuda_write) {
      if (stream_ == nullptr || stream == nullptr || stream != stream_) {
        record_diagnostic(&slot,
                          TensorOutputOperation::kBeginWrite,
                          ErrorCode::kNotSupported,
                          TensorOutputDiagnosticReason::kLoanStreamDeviceMismatch);
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
    } else if (plan_.memory_kind == MemoryKind::kCudaDevice) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kBeginWrite,
                        ErrorCode::kNotSupported,
                        TensorOutputDiagnosticReason::kLoanStreamDeviceMismatch);
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    slot.writer_cuda = cuda_write;
    slot.writer_stream = stream;
    slot.phase = SlotPhase::kWriterActive;
    add_claim(slot, TensorSlotClaimReason::kProducerWriter);
    if (slot.phase == SlotPhase::kQuarantined) {
      consume_loan(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return TensorOutputLoanAccess::write_guard(&slot, &kWriteOps, slot.self.lock());
  }

  [[nodiscard]] expected<void, Error> commit_write(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (slot.phase != SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kGuardDoubleCommit);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool initially_invalid =
        closed_ || !current_activation(slot) || slot.cancel_pending || !slot.loan_live;
    if (!initially_invalid) {
      inject_test_corruption(slot);
      if (!identity_valid(slot)) {
        fail_identity(slot, TensorOutputOperation::kCommitWrite);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    // write_host() completes before commit and exposes no asynchronous producer work. Recording
    // the pool stream in that case would manufacture a remote CUDA wait and defer transport
    // reference retirement for an otherwise ready host buffer.
    const cudaStream_t readiness_stream = slot.writer_stream;
    const bool readiness_required = slot.writer_cuda;
    const bool force_readiness_failure =
        !initially_invalid &&
        consume_test_fault(RuntimeTensorOutputTestFault::kCommitReadinessFailure);
    bool readiness_ok = !force_readiness_failure;
    if (readiness_ok && readiness_required) {
      lock.unlock();
      readiness_ok = slot.lifetime->record_readiness(readiness_stream);
      lock.lock();
    }
    if (slot.phase != SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (closed_ || !current_activation(slot) || slot.cancel_pending || !slot.loan_live) {
      const ErrorCode invalid_code = closed_ ? ErrorCode::kClosed : ErrorCode::kInvalidArgument;
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        invalid_code,
                        !slot.loan_live
                            ? TensorOutputDiagnosticReason::kGuardOutlivedLoan
                            : TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch);
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
      consume_loan(slot);
      add_claim(slot, TensorSlotClaimReason::kCancellation);
      if (!readiness_ok) {
        record_diagnostic(&slot,
                          TensorOutputOperation::kCommitWrite,
                          ErrorCode::kFailure,
                          TensorOutputDiagnosticReason::kLoanReadinessRecordFailed);
        quarantine(slot);
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      slot.phase = SlotPhase::kReady;
      begin_retirement(slot, lock);
      return make_unexpected(Error{invalid_code});
    }
    if (!readiness_ok) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCommitWrite,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanReadinessRecordFailed);
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
      consume_loan(slot);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
    slot.phase = SlotPhase::kReady;
    slot.writer_stream = nullptr;
    return {};
  }

  void abandon_write(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (slot.phase != SlotPhase::kWriterActive) {
      return;
    }
    record_diagnostic(&slot,
                      TensorOutputOperation::kDropWrite,
                      ErrorCode::kInvalidArgument,
                      TensorOutputDiagnosticReason::kGuardDroppedWithoutCommit);
    const bool writer_cuda = slot.writer_cuda;
    cudaStream_t writer_stream = slot.writer_stream;
    bool readiness_ok = true;
    if (writer_cuda) {
      lock.unlock();
      readiness_ok = slot.lifetime->record_readiness(writer_stream);
      lock.lock();
    }
    if (!readiness_ok) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kDropWrite,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanCancelSafetyRecordFailed);
      remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
      consume_loan(slot);
      quarantine(slot);
      return;
    }
    remove_claim(slot, TensorSlotClaimReason::kProducerWriter);
    consume_loan(slot);
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    slot.phase = SlotPhase::kReady;
    begin_retirement(slot, lock);
  }

  void abandon_loan(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (!slot.loan_live) {
      return;
    }
    consume_loan(slot);
    if (slot.phase == SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCancel,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanDestroyedWithActiveWriter);
      slot.cancel_pending = true;
      add_claim(slot, TensorSlotClaimReason::kCancellation);
      return;
    }
    if (slot.phase == SlotPhase::kLoaned || slot.phase == SlotPhase::kReady) {
      add_claim(slot, TensorSlotClaimReason::kCancellation);
      begin_retirement(slot, lock);
    }
  }

  [[nodiscard]] expected<void, Error> cancel(OutputSlot& slot) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (!slot.loan_live || (slot.phase != SlotPhase::kLoaned && slot.phase != SlotPhase::kReady &&
                            slot.phase != SlotPhase::kWriterActive)) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCancel,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanDoubleCancel);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    consume_loan(slot);
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    if (slot.phase == SlotPhase::kWriterActive) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kCancel,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanWriterActive);
      slot.cancel_pending = true;
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    begin_retirement(slot, lock);
    return {};
  }

  void add_publication_claims(OutputSlot& slot) noexcept {
    constexpr std::array publication_reasons{
        TensorSlotClaimReason::kAcceptedPublication,
        TensorSlotClaimReason::kHolomqTransportOrFanout,
        TensorSlotClaimReason::kSubscriberBorrowOrImport,
        TensorSlotClaimReason::kConsumerCudaRelease,
        TensorSlotClaimReason::kRelayOrReexport,
        TensorSlotClaimReason::kTemporalCache,
        TensorSlotClaimReason::kPendingOrSelectedWindow,
        TensorSlotClaimReason::kReuseLast,
        TensorSlotClaimReason::kDrainAllActivation,
        TensorSlotClaimReason::kDlpackOrFrameworkBorrow,
        TensorSlotClaimReason::kCopyOrRecorderMaterialization,
    };
    for (const auto reason : publication_reasons) {
      if (claim_admitted(reason)) {
        add_claim(slot, reason);
      }
    }
  }

  [[nodiscard]] expected<void, Error> finish_publication(OutputSlot& slot,
                                                         expected<void, Error> result) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (slot.phase != SlotPhase::kSubmitPending) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }

    const std::uint32_t handles = slot.tensor_handles.load(std::memory_order_acquire);
    if (result) {
      add_publication_claims(slot);
      if (slot.phase == SlotPhase::kQuarantined) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      slot.phase = SlotPhase::kRetained;
      if (handles == 0U) {
        slot.retirement_requested.store(false, std::memory_order_release);
        begin_retirement(slot, lock);
      }
      return {};
    }

    const ErrorCode code = result.error().code;
    if (publication_error_consumes_authority(code)) {
      const TensorOutputDiagnosticReason reason =
          code == ErrorCode::kPublicationPartial
              ? TensorOutputDiagnosticReason::kPublicationPartial
              : TensorOutputDiagnosticReason::kPublicationIndeterminate;
      record_diagnostic(&slot, TensorOutputOperation::kEmit, code, reason);
      add_publication_claims(slot);
      slot.phase = SlotPhase::kPublicationTerminal;
      if (handles == 0U) {
        slot.retirement_requested.store(false, std::memory_order_release);
        quarantine(slot);
      }
      return make_unexpected(std::move(result).error());
    }

    // A proven pre-submit result cannot retain a recipient-owned Tensor handle. A live handle at
    // this boundary contradicts the frozen publication phase, so fail closed without retry/reuse.
    if (handles != 0U) {
      record_diagnostic(&slot,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kFailure,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      quarantine(slot);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    slot.retirement_requested.store(false, std::memory_order_release);
    slot.phase = SlotPhase::kReady;
    if (test_control_ != nullptr) {
      record_retry_cancellation(slot);
    }
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    begin_retirement(slot, lock);
    if (code == contract_.publication.pre_submit_error || code == ErrorCode::kClosed) {
      return make_unexpected(Error{code});
    }
    if (code == ErrorCode::kNotReady || code == ErrorCode::kStaleEndpoint) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return make_unexpected(Error{code});
  }

  void cancel_invalid_emit(OutputSlot& slot, std::unique_lock<std::mutex>& lock) noexcept {
    consume_loan(slot);
    add_claim(slot, TensorSlotClaimReason::kCancellation);
    if (slot.phase == SlotPhase::kWriterActive) {
      slot.cancel_pending = true;
    } else {
      begin_retirement(slot, lock);
    }
  }

  [[nodiscard]] TensorOutputDiagnosticReason emit_readiness_reason(
      const OutputSlot& slot) const noexcept {
    if (!callback_activation_current(slot)) {
      return TensorOutputDiagnosticReason::kLoanActivationGenerationMismatch;
    }
    if (slot.phase == SlotPhase::kWriterActive) {
      return TensorOutputDiagnosticReason::kLoanWriterActive;
    }
    return !slot.loan_live ? TensorOutputDiagnosticReason::kLoanDoubleEmit
                           : TensorOutputDiagnosticReason::kLoanReadinessNotCommitted;
  }

  [[nodiscard]] expected<std::shared_ptr<DLManagedTensorContext>, Error> prepare_emit(
      OutputSlot& selected, bool force_closed) noexcept {
    std::unique_lock lock(mutex_);
    reconcile_deferred_releases(lock);
    if (closed_ || force_closed) {
      record_diagnostic(&selected,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kClosed,
                        TensorOutputDiagnosticReason::kLoanInvalidStateTransition);
      cancel_invalid_emit(selected, lock);
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    inject_test_corruption(selected);
    if (!plan_identity_current(selected)) {
      record_diagnostic(&selected,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanPlanGenerationMismatch);
      cancel_invalid_emit(selected, lock);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!callback_activation_current(selected) || !selected.loan_live ||
        selected.phase != SlotPhase::kReady) {
      record_diagnostic(&selected,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kInvalidArgument,
                        emit_readiness_reason(selected));
      cancel_invalid_emit(selected, lock);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!identity_valid(selected)) {
      fail_identity(selected, TensorOutputOperation::kEmit);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    consume_loan(selected);
    selected.phase = SlotPhase::kSubmitPending;
    return selected.context;
  }

  [[nodiscard]] expected<void, Error> emit(TensorOutputLoan loan, const void* descriptor,
                                           EmitOptions options) noexcept {
    const bool force_closed = consume_test_fault(RuntimeTensorOutputTestFault::kEmitClosedBinding);
    auto* const selected =
        static_cast<OutputSlot*>(TensorOutputLoanAccess::state_for_ops(loan, &kLoanOps));
    std::optional<OutputPoolCallbackClaim> selected_pool;
    if (selected != nullptr) {
      selected_pool.emplace(*selected);
    }
    if (!selected_pool || !*selected_pool || selected_pool->pool() != this ||
        selected->index >= slots_.size() || slots_[selected->index].get() != selected) {
      const std::lock_guard lock(mutex_);
      record_diagnostic(nullptr,
                        TensorOutputOperation::kEmit,
                        ErrorCode::kInvalidArgument,
                        TensorOutputDiagnosticReason::kLoanOutputMismatch);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool descriptor_schema =
        type_ops_->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema;
    if (descriptor_schema != (descriptor != nullptr)) {
      std::unique_lock lock(mutex_);
      reconcile_deferred_releases(lock);
      cancel_invalid_emit(*selected, lock);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    auto context = prepare_emit(*selected, force_closed);
    if (!context) {
      return make_unexpected(std::move(context.error()));
    }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      Tensor tensor =
          TensorAccess::from_context(std::move(*context), TensorAuthority::kLocalWritable);
      auto published =
          type_ops_->publish_tensor_loan(declaration_, descriptor, std::move(tensor), options);
      if (published && consume_test_fault(RuntimeTensorOutputTestFault::kPostSubmitIndeterminate)) {
        published = make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      } else if (published &&
                 consume_test_fault(RuntimeTensorOutputTestFault::kPostSubmitPartial)) {
        published = make_unexpected(make_publication_partial_error(0U, 1U));
      }
      return finish_publication(*selected, std::move(published));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      std::unique_lock lock(mutex_);
      reconcile_deferred_releases(lock);
      quarantine(*selected);
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  static expected<TensorOutputLoan, Error> allocate_callback(void* state,
                                                             TensorLoanRequest request) noexcept {
    return static_cast<OutputPool*>(state)->allocate(request);
  }

  static expected<void, Error> emit_callback(void* state, TensorOutputLoan loan,
                                             const void* descriptor, EmitOptions options) noexcept {
    return static_cast<OutputPool*>(state)->emit(std::move(loan), descriptor, options);
  }

  static void abandon_loan_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    if (pool) {
      pool.pool()->abandon_loan(slot);
    }
  }

  static TensorShapeView shape_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->shape;
  }

  static TensorByteStridesView strides_callback(const void* state) noexcept {
    const auto& slot = *static_cast<const OutputSlot*>(state);
    return slot.explicit_strides ? TensorByteStridesView{slot.byte_strides}
                                 : TensorByteStridesView{std::nullopt};
  }

  static DLDataType dtype_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->dtype;
  }

  static DeviceId device_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->device;
  }

  static MemoryKind memory_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->memory_kind;
  }

  static std::size_t bytes_callback(const void* state) noexcept {
    return static_cast<const OutputSlot*>(state)->byte_size;
  }

  static expected<TensorOutputWriteGuard, Error> write_callback(
      void* state, ::cuda::stream_ref stream) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->begin_write(slot, stream.get(), true)
                : expected<TensorOutputWriteGuard, Error>{
                      make_unexpected(Error{ErrorCode::kClosed})};
  }

  static expected<TensorOutputWriteGuard, Error> write_host_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->begin_write(slot, nullptr, false)
                : expected<TensorOutputWriteGuard, Error>{
                      make_unexpected(Error{ErrorCode::kClosed})};
  }

  static expected<void, Error> cancel_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->cancel(slot)
                : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
  }

  static void abandon_write_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    if (pool) {
      pool.pool()->abandon_write(slot);
    }
  }

  static void* data_callback(const void* state) noexcept {
    const auto& slot = *static_cast<const OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? slot.context->tensor.dl_tensor.data : nullptr;
  }

  static std::optional<::cuda::stream_ref> stream_callback(const void* state) noexcept {
    const auto& slot = *static_cast<const OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool && slot.writer_cuda
               ? std::optional<::cuda::stream_ref>{::cuda::stream_ref{slot.writer_stream}}
               : std::nullopt;
  }

  static expected<void, Error> commit_callback(void* state) noexcept {
    auto& slot = *static_cast<OutputSlot*>(state);
    const OutputPoolCallbackClaim pool(slot);
    return pool ? pool.pool()->commit_write(slot)
                : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
  }

  inline static const TensorOutputBindingOps kBindingOps{
      .allocate = &allocate_callback,
      .emit_loan = &emit_callback,
  };

  inline static const TensorOutputLoanOps kLoanOps{
      .abandon = &abandon_loan_callback,
      .shape = &shape_callback,
      .byte_strides = &strides_callback,
      .dtype = &dtype_callback,
      .device_id = &device_callback,
      .memory_kind = &memory_callback,
      .byte_size = &bytes_callback,
      .write = &write_callback,
      .write_host = &write_host_callback,
      .cancel = &cancel_callback,
  };

  inline static const TensorOutputWriteGuardOps kWriteOps{
      .abandon = &abandon_write_callback,
      .data = &data_callback,
      .byte_size = &bytes_callback,
      .stream = &stream_callback,
      .commit = &commit_callback,
  };

  TensorAllocationPlanRecord plan_;
  OutputPlanContract contract_;
  VertexId vertex_{};
  OperatorInstanceId owner_{};
  void* declaration_{};
  const RuntimePortTypeOps* type_ops_{};
  cudaStream_t stream_{};
  bool elide_same_stream_readiness_{};
  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  std::uint64_t pool_generation_{};
  std::uint64_t allocation_generation_{};
  std::uint32_t active_slot_count_{};
  RuntimeTensorPoolAllocationAction allocation_action_{RuntimeTensorPoolAllocationAction::kFresh};
  std::shared_ptr<RuntimeTensorOutputTestControl> test_control_;
  std::size_t test_pool_index_{};
  std::optional<std::uint32_t> descriptor_tensor_index_;
  std::mutex mutex_;
  std::vector<std::shared_ptr<OutputSlot>> slots_;
  std::vector<std::shared_ptr<TensorMemoryReference>> references_;
  std::shared_ptr<OutputPoolReferenceControl> reference_control_;
  std::shared_ptr<holomq::BufferPool> export_pool_;
  std::vector<std::uint32_t> free_slots_;
  std::vector<std::uint32_t> retiring_slots_;
  std::atomic<OutputSlot*> deferred_head_{};
  std::vector<TensorOutputDiagnosticRecord> diagnostic_storage_;
  TensorOutputDiagnosticBuffer diagnostics_;
  RuntimeTensorOutputTestControl::PrimitiveMetrics metrics_;
  std::uint64_t zero_depth_started_{};
  std::uint64_t zero_depth_accumulated_ns_{};
  std::uint32_t live_loans_{};
  std::uint32_t retirement_records_in_flight_{};
  bool metrics_initialized_{};
  bool zero_depth_active_{};
  bool retirement_query_active_{};
  bool closed_{};
  bool bound_{};
  bool ever_bound_{};
  bool restart_fenced_{};
  bool lifecycle_transition_active_{};
  bool materialization_recorded_{};
};

template <typename Base>
void SlotMemoryReference<Base>::retain_tensor_handle() noexcept {
  if (auto* const pool = control_ != nullptr ? control_->acquire() : nullptr; pool != nullptr) {
    pool->retain_tensor(slot_);
    control_->release();
  }
}

template <typename Base>
void SlotMemoryReference<Base>::release_tensor_handle() noexcept {
  if (auto* const pool = control_ != nullptr ? control_->acquire() : nullptr; pool != nullptr) {
    pool->release_tensor(slot_);
    control_->release();
  }
}

struct TensorBackedOutputDeclaration {
  void* declaration{};
  const RuntimePortTypeOps* type_ops{};
};

[[nodiscard]] TensorBackedOutputDeclaration find_output_declaration(
    RuntimeOperatorInstance& instance, std::string_view local_name) noexcept {
  if (instance.replay_spec == nullptr) {
    return {};
  }
  for (std::size_t index = 0U;
       index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
       ++index) {
    const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
    if (replay.direction == PortDirection::kOutput && replay.name == local_name &&
        replay.declaration != nullptr && replay.runtime_type_ops != nullptr &&
        replay.runtime_type_ops->publish_tensor_loan != nullptr &&
        replay.runtime_type_ops->bind_tensor_output != nullptr &&
        replay.runtime_type_ops->unbind_tensor_output != nullptr) {
      return TensorBackedOutputDeclaration{
          .declaration = const_cast<void*>(replay.declaration),
          .type_ops = replay.runtime_type_ops,
      };
    }
  }
  return {};
}

[[nodiscard]] TensorAllocationPlanRecord tensor_field_output_allocation(
    const TensorFieldAllocationPlanRecord& field) {
  return TensorAllocationPlanRecord{
      .output = field.output,
      .plan_generation = field.plan_generation,
      .run_generation = field.run_generation,
      .pool_id = field.pool_id,
      .allocator_root_id = field.allocator_root_id,
      .allocator_process_group = field.allocator_process_group,
      .bounds = field.bounds,
      .axes = field.axes,
      .layout = field.layout,
      .memory_kind = field.memory_kind,
      .device = field.device,
      .slot_count = field.slot_count,
      .aggregate_byte_capacity = field.aggregate_byte_capacity,
      .max_outstanding_loans = field.max_outstanding_loans,
      .max_emits_per_compute = field.max_emits_per_compute,
      .control_capacity = field.control_capacity,
      .readiness_capacity = field.readiness_capacity,
      .release_capacity = field.release_capacity,
      .retirement_capacity = field.retirement_capacity,
      .quarantine_capacity = field.quarantine_capacity,
      .materialize_before = field.materialize_before,
      .effective_profile = field.effective_profile,
      .disposition = field.disposition,
      .fallback_rejected = field.fallback_rejected,
  };
}

[[nodiscard]] expected<bool, Error> provider_binding_replayed(
    const ExecutionPlanState& plan, const ProviderAllocationPlanRecord& provider,
    const std::optional<LifecycleStage> restart_from = std::nullopt) noexcept {
  if (!provider.first_sizing_dependency || provider.sizing_dependency_count == 0U) {
    return make_unexpected(tensor_resource_error("rate provider dependency invalid"));
  }
  const std::size_t first = provider.first_sizing_dependency.value - 1U;
  if (first > plan.pool_capacity_dependencies.size() ||
      provider.sizing_dependency_count > plan.pool_capacity_dependencies.size() - first) {
    return make_unexpected(tensor_resource_error("rate provider dependency invalid"));
  }
  const auto dependencies =
      std::span{plan.pool_capacity_dependencies}.subspan(first, provider.sizing_dependency_count);
  for (const auto& dependency : dependencies) {
    if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
      continue;
    }
    const auto recipe = std::ranges::find(plan.queue_sizing_recipes,
                                          dependency.queue_sizing_recipe,
                                          &QueueSizingRecipePlanRecord::id);
    if (recipe == plan.queue_sizing_recipes.end()) {
      return make_unexpected(tensor_resource_error("rate provider dependency invalid"));
    }
    if (recipe->latest_binding_stage.has_value()) {
      // The immediately preceding presence check guards this cold-plan access.
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      const LifecycleStage latest_binding_stage = recipe->latest_binding_stage.value();
      if (!restart_from || static_cast<std::uint8_t>(*restart_from) <=
                               static_cast<std::uint8_t>(latest_binding_stage)) {
        return expected<bool, Error>{true};
      }
    }
  }
  return expected<bool, Error>{false};
}

// One Operator owns this output binding, and its single active compute callback is the only
// publication authority. The count-sized request/Tensor/slot scratch below is therefore safe to
// reuse without additional same-output synchronization under the current Holoscan compute model.
class TensorSchemaOutputPool final {
 public:
  TensorSchemaOutputPool(const TensorSchemaOutputPool&) = delete;
  TensorSchemaOutputPool& operator=(const TensorSchemaOutputPool&) = delete;

  ~TensorSchemaOutputPool() {
    detach();
    close();
  }

  [[nodiscard]] static expected<std::unique_ptr<TensorSchemaOutputPool>, Error> create(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance, void* declaration,
      const RuntimePortTypeOps* type_ops, const RuntimeCudaExecutionResources& cuda_resources,
      std::span<const DeviceResource> devices,
      const std::span<const TensorFieldAllocationPlanRecord> allocations,
      const RunGenerationId run_generation,
      const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
      const std::size_t test_pool_offset) {
    if (declaration == nullptr || type_ops == nullptr ||
        type_ops->descriptor_tensor_static_count != allocations.size() ||
        type_ops->publish_tensor_loans == nullptr || type_ops->bind_tensor_output == nullptr ||
        type_ops->unbind_tensor_output == nullptr || allocations.empty() ||
        allocations.size() > kMaximumTensorSchemaLoanFields) {
      return make_unexpected(tensor_resource_error("invalid Tensor schema output-pool contract"));
    }
    auto result = std::unique_ptr<TensorSchemaOutputPool>{new TensorSchemaOutputPool{
        instance.operator_id, allocations.front().output, declaration, type_ops}};
    result->fields_.reserve(allocations.size());
    result->field_providers_.reserve(allocations.size());
    result->field_ids_.resize(allocations.size());
    result->field_dtypes_.resize(allocations.size());
    result->canonical_requests_.resize(allocations.size());
    result->tensor_scratch_.resize(allocations.size());
    result->slot_scratch_.resize(allocations.size());
    for (std::size_t index = 0U; index < allocations.size(); ++index) {
      const auto& field = allocations[index];
      if (field.output != result->output_ || field.tensor_index != index ||
          (index != 0U && field.field_id <= allocations[index - 1U].field_id)) {
        return make_unexpected(
            tensor_resource_error("noncanonical Tensor schema field allocation"));
      }
      const auto provider =
          std::ranges::find_if(plan.provider_allocation_plans, [&](const auto& row) {
            return row.allocation_kind == ProviderAllocationKind::kTensorFieldOutput &&
                   row.geometry_mode == AllocationGeometryMode::kCompileConcrete &&
                   row.disposition == PlanDisposition::kAccepted && row.output == field.output &&
                   row.tensor_field_id == field.field_id &&
                   row.tensor_index == field.tensor_index && row.pool_id == field.pool_id &&
                   !row.tensor_template;
          });
      if (provider == plan.provider_allocation_plans.end() ||
          std::ranges::count_if(plan.provider_allocation_plans, [&](const auto& row) {
            return row.allocation_kind == ProviderAllocationKind::kTensorFieldOutput &&
                   row.disposition == PlanDisposition::kAccepted && row.output == field.output &&
                   row.tensor_field_id == field.field_id &&
                   row.tensor_index == field.tensor_index && row.pool_id == field.pool_id;
          }) != 1) {
        return make_unexpected(
            tensor_resource_error("Tensor schema field provider mismatch", "pool", field.pool_id));
      }
      const TensorAllocationPlanRecord allocation = tensor_field_output_allocation(field);
      auto contract = validate_output_plan_contract(plan, allocation);
      if (!contract) {
        return make_unexpected(std::move(contract.error()));
      }
      auto stream = tensor_output_stream(plan, cuda_resources, devices, allocation);
      if (!stream) {
        return make_unexpected(std::move(stream.error()));
      }
      const bool elide_same_stream_readiness =
          tensor_can_elide_native_readiness(plan, cuda_resources, allocation, *stream);
      auto pool = OutputPool::create(plan,
                                     allocation,
                                     std::move(*contract),
                                     instance,
                                     declaration,
                                     type_ops,
                                     *stream,
                                     elide_same_stream_readiness,
                                     run_generation,
                                     test_control,
                                     test_pool_offset + index,
                                     false,
                                     1U,
                                     1U,
                                     static_cast<std::uint32_t>(index));
      if (!pool) {
        return make_unexpected(std::move(pool.error()));
      }
      result->field_ids_[index] = field.field_id;
      result->field_dtypes_[index] = (*pool)->exact_bundle_dtype();
      result->field_providers_.push_back(provider->id);
      result->fields_.push_back(std::move(*pool));
    }
    result->bind();
    return result;
  }

  [[nodiscard]] OperatorInstanceId owner() const noexcept { return owner_; }
  [[nodiscard]] PortId output() const noexcept { return output_; }
  [[nodiscard]] std::size_t field_count() const noexcept { return fields_.size(); }
  [[nodiscard]] ProviderAllocationPlanRecordId field_provider(
      const std::size_t index) const noexcept {
    return index < field_providers_.size() ? field_providers_[index]
                                           : ProviderAllocationPlanRecordId{};
  }
  [[nodiscard]] std::uint32_t field_physical_slot_count(const std::size_t index) const noexcept {
    return index < fields_.size() ? fields_[index]->physical_slot_count() : 0U;
  }

  [[nodiscard]] expected<std::size_t, Error> replayed_rate_provider_count(
      const ExecutionPlanState& plan,
      const std::optional<LifecycleStage> restart_from = std::nullopt) const noexcept {
    auto selected = std::size_t{};
    for (const auto provider_id : field_providers_) {
      const auto provider = std::ranges::find(
          plan.provider_allocation_plans, provider_id, &ProviderAllocationPlanRecord::id);
      if (provider == plan.provider_allocation_plans.end()) {
        return make_unexpected(tensor_resource_error("Tensor schema field provider mismatch"));
      }
      auto replayed = provider_binding_replayed(plan, *provider, restart_from);
      if (!replayed) {
        return make_unexpected(std::move(replayed.error()));
      }
      selected += static_cast<std::size_t>(*replayed);
    }
    if (selected != 0U && selected != fields_.size()) {
      return make_unexpected(tensor_resource_error("partial Tensor schema rate provider bundle"));
    }
    return expected<std::size_t, Error>{selected};
  }

  [[nodiscard]] expected<void, Error> configure_startup_active_slots(
      const std::span<const std::uint32_t> active_entries) noexcept {
    auto preflight = preflight_startup_active_slots(active_entries);
    if (!preflight) {
      return preflight;
    }
    publish_startup_active_slots(active_entries);
    return {};
  }

  [[nodiscard]] expected<void, Error> preflight_startup_active_slots(
      const std::span<const std::uint32_t> active_entries) noexcept {
    if (active_entries.size() != fields_.size()) {
      return make_unexpected(tensor_resource_error("Tensor schema active-slot inventory invalid"));
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto preflight = fields_[index]->preflight_startup_active_slots(active_entries[index]);
      if (!preflight) {
        return make_unexpected(std::move(preflight.error()));
      }
    }
    return {};
  }

  void publish_startup_active_slots(const std::span<const std::uint32_t> active_entries) noexcept {
    assert(active_entries.size() == fields_.size());
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      fields_[index]->publish_startup_active_slots(active_entries[index]);
    }
  }

  [[nodiscard]] expected<bool, Error> can_reuse_retained(
      const std::span<const std::uint32_t> active_entries) noexcept {
    if (active_entries.size() != fields_.size()) {
      return make_unexpected(tensor_resource_error("Tensor schema active-slot inventory invalid"));
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto reusable =
          fields_[index]->can_reuse_retained(fields_[index]->allocation(), active_entries[index]);
      if (!reusable) {
        return make_unexpected(std::move(reusable.error()));
      }
      if (!*reusable) {
        return expected<bool, Error>{false};
      }
    }
    return expected<bool, Error>{true};
  }

  [[nodiscard]] bool restart_rate_publication_ready(
      const std::span<const std::uint32_t> active_entries) noexcept {
    if (active_entries.size() != fields_.size()) {
      return false;
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      if (!fields_[index]->restart_reuse_publication_ready(active_entries[index])) {
        return false;
      }
    }
    return true;
  }

  void publish_rate_reuse(
      const std::span<const std::uint32_t> active_entries,
      const std::span<const RuntimeTensorPoolAllocationAction> actions) noexcept {
    assert(active_entries.size() == fields_.size());
    assert(actions.size() == fields_.size());
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      fields_[index]->publish_rate_reuse(active_entries[index], actions[index], false);
    }
    bind();
  }

  [[nodiscard]] expected<std::unique_ptr<TensorSchemaOutputPool>, Error> recreate_unbound(
      const ExecutionPlanState& plan, RuntimeOperatorInstance& instance) const {
    auto result = std::unique_ptr<TensorSchemaOutputPool>{
        new TensorSchemaOutputPool{owner_, output_, declaration_, type_ops_}};
    result->fields_.reserve(fields_.size());
    result->field_providers_ = field_providers_;
    result->field_ids_.resize(fields_.size());
    result->field_dtypes_ = field_dtypes_;
    result->canonical_requests_.resize(fields_.size());
    result->tensor_scratch_.resize(fields_.size());
    result->slot_scratch_.resize(fields_.size());
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto field = fields_[index]->recreate_unbound(plan, instance);
      if (!field) {
        return make_unexpected(std::move(field.error()));
      }
      result->field_ids_[index] = field_ids_[index];
      result->fields_.push_back(std::move(*field));
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> begin_restart() noexcept {
    std::size_t prepared = 0U;
    for (; prepared < fields_.size(); ++prepared) {
      auto fenced = fields_[prepared]->begin_restart();
      if (!fenced) {
        for (std::size_t index = 0U; index < prepared; ++index) {
          static_cast<void>(fields_[index]->reopen(false));
        }
        return make_unexpected(std::move(fenced.error()));
      }
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> reopen() noexcept {
    for (auto& field : fields_) {
      auto reopened = field->reopen(false);
      if (!reopened) {
        return make_unexpected(std::move(reopened.error()));
      }
    }
    bind();
    return {};
  }

  [[nodiscard]] bool restart_publication_ready() noexcept {
    return std::ranges::all_of(
        fields_, [](const auto& field) { return field->restart_publication_ready(); });
  }

  void publish_reopen() noexcept {
    for (auto& field : fields_) {
      field->publish_reopen(false);
    }
    bind();
  }

  void close() noexcept {
    for (auto& field : fields_) {
      field->close();
    }
  }

  [[nodiscard]] expected<bool, Error> advance_terminal_retirement() noexcept {
    auto complete = true;
    for (auto& field : fields_) {
      auto advanced = field->advance_terminal_retirement();
      if (!advanced) {
        return make_unexpected(std::move(advanced).error());
      }
      if (!*advanced) {
        complete = false;
      }
    }
    return expected<bool, Error>{complete};
  }

  [[nodiscard]] expected<bool, Error> preflight_terminal_close() noexcept {
    auto ready = true;
    for (auto& field : fields_) {
      auto preflight = field->preflight_terminal_close();
      if (!preflight) {
        return make_unexpected(std::move(preflight).error());
      }
      if (!*preflight) {
        ready = false;
      }
    }
    return expected<bool, Error>{ready};
  }

  void detach() noexcept {
    if (bound_) {
      type_ops_->unbind_tensor_output(declaration_);
      bound_ = false;
    }
  }

  void bind_output() noexcept { bind(); }

 private:
  TensorSchemaOutputPool(OperatorInstanceId owner, PortId output, void* declaration,
                         const RuntimePortTypeOps* type_ops) noexcept
      : owner_(owner), output_(output), declaration_(declaration), type_ops_(type_ops) {}

  void bind() noexcept {
    if (!bound_) {
      type_ops_->bind_tensor_output(declaration_,
                                    TensorOutputBinding{
                                        .state = this,
                                        .ops = &kBindingOps,
                                        .tensor_field_dtypes = field_dtypes_,
                                    });
      bound_ = true;
    }
    for (auto& field : fields_) {
      field->mark_bundle_binding_published();
    }
  }

  [[nodiscard]] expected<TensorSchemaLoanBundleStorage, Error> allocate_tensors(
      const TensorSchemaLoanRequestList& requests) noexcept {
    if (requests.field_count != fields_.size()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_COUNT_MISMATCH"});
    }
    auto& canonical = canonical_requests_;
    std::fill(canonical.begin(), canonical.end(), nullptr);
    for (std::size_t index = 0U; index < requests.field_count; ++index) {
      const auto& request = requests.fields[index];
      if (request.tensor_index >= fields_.size() ||
          field_ids_[request.tensor_index] != request.field_id) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_SELECTOR_INVALID"});
      }
      if (canonical[request.tensor_index] != nullptr) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_DUPLICATE"});
      }
      canonical[request.tensor_index] = &request;
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      if (canonical[index] == nullptr ||
          !fields_[index]->accepts_bundle_request(canonical[index]->tensor)) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_REQUEST_INVALID"});
      }
    }

    TensorSchemaLoanBundleStorage result;
    result.field_count = static_cast<std::uint32_t>(fields_.size());
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto loan = fields_[index]->allocate_bundle_field(canonical[index]->tensor);
      if (!loan) {
        return make_unexpected(std::move(loan.error()));
      }
      result.field_ids[index] = field_ids_[index];
      result.fields[index].emplace(std::move(*loan));
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> emit_tensors(TensorSchemaLoanBundleStorage bundle,
                                                   const void* descriptor,
                                                   EmitOptions options) noexcept {
    if (descriptor == nullptr || bundle.field_count != fields_.size()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_PUBLICATION_INVALID"});
    }
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      if (bundle.field_ids[index] != field_ids_[index] || !bundle.fields[index]) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_PREFLIGHT_FAILED"});
      }
      // The preceding bundle-field check proves this loan is present.
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      if (!fields_[index]->preflight_bundle_field(bundle.fields[index].value())) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "TENSOR_SCHEMA_BUNDLE_FIELD_PREFLIGHT_FAILED"});
      }
    }

    auto& tensors = tensor_scratch_;
    auto& slots = slot_scratch_;
    std::fill(slots.begin(), slots.end(), nullptr);
    std::size_t prepared_count = 0U;
    for (; prepared_count < fields_.size(); ++prepared_count) {
      // The complete inventory was proven engaged by the preflight loop above. Rechecking here
      // would add a redundant per-field branch to the schema publication path.
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto prepared = fields_[prepared_count]->prepare_bundle_field(*bundle.fields[prepared_count]);
      if (!prepared) {
        const expected<void, Error> failure = make_unexpected(
            Error{.code = prepared.error().code, .message = prepared.error().message});
        for (std::size_t index = 0U; index < prepared_count; ++index) {
          tensors[index] = Tensor{};
          static_cast<void>(fields_[index]->finish_bundle_field(slots[index], failure));
        }
        return make_unexpected(std::move(prepared.error()));
      }
      slots[prepared_count] = prepared->slot;
      tensors[prepared_count] = std::move(prepared->tensor);
    }

    const std::span<const Tensor> authoritative{tensors.data(), fields_.size()};
    const expected<void, Error> published =
        type_ops_->publish_tensor_loans(declaration_, descriptor, authoritative, options);
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      tensors[index] = Tensor{};
    }
    expected<void, Error> finished = published;
    for (std::size_t index = 0U; index < fields_.size(); ++index) {
      auto field_finished = fields_[index]->finish_bundle_field(slots[index], published);
      if (!field_finished && finished) {
        finished = make_unexpected(std::move(field_finished.error()));
      }
    }
    return finished;
  }

  static expected<TensorSchemaLoanBundleStorage, Error> allocate_tensors_callback(
      void* state, const TensorSchemaLoanRequestList& requests) noexcept {
    return static_cast<TensorSchemaOutputPool*>(state)->allocate_tensors(requests);
  }

  static expected<void, Error> emit_tensors_callback(void* state,
                                                     TensorSchemaLoanBundleStorage bundle,
                                                     const void* descriptor,
                                                     EmitOptions options) noexcept {
    return static_cast<TensorSchemaOutputPool*>(state)->emit_tensors(
        std::move(bundle), descriptor, options);
  }

  inline static const TensorOutputBindingOps kBindingOps{
      .allocate = nullptr,
      .emit_loan = nullptr,
      .allocate_tensors = &allocate_tensors_callback,
      .emit_tensors = &emit_tensors_callback,
  };

  OperatorInstanceId owner_;
  PortId output_;
  void* declaration_{};
  const RuntimePortTypeOps* type_ops_{};
  std::vector<std::unique_ptr<OutputPool>> fields_;
  std::vector<ProviderAllocationPlanRecordId> field_providers_;
  std::vector<std::uint16_t> field_ids_;
  std::vector<std::optional<DLDataType>> field_dtypes_;
  std::vector<const TensorSchemaLoanRequest*> canonical_requests_;
  std::vector<Tensor> tensor_scratch_;
  std::vector<OutputSlot*> slot_scratch_;
  bool bound_{};
};

[[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> materialize_output_pool(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources, std::span<const DeviceResource> devices,
    std::span<const TensorAllocationPlanRecord> allocation_inventory,
    RuntimeRateMaterializationView materialization, const TensorAllocationPlanRecord& allocation,
    RunGenerationId run_generation, ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control, std::size_t pool_index) {
  if (std::count_if(allocation_inventory.begin(), allocation_inventory.end(), [&](const auto& row) {
        return row.output == allocation.output || row.pool_id == allocation.pool_id;
      }) != 1) {
    return make_unexpected(tensor_resource_error(
        "duplicate output-pool identity", "output", allocation.output.value()));
  }
  const auto provider = std::ranges::find(
      plan.provider_allocation_plans, allocation.pool_id, &ProviderAllocationPlanRecord::pool_id);
  const RuntimeTensorAllocationGeneration* generated_allocation = nullptr;
  if (provider != plan.provider_allocation_plans.end() &&
      provider->geometry_mode == AllocationGeometryMode::kLifecycleCandidate) {
    const auto generated = std::ranges::find(materialization.tensor_allocations,
                                             provider->id,
                                             &RuntimeTensorAllocationGeneration::provider);
    if (generated == materialization.tensor_allocations.end() ||
        std::ranges::count(materialization.tensor_allocations,
                           provider->id,
                           &RuntimeTensorAllocationGeneration::provider) != 1) {
      return make_unexpected(tensor_resource_error(
          "lifecycle candidate allocation missing", "pool", allocation.pool_id));
    }
    generated_allocation = &*generated;
  }
  if (generated_allocation != nullptr) {
    auto admitted = validate_lifecycle_candidate_retained_contract(plan, *provider, allocation);
    if (!admitted) {
      return make_unexpected(std::move(admitted.error()));
    }
  }
  auto contract =
      validate_output_plan_contract(plan, allocation, materialization, generated_allocation);
  if (!contract) {
    return make_unexpected(std::move(contract.error()));
  }
  const auto* port = find_port_by_id_and_direction(plan, allocation.output, PortDirection::kOutput);
  if (port == nullptr || port->owner_kind != PortOwnerKind::kOperator) {
    return make_unexpected(tensor_resource_error(
        "output port inventory mismatch", "output", allocation.output.value()));
  }
  if (!worker_owns_vertex(plan, process_group, port->owner)) {
    return std::unique_ptr<OutputPool>{};
  }
  const auto instance =
      std::find_if(operators.begin(), operators.end(), [&](const auto& candidate) {
        return candidate.vertex == port->owner;
      });
  if (instance == operators.end() || instance->replay_spec == nullptr) {
    return make_unexpected(tensor_resource_error(
        "output operator inventory mismatch", "output", allocation.output.value()));
  }
  const auto declaration = find_output_declaration(*instance, port->local_name);
  if (declaration.declaration == nullptr || declaration.type_ops == nullptr) {
    return make_unexpected(
        tensor_resource_error("output declaration mismatch", "output", allocation.output.value()));
  }
  auto stream = tensor_output_stream(plan, cuda_resources, devices, allocation);
  if (!stream) {
    return make_unexpected(std::move(stream.error()));
  }
  const bool elide_same_stream_readiness =
      tensor_can_elide_native_readiness(plan, cuda_resources, allocation, *stream);
  return OutputPool::create(plan,
                            allocation,
                            std::move(*contract),
                            *instance,
                            declaration.declaration,
                            declaration.type_ops,
                            *stream,
                            elide_same_stream_readiness,
                            run_generation,
                            test_control,
                            pool_index);
}

[[nodiscard]] expected<void, Error> materialize_output_pools(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources, std::span<const DeviceResource> devices,
    std::span<const TensorAllocationPlanRecord> allocation_inventory,
    RuntimeRateMaterializationView materialization, RunGenerationId run_generation,
    ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    std::vector<std::unique_ptr<OutputPool>>& outputs) {
  outputs.reserve(allocation_inventory.size());
  for (const auto& allocation : allocation_inventory) {
    auto pool = materialize_output_pool(plan,
                                        operators,
                                        cuda_resources,
                                        devices,
                                        allocation_inventory,
                                        materialization,
                                        allocation,
                                        run_generation,
                                        process_group,
                                        test_control,
                                        outputs.size());
    if (!pool) {
      return make_unexpected(std::move(pool.error()));
    }
    if (*pool != nullptr) {
      outputs.push_back(std::move(*pool));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> materialize_tensor_schema_output_pools(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources, std::span<const DeviceResource> devices,
    RunGenerationId run_generation, ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    std::vector<std::unique_ptr<TensorSchemaOutputPool>>& outputs) {
  std::size_t begin = 0U;
  while (begin < plan.tensor_field_allocations.size()) {
    std::size_t end = begin + 1U;
    while (end < plan.tensor_field_allocations.size() &&
           plan.tensor_field_allocations[end].output ==
               plan.tensor_field_allocations[begin].output) {
      ++end;
    }
    const auto allocations =
        std::span<const TensorFieldAllocationPlanRecord>{plan.tensor_field_allocations}.subspan(
            begin, end - begin);
    const PortId output = allocations.front().output;
    const auto* port = find_port_by_id_and_direction(plan, output, PortDirection::kOutput);
    if (port == nullptr || port->owner_kind != PortOwnerKind::kOperator) {
      return make_unexpected(
          tensor_resource_error("field output port inventory mismatch", "output", output.value()));
    }
    if (worker_owns_vertex(plan, process_group, port->owner)) {
      const auto instance =
          std::find_if(operators.begin(), operators.end(), [&](const auto& candidate) {
            return candidate.vertex == port->owner;
          });
      if (instance == operators.end() || instance->replay_spec == nullptr) {
        return make_unexpected(tensor_resource_error(
            "field output operator inventory mismatch", "output", output.value()));
      }
      const auto declaration = find_output_declaration(*instance, port->local_name);
      if (declaration.declaration == nullptr || declaration.type_ops == nullptr) {
        return make_unexpected(
            tensor_resource_error("field output declaration mismatch", "output", output.value()));
      }
      auto pool = TensorSchemaOutputPool::create(plan,
                                                 *instance,
                                                 declaration.declaration,
                                                 declaration.type_ops,
                                                 cuda_resources,
                                                 devices,
                                                 allocations,
                                                 run_generation,
                                                 test_control,
                                                 plan.tensor_allocations.size() + begin);
      if (!pool) {
        return make_unexpected(std::move(pool.error()));
      }
      outputs.push_back(std::move(*pool));
    }
    begin = end;
  }
  return {};
}

template <typename Pool>
struct PreparedRestartOutput {
  std::size_t output_index{};
  std::unique_ptr<Pool> candidate;
  ProviderAllocationPlanRecordId rate_provider{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  std::uint32_t active_entries{};
  RuntimeTensorPoolAllocationAction action{RuntimeTensorPoolAllocationAction::kFresh};
  std::array<std::uint32_t, kMaximumTensorSchemaLoanFields> field_active_entries{};
  std::array<RuntimeTensorPoolAllocationAction, kMaximumTensorSchemaLoanFields> field_actions{};
  std::uint32_t rate_provider_count{};
  bool rate_sensitive{};
  bool rate_prepared{};
  bool replacement_allowed{};
};

[[nodiscard]] expected<ProviderAllocationPlanRecordId, Error> rate_sensitive_tensor_provider(
    const ExecutionPlanState& plan, const OutputPool& output,
    const std::optional<LifecycleStage> restart_from = std::nullopt) noexcept {
  const ProviderAllocationPlanRecord* selected{};
  for (const auto& provider : plan.provider_allocation_plans) {
    if (provider.allocation_kind != ProviderAllocationKind::kTensorOutput ||
        provider.output != output.output() || provider.pool_id != output.allocation().pool_id) {
      continue;
    }
    if (selected != nullptr) {
      return make_unexpected(tensor_resource_error("duplicate rate-sensitive Tensor provider"));
    }
    selected = &provider;
  }
  if (selected == nullptr) {
    return ProviderAllocationPlanRecordId{};
  }
  if (selected->disposition != PlanDisposition::kAccepted || selected->reason ||
      !selected->first_sizing_dependency || selected->sizing_dependency_count == 0U) {
    return make_unexpected(tensor_resource_error("rate-sensitive Tensor provider invalid"));
  }
  auto replayed = provider_binding_replayed(plan, *selected, restart_from);
  if (!replayed) {
    return make_unexpected(std::move(replayed.error()));
  }
  return *replayed ? selected->id : ProviderAllocationPlanRecordId{};
}

[[nodiscard]] expected<ProviderAllocationPlanRecordId, Error> tensor_output_provider(
    const ExecutionPlanState& plan, const OutputPool& output) noexcept {
  const ProviderAllocationPlanRecord* selected{};
  for (const auto& provider : plan.provider_allocation_plans) {
    if (provider.allocation_kind != ProviderAllocationKind::kTensorOutput ||
        provider.output != output.output() || provider.pool_id != output.allocation().pool_id) {
      continue;
    }
    if (selected != nullptr) {
      return make_unexpected(tensor_resource_error("duplicate Tensor output provider"));
    }
    selected = &provider;
  }
  if (selected == nullptr) {
    return ProviderAllocationPlanRecordId{};
  }
  if (selected->disposition != PlanDisposition::kAccepted || selected->reason ||
      !selected->first_sizing_dependency || selected->sizing_dependency_count == 0U) {
    return make_unexpected(tensor_resource_error("Tensor output provider invalid"));
  }
  return selected->id;
}

[[nodiscard]] AllocationAction allocation_action(
    RuntimeTensorPoolAllocationAction action) noexcept {
  switch (action) {
    case RuntimeTensorPoolAllocationAction::kFresh:
      return AllocationAction::kFresh;
    case RuntimeTensorPoolAllocationAction::kReusedExact:
      return AllocationAction::kReusedExact;
    case RuntimeTensorPoolAllocationAction::kReusedLarger:
      return AllocationAction::kReusedLarger;
    case RuntimeTensorPoolAllocationAction::kReplaced:
      return AllocationAction::kReplaced;
  }
  return AllocationAction::kRejected;
}

[[nodiscard]] expected<void, Error> report_provider_materialization(
    RuntimeRateMaterializationView materialization, ProviderAllocationPlanRecordId provider,
    RuntimeTensorPoolAllocationAction action, const std::uint32_t physical_slot_count,
    const bool preflight_only = false) noexcept {
  if (materialization.provider_results.empty()) {
    return make_unexpected(tensor_resource_error("rate provider evidence missing"));
  }
  const auto result = std::ranges::find(
      materialization.provider_results, provider, &RuntimeProviderMaterializationResult::provider);
  if (result == materialization.provider_results.end() ||
      std::ranges::count(materialization.provider_results,
                         provider,
                         &RuntimeProviderMaterializationResult::provider) != 1 ||
      physical_slot_count == 0U) {
    return make_unexpected(tensor_resource_error("rate provider evidence invalid"));
  }
  const RuntimeProviderMaterializationResult selected = with_provider_materialization(
      *result, provider, allocation_action(action), physical_slot_count);
  if (result->reported &&
      (result->provider != selected.provider || result->action != selected.action ||
       result->compatibility_reason != selected.compatibility_reason ||
       result->materialized_entries != selected.materialized_entries ||
       result->stride_bytes_per_entry != selected.stride_bytes_per_entry ||
       result->reserved_bytes_total != selected.reserved_bytes_total)) {
    return make_unexpected(tensor_resource_error("rate provider evidence drift"));
  }
  if (!preflight_only) {
    *result = selected;
  }
  return {};
}

[[nodiscard]] expected<void, Error> report_provider_materialization(
    RuntimeRateMaterializationView materialization, ProviderAllocationPlanRecordId provider,
    RuntimeTensorPoolAllocationAction action, const OutputPool& output) noexcept {
  return report_provider_materialization(
      materialization, provider, action, output.physical_slot_count());
}

[[nodiscard]] expected<bool, Error> provider_reuse_permitted(
    const ExecutionPlanState& plan, const ProviderAllocationPlanRecord& provider) noexcept {
  if (!provider.first_sizing_dependency || provider.sizing_dependency_count == 0U) {
    return make_unexpected(tensor_resource_error("restart rate candidate dependency invalid"));
  }
  const std::size_t first = provider.first_sizing_dependency.value - 1U;
  if (first > plan.pool_capacity_dependencies.size() ||
      provider.sizing_dependency_count > plan.pool_capacity_dependencies.size() - first) {
    return make_unexpected(tensor_resource_error("restart rate candidate dependency invalid"));
  }
  const auto dependencies =
      std::span{plan.pool_capacity_dependencies}.subspan(first, provider.sizing_dependency_count);
  auto queue_dependency = false;
  for (const auto& dependency : dependencies) {
    if (dependency.pool_id != provider.pool_id) {
      return make_unexpected(tensor_resource_error("restart rate candidate dependency invalid"));
    }
    if (dependency.source == PoolCapacityDependencySource::kPredecessorContributor) {
      continue;
    }
    if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
      return make_unexpected(tensor_resource_error("restart rate candidate dependency invalid"));
    }
    queue_dependency = true;
    const auto recipe = std::ranges::find(plan.queue_sizing_recipes,
                                          dependency.queue_sizing_recipe,
                                          &QueueSizingRecipePlanRecord::id);
    if (recipe == plan.queue_sizing_recipes.end() ||
        std::ranges::count(plan.queue_sizing_recipes,
                           dependency.queue_sizing_recipe,
                           &QueueSizingRecipePlanRecord::id) != 1) {
      return make_unexpected(tensor_resource_error("restart rate candidate dependency invalid"));
    }
    if (recipe->reuse_policy != ProviderReusePolicy::kCompatibleExactOrLarger) {
      return expected<bool, Error>{false};
    }
  }
  if (!queue_dependency) {
    return make_unexpected(tensor_resource_error("restart rate candidate dependency invalid"));
  }
  return expected<bool, Error>{true};
}

template <typename Pool>
void close_output_family(std::vector<std::unique_ptr<Pool>>& outputs) noexcept {
  for (auto& output : outputs) {
    output->close();
  }
}

template <typename Pool>
void detach_output_family(std::vector<std::unique_ptr<Pool>>& outputs) noexcept {
  for (auto& output : outputs) {
    output->detach();
  }
}

template <typename Pool>
[[nodiscard]] std::size_t affected_output_count(
    const std::vector<std::unique_ptr<Pool>>& outputs,
    std::span<const OperatorInstanceId> members) noexcept {
  return static_cast<std::size_t>(std::ranges::count_if(outputs, [&](const auto& output) {
    return std::ranges::find(members, output->owner()) != members.end();
  }));
}

[[nodiscard]] RuntimeOperatorInstance* restart_operator(
    std::span<RuntimeOperatorInstance> operators, OperatorInstanceId owner) noexcept {
  const auto instance = std::ranges::find(operators, owner, &RuntimeOperatorInstance::operator_id);
  return instance != operators.end() ? &*instance : nullptr;
}

[[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> recreate_restart_output(
    const ExecutionPlanState& plan, OutputPool& output, RuntimeOperatorInstance& instance) {
  expected<std::unique_ptr<OutputPool>, Error> created = make_unexpected(tensor_resource_error(
      "restart output recreation not attempted", "output", output.output().value()));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    created = output.recreate_unbound(plan, instance);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(tensor_resource_error(
        "restart output recreation threw", "output", output.output().value()));
  }
#endif
  return created;
}

[[nodiscard]] expected<std::unique_ptr<OutputPool>, Error> recreate_restart_output(
    const ExecutionPlanState& plan, OutputPool& output, RuntimeOperatorInstance& instance,
    const TensorAllocationPlanRecord& allocation, const std::uint32_t active_entries) {
  expected<std::unique_ptr<OutputPool>, Error> created = make_unexpected(tensor_resource_error(
      "restart rate output recreation not attempted", "output", output.output().value()));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    created = output.recreate_unbound(plan, instance, allocation, active_entries);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(tensor_resource_error(
        "restart rate output recreation threw", "output", output.output().value()));
  }
#endif
  return created;
}

[[nodiscard]] expected<std::unique_ptr<TensorSchemaOutputPool>, Error> recreate_restart_output(
    const ExecutionPlanState& plan, TensorSchemaOutputPool& output,
    RuntimeOperatorInstance& instance) {
  expected<std::unique_ptr<TensorSchemaOutputPool>, Error> created =
      make_unexpected(tensor_resource_error("restart Tensor schema output recreation not attempted",
                                            "output",
                                            output.output().value()));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    created = output.recreate_unbound(plan, instance);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(tensor_resource_error(
        "restart Tensor schema output recreation threw", "output", output.output().value()));
  }
#endif
  return created;
}

template <typename Pool>
// Restart preflight, replacement creation, and rollback authority are one cold transaction. Keep
// their branches together so every failure visibly reopens the original binding.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> prepare_restart_family(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    std::span<const OperatorInstanceId> members, LifecycleStage from_stage, bool recreate,
    std::vector<std::unique_ptr<Pool>>& outputs,
    std::vector<PreparedRestartOutput<Pool>>& prepared_outputs) {
  for (std::size_t index = 0U; index < outputs.size(); ++index) {
    auto& output = outputs[index];
    if (std::ranges::find(members, output->owner()) == members.end()) {
      continue;
    }
    // Each template instantiation mutates exactly one discriminator; the other remains the
    // canonical zero value used by the common prepared-output record.
    // NOLINTNEXTLINE(misc-const-correctness)
    ProviderAllocationPlanRecordId rate_provider{};
    // NOLINTNEXTLINE(misc-const-correctness)
    std::uint32_t rate_provider_count{};
    bool rate_sensitive{};
    if constexpr (std::is_same_v<Pool, OutputPool>) {
      auto selected = rate_sensitive_tensor_provider(plan, *output, from_stage);
      if (!selected) {
        return make_unexpected(std::move(selected.error()));
      }
      rate_provider = *selected;
      rate_sensitive = static_cast<bool>(rate_provider);
    } else {
      auto replayed = output->replayed_rate_provider_count(plan, from_stage);
      if (!replayed || *replayed > std::numeric_limits<std::uint32_t>::max()) {
        return make_unexpected(
            replayed ? tensor_resource_error("Tensor schema rate provider count overflow")
                     : std::move(replayed.error()));
      }
      rate_provider_count = static_cast<std::uint32_t>(*replayed);
      rate_sensitive = rate_provider_count != 0U;
    }
    auto fenced = output->begin_restart();
    if (!fenced) {
      return make_unexpected(std::move(fenced.error()));
    }
    std::unique_ptr<Pool> candidate;
    if (recreate && !rate_sensitive) {
      auto* const instance = restart_operator(operators, output->owner());
      if (instance == nullptr) {
        static_cast<void>(output->reopen());
        return make_unexpected(tensor_resource_error(
            "restart operator inventory mismatch", "output", output->output().value()));
      }
      auto created = recreate_restart_output(plan, *output, *instance);
      if (!created) {
        static_cast<void>(output->reopen());
        return make_unexpected(std::move(created.error()));
      }
      candidate = std::move(*created);
    }
    prepared_outputs.push_back(PreparedRestartOutput<Pool>{
        .output_index = index,
        .candidate = std::move(candidate),
        .rate_provider = rate_provider,
        .rate_provider_count = rate_provider_count,
        .rate_sensitive = rate_sensitive,
        .replacement_allowed = recreate,
    });
  }
  return {};
}

template <typename Pool>
void rollback_prepared_family(std::vector<std::unique_ptr<Pool>>& outputs,
                              std::vector<PreparedRestartOutput<Pool>>& prepared_outputs) noexcept {
  for (const auto& prepared : prepared_outputs) {
    static_cast<void>(outputs[prepared.output_index]->reopen());
  }
  prepared_outputs.clear();
}

template <typename Pool>
[[nodiscard]] expected<void, Error> preflight_restart_family(
    std::vector<std::unique_ptr<Pool>>& outputs,
    const std::vector<PreparedRestartOutput<Pool>>& prepared_outputs) noexcept {
  for (const auto& prepared : prepared_outputs) {
    bool ready{};
    if constexpr (std::is_same_v<Pool, OutputPool>) {
      if (prepared.rate_provider) {
        ready = prepared.rate_prepared &&
                (prepared.candidate != nullptr
                     ? outputs[prepared.output_index]->restart_publication_ready()
                     : outputs[prepared.output_index]->restart_reuse_publication_ready(
                           prepared.active_entries));
      } else {
        ready = outputs[prepared.output_index]->restart_publication_ready();
      }
    } else {
      if (prepared.rate_sensitive) {
        const auto active_entries =
            std::span<const std::uint32_t>{prepared.field_active_entries}.first(
                prepared.rate_provider_count);
        ready = prepared.candidate == nullptr && prepared.rate_prepared &&
                prepared.rate_provider_count == outputs[prepared.output_index]->field_count() &&
                outputs[prepared.output_index]->restart_rate_publication_ready(active_entries);
      } else {
        ready = outputs[prepared.output_index]->restart_publication_ready();
      }
    }
    if (!ready) {
      return make_unexpected(
          tensor_resource_error("restart publication preflight failed",
                                "output",
                                outputs[prepared.output_index]->output().value()));
    }
  }
  return {};
}

void bind_restart_output(OutputPool& output) noexcept {
  output.bind();
}

void bind_restart_output(TensorSchemaOutputPool& output) noexcept {
  output.bind_output();
}

template <typename Pool>
void publish_restart_family(std::vector<std::unique_ptr<Pool>>& outputs,
                            std::vector<std::unique_ptr<Pool>>& retired_outputs,
                            std::vector<PreparedRestartOutput<Pool>>& prepared_outputs) noexcept {
  for (auto& prepared : prepared_outputs) {
    auto& predecessor = outputs[prepared.output_index];
    if (prepared.candidate != nullptr) {
      predecessor->detach();
      bind_restart_output(*prepared.candidate);
      retired_outputs.push_back(std::move(predecessor));
      predecessor = std::move(prepared.candidate);
    } else {
      if constexpr (std::is_same_v<Pool, OutputPool>) {
        if (prepared.rate_provider) {
          predecessor->publish_rate_reuse(prepared.active_entries, prepared.action);
          continue;
        }
      } else {
        if (prepared.rate_sensitive) {
          predecessor->publish_rate_reuse(
              std::span<const std::uint32_t>{prepared.field_active_entries}.first(
                  prepared.rate_provider_count),
              std::span<const RuntimeTensorPoolAllocationAction>{prepared.field_actions}.first(
                  prepared.rate_provider_count));
          continue;
        }
      }
      predecessor->publish_reopen();
    }
  }
  prepared_outputs.clear();
}

template <typename Pool>
[[nodiscard]] expected<void, Error> rollback_restart_family(
    std::vector<std::unique_ptr<Pool>>& outputs,
    std::vector<PreparedRestartOutput<Pool>>& prepared_outputs) noexcept {
  for (auto& prepared : prepared_outputs) {
    auto reopened = outputs[prepared.output_index]->reopen();
    if (!reopened) {
      return make_unexpected(std::move(reopened.error()));
    }
    prepared.candidate.reset();
  }
  return {};
}

}  // namespace

class RuntimeTensorOutputResources::Impl final {
 public:
  const ExecutionPlanState* plan{};
  std::span<RuntimeOperatorInstance> operators;
  RunGenerationId run_generation{};
  std::vector<std::unique_ptr<OutputPool>> outputs;
  std::vector<std::unique_ptr<TensorSchemaOutputPool>> tensor_schema_outputs;
  std::vector<std::unique_ptr<OutputPool>> retired_outputs;
  std::vector<std::unique_ptr<TensorSchemaOutputPool>> retired_tensor_schema_outputs;
  std::vector<PreparedRestartOutput<OutputPool>> restart_outputs;
  std::vector<PreparedRestartOutput<TensorSchemaOutputPool>> restart_tensor_schema_outputs;
  bool restart_active{};
};

RuntimeTensorOutputResources::RuntimeTensorOutputResources(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeTensorOutputResources::~RuntimeTensorOutputResources() {
  close();
  detach_ports();
}

expected<std::unique_ptr<RuntimeTensorOutputResources>, Error> RuntimeTensorOutputResources::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    const RuntimeCudaExecutionResources& cuda_resources,
    std::span<const RuntimeTensorDeviceResource> devices, RunGenerationId run_generation,
    ProcessGroupId process_group,
    const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
    RuntimeRateMaterializationView materialization,
    RuntimeTensorOutputMaterializationPlan materialization_plan) {
  auto impl = std::make_unique<Impl>();
  impl->plan = &plan;
  impl->operators = operators;
  impl->run_generation = run_generation;
  auto& allocation_inventory = materialization_plan.allocations;
  impl->restart_outputs.reserve(allocation_inventory.size());
  impl->restart_tensor_schema_outputs.reserve(plan.tensor_field_allocations.size());
  if (plan.lifecycle_transaction_capacity) {
    const std::size_t receipt_capacity =
        plan.lifecycle_transaction_capacity->configured.max_receipts;
    impl->retired_outputs.reserve(allocation_inventory.size() * receipt_capacity);
    impl->retired_tensor_schema_outputs.reserve(plan.tensor_field_allocations.size() *
                                                receipt_capacity);
  }
  auto output_pools = materialize_output_pools(plan,
                                               operators,
                                               cuda_resources,
                                               devices,
                                               allocation_inventory,
                                               materialization,
                                               run_generation,
                                               process_group,
                                               test_control,
                                               impl->outputs);
  if (!output_pools) {
    return make_unexpected(std::move(output_pools.error()));
  }
  auto tensor_schema_output_pools =
      materialize_tensor_schema_output_pools(plan,
                                             operators,
                                             cuda_resources,
                                             devices,
                                             run_generation,
                                             process_group,
                                             test_control,
                                             impl->tensor_schema_outputs);
  if (!tensor_schema_output_pools) {
    return make_unexpected(std::move(tensor_schema_output_pools.error()));
  }
  return std::unique_ptr<RuntimeTensorOutputResources>{
      new RuntimeTensorOutputResources{std::move(impl)}};
}

expected<RuntimeTensorOutputMaterializationPlan, Error> plan_runtime_tensor_output_materialization(
    const ExecutionPlanState& plan, RunGenerationId run_generation, ProcessGroupId process_group,
    RuntimeRateMaterializationView materialization) {
  const auto worker_owns_output = [&](const PortId output) noexcept {
    const auto* port = find_port_by_id_and_direction(plan, output, PortDirection::kOutput);
    return port != nullptr && port->owner_kind == PortOwnerKind::kOperator &&
           worker_owns_vertex(plan, process_group, port->owner);
  };
  const std::size_t lifecycle_candidate_count = static_cast<std::size_t>(
      std::ranges::count_if(plan.provider_allocation_plans, [&](const auto& provider) {
        return provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
               provider.allocation_kind == ProviderAllocationKind::kTensorOutput &&
               provider.disposition == PlanDisposition::kAccepted && provider.output &&
               worker_owns_output(*provider.output);
      }));
  if (lifecycle_candidate_count != materialization.tensor_allocations.size()) {
    return make_unexpected(tensor_resource_error("lifecycle candidate inventory mismatch"));
  }
  RuntimeTensorOutputMaterializationPlan result;
  auto& allocation_inventory = result.allocations;
  allocation_inventory.reserve(plan.tensor_allocations.size() +
                               materialization.tensor_allocations.size());
  std::ranges::copy_if(
      plan.tensor_allocations,
      std::back_inserter(allocation_inventory),
      [&](const auto& allocation) { return worker_owns_output(allocation.output); });
  auto lifecycle_rate_generation = RateGenerationId{};
  for (const auto& generated : materialization.tensor_allocations) {
    const auto provider = std::ranges::find(
        plan.provider_allocation_plans, generated.provider, &ProviderAllocationPlanRecord::id);
    if (provider == plan.provider_allocation_plans.end() ||
        std::ranges::count(plan.provider_allocation_plans,
                           generated.provider,
                           &ProviderAllocationPlanRecord::id) != 1 ||
        provider->geometry_mode != AllocationGeometryMode::kLifecycleCandidate ||
        provider->allocation_kind != ProviderAllocationKind::kTensorOutput || !provider->output) {
      return make_unexpected(tensor_resource_error("lifecycle candidate inventory mismatch"));
    }
    // The provider inventory check above proves the output identity is present.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const PortId provider_output = provider->output.value();
    if (!worker_owns_output(provider_output) || generated.run_generation != run_generation ||
        generated.rate_generation == 0U || provider->pool_id != generated.allocation.pool_id ||
        provider_output != generated.allocation.output) {
      return make_unexpected(tensor_resource_error("lifecycle candidate inventory mismatch"));
    }
    auto admitted =
        validate_lifecycle_candidate_retained_contract(plan, *provider, generated.allocation);
    if (!admitted) {
      return make_unexpected(std::move(admitted.error()));
    }
    if (lifecycle_rate_generation == 0U) {
      lifecycle_rate_generation = generated.rate_generation;
    } else if (lifecycle_rate_generation != generated.rate_generation) {
      return make_unexpected(tensor_resource_error("lifecycle candidate generation mismatch"));
    }
    allocation_inventory.push_back(generated.allocation);
  }
  if (plan.lifecycle_transaction_capacity) {
    const std::size_t receipt_capacity =
        plan.lifecycle_transaction_capacity->configured.max_receipts;
    if (receipt_capacity != 0U &&
        (allocation_inventory.size() > std::numeric_limits<std::size_t>::max() / receipt_capacity ||
         plan.tensor_field_allocations.size() >
             std::numeric_limits<std::size_t>::max() / receipt_capacity)) {
      return make_unexpected(tensor_resource_error("restart retirement capacity overflow"));
    }
  }
  return result;
}

void RuntimeTensorOutputResources::close() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  close_output_family(impl_->outputs);
  close_output_family(impl_->tensor_schema_outputs);
  close_output_family(impl_->retired_outputs);
  close_output_family(impl_->retired_tensor_schema_outputs);
}

expected<bool, Error> RuntimeTensorOutputResources::preflight_owner_terminal_close(
    const OperatorInstanceId owner) noexcept {
  if (impl_ == nullptr || owner == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto ready = true;
  const auto preflight_outputs = [&ready, owner](auto& outputs) -> expected<void, Error> {
    for (auto& output : outputs) {
      if (output->owner() != owner) {
        continue;
      }
      auto preflight = output->preflight_terminal_close();
      if (!preflight) {
        return make_unexpected(std::move(preflight).error());
      }
      if (!*preflight) {
        ready = false;
      }
    }
    return {};
  };
  if (auto preflight = preflight_outputs(impl_->outputs); !preflight) {
    return make_unexpected(std::move(preflight).error());
  }
  if (auto preflight = preflight_outputs(impl_->tensor_schema_outputs); !preflight) {
    return make_unexpected(std::move(preflight).error());
  }
  if (auto preflight = preflight_outputs(impl_->retired_outputs); !preflight) {
    return make_unexpected(std::move(preflight).error());
  }
  if (auto preflight = preflight_outputs(impl_->retired_tensor_schema_outputs); !preflight) {
    return make_unexpected(std::move(preflight).error());
  }
  return expected<bool, Error>{ready};
}

expected<bool, Error> RuntimeTensorOutputResources::advance_owner_terminal_retirement(
    const OperatorInstanceId owner) noexcept {
  if (impl_ == nullptr || owner == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto complete = true;
  const auto advance_outputs = [&complete, owner](auto& outputs) -> expected<void, Error> {
    for (auto& output : outputs) {
      if (output->owner() != owner) {
        continue;
      }
      auto advanced = output->advance_terminal_retirement();
      if (!advanced) {
        return make_unexpected(std::move(advanced).error());
      }
      if (!*advanced) {
        complete = false;
      }
    }
    return {};
  };
  if (auto advanced = advance_outputs(impl_->outputs); !advanced) {
    return make_unexpected(std::move(advanced).error());
  }
  if (auto advanced = advance_outputs(impl_->tensor_schema_outputs); !advanced) {
    return make_unexpected(std::move(advanced).error());
  }
  if (auto advanced = advance_outputs(impl_->retired_outputs); !advanced) {
    return make_unexpected(std::move(advanced).error());
  }
  if (auto advanced = advance_outputs(impl_->retired_tensor_schema_outputs); !advanced) {
    return make_unexpected(std::move(advanced).error());
  }
  return expected<bool, Error>{complete};
}

void RuntimeTensorOutputResources::detach_ports() noexcept {
  if (impl_ == nullptr) {
    return;
  }
  detach_output_family(impl_->outputs);
  detach_output_family(impl_->tensor_schema_outputs);
  detach_output_family(impl_->retired_outputs);
  detach_output_family(impl_->retired_tensor_schema_outputs);
}

// Startup rate publication is a three-phase transaction across direct and schema output families;
// keeping the phases together makes the no-publication-before-complete-preflight invariant clear.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeTensorOutputResources::prepare_rate_candidate_startup(
    RuntimeRateMaterializationView materialization) noexcept {
  if (impl_ == nullptr || impl_->plan == nullptr || impl_->restart_active ||
      impl_->run_generation == 0U) {
    return make_unexpected(tensor_resource_error("startup rate candidate state invalid"));
  }
  enum class StartupRatePhase : std::uint8_t { kPreflight, kRecord, kPublish };
  auto phase_rate_generation = RateGenerationId{};
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  const auto prepare_direct = [&](const StartupRatePhase phase) -> expected<void, Error> {
    for (auto& output : impl_->outputs) {
      auto selected = tensor_output_provider(*impl_->plan, *output);
      if (!selected) {
        return make_unexpected(std::move(selected.error()));
      }
      const ProviderAllocationPlanRecordId provider_id = *selected;
      if (!provider_id) {
        continue;
      }
      const auto provider = std::ranges::find(
          impl_->plan->provider_allocation_plans, provider_id, &ProviderAllocationPlanRecord::id);
      if (provider == impl_->plan->provider_allocation_plans.end()) {
        return make_unexpected(tensor_resource_error("startup Tensor provider missing"));
      }
      auto lifecycle_selected = provider_binding_replayed(*impl_->plan, *provider);
      if (!lifecycle_selected) {
        return make_unexpected(std::move(lifecycle_selected.error()));
      }
      const auto capacity = std::ranges::find(
          materialization.pool_capacities, provider_id, &RuntimePoolCapacityGeneration::provider);
      const auto result = std::ranges::find(materialization.provider_results,
                                            provider_id,
                                            &RuntimeProviderMaterializationResult::provider);
      if (capacity == materialization.pool_capacities.end() ||
          std::ranges::count(materialization.pool_capacities,
                             provider_id,
                             &RuntimePoolCapacityGeneration::provider) != 1 ||
          result == materialization.provider_results.end() ||
          std::ranges::count(materialization.provider_results,
                             provider_id,
                             &RuntimeProviderMaterializationResult::provider) != 1 ||
          capacity->run_generation != impl_->run_generation || capacity->rate_generation == 0U ||
          (capacity->admitted_entry_ceiling &&
           capacity->required_entries > *capacity->admitted_entry_ceiling) ||
          (capacity->required_entries == 0U && result->reported)) {
        return make_unexpected(tensor_resource_error("startup rate candidate identity invalid"));
      }
      if (phase_rate_generation == 0U) {
        phase_rate_generation = capacity->rate_generation;
      } else if (phase_rate_generation != capacity->rate_generation) {
        return make_unexpected(tensor_resource_error("startup rate generation mismatch"));
      }
      if (capacity->required_entries == 0U) {
        if (!*lifecycle_selected) {
          return make_unexpected(tensor_resource_error("startup Tensor capacity unresolved"));
        }
        continue;
      }
      if (capacity->required_entries > output->allocation().slot_count) {
        return make_unexpected(
            tensor_resource_error("startup Tensor active-slot limit exceeds backing",
                                  "output",
                                  output->output().value()));
      }
      if (phase == StartupRatePhase::kPreflight) {
        if (*lifecycle_selected) {
          auto preflight = output->preflight_startup_active_slots(capacity->required_entries);
          if (!preflight) {
            return make_unexpected(std::move(preflight.error()));
          }
        }
        auto preflight = report_provider_materialization(materialization,
                                                         provider_id,
                                                         RuntimeTensorPoolAllocationAction::kFresh,
                                                         output->physical_slot_count(),
                                                         true);
        if (!preflight) {
          return make_unexpected(std::move(preflight.error()));
        }
      } else if (phase == StartupRatePhase::kRecord) {
        auto reported = report_provider_materialization(
            materialization, provider_id, RuntimeTensorPoolAllocationAction::kFresh, *output);
        if (!reported) {
          return make_unexpected(std::move(reported.error()));
        }
      } else if (*lifecycle_selected) {
        output->publish_startup_active_slots(capacity->required_entries);
      }
    }
    return {};
  };

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  const auto prepare_schema = [&](const StartupRatePhase phase) -> expected<void, Error> {
    for (auto& output : impl_->tensor_schema_outputs) {
      auto lifecycle_provider_count = output->replayed_rate_provider_count(*impl_->plan);
      if (!lifecycle_provider_count) {
        return make_unexpected(std::move(lifecycle_provider_count.error()));
      }
      const std::size_t provider_count = output->field_count();
      if (provider_count == 0U) {
        continue;
      }
      if ((*lifecycle_provider_count != 0U && *lifecycle_provider_count != provider_count) ||
          provider_count > kMaximumTensorSchemaLoanFields) {
        return make_unexpected(tensor_resource_error("startup field provider inventory invalid"));
      }
      const bool lifecycle_selected = *lifecycle_provider_count != 0U;
      std::array<std::uint32_t, kMaximumTensorSchemaLoanFields> active_entries{};
      RateGenerationId family_rate_generation{};
      bool any_resolved{};
      bool any_unresolved{};
      for (std::size_t index = 0U; index < provider_count; ++index) {
        const ProviderAllocationPlanRecordId provider_id = output->field_provider(index);
        const auto capacity = std::ranges::find(
            materialization.pool_capacities, provider_id, &RuntimePoolCapacityGeneration::provider);
        const auto result = std::ranges::find(materialization.provider_results,
                                              provider_id,
                                              &RuntimeProviderMaterializationResult::provider);
        if (!provider_id || capacity == materialization.pool_capacities.end() ||
            std::ranges::count(materialization.pool_capacities,
                               provider_id,
                               &RuntimePoolCapacityGeneration::provider) != 1 ||
            result == materialization.provider_results.end() ||
            std::ranges::count(materialization.provider_results,
                               provider_id,
                               &RuntimeProviderMaterializationResult::provider) != 1 ||
            capacity->run_generation != impl_->run_generation || capacity->rate_generation == 0U ||
            (capacity->required_entries == 0U && result->reported)) {
          return make_unexpected(tensor_resource_error("startup field rate candidate invalid"));
        }
        if (phase_rate_generation == 0U) {
          phase_rate_generation = capacity->rate_generation;
        } else if (phase_rate_generation != capacity->rate_generation) {
          return make_unexpected(tensor_resource_error("startup rate generation mismatch"));
        }
        if (family_rate_generation == 0U) {
          family_rate_generation = capacity->rate_generation;
        } else if (family_rate_generation != capacity->rate_generation) {
          return make_unexpected(tensor_resource_error("startup field rate generation mismatch"));
        }
        if (capacity->required_entries == 0U) {
          if (!lifecycle_selected) {
            return make_unexpected(tensor_resource_error("startup field capacity unresolved"));
          }
          any_unresolved = true;
          continue;
        }
        if (!capacity->admitted_entry_ceiling ||
            capacity->required_entries > *capacity->admitted_entry_ceiling ||
            *capacity->admitted_entry_ceiling > output->field_physical_slot_count(index)) {
          return make_unexpected(tensor_resource_error("startup field rate candidate invalid"));
        }
        any_resolved = true;
        active_entries[index] = capacity->required_entries;
        if (phase == StartupRatePhase::kPreflight) {
          auto preflight =
              report_provider_materialization(materialization,
                                              provider_id,
                                              RuntimeTensorPoolAllocationAction::kFresh,
                                              output->field_physical_slot_count(index),
                                              true);
          if (!preflight) {
            return make_unexpected(std::move(preflight.error()));
          }
        }
      }
      if (any_resolved && any_unresolved) {
        return make_unexpected(tensor_resource_error("partial startup field rate candidate"));
      }
      if (!any_resolved) {
        continue;
      }
      const auto selected_entries =
          std::span<const std::uint32_t>{active_entries}.first(provider_count);
      if (phase == StartupRatePhase::kPreflight) {
        if (lifecycle_selected) {
          auto preflight = output->preflight_startup_active_slots(selected_entries);
          if (!preflight) {
            return make_unexpected(std::move(preflight.error()));
          }
        }
      } else if (phase == StartupRatePhase::kRecord) {
        for (std::size_t index = 0U; index < provider_count; ++index) {
          auto reported = report_provider_materialization(materialization,
                                                          output->field_provider(index),
                                                          RuntimeTensorPoolAllocationAction::kFresh,
                                                          output->field_physical_slot_count(index));
          if (!reported) {
            return make_unexpected(std::move(reported.error()));
          }
        }
      } else if (lifecycle_selected) {
        output->publish_startup_active_slots(selected_entries);
      }
    }
    return {};
  };

  constexpr std::array phases{
      StartupRatePhase::kPreflight, StartupRatePhase::kRecord, StartupRatePhase::kPublish};
  for (const StartupRatePhase phase : phases) {
    phase_rate_generation = 0U;
    if (auto prepared = prepare_direct(phase); !prepared) {
      return make_unexpected(std::move(prepared.error()));
    }
    if (auto prepared = prepare_schema(phase); !prepared) {
      return make_unexpected(std::move(prepared.error()));
    }
  }
  return {};
}

// Restart fencing is one transaction across active, retired, and schema Tensor resources.
expected<void, Error> RuntimeTensorOutputResources::begin_restart(
    std::span<const OperatorInstanceId> members, LifecycleStage from_stage) noexcept {
  if (impl_ == nullptr || impl_->plan == nullptr || impl_->restart_active || members.empty() ||
      !is_lifecycle_forward_stage(from_stage)) {
    return make_unexpected(tensor_resource_error("restart request state invalid"));
  }
  impl_->restart_active = true;
  impl_->restart_outputs.clear();
  impl_->restart_tensor_schema_outputs.clear();
  const bool recreate =
      static_cast<std::uint8_t>(from_stage) <= static_cast<std::uint8_t>(LifecycleStage::kAllocate);
  const std::size_t affected_count = affected_output_count(impl_->outputs, members);
  const std::size_t affected_tensor_schema_count =
      affected_output_count(impl_->tensor_schema_outputs, members);
  std::size_t affected_tensor_schema_replacement_count{};
  if (recreate) {
    for (const auto& output : impl_->tensor_schema_outputs) {
      if (std::ranges::find(members, output->owner()) == members.end()) {
        continue;
      }
      auto replayed = output->replayed_rate_provider_count(*impl_->plan, from_stage);
      if (!replayed) {
        impl_->restart_active = false;
        return make_unexpected(std::move(replayed.error()));
      }
      affected_tensor_schema_replacement_count += static_cast<std::size_t>(*replayed == 0U);
    }
  }
  const bool prepared_capacity_exhausted =
      affected_count > impl_->restart_outputs.capacity() ||
      affected_tensor_schema_count > impl_->restart_tensor_schema_outputs.capacity();
  const bool retirement_capacity_exhausted =
      recreate &&
      (affected_count > impl_->retired_outputs.capacity() - impl_->retired_outputs.size() ||
       affected_tensor_schema_replacement_count > impl_->retired_tensor_schema_outputs.capacity() -
                                                      impl_->retired_tensor_schema_outputs.size());
  if (prepared_capacity_exhausted) {
    impl_->restart_active = false;
    return make_unexpected(tensor_resource_error("restart preparation capacity exhausted"));
  }
  if (retirement_capacity_exhausted) {
    impl_->restart_active = false;
    return make_unexpected(tensor_resource_error("restart retirement capacity exhausted"));
  }

  const auto rollback_prepared = [&]() noexcept {
    rollback_prepared_family(impl_->outputs, impl_->restart_outputs);
    rollback_prepared_family(impl_->tensor_schema_outputs, impl_->restart_tensor_schema_outputs);
    impl_->restart_active = false;
  };
  auto direct = prepare_restart_family(*impl_->plan,
                                       impl_->operators,
                                       members,
                                       from_stage,
                                       recreate,
                                       impl_->outputs,
                                       impl_->restart_outputs);
  if (!direct) {
    rollback_prepared();
    return make_unexpected(std::move(direct.error()));
  }
  auto schema = prepare_restart_family(*impl_->plan,
                                       impl_->operators,
                                       members,
                                       from_stage,
                                       recreate,
                                       impl_->tensor_schema_outputs,
                                       impl_->restart_tensor_schema_outputs);
  if (!schema) {
    rollback_prepared();
    return make_unexpected(std::move(schema.error()));
  }
  return {};
}

// Restart candidate preparation validates and records every replacement before later commit or
// rollback. Splitting the family transaction would duplicate its cleanup authority.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeTensorOutputResources::prepare_rate_candidate_restart(
    RuntimeRateMaterializationView materialization) noexcept {
  if (impl_ == nullptr || impl_->plan == nullptr || !impl_->restart_active ||
      impl_->run_generation == 0U) {
    return make_unexpected(tensor_resource_error("restart rate candidate state invalid"));
  }
  const auto discard_prepared = [&]() noexcept {
    for (auto& prepared : impl_->restart_outputs) {
      if (prepared.rate_provider) {
        prepared.candidate.reset();
        prepared.run_generation = 0U;
        prepared.rate_generation = 0U;
        prepared.active_entries = 0U;
        prepared.action = RuntimeTensorPoolAllocationAction::kFresh;
        prepared.rate_prepared = false;
      }
    }
    for (auto& prepared : impl_->restart_tensor_schema_outputs) {
      if (prepared.rate_sensitive) {
        prepared.candidate.reset();
        prepared.run_generation = 0U;
        prepared.rate_generation = 0U;
        prepared.field_active_entries.fill(0U);
        prepared.field_actions.fill(RuntimeTensorPoolAllocationAction::kFresh);
        prepared.rate_prepared = false;
      }
    }
  };
  const auto fail = [&](Error error) -> expected<void, Error> {
    discard_prepared();
    return make_unexpected(std::move(error));
  };

  for (auto& prepared : impl_->restart_outputs) {
    if (!prepared.rate_provider) {
      continue;
    }
    if (prepared.output_index >= impl_->outputs.size()) {
      return fail(tensor_resource_error("restart rate candidate inventory invalid"));
    }
    const auto capacity = std::ranges::find(materialization.pool_capacities,
                                            prepared.rate_provider,
                                            &RuntimePoolCapacityGeneration::provider);
    if (capacity == materialization.pool_capacities.end() ||
        std::ranges::count(materialization.pool_capacities,
                           prepared.rate_provider,
                           &RuntimePoolCapacityGeneration::provider) != 1 ||
        capacity->run_generation != impl_->run_generation || capacity->rate_generation == 0U ||
        (capacity->admitted_entry_ceiling &&
         capacity->required_entries > *capacity->admitted_entry_ceiling)) {
      return fail(tensor_resource_error("restart rate candidate identity invalid"));
    }
    if (capacity->required_entries == 0U) {
      continue;
    }
    auto& predecessor = impl_->outputs[prepared.output_index];
    const auto provider = std::ranges::find(impl_->plan->provider_allocation_plans,
                                            prepared.rate_provider,
                                            &ProviderAllocationPlanRecord::id);
    if (provider == impl_->plan->provider_allocation_plans.end() ||
        std::ranges::count(impl_->plan->provider_allocation_plans,
                           prepared.rate_provider,
                           &ProviderAllocationPlanRecord::id) != 1 ||
        provider->allocation_kind != ProviderAllocationKind::kTensorOutput ||
        provider->disposition != PlanDisposition::kAccepted ||
        provider->output != predecessor->output() ||
        provider->pool_id != predecessor->allocation().pool_id) {
      return fail(tensor_resource_error("restart rate candidate provider mismatch"));
    }

    const TensorAllocationPlanRecord* allocation{};
    const RuntimeTensorAllocationGeneration* generated{};
    if (provider->geometry_mode == AllocationGeometryMode::kLifecycleCandidate) {
      const auto candidate = std::ranges::find(materialization.tensor_allocations,
                                               prepared.rate_provider,
                                               &RuntimeTensorAllocationGeneration::provider);
      if (candidate == materialization.tensor_allocations.end() ||
          std::ranges::count(materialization.tensor_allocations,
                             prepared.rate_provider,
                             &RuntimeTensorAllocationGeneration::provider) != 1 ||
          candidate->run_generation != capacity->run_generation ||
          candidate->rate_generation != capacity->rate_generation ||
          candidate->allocation.slot_count != capacity->required_entries) {
        return fail(tensor_resource_error("restart rate candidate allocation missing"));
      }
      generated = &*candidate;
      allocation = &candidate->allocation;
    } else if (provider->geometry_mode == AllocationGeometryMode::kCompileConcrete) {
      const auto candidate =
          std::ranges::find_if(impl_->plan->tensor_allocations, [&](const auto& row) {
            return row.output == predecessor->output() && row.pool_id == provider->pool_id;
          });
      if (candidate == impl_->plan->tensor_allocations.end() ||
          std::ranges::count_if(impl_->plan->tensor_allocations,
                                [&](const auto& row) {
                                  return row.output == predecessor->output() &&
                                         row.pool_id == provider->pool_id;
                                }) != 1 ||
          capacity->required_entries > candidate->slot_count) {
        return fail(tensor_resource_error("restart rate candidate allocation missing"));
      }
      allocation = &*candidate;
    } else {
      return fail(tensor_resource_error("restart rate candidate provider mismatch"));
    }

    if (provider->geometry_mode == AllocationGeometryMode::kLifecycleCandidate) {
      auto admitted =
          validate_lifecycle_candidate_retained_contract(*impl_->plan, *provider, *allocation);
      if (!admitted) {
        return fail(std::move(admitted.error()));
      }
    }

    auto contract =
        generated != nullptr
            ? validate_output_plan_contract(*impl_->plan, *allocation, materialization, generated)
            : validate_output_plan_contract(*impl_->plan, *allocation);
    if (!contract) {
      return fail(std::move(contract.error()));
    }
    if (prepared.rate_prepared) {
      if (prepared.run_generation != capacity->run_generation ||
          prepared.rate_generation != capacity->rate_generation ||
          prepared.active_entries != capacity->required_entries) {
        return fail(tensor_resource_error("restart rate candidate binding drift"));
      }
      const OutputPool& materialized =
          prepared.candidate != nullptr ? *prepared.candidate : *predecessor;
      if (auto reported = report_provider_materialization(
              materialization, prepared.rate_provider, prepared.action, materialized);
          !reported) {
        return fail(std::move(reported.error()));
      }
      continue;
    }

    auto reuse_permitted = provider_reuse_permitted(*impl_->plan, *provider);
    if (!reuse_permitted) {
      return fail(std::move(reuse_permitted.error()));
    }
    auto reusable = predecessor->can_reuse_retained(*allocation, capacity->required_entries);
    if (!reusable) {
      return fail(std::move(reusable.error()));
    }
    if (*reuse_permitted && *reusable) {
      prepared.action = predecessor->allocation().slot_count == capacity->required_entries
                            ? RuntimeTensorPoolAllocationAction::kReusedExact
                            : RuntimeTensorPoolAllocationAction::kReusedLarger;
    } else {
      if (!capacity->admitted_entry_ceiling) {
        return fail(Error{ErrorCode::kResourceExhausted, "QUEUE_REPLACEMENT_NOT_ADMITTED"});
      }
      if (!prepared.replacement_allowed) {
        return fail(Error{ErrorCode::kResourceExhausted, "QUEUE_REPLACEMENT_NOT_ADMITTED"});
      }
      RuntimeOperatorInstance* const instance =
          restart_operator(impl_->operators, predecessor->owner());
      if (instance == nullptr) {
        return fail(tensor_resource_error(
            "restart operator inventory mismatch", "output", predecessor->output().value()));
      }
      auto candidate = recreate_restart_output(
          *impl_->plan, *predecessor, *instance, *allocation, capacity->required_entries);
      if (!candidate) {
        return fail(std::move(candidate.error()));
      }
      prepared.candidate = std::move(*candidate);
      prepared.action = RuntimeTensorPoolAllocationAction::kReplaced;
    }
    prepared.run_generation = capacity->run_generation;
    prepared.rate_generation = capacity->rate_generation;
    prepared.active_entries = capacity->required_entries;
    prepared.rate_prepared = true;
    const OutputPool& materialized =
        prepared.candidate != nullptr ? *prepared.candidate : *predecessor;
    if (auto reported = report_provider_materialization(
            materialization, prepared.rate_provider, prepared.action, materialized);
        !reported) {
      return fail(std::move(reported.error()));
    }
  }

  for (auto& prepared : impl_->restart_tensor_schema_outputs) {
    if (!prepared.rate_sensitive) {
      continue;
    }
    if (prepared.output_index >= impl_->tensor_schema_outputs.size() ||
        prepared.candidate != nullptr || prepared.rate_provider_count == 0U ||
        prepared.rate_provider_count > kMaximumTensorSchemaLoanFields) {
      return fail(tensor_resource_error("restart field rate candidate inventory invalid"));
    }
    auto& predecessor = impl_->tensor_schema_outputs[prepared.output_index];
    if (prepared.rate_provider_count != predecessor->field_count()) {
      return fail(tensor_resource_error("restart field rate provider inventory invalid"));
    }

    std::array<std::uint32_t, kMaximumTensorSchemaLoanFields> active_entries{};
    std::array<RuntimeTensorPoolAllocationAction, kMaximumTensorSchemaLoanFields> actions{};
    RateGenerationId rate_generation{};
    bool any_resolved{};
    bool any_unresolved{};
    for (std::size_t index = 0U; index < prepared.rate_provider_count; ++index) {
      const ProviderAllocationPlanRecordId provider_id = predecessor->field_provider(index);
      const auto provider = std::ranges::find(
          impl_->plan->provider_allocation_plans, provider_id, &ProviderAllocationPlanRecord::id);
      const auto capacity = std::ranges::find(
          materialization.pool_capacities, provider_id, &RuntimePoolCapacityGeneration::provider);
      if (!provider_id || provider == impl_->plan->provider_allocation_plans.end() ||
          std::ranges::count(impl_->plan->provider_allocation_plans,
                             provider_id,
                             &ProviderAllocationPlanRecord::id) != 1 ||
          provider->allocation_kind != ProviderAllocationKind::kTensorFieldOutput ||
          provider->geometry_mode != AllocationGeometryMode::kCompileConcrete ||
          provider->disposition != PlanDisposition::kAccepted ||
          provider->output != predecessor->output() || provider->tensor_template ||
          capacity == materialization.pool_capacities.end() ||
          std::ranges::count(materialization.pool_capacities,
                             provider_id,
                             &RuntimePoolCapacityGeneration::provider) != 1 ||
          capacity->run_generation != impl_->run_generation || capacity->rate_generation == 0U) {
        return fail(tensor_resource_error("restart field rate candidate invalid"));
      }
      if (rate_generation == 0U) {
        rate_generation = capacity->rate_generation;
      } else if (rate_generation != capacity->rate_generation) {
        return fail(tensor_resource_error("restart field rate generation mismatch"));
      }
      if (capacity->required_entries == 0U) {
        any_unresolved = true;
        continue;
      }
      if (!capacity->admitted_entry_ceiling ||
          capacity->required_entries > *capacity->admitted_entry_ceiling ||
          *capacity->admitted_entry_ceiling > predecessor->field_physical_slot_count(index)) {
        return fail(tensor_resource_error("restart field rate candidate invalid"));
      }
      auto reuse_permitted = provider_reuse_permitted(*impl_->plan, *provider);
      if (!reuse_permitted) {
        return fail(std::move(reuse_permitted.error()));
      }
      if (!*reuse_permitted) {
        return fail(Error{ErrorCode::kResourceExhausted, "QUEUE_REPLACEMENT_NOT_ADMITTED"});
      }
      any_resolved = true;
      active_entries[index] = capacity->required_entries;
      actions[index] = predecessor->field_physical_slot_count(index) == capacity->required_entries
                           ? RuntimeTensorPoolAllocationAction::kReusedExact
                           : RuntimeTensorPoolAllocationAction::kReusedLarger;
    }

    if (any_resolved && any_unresolved) {
      return fail(tensor_resource_error("partial restart field rate candidate"));
    }
    if (!any_resolved) {
      continue;
    }

    const auto selected_entries =
        std::span<const std::uint32_t>{active_entries}.first(prepared.rate_provider_count);
    if (prepared.rate_prepared) {
      if (prepared.run_generation != impl_->run_generation ||
          prepared.rate_generation != rate_generation ||
          !std::ranges::equal(selected_entries,
                              std::span<const std::uint32_t>{prepared.field_active_entries}.first(
                                  prepared.rate_provider_count)) ||
          !std::ranges::equal(
              std::span<const RuntimeTensorPoolAllocationAction>{actions}.first(
                  prepared.rate_provider_count),
              std::span<const RuntimeTensorPoolAllocationAction>{prepared.field_actions}.first(
                  prepared.rate_provider_count))) {
        return fail(tensor_resource_error("restart field rate candidate binding drift"));
      }
      for (std::size_t index = 0U; index < prepared.rate_provider_count; ++index) {
        auto reported =
            report_provider_materialization(materialization,
                                            predecessor->field_provider(index),
                                            actions[index],
                                            predecessor->field_physical_slot_count(index));
        if (!reported) {
          return fail(std::move(reported.error()));
        }
      }
      continue;
    }

    auto reusable = predecessor->can_reuse_retained(selected_entries);
    if (!reusable) {
      return fail(std::move(reusable.error()));
    }
    if (!*reusable) {
      return fail(Error{ErrorCode::kResourceExhausted, "QUEUE_REPLACEMENT_NOT_ADMITTED"});
    }
    for (std::size_t index = 0U; index < prepared.rate_provider_count; ++index) {
      auto preflight =
          report_provider_materialization(materialization,
                                          predecessor->field_provider(index),
                                          actions[index],
                                          predecessor->field_physical_slot_count(index),
                                          true);
      if (!preflight) {
        return fail(std::move(preflight.error()));
      }
    }
    for (std::size_t index = 0U; index < prepared.rate_provider_count; ++index) {
      auto reported =
          report_provider_materialization(materialization,
                                          predecessor->field_provider(index),
                                          actions[index],
                                          predecessor->field_physical_slot_count(index));
      if (!reported) {
        return fail(std::move(reported.error()));
      }
    }
    prepared.run_generation = impl_->run_generation;
    prepared.rate_generation = rate_generation;
    prepared.field_active_entries = active_entries;
    prepared.field_actions = actions;
    prepared.rate_prepared = true;
  }
  return {};
}

expected<void, Error> RuntimeTensorOutputResources::publish_restart() noexcept {
  auto preflight = preflight_restart_publication();
  if (!preflight) {
    return preflight;
  }
  publish_restart_family(impl_->outputs, impl_->retired_outputs, impl_->restart_outputs);
  publish_restart_family(impl_->tensor_schema_outputs,
                         impl_->retired_tensor_schema_outputs,
                         impl_->restart_tensor_schema_outputs);
  impl_->restart_active = false;
  return {};
}

expected<void, Error> RuntimeTensorOutputResources::preflight_restart_publication() noexcept {
  if (impl_ == nullptr || !impl_->restart_active) {
    return make_unexpected(tensor_resource_error("restart publication state invalid"));
  }
  auto direct = preflight_restart_family(impl_->outputs, impl_->restart_outputs);
  if (!direct) {
    return direct;
  }
  return preflight_restart_family(impl_->tensor_schema_outputs,
                                  impl_->restart_tensor_schema_outputs);
}

expected<void, Error> RuntimeTensorOutputResources::rollback_restart() noexcept {
  if (impl_ == nullptr || !impl_->restart_active) {
    return make_unexpected(tensor_resource_error("restart rollback state invalid"));
  }
  auto direct = rollback_restart_family(impl_->outputs, impl_->restart_outputs);
  if (!direct) {
    return direct;
  }
  auto schema =
      rollback_restart_family(impl_->tensor_schema_outputs, impl_->restart_tensor_schema_outputs);
  if (!schema) {
    return schema;
  }
  impl_->restart_outputs.clear();
  impl_->restart_tensor_schema_outputs.clear();
  impl_->restart_active = false;
  return {};
}

std::size_t RuntimeTensorOutputResources::output_pool_count() const noexcept {
  return impl_ != nullptr ? impl_->outputs.size() : 0U;
}

}  // namespace holoscan::detail
