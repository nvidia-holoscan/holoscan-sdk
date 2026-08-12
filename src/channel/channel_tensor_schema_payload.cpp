/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <holomq/buffer_pool.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/inbound_message.hpp>
#include <holomq/outbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/subscriber.hpp>
#include <holomq/wait_set.hpp>

#include "../core/domain/tensor_access.hpp"
#include "catalog_tensor_receive_control.hpp"
#include "channel_common.hpp"
#include "channel_payload_limits.hpp"
#include "channel_tensor_schema_payload.hpp"
#include "channel_tensor_sidecar_store.hpp"
#include "holomq_tensor.hpp"
#include "payload_descriptor_wire.hpp"

namespace holoscan {

using channel_internal::from_holomq_error;
using channel_internal::from_holomq_publication_result;
using channel_internal::from_holomq_result;
using channel_internal::InlinePayloadBucket;
using channel_internal::make_channel_error;
using channel_internal::PublisherPressureOptions;
using channel_internal::replaces_oldest_unread_sample;
using channel_internal::to_holomq_publisher_options;
using channel_internal::to_holomq_service_options;
using channel_internal::to_holomq_subscriber_options;

namespace {

[[nodiscard]] bool dtype_equal(const DLDataType lhs, const DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

[[nodiscard]] bool descriptor_offsets_unique(
    const std::span<const std::size_t> descriptor_offsets) noexcept {
  for (auto current = descriptor_offsets.begin(); current != descriptor_offsets.end(); ++current) {
    if (std::find(descriptor_offsets.begin(), current, *current) != current) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool tensor_matches_staging_memory(const DLDevice device,
                                                 const MemoryKind memory_kind,
                                                 const DeviceId planned_device) noexcept {
  switch (memory_kind) {
    case MemoryKind::kHost:
      return device.device_type == kDLCPU && device.device_id == 0 && !planned_device.valid();
    case MemoryKind::kPinnedHost:
      return device.device_type == kDLCUDAHost && device.device_id == 0 && planned_device.valid();
    case MemoryKind::kCudaDevice:
      return device.device_type == kDLCUDA && planned_device.valid() &&
             device.device_id == planned_device.value;
    default:
      return false;
  }
}

[[nodiscard]] expected<channel_internal::HoloMQTensorDescriptor, Error>
make_staged_tensor_descriptor(const Tensor& tensor, const holomq::BufferHandle& handle) {
  std::array<std::int64_t, channel_internal::kHoloMQTensorDescriptorMaxRank> byte_strides{};
  TensorByteStridesView byte_strides_view{std::nullopt};
  const auto dlpack_strides = tensor.dlpack_strides_span();
  if (dlpack_strides.size() > byte_strides.size()) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument, "Tensor schema rank exceeds capacity"));
  }
  if (!dlpack_strides.empty()) {
    const std::int64_t item_size = static_cast<std::int64_t>(tensor.itemsize());
    for (std::size_t index = 0U; index < dlpack_strides.size(); ++index) {
      byte_strides[index] = dlpack_strides[index] * item_size;
    }
    byte_strides_view = TensorByteStridesSpan{byte_strides.data(), dlpack_strides.size()};
  }
  return channel_internal::make_holomq_tensor_descriptor(
      tensor.shape_span(),
      tensor.dtype(),
      byte_strides_view,
      static_cast<std::uint64_t>(tensor.nbytes()),
      handle,
      /*read_only=*/true,
      tensor.device().device_type == kDLCUDA ? tensor.device().device_id : -1);
}

class ScopedCUDADevice {
 public:
  explicit ScopedCUDADevice(const DeviceId requested) noexcept {
    if (cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    selected_ = requested.valid() ? requested.value : previous_;
    if (selected_ == previous_ || cudaSetDevice(selected_) == cudaSuccess) {
      active_ = true;
    }
  }
  ScopedCUDADevice(const ScopedCUDADevice&) = delete;
  ScopedCUDADevice& operator=(const ScopedCUDADevice&) = delete;
  ~ScopedCUDADevice() {
    if (active_ && selected_ != previous_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }
  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_{};
  int selected_{};
  bool active_{};
};

struct TensorSchemaReceiveControlDeleter {
  void operator()(channel_internal::CatalogTensorReceiveControl* const control) const noexcept {
    if (control != nullptr) {
      control->release();
    }
  }
};

using TensorSchemaReceiveControlPtr = std::unique_ptr<channel_internal::CatalogTensorReceiveControl,
                                                      TensorSchemaReceiveControlDeleter>;

class TensorSchemaStagingPool final : public std::enable_shared_from_this<TensorSchemaStagingPool> {
 public:
  struct Slot {
    TensorSchemaStagingPool* owner{};
    std::size_t index{};
    holomq::Buffer buffer;
    cudaEvent_t copy_complete{};
    bool in_use{};
    bool awaiting_import_release{};
    bool copy_pending{};
  };

  struct Pending {
    Tensor tensor;
    Slot* slot{};
    cudaStream_t copy_stream{};
    bool device_copy_enqueued{};
    channel_internal::TensorSchemaPayloadRetainedState retained_state;
  };

  [[nodiscard]] static expected<std::shared_ptr<TensorSchemaStagingPool>, Error> create(
      [[maybe_unused]] holomq::Context& context,
      const channel_internal::TensorSchemaStagingOptions& options) {
    const bool supported_memory = options.memory_kind == MemoryKind::kHost ||
                                  options.memory_kind == MemoryKind::kPinnedHost ||
                                  options.memory_kind == MemoryKind::kCudaDevice;
    const bool device_matches =
        options.memory_kind == MemoryKind::kHost ? !options.device.valid() : options.device.valid();
    if (options.max_byte_span == 0U ||
        options.max_byte_span > std::numeric_limits<std::size_t>::max() ||
        options.slot_count == 0U || !supported_memory || !device_matches ||
        options.foreign_emit_policy != InterfaceTensorEmitPolicy::kCopyIfNeeded) {
      return make_unexpected(make_channel_error(
          ErrorCode::kNotSupported,
          "Tensor schema staging requires bounded host, pinned-host, or CUDA-device "
          "copy-if-needed storage"));
    }
    const ScopedCUDADevice selected{options.device};
    if (!selected.active()) {
      return make_unexpected(make_channel_error(ErrorCode::kNotSupported,
                                                "Tensor schema staging device is unavailable"));
    }
    // Host export uses a CUDA VMM generation sidecar. Device export additionally materializes its
    // payload mapping on the compiler-selected device.
    if (cudaFree(nullptr) != cudaSuccess) {
      return make_unexpected(make_channel_error(ErrorCode::kNotSupported,
                                                "Tensor schema staging requires CUDA VMM support"));
    }
    // The ordinary Channel context does not expose an FD-pass server. As with raw Tensor export,
    // create that infrastructure only for the bounded exportable pool. BufferPool retains the
    // private Context implementation after this temporary shell is destroyed.
    auto export_context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
    if (!export_context) {
      return make_unexpected(from_holomq_error(std::move(export_context).error(),
                                               "failed to create Tensor schema staging context"));
    }
    auto buffers = options.memory_kind == MemoryKind::kCudaDevice
                       ? export_context->create_device_buffer_pool(
                             static_cast<std::size_t>(options.max_byte_span), options.slot_count)
                   : options.memory_kind == MemoryKind::kPinnedHost
                       ? export_context->create_host_pinned_buffer_pool(
                             static_cast<std::size_t>(options.max_byte_span),
                             options.slot_count,
                             CU_MEMHOSTREGISTER_PORTABLE)
                       : export_context->create_host_buffer_pool(
                             static_cast<std::size_t>(options.max_byte_span), options.slot_count);
    if (!buffers) {
      return make_unexpected(from_holomq_error(std::move(buffers).error(),
                                               "failed to create Tensor schema staging pool"));
    }
    const holomq::BufferMemoryType expected_memory =
        options.memory_kind == MemoryKind::kCudaDevice
            ? holomq::BufferMemoryType::Device
            : (options.memory_kind == MemoryKind::kPinnedHost
                   ? holomq::BufferMemoryType::HostPinned
                   : holomq::BufferMemoryType::HostPageable);
    const auto pool_device = buffers->cuda_device_ordinal();
    if (buffers->memory_type() != expected_memory ||
        (options.memory_kind == MemoryKind::kCudaDevice &&
         (!pool_device || *pool_device != options.device.value))) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError, "Tensor schema staging pool violates its memory contract"));
    }
    try {
      auto result = std::shared_ptr<TensorSchemaStagingPool>(
          new TensorSchemaStagingPool(std::move(*buffers), options));
      result->slots_.resize(options.slot_count);
      result->free_slots_.reserve(options.slot_count);
      for (std::size_t index = 0U; index < result->slots_.size(); ++index) {
        auto buffer = result->pool_.acquire();
        if (!buffer || !*buffer || buffer->ptr() == nullptr ||
            buffer->size_bytes() < options.max_byte_span) {
          return make_unexpected(make_channel_error(
              ErrorCode::kResourceExhausted, "Tensor schema staging buffer cold binding failed"));
        }
        result->slots_[index].owner = result.get();
        result->slots_[index].index = index;
        result->slots_[index].buffer = std::move(*buffer);
        if (result->stages_device_memory() &&
            cudaEventCreateWithFlags(&result->slots_[index].copy_complete,
                                     cudaEventDisableTiming) != cudaSuccess) {
          return make_unexpected(
              make_channel_error(ErrorCode::kResourceExhausted,
                                 "Tensor schema staging completion-event allocation failed"));
        }
        result->free_slots_.push_back(result->slots_.size() - index - 1U);
      }
      return result;
    } catch (const std::bad_alloc&) {
      return make_unexpected(make_channel_error(ErrorCode::kResourceExhausted,
                                                "Tensor schema staging control allocation failed"));
    } catch (const std::length_error&) {
      return make_unexpected(
          make_channel_error(ErrorCode::kResourceExhausted,
                             "Tensor schema staging capacity exceeds allocator limits"));
    }
  }

  [[nodiscard]] bool stages_device_memory() const noexcept {
    return options_.memory_kind == MemoryKind::kCudaDevice;
  }

  [[nodiscard]] DeviceId device() const noexcept { return options_.device; }

  ~TensorSchemaStagingPool() {
    if (!stages_device_memory()) {
      return;
    }
    const ScopedCUDADevice selected{options_.device};
    if (!selected.active()) {
      return;
    }
    for (Slot& slot : slots_) {
      if (slot.copy_complete != nullptr) {
        static_cast<void>(cudaEventDestroy(slot.copy_complete));
      }
    }
  }

  [[nodiscard]] expected<Pending, Error> stage_copy(const Tensor& tensor,
                                                    cudaStream_t joined_stream) {
    const DLDevice device = tensor.device();
    if (tensor.data() == nullptr || tensor.nbytes() <= 0 || !tensor.is_contiguous() ||
        !tensor_matches_staging_memory(device, options_.memory_kind, options_.device) ||
        static_cast<std::uint64_t>(tensor.nbytes()) > options_.max_byte_span ||
        (options_.rank && tensor.shape_span().size() != *options_.rank) ||
        (options_.dtype && !dtype_equal(tensor.dtype(), *options_.dtype))) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument,
                             "Tensor schema source does not match its bounded staging contract"));
    }

    cudaStream_t producer_stream =
        stages_device_memory() ? detail::TensorAccess::readiness_stream(tensor) : nullptr;
    cudaStream_t copy_stream = joined_stream != nullptr ? joined_stream : producer_stream;
    if (stages_device_memory() && copy_stream == nullptr) {
      return make_unexpected(make_channel_error(
          ErrorCode::kNotSupported, "CUDA Tensor schema staging requires a producer stream"));
    }
    if (stages_device_memory() && joined_stream != nullptr) {
      auto synchronized = tensor.synchronize_for_consumer_stream(joined_stream);
      if (!synchronized) {
        return make_unexpected(std::move(synchronized).error());
      }
    }

    Slot* slot{};
    {
      const std::lock_guard lock(mutex_);
      for (Slot& candidate : slots_) {
        static_cast<void>(try_recycle(candidate));
      }
      if (free_slots_.empty()) {
        return make_unexpected(make_channel_error(ErrorCode::kBackpressured,
                                                  "Tensor schema staging pool has no free slot"));
      }
      slot = &slots_[free_slots_.back()];
      free_slots_.pop_back();
      if (!slot->buffer || slot->buffer.ptr() == nullptr ||
          slot->buffer.size_bytes() < static_cast<std::size_t>(tensor.nbytes())) {
        free_slots_.push_back(slot->index);
        return make_unexpected(
            make_channel_error(ErrorCode::kProtocolError,
                               "Tensor schema staging pool returned an undersized buffer lease"));
      }
      slot->in_use = true;
    }

    bool device_copy_enqueued = false;
    expected<void, Error> copied;
    if (stages_device_memory()) {
      const ScopedCUDADevice selected{options_.device};
      if (!selected.active() || cudaMemcpyAsync(slot->buffer.ptr(),
                                                tensor.data(),
                                                static_cast<std::size_t>(tensor.nbytes()),
                                                cudaMemcpyDeviceToDevice,
                                                copy_stream) != cudaSuccess) {
        copied = make_unexpected(
            make_channel_error(ErrorCode::kFailure, "Tensor schema device staging copy failed"));
      } else {
        if (cudaEventRecord(slot->copy_complete, copy_stream) != cudaSuccess) {
          static_cast<void>(cudaStreamSynchronize(copy_stream));
          copied = make_unexpected(make_channel_error(
              ErrorCode::kFailure, "Tensor schema staging completion record failed"));
        } else {
          slot->copy_pending = true;
          device_copy_enqueued = true;
        }
      }
    } else {
      // Host and pinned-host staging is complete before publication: copy_to_host() first waits
      // for any source readiness authority and then performs the host memcpy synchronously.
      copied = tensor.copy_to_host(
          slot->buffer.ptr(), static_cast<std::size_t>(tensor.nbytes()), nullptr);
    }
    if (!copied) {
      release_slot(slot);
      return make_unexpected(std::move(copied).error());
    }
    return Pending{
        .tensor = tensor,
        .slot = slot,
        .copy_stream = copy_stream,
        .device_copy_enqueued = device_copy_enqueued,
        .retained_state =
            channel_internal::TensorSchemaPayloadRetainedState{
                shared_from_this(), slot, &TensorSchemaStagingPool::release_slot},
    };
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadFieldReservation, Error> finalize(
      Pending pending) {
    // Every device-staged field exports its own readiness-bearing handle. When multiple fields
    // share a joined stream, the publisher attaches that stream's shared owner to each completed
    // reservation so neither Tensor deallocation nor transport retirement can outlive the stream.
    expected<holomq::BufferHandle, holomq::Error> handle =
        pending.slot->buffer.export_handle_unsynchronized();
    if (pending.device_copy_enqueued) {
      handle = pending.slot->buffer.export_handle(pending.copy_stream);
    }
    if (!handle) {
      if (pending.device_copy_enqueued) {
        static_cast<void>(cudaStreamSynchronize(pending.copy_stream));
      }
      return make_unexpected(from_holomq_error(std::move(handle).error(),
                                               "Tensor schema staging buffer export failed"));
    }
    auto descriptor = make_staged_tensor_descriptor(pending.tensor, *handle);
    if (!descriptor) {
      if (pending.device_copy_enqueued) {
        static_cast<void>(cudaStreamSynchronize(pending.copy_stream));
      }
      return make_unexpected(std::move(descriptor).error());
    }
    std::optional<Tensor> retained_source;
    if (pending.device_copy_enqueued) {
      retained_source.emplace(std::move(pending.tensor));
      if (!retained_source->set_deallocation_stream(pending.copy_stream)) {
        static_cast<void>(cudaStreamSynchronize(pending.copy_stream));
        return make_unexpected(make_channel_error(
            ErrorCode::kFailure, "Tensor schema staging failed to retain its device source"));
      }
    }
    return channel_internal::TensorSchemaPayloadFieldReservation{
        .descriptor =
            channel_internal::payload_descriptor_from_holomq_tensor_descriptor(*descriptor),
        .retained_tensor = std::move(retained_source),
        .retained_state = std::move(pending.retained_state),
    };
  }

 private:
  TensorSchemaStagingPool(holomq::BufferPool pool,
                          channel_internal::TensorSchemaStagingOptions options) noexcept
      : pool_(std::move(pool)), options_(std::move(options)) {}

  [[nodiscard]] bool try_recycle(Slot& slot) noexcept {
    if (!slot.awaiting_import_release || slot.in_use) {
      return false;
    }
    if (slot.copy_pending) {
      const ScopedCUDADevice selected{options_.device};
      if (!selected.active() || cudaEventQuery(slot.copy_complete) != cudaSuccess) {
        return false;
      }
      slot.copy_pending = false;
    }
    if (!slot.buffer.prepare_for_reuse()) {
      return false;
    }
    slot.awaiting_import_release = false;
    free_slots_.push_back(slot.index);
    return true;
  }

  static void release_slot(void* state) noexcept {
    auto* const slot = static_cast<Slot*>(state);
    if (slot == nullptr || slot->owner == nullptr) {
      return;
    }
    const std::lock_guard lock(slot->owner->mutex_);
    if (!slot->in_use) {
      return;
    }
    slot->in_use = false;
    slot->awaiting_import_release = true;
    static_cast<void>(slot->owner->try_recycle(*slot));
  }

  holomq::BufferPool pool_;
  channel_internal::TensorSchemaStagingOptions options_;
  std::mutex mutex_;
  std::vector<Slot> slots_;
  std::vector<std::size_t> free_slots_;
};

// One Operator owns this graph-bound output, and its single active compute callback is the only
// publication authority. The joined stream therefore orders fields and successive publications;
// no additional same-output mutex is needed under the current Holoscan compute model.
class TensorSchemaJoinedReadinessStream final {
 public:
  [[nodiscard]] static expected<std::shared_ptr<TensorSchemaJoinedReadinessStream>, Error> create(
      const DeviceId device) {
    const ScopedCUDADevice selected{device};
    if (!selected.active()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kNotSupported, "Tensor schema joined-readiness device is unavailable"));
    }
    cudaStream_t stream{};
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      return make_unexpected(
          make_channel_error(ErrorCode::kResourceExhausted,
                             "Tensor schema joined-readiness stream allocation failed"));
    }
    try {
      return std::make_shared<TensorSchemaJoinedReadinessStream>(device, stream);
    } catch (const std::bad_alloc&) {
      static_cast<void>(cudaStreamDestroy(stream));
      return make_unexpected(make_channel_error(
          ErrorCode::kResourceExhausted, "Tensor schema joined-readiness owner allocation failed"));
    }
  }

  TensorSchemaJoinedReadinessStream(const DeviceId device, cudaStream_t stream) noexcept
      : device_(device), stream_(stream) {}

  TensorSchemaJoinedReadinessStream(const TensorSchemaJoinedReadinessStream&) = delete;
  TensorSchemaJoinedReadinessStream& operator=(const TensorSchemaJoinedReadinessStream&) = delete;

  ~TensorSchemaJoinedReadinessStream() {
    const ScopedCUDADevice selected{device_};
    if (selected.active()) {
      static_cast<void>(cudaStreamSynchronize(stream_));
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  DeviceId device_{};
  cudaStream_t stream_{};
};

class TensorSchemaReservationPool final
    : public std::enable_shared_from_this<TensorSchemaReservationPool> {
 public:
  struct Slot {
    TensorSchemaReservationPool* pool{};
    std::vector<channel_internal::TensorSchemaPayloadFieldReservation> fields;
    bool in_use{};
  };

  [[nodiscard]] static expected<std::shared_ptr<TensorSchemaReservationPool>, Error> create(
      const std::size_t field_count, const std::size_t capacity) {
    if (field_count <= 1U || field_count > channel_internal::kMaximumDescriptorTensorFields ||
        capacity == 0U) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "Tensor schema reservation pool contract is invalid"));
    }
    try {
      auto result = std::shared_ptr<TensorSchemaReservationPool>(new TensorSchemaReservationPool);
      result->slots_.resize(capacity);
      for (auto& slot : result->slots_) {
        slot.pool = result.get();
        slot.fields.resize(field_count - 1U);
      }
      return result;
    } catch (const std::bad_alloc&) {
      return make_unexpected(make_channel_error(
          ErrorCode::kResourceExhausted, "Tensor schema reservation pool allocation failed"));
    }
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error> acquire() {
    const std::scoped_lock lock{mutex_};
    const auto available =
        std::find_if(slots_.begin(), slots_.end(), [](const auto& slot) { return !slot.in_use; });
    if (available == slots_.end()) {
      // Expected transient pressure is reaped and retried by reserve_tensors().
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    available->in_use = true;
    channel_internal::TensorSchemaPayloadReservation result{
        .field_count = static_cast<std::uint32_t>(available->fields.size() + 1U),
        .additional_fields = available->fields.data(),
        .field_storage = channel_internal::TensorSchemaPayloadRetainedState{
            shared_from_this(), &*available, &release_slot}};
    return result;
  }

 private:
  static void release_slot(void* state) noexcept {
    auto& slot = *static_cast<Slot*>(state);
    const std::scoped_lock lock{slot.pool->mutex_};
    for (auto& field : slot.fields) {
      field = {};
    }
    slot.in_use = false;
  }

  std::mutex mutex_;
  std::vector<Slot> slots_;
};

class TensorSchemaPayloadReleaseCallback final {
 public:
  TensorSchemaPayloadReleaseCallback(
      std::shared_ptr<TensorSidecarStore> sidecar_store,
      channel_internal::TensorSchemaPayloadReservation&& source) noexcept
      : sidecar_store_(std::move(sidecar_store)), reservation_(take_reservation(source)) {}

  TensorSchemaPayloadReleaseCallback(const TensorSchemaPayloadReleaseCallback&) = delete;
  TensorSchemaPayloadReleaseCallback& operator=(const TensorSchemaPayloadReleaseCallback&) = delete;

  TensorSchemaPayloadReleaseCallback(TensorSchemaPayloadReleaseCallback&& other) noexcept
      : sidecar_store_(std::move(other.sidecar_store_)),
        reservation_(take_reservation(other.reservation_)) {}

  TensorSchemaPayloadReleaseCallback& operator=(
      TensorSchemaPayloadReleaseCallback&& other) noexcept {
    if (this != &other) {
      reset();
      sidecar_store_ = std::move(other.sidecar_store_);
      reservation_ = take_reservation(other.reservation_);
    }
    return *this;
  }

  ~TensorSchemaPayloadReleaseCallback() noexcept { reset(); }

  void operator()() noexcept { reset(); }

 private:
  [[nodiscard]] static channel_internal::TensorSchemaPayloadReservation take_reservation(
      channel_internal::TensorSchemaPayloadReservation& source) noexcept {
    return channel_internal::TensorSchemaPayloadReservation{
        .field_count = std::exchange(source.field_count, 0U),
        .first_field = std::move(source.first_field),
        .additional_fields = std::exchange(source.additional_fields, nullptr),
        .field_storage = std::move(source.field_storage),
    };
  }

  void reset() noexcept {
    for (std::size_t index = 0U; index < reservation_.field_count; ++index) {
      const auto* field = reservation_.field_at(index);
      if (field == nullptr) {
        continue;
      }
      const PayloadDescriptor& descriptor = field->descriptor;
      if (descriptor.payload_kind == ChannelPayloadKind::kLocalBufferDescriptor) {
        sidecar_store_->release_all(channel_descriptor_from_payload_descriptor(descriptor));
      }
    }
    reservation_ = {};
  }

  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  channel_internal::TensorSchemaPayloadReservation reservation_;
};

[[nodiscard]] expected<void, Error> validate_schema_device_matches_buffer_handle(
    const DLDevice& schema_device, const holomq::BufferHandle& buffer_handle) {
  if (!buffer_handle) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument, "Tensor schema buffer handle is empty"));
  }

  DLDeviceType expected_device_type = kDLCPU;
  switch (buffer_handle.memory_type()) {
    case holomq::BufferMemoryType::Device:
      expected_device_type = kDLCUDA;
      break;
    case holomq::BufferMemoryType::HostPageable:
      expected_device_type = kDLCPU;
      break;
    case holomq::BufferMemoryType::HostPinned:
      expected_device_type = kDLCUDAHost;
      break;
    case holomq::BufferMemoryType::None:
    default:
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema buffer handle has no memory type"));
  }

  if (schema_device.device_type != expected_device_type ||
      (expected_device_type != kDLCUDA && schema_device.device_id != 0)) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema device does not match HoloMQ buffer handle memory kind"));
  }
  return {};
}

template <std::size_t Size>
class TensorSchemaPayloadPublisherImpl final
    : public channel_internal::TensorSchemaPayloadPublisher {
 public:
  TensorSchemaPayloadPublisherImpl(
      holomq::Publisher<InlinePayloadBucket<Size>>&& publisher,
      std::shared_ptr<TensorSidecarStore> sidecar_store,
      std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools,
      std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream,
      std::shared_ptr<TensorSchemaReservationPool> reservation_pool, ChannelOptions options,
      const bool graph_retry_boundary) noexcept
      : reservation_pool_(std::move(reservation_pool)),
        joined_stream_(std::move(joined_stream)),
        staging_pools_(std::move(staging_pools)),
        sidecar_store_(std::move(sidecar_store)),
        publisher_(std::move(publisher)),
        options_(options),
        graph_retry_boundary_(graph_retry_boundary) {}

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error> reserve_tensors(
      const std::span<const Tensor> tensors, const bool require_direct_export) override {
    if (tensors.empty() || tensors.size() > channel_internal::kMaximumDescriptorTensorFields ||
        (!staging_pools_.empty() && staging_pools_.size() != tensors.size())) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument,
                             "Tensor schema reservation field count violates its endpoint"));
    }
    channel_internal::TensorSchemaPayloadReservation reservation{
        .field_count = static_cast<std::uint32_t>(tensors.size())};
    if (tensors.size() > 1U) {
      if (reservation_pool_ == nullptr) {
        return make_unexpected(make_channel_error(ErrorCode::kProtocolError,
                                                  "Tensor schema reservation pool is unavailable"));
      }
      auto created = reservation_pool_->acquire();
      if (!created && created.error().code == ErrorCode::kResourceExhausted) {
        // Entering the transport allocation path synchronously dispatches returned-sample
        // callbacks. That can release a reservation retained by an earlier publication before
        // this bounded pool reports pressure to the graph.
        static_cast<void>(publisher_.loan());
        created = reservation_pool_->acquire();
        if (!created && created.error().code == ErrorCode::kResourceExhausted) {
          return make_unexpected(make_channel_error(
              ErrorCode::kResourceExhausted,
              "Tensor schema reservation pool remains exhausted after release-callback reap "
              "retry"));
        }
      }
      if (!created) {
        return make_unexpected(std::move(created).error());
      }
      reservation = std::move(*created);
    }
    if (options_.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle) {
      auto direct_export = reserve_direct_tensors(tensors, reservation);
      if (direct_export) {
        return reservation;
      }
      clear_field_reservations(reservation);
      if (require_direct_export || staging_pools_.empty()) {
        return make_unexpected(std::move(direct_export).error());
      }
      return reserve_staged_tensors(tensors, std::move(reservation));
    }
    return reserve_local_tensors(tensors, std::move(reservation));
  }

 private:
  [[nodiscard]] static expected<void, Error> reserve_direct_tensors(
      const std::span<const Tensor> tensors,
      channel_internal::TensorSchemaPayloadReservation& reservation) {
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      auto descriptor = channel_internal::export_holomq_tensor_descriptor(tensors[index]);
      if (!descriptor) {
        return make_unexpected(std::move(descriptor).error());
      }
      *reservation.field_at(index) = channel_internal::TensorSchemaPayloadFieldReservation{
          .descriptor =
              channel_internal::payload_descriptor_from_holomq_tensor_descriptor(*descriptor),
          .retained_tensor = tensors[index],
      };
    }
    return {};
  }

  static void clear_field_reservations(
      channel_internal::TensorSchemaPayloadReservation& reservation) noexcept {
    for (std::size_t index = 0U; index < reservation.field_count; ++index) {
      *reservation.field_at(index) = {};
    }
  }

  static void synchronize_stream(cudaStream_t stream) noexcept {
    if (stream != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream));
    }
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error>
  reserve_staged_tensors(const std::span<const Tensor> tensors,
                         channel_internal::TensorSchemaPayloadReservation reservation) {
    cudaStream_t joined_stream = joined_stream_ != nullptr ? joined_stream_->get() : nullptr;
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      auto staged = staging_pools_[index]->stage_copy(tensors[index], joined_stream);
      if (!staged) {
        synchronize_stream(joined_stream);
        return make_unexpected(std::move(staged).error());
      }
      auto field = staging_pools_[index]->finalize(std::move(*staged));
      if (!field) {
        synchronize_stream(joined_stream);
        return make_unexpected(std::move(field).error());
      }
      if (joined_stream_ != nullptr) {
        field->readiness_owner = joined_stream_;
      }
      *reservation.field_at(index) = std::move(*field);
    }
    return reservation;
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error>
  reserve_local_tensors(const std::span<const Tensor> tensors,
                        channel_internal::TensorSchemaPayloadReservation reservation) {
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      auto descriptor = sidecar_store_->insert(Tensor{tensors[index]});
      if (!descriptor) {
        release_reservation(reservation);
        return make_unexpected(std::move(descriptor).error());
      }
      reservation.field_at(index)->descriptor = payload_descriptor_from_descriptor(*descriptor);
    }
    return reservation;
  }

 public:
  [[nodiscard]] expected<std::uint32_t, Error> publish_payload(
      const std::span<const std::byte> payload,
      channel_internal::TensorSchemaPayloadReservation reservation) override {
    if (payload.empty() || payload.size() > Size ||
        payload.size() > std::numeric_limits<std::uint32_t>::max()) {
      release_reservation(reservation);
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema payload is larger than its bucket"));
    }
    auto message = publisher_.loan();
    if (!message) {
      release_reservation(reservation);
      return make_unexpected(from_holomq_error(std::move(message).error()));
    }

    auto& bucket = message->payload();
    std::memcpy(bucket.bytes.data(), payload.data(), payload.size());
    bucket.payload_size = static_cast<std::uint32_t>(payload.size());
    return send_message(std::move(*message), std::move(reservation));
  }

  [[nodiscard]] expected<std::uint32_t, Error> publish_encoded_payload(
      const channel_internal::TensorSchemaPayloadEncoder& encoder,
      channel_internal::TensorSchemaPayloadReservation reservation) override {
    if (encoder.state == nullptr || encoder.encode == nullptr || encoder.capacity == 0U ||
        encoder.capacity > Size) {
      release_reservation(reservation);
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "Tensor schema encoder exceeds its bounded bucket"));
    }
    auto message = publisher_.loan();
    if (!message) {
      release_reservation(reservation);
      return make_unexpected(from_holomq_error(std::move(message).error()));
    }
    auto& bucket = message->payload();
    auto encoded = encoder.encode(
        encoder.state, std::span<std::byte>{bucket.bytes}.first(encoder.capacity), reservation);
    if (!encoded || *encoded == 0U || *encoded > encoder.capacity ||
        *encoded > std::numeric_limits<std::uint32_t>::max()) {
      release_reservation(reservation);
      return make_unexpected(
          encoded ? make_channel_error(ErrorCode::kProtocolError,
                                       "Tensor schema encoder returned an invalid size")
                  : std::move(encoded).error());
    }
    bucket.payload_size = static_cast<std::uint32_t>(*encoded);
    return send_message(std::move(*message), std::move(reservation));
  }

 private:
  [[nodiscard]] expected<std::uint32_t, Error> send_message(
      holomq::OutboundMessage<InlinePayloadBucket<Size>> message,
      channel_internal::TensorSchemaPayloadReservation reservation) {
    expected<int, holomq::Error> sent;
    try {
      sent = publisher_.send(
          std::move(message),
          TensorSchemaPayloadReleaseCallback{sidecar_store_, std::move(reservation)});
    } catch (const std::length_error&) {
      release_reservation(reservation);
      return channel_internal::make_unexpected_nothrow<std::uint32_t>(
          ErrorCode::kResourceExhausted,
          "Tensor schema release-callback registration exceeds allocator limits");
    } catch (const std::bad_alloc&) {
      release_reservation(reservation);
      return channel_internal::make_unexpected_nothrow<std::uint32_t>(
          ErrorCode::kResourceExhausted,
          "Tensor schema release-callback registration allocation failed");
    } catch (...) {
      release_reservation(reservation);
      return channel_internal::make_unexpected_nothrow<std::uint32_t>(
          ErrorCode::kFailure, "Tensor schema release-callback registration failed");
    }
    return from_holomq_publication_result(
        std::move(sent), options_.expected_recipient_count, graph_retry_boundary_);
  }

 public:
  void release_tensor(
      channel_internal::TensorSchemaPayloadReservation reservation) noexcept override {
    release_reservation(reservation);
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept override {
    return from_holomq_result(publisher_.update_connections());
  }

  [[nodiscard]] bool has_subscribers() const noexcept override {
    return publisher_.has_subscribers();
  }

  [[nodiscard]] expected<MemoryBuffer::Options, Error> tensor_memory_options() const override {
    MemoryBuffer::Options options{};
    options.max_fanout_release_streams = options_.expected_recipient_count;
    return options;
  }

 private:
  std::shared_ptr<TensorSchemaReservationPool> reservation_pool_;
  void release_reservation(channel_internal::TensorSchemaPayloadReservation& reservation) noexcept {
    for (std::size_t index = 0U; index < reservation.field_count; ++index) {
      const auto* field = reservation.field_at(index);
      if (field == nullptr) {
        continue;
      }
      const PayloadDescriptor& descriptor = field->descriptor;
      if (descriptor.payload_kind == ChannelPayloadKind::kLocalBufferDescriptor) {
        sidecar_store_->release_all(channel_descriptor_from_payload_descriptor(descriptor));
      }
    }
    reservation = {};
  }

  std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream_;
  std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools_;
  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  holomq::Publisher<InlinePayloadBucket<Size>> publisher_;
  ChannelOptions options_;
  bool graph_retry_boundary_{};
};

template <std::size_t Size>
class TensorSchemaPayloadSubscriberImpl final
    : public channel_internal::TensorSchemaPayloadSubscriber {
 public:
  TensorSchemaPayloadSubscriberImpl(
      holomq::Subscriber<InlinePayloadBucket<Size>>&& subscriber,
      std::shared_ptr<TensorSidecarStore> sidecar_store, std::size_t recipient_index,
      std::vector<TensorSchemaReceiveControlPtr> receive_controls) noexcept
      : subscriber_(std::move(subscriber)),
        sidecar_store_(std::move(sidecar_store)),
        recipient_index_(recipient_index),
        receive_controls_(std::move(receive_controls)) {}

  ~TensorSchemaPayloadSubscriberImpl() override = default;

  [[nodiscard]] expected<bool, Error> has_samples() const override {
    return from_holomq_result(subscriber_.has_samples());
  }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void* wait_set) override {
    if (wait_set == nullptr || attachment_.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto attached = static_cast<holomq::WaitSet*>(wait_set)->attach(subscriber_);
    if (!attached) {
      return make_unexpected(from_holomq_error(std::move(attached).error()));
    }
    const std::uint64_t handle = attached->handle().id();
    attachment_.emplace(std::move(*attached));
    return handle;
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReceiveGuard, Error> receive_payload(
      std::span<std::byte> payload) override {
    auto borrowed = borrow_payload();
    if (!borrowed) {
      return make_unexpected(std::move(borrowed).error());
    }
    if (borrowed->payload().size() > payload.size()) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema destination is too small"));
    }
    std::memcpy(payload.data(), borrowed->payload().data(), borrowed->payload().size());
    return borrowed;
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReceiveGuard, Error> borrow_payload()
      override {
    if (inbound_.has_value()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kFailure, "Tensor schema payload already has an active transport loan"));
    }

    auto inbound = subscriber_.try_take();
    if (!inbound) {
      return make_unexpected(from_holomq_error(std::move(inbound).error()));
    }
    inbound_.emplace(std::move(inbound).value());
    const auto& bucket = inbound_->payload();
    if (bucket.payload_size == 0U || bucket.payload_size > Size) {
      inbound_.reset();
      return make_unexpected(
          make_channel_error(ErrorCode::kProtocolError, "Tensor schema payload length is invalid"));
    }
    return channel_internal::TensorSchemaPayloadReceiveGuard{
        *this, std::span<const std::byte>{bucket.bytes}.first(bucket.payload_size)};
  }

  [[nodiscard]] expected<Tensor, Error> acquire_tensor(
      const PayloadDescriptor& descriptor) override {
    if (descriptor.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor schema sidecar acquisition requires a local buffer descriptor"));
    }
    return sidecar_store_->acquire(channel_descriptor_from_payload_descriptor(descriptor),
                                   recipient_index_);
  }

  [[nodiscard]] expected<Tensor, Error> import_buffer_handle_tensor(
      const PayloadDescriptor& descriptor, const TensorViewMetadata& metadata,
      const std::uint32_t field_index) override {
    if (descriptor.payload_kind != ChannelPayloadKind::kBufferHandle) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument,
                             "Tensor schema import requires a HoloMQ buffer-handle descriptor"));
    }
    if (metadata.rank > TensorViewMetadata::kMaxRank) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema metadata rank exceeds capacity"));
    }
    if (descriptor.byte_size != metadata.byte_size) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor schema metadata byte size does not match the buffer-handle descriptor"));
    }
    auto buffer_handle =
        channel_internal::holomq_buffer_handle_from_payload_reference(descriptor.buffer);
    if (!buffer_handle) {
      return make_unexpected(std::move(buffer_handle).error());
    }
    if (auto valid_device =
            validate_schema_device_matches_buffer_handle(metadata.device, buffer_handle.value());
        !valid_device) {
      return make_unexpected(std::move(valid_device).error());
    }

    const TensorShapeView shape{metadata.shape.data(), metadata.rank};
    const TensorByteStridesView byte_strides =
        metadata.has_explicit_byte_strides ? TensorByteStridesView{TensorByteStridesSpan{
                                                 metadata.byte_strides.data(), metadata.rank}}
                                           : TensorByteStridesView{std::nullopt};
    auto tensor_descriptor = channel_internal::make_holomq_tensor_descriptor(
        shape,
        metadata.dtype,
        byte_strides,
        metadata.byte_size,
        buffer_handle.value(),
        /*read_only=*/true,
        metadata.device.device_type == kDLCUDA ? metadata.device.device_id : -1);
    if (!tensor_descriptor) {
      return make_unexpected(std::move(tensor_descriptor).error());
    }
    if (field_index >= receive_controls_.size()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "Tensor schema field index exceeds receive pool count"));
    }
    // The field-local control retains imported mappings, Tensor contexts, and rank metadata.
    // Readiness is still joined lazily to the exact activation stream by the returned Tensor.
    return receive_controls_[field_index]->materialize(*tensor_descriptor);
  }

  void release_recipient_claim(const PayloadDescriptor& descriptor) noexcept override {
    if (descriptor.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return;
    }
    sidecar_store_->release_recipient_claim(channel_descriptor_from_payload_descriptor(descriptor),
                                            recipient_index_);
  }

  void release_payload_loan() noexcept override { inbound_.reset(); }

 private:
  holomq::Subscriber<InlinePayloadBucket<Size>> subscriber_;
  std::optional<holomq::WaitSetAttachment> attachment_;
  std::optional<holomq::InboundMessage<InlinePayloadBucket<Size>>> inbound_;
  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  std::size_t recipient_index_{};
  std::vector<TensorSchemaReceiveControlPtr> receive_controls_;
};

template <std::size_t Size>
[[nodiscard]] PayloadDescriptor payload_descriptor_from_bucket(
    const InlinePayloadBucket<Size>& bucket, const std::size_t descriptor_offset) noexcept {
  if (descriptor_offset + sizeof(PayloadDescriptor) <= Size) {
    auto descriptor = channel_internal::decode_payload_descriptor_wire(
        std::span<const std::byte>{bucket.bytes}.subspan(descriptor_offset,
                                                         sizeof(PayloadDescriptor)));
    if (descriptor) {
      return *descriptor;
    }
  }
  return {};
}

template <std::size_t Size>
[[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
create_tensor_schema_payload_publisher_impl(
    holomq::Context& context, const std::string_view channel_name, const ChannelOptions& options,
    std::shared_ptr<TensorSidecarStore> sidecar_store, const std::size_t payload_size,
    const std::vector<std::size_t>& descriptor_offsets,
    std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools,
    std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream,
    const PublisherPressureOptions pressure) {
  if (payload_size > Size || descriptor_offsets.empty() ||
      descriptor_offsets.size() > channel_internal::kMaximumDescriptorTensorFields ||
      !descriptor_offsets_unique(descriptor_offsets) ||
      std::any_of(descriptor_offsets.begin(), descriptor_offsets.end(), [&](const auto offset) {
        return offset > payload_size || sizeof(PayloadDescriptor) > payload_size - offset;
      })) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "Tensor schema descriptor offsets must be unique and inside payload bounds"));
  }

  auto publisher_builder = context.publisher_builder(std::string(channel_name))
                               .service_options(to_holomq_service_options(options))
                               .publisher_options(to_holomq_publisher_options(options, pressure))
                               .template payload_type<InlinePayloadBucket<Size>>();
  if (replaces_oldest_unread_sample(options.queue_policy) &&
      options.replacement_attribution_required) {
    publisher_builder =
        std::move(publisher_builder)
            .on_sample_replaced([sidecar_store, descriptor_offsets](
                                    const InlinePayloadBucket<Size>& bucket,
                                    const holomq::SampleReplacedInfo& info) noexcept {
              for (const std::size_t descriptor_offset : descriptor_offsets) {
                PayloadDescriptor payload_descriptor{};
                std::memcpy(&payload_descriptor,
                            bucket.bytes.data() + descriptor_offset,
                            sizeof(payload_descriptor));
                if (payload_descriptor.payload_kind == ChannelPayloadKind::kLocalBufferDescriptor) {
                  sidecar_store->release_replaced(
                      channel_descriptor_from_payload_descriptor(payload_descriptor),
                      info.receiver_id);
                }
              }
            });
  }

  auto publisher = std::move(publisher_builder).create();
  if (!publisher) {
    return make_unexpected(from_holomq_error(std::move(publisher).error(),
                                             "failed to create Tensor schema channel publisher"));
  }
  std::shared_ptr<TensorSchemaReservationPool> reservation_pool;
  if (descriptor_offsets.size() > 1U) {
    auto sidecar_capacity = channel_internal::tensor_sidecar_record_capacity(options);
    if (!sidecar_capacity ||
        options.publisher_history_depth >
            std::numeric_limits<std::size_t>::max() - (sidecar_capacity ? *sidecar_capacity : 0U)) {
      return make_unexpected(
          sidecar_capacity
              ? make_channel_error(ErrorCode::kInvalidArgument,
                                   "Tensor schema reservation capacity exceeds size range")
              : std::move(sidecar_capacity).error());
    }
    auto created_pool = TensorSchemaReservationPool::create(
        descriptor_offsets.size(), *sidecar_capacity + options.publisher_history_depth);
    if (!created_pool) {
      return make_unexpected(std::move(created_pool).error());
    }
    reservation_pool = std::move(*created_pool);
  }
  return std::make_unique<TensorSchemaPayloadPublisherImpl<Size>>(
      std::move(publisher).value(),
      std::move(sidecar_store),
      std::move(staging_pools),
      std::move(joined_stream),
      std::move(reservation_pool),
      options,
      pressure.enables_graph_retry_boundary(options.publisher_history_depth));
}

template <std::size_t Size>
[[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
create_tensor_schema_payload_subscriber_impl(holomq::Context& context,
                                             const std::string_view channel_name,
                                             const ChannelOptions& options,
                                             std::shared_ptr<TensorSidecarStore> sidecar_store,
                                             std::shared_ptr<void> runtime_lifetime,
                                             const std::uint32_t tensor_field_count) {
  auto subscriber = context.subscriber_builder(std::string(channel_name))
                        .service_options(to_holomq_service_options(options))
                        .subscriber_options(to_holomq_subscriber_options(options))
                        .template payload_type<InlinePayloadBucket<Size>>()
                        .create();
  if (!subscriber) {
    return make_unexpected(from_holomq_error(std::move(subscriber).error(),
                                             "failed to create Tensor schema channel subscriber"));
  }

  auto recipient_index = sidecar_store->register_recipient(subscriber->recipient_id());
  if (!recipient_index) {
    return make_unexpected(std::move(recipient_index).error());
  }
  auto receive_capacity = channel_internal::tensor_sidecar_record_capacity(options);
  if (!receive_capacity || *receive_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(receive_capacity
                               ? make_channel_error(ErrorCode::kInvalidArgument,
                                                    "Tensor schema receive capacity is too large")
                               : std::move(receive_capacity).error());
  }
  LogicalTensorRequirementsRecord receive_contract{};
  receive_contract.max_rank = TensorViewMetadata::kMaxRank;
  std::vector<TensorSchemaReceiveControlPtr> receive_controls;
  receive_controls.reserve(tensor_field_count);
  for (std::uint32_t index = 0U; index < tensor_field_count; ++index) {
    auto control = channel_internal::CatalogTensorReceiveControl::create(
        runtime_lifetime, context, receive_contract, static_cast<std::uint32_t>(*receive_capacity));
    if (!control) {
      return make_unexpected(std::move(control).error());
    }
    receive_controls.emplace_back(control->release());
  }
  return std::make_unique<TensorSchemaPayloadSubscriberImpl<Size>>(std::move(subscriber).value(),
                                                                   std::move(sidecar_store),
                                                                   recipient_index.value(),
                                                                   std::move(receive_controls));
}
struct TensorSchemaPayloadPublisherFactory {
  holomq::Context& context;
  std::string_view channel_name;
  const ChannelOptions& options;
  std::shared_ptr<TensorSidecarStore> sidecar_store;
  std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools;
  std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream;
  std::size_t payload_size;
  std::vector<std::size_t> descriptor_offsets;
  PublisherPressureOptions pressure;

  template <std::size_t Size>
  [[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
  operator()() const {
    return create_tensor_schema_payload_publisher_impl<Size>(context,
                                                             channel_name,
                                                             options,
                                                             sidecar_store,
                                                             payload_size,
                                                             descriptor_offsets,
                                                             staging_pools,
                                                             joined_stream,
                                                             pressure);
  }
};

struct TensorSchemaPayloadSubscriberFactory {
  holomq::Context& context;
  std::string_view channel_name;
  const ChannelOptions& options;
  std::shared_ptr<TensorSidecarStore> sidecar_store;
  std::shared_ptr<void> runtime_lifetime;
  std::uint32_t tensor_field_count;

  template <std::size_t Size>
  [[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
  operator()() const {
    return create_tensor_schema_payload_subscriber_impl<Size>(
        context, channel_name, options, sidecar_store, runtime_lifetime, tensor_field_count);
  }
};

template <typename Factory>
[[nodiscard]] auto dispatch_tensor_schema_payload_bucket(const std::size_t payload_size,
                                                         Factory& factory)
    -> decltype(factory.template operator()<8U>()) {
  const auto bucket_size = channel_internal::inline_payload_bucket_size(payload_size);
  if (!bucket_size.has_value()) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument,
        "Tensor schema channel payload size must be in the range [1, " +
            std::to_string(channel_internal::kMaximumInlinePayloadBytes) + "] bytes"));
  }

  switch (bucket_size.value()) {
    case 8U:
      return factory.template operator()<8U>();
    case 16U:
      return factory.template operator()<16U>();
    case 32U:
      return factory.template operator()<32U>();
    case 64U:
      return factory.template operator()<64U>();
    case 128U:
      return factory.template operator()<128U>();
    case 256U:
      return factory.template operator()<256U>();
    case 512U:
      return factory.template operator()<512U>();
    case 1024U:
      return factory.template operator()<1024U>();
    case 2048U:
      return factory.template operator()<2048U>();
    case 4096U:
      return factory.template operator()<4096U>();
    case 8192U:
      return factory.template operator()<8192U>();
    case 16384U:
      return factory.template operator()<16384U>();
    case 32768U:
      return factory.template operator()<32768U>();
    default:
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "unsupported Tensor schema channel payload bucket size"));
  }
}

}  // namespace

namespace channel_internal {

expected<std::unique_ptr<TensorSchemaPayloadPublisher>, Error>
create_tensor_schema_payload_publisher(holomq::Context& context,
                                       const std::string_view channel_name,
                                       const std::size_t payload_size,
                                       const std::span<const std::size_t> descriptor_offsets,
                                       const ChannelOptions& options,
                                       std::shared_ptr<TensorSidecarStore> sidecar_store,
                                       const std::span<const TensorSchemaStagingOptions> staging,
                                       const PublisherPressureOptions pressure) {
  if (descriptor_offsets.empty() || !descriptor_offsets_unique(descriptor_offsets) ||
      (!staging.empty() && staging.size() != descriptor_offsets.size())) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema descriptor offsets must be unique and match staging"));
  }
  std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools;
  staging_pools.reserve(staging.size());
  std::optional<DeviceId> joined_device;
  for (const auto& field : staging) {
    auto created = TensorSchemaStagingPool::create(context, field);
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    if ((*created)->stages_device_memory()) {
      if (joined_device && *joined_device != (*created)->device()) {
        return make_unexpected(make_channel_error(
            ErrorCode::kNotSupported,
            "Tensor schema joined readiness requires one CUDA device per endpoint"));
      }
      joined_device = (*created)->device();
    }
    staging_pools.push_back(std::move(*created));
  }
  std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream;
  if (staging.size() > 1U && joined_device) {
    auto created = TensorSchemaJoinedReadinessStream::create(*joined_device);
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    joined_stream = std::move(*created);
  }
  TensorSchemaPayloadPublisherFactory factory{
      context,
      channel_name,
      options,
      std::move(sidecar_store),
      std::move(staging_pools),
      std::move(joined_stream),
      payload_size,
      {descriptor_offsets.begin(), descriptor_offsets.end()},
      pressure};
  return dispatch_tensor_schema_payload_bucket(payload_size, factory);
}

expected<std::unique_ptr<TensorSchemaPayloadSubscriber>, Error>
create_tensor_schema_payload_subscriber(
    holomq::Context& context, const std::string_view channel_name, const std::size_t payload_size,
    const ChannelOptions& options, std::shared_ptr<TensorSidecarStore> sidecar_store,
    std::shared_ptr<void> runtime_lifetime, const std::uint32_t tensor_field_count) {
  if (!runtime_lifetime || tensor_field_count == 0U ||
      tensor_field_count > channel_internal::kMaximumDescriptorTensorFields) {
    return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                              "Tensor schema receive pool contract is invalid"));
  }
  if (options.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle &&
      cudaFree(nullptr) != cudaSuccess) {
    return make_unexpected(make_channel_error(
        ErrorCode::kNotSupported, "Tensor schema BufferHandle import requires CUDA VMM support"));
  }
  TensorSchemaPayloadSubscriberFactory factory{context,
                                               channel_name,
                                               options,
                                               std::move(sidecar_store),
                                               std::move(runtime_lifetime),
                                               tensor_field_count};
  return dispatch_tensor_schema_payload_bucket(payload_size, factory);
}

}  // namespace channel_internal
}  // namespace holoscan
