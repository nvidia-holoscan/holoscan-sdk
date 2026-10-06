/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
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
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fmt/format.h>

#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/tensor_loan.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include <holomq/buffer_pool.hpp>
#include <holomq/context_builder.hpp>

#include "catalog_tensor_clock_health.hpp"
#include "channel_common.hpp"
#include "channel_inline_payload.hpp"
#include "channel_runtime_impl.hpp"
#include "channel_tensor_loan_internal.hpp"
#include "channel_tensor_test_access.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/domain/tensor_layout_validation.hpp"
#include "endpoint_catalog_state.hpp"
#include "holomq_tensor.hpp"

namespace holoscan {

namespace {

std::atomic<channel_internal::TensorPublisherTestAccess::RequiredPublisherWaitBarrier*>
    g_required_publisher_wait_barrier{};

[[nodiscard]] constexpr bool catalog_publication_consumes_authority(const ErrorCode code) noexcept {
  return publication_error_consumes_authority(code) || code == ErrorCode::kProtocolError ||
         code == ErrorCode::kInvalidArgument;
}

class ScopedCUDADevice final {
 public:
  explicit ScopedCUDADevice(const std::optional<DeviceId> requested) noexcept {
    if (cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    selected_ = previous_;
    if (requested) {
      if (!requested->valid()) {
        return;
      }
      selected_ = requested->value;
    }
    if (selected_ != previous_ && cudaSetDevice(selected_) != cudaSuccess) {
      return;
    }
    active_ = true;
  }

  ScopedCUDADevice(const ScopedCUDADevice&) = delete;
  ScopedCUDADevice& operator=(const ScopedCUDADevice&) = delete;

  ~ScopedCUDADevice() {
    if (active_ && selected_ != previous_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] DeviceId selected_device() const noexcept {
    return active_ ? DeviceId{selected_} : DeviceId{};
  }

 private:
  int previous_{};
  int selected_{};
  bool active_{};
};

enum class TensorLoanSlotPhase : std::uint8_t {
  kFree,
  kLoaned,
  kWriting,
  kReady,
  kCancelPending,
  kPublished,
  kQuarantined,
};

class CatalogTensorPublisherControl;

class CatalogTensorSlotMemoryReference final
    : public channel_internal::HoloMQExportableTensorMemoryReference {
 public:
  CatalogTensorSlotMemoryReference(CatalogTensorPublisherControl* owner,
                                   std::size_t slot_index) noexcept
      : owner_(owner), slot_index_(slot_index) {}

  void retain_tensor_handle() noexcept override {
    auto count = handle_count_.load(std::memory_order_relaxed);
    while (count != std::numeric_limits<std::uint32_t>::max()) {
      if (handle_count_.compare_exchange_weak(
              count, count + 1U, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return;
      }
    }
    // Tensor copies cannot report an error. Saturate permanently instead of wrapping and
    // accidentally making a still-referenced slot reusable.
    overflowed_.store(true, std::memory_order_release);
  }

  void release_tensor_handle() noexcept override;

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept override {
    if (!buffer_) {
      return channel_internal::make_unexpected_nothrow<holomq::BufferHandle>(
          ErrorCode::kInvalidArgument, "external Tensor slot is not bound");
    }
    cudaStream_t readiness_stream = readiness_stream_.load(std::memory_order_acquire);
    auto handle = readiness_stream == nullptr ? buffer_->export_handle_unsynchronized()
                                              : buffer_->export_handle(readiness_stream);
    if (!handle) {
      return make_unexpected(channel_internal::from_holomq_error(
          std::move(handle).error(), "failed to export external Tensor slot"));
    }
    return std::move(handle).value();
  }

  [[nodiscard]] expected<void, Error> bind(holomq::Buffer buffer) noexcept {
    if (buffer_ || !buffer) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    readiness_stream_.store(nullptr, std::memory_order_release);
    buffer_.emplace(std::move(buffer));
    return {};
  }

  void release_buffer() noexcept {
    readiness_stream_.store(nullptr, std::memory_order_release);
    buffer_.reset();
  }

  void prepare_reuse() noexcept { readiness_stream_.store(nullptr, std::memory_order_release); }

  [[nodiscard]] bool has_buffer() const noexcept { return buffer_.has_value(); }

  [[nodiscard]] void* data() const noexcept {
    return buffer_ ? const_cast<void*>(buffer_->ptr()) : nullptr;
  }

  bool set_deallocation_stream(cudaStream_t stream) override {
    if (!buffer_) {
      return false;
    }
    return stream == nullptr || buffer_->enqueue_ready_set(stream);
  }

  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override {
    return buffer_.has_value();
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    if (!buffer_) {
      return false;
    }
    readiness_stream_.store(stream, std::memory_order_release);
    return stream == nullptr || buffer_->enqueue_ready_set(stream);
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return buffer_.has_value();
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return readiness_stream_.load(std::memory_order_acquire);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    return synchronize_cuda_streams(readiness_stream_.load(std::memory_order_acquire),
                                    consumer_stream);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    return synchronize_cuda_stream_for_host(readiness_stream_.load(std::memory_order_acquire));
  }

  void arm(std::uint64_t generation) noexcept {
    generation_.store(generation, std::memory_order_relaxed);
    armed_.store(true, std::memory_order_release);
  }

  void disarm() noexcept { armed_.store(false, std::memory_order_release); }

 private:
  CatalogTensorPublisherControl* owner_{};
  std::size_t slot_index_{};
  std::atomic<std::uint32_t> handle_count_{0U};
  std::atomic<std::uint64_t> generation_{0U};
  std::atomic<bool> armed_{false};
  std::atomic<bool> overflowed_{false};
  std::optional<holomq::Buffer> buffer_;
  std::atomic<cudaStream_t> readiness_stream_{nullptr};
};

struct CatalogTensorLoanSlot {
  CatalogTensorPublisherControl* owner{};
  std::size_t index{};
  std::uint64_t generation{1U};
  TensorLoanSlotPhase phase{TensorLoanSlotPhase::kFree};
  bool loan_handle_live{};
  std::size_t byte_size{};
  DLDataType dtype{};
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> byte_strides;
  std::shared_ptr<DLManagedTensorContext> context;
  CatalogTensorSlotMemoryReference* memory_reference{};
  Tensor template_tensor;
  Tensor retained_copy_source;
  std::byte* storage{};
  cudaStream_t writer_stream{};
  cudaEvent_t producer_ready_event{};
};

struct NormalizedTensorLoanRequest {
  std::size_t byte_size{};
  std::size_t item_size{};
};

struct ValidatedTensorEndpoint {
  const TensorBoundaryContractRecord* tensor{};
  MemoryKind memory_kind{MemoryKind::kUnknown};
  std::uint32_t max_rank{};
  std::uint64_t max_byte_span{};
  std::uint32_t max_loaned_samples{};
  std::uint32_t max_borrowed_samples{};
  std::uint32_t publisher_pool_slots{};
  std::chrono::nanoseconds max_wait{};
  ClockHealthOptions clock_health{};
};

[[nodiscard]] bool dtype_equal(const DLDataType lhs, const DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

class CatalogTensorPublisherControl {
 public:
  CatalogTensorPublisherControl(ManualTensorPublisher publisher,
                                std::unique_ptr<channel_internal::InlinePayloadPublisher> health,
                                holomq::BufferPool pool, const ValidatedTensorEndpoint& endpoint,
                                const DeviceId synchronization_device, const ClockId capture_clock,
                                const std::uint64_t catalog_clock_count,
                                const std::uint64_t producer_generation)
      : publisher_(std::move(publisher)),
        health_(std::move(health)),
        pool_(std::move(pool)),
        capture_clock_(capture_clock),
        catalog_clock_count_(catalog_clock_count),
        memory_kind_(endpoint.memory_kind),
        device_id_(synchronization_device),
        required_dtype_(endpoint.tensor->dtype),
        exact_rank_(endpoint.tensor->rank),
        max_rank_(endpoint.max_rank),
        max_byte_span_(endpoint.max_byte_span),
        axis_bounds_(endpoint.tensor->axes),
        stride_policy_(endpoint.tensor->strides),
        foreign_emit_policy_(endpoint.tensor->foreign_emit_policy),
        clock_health_(endpoint.clock_health),
        max_loaned_samples_(endpoint.max_loaned_samples),
        slot_count_(endpoint.publisher_pool_slots),
        producer_generation_(producer_generation) {}

  CatalogTensorPublisherControl(const CatalogTensorPublisherControl&) = delete;
  CatalogTensorPublisherControl& operator=(const CatalogTensorPublisherControl&) = delete;

  ~CatalogTensorPublisherControl() {
    if (!device_id_.valid()) {
      return;
    }
    const ScopedCUDADevice selected{device_id_};
    if (!selected.active()) {
      return;
    }
    for (auto& slot : slots_) {
      if (slot.producer_ready_event != nullptr) {
        static_cast<void>(cudaEventDestroy(slot.producer_ready_event));
      }
    }
  }

  [[nodiscard]] expected<void, Error> initialize() {
    const holomq::BufferMemoryType expected_pool_type =
        memory_kind_ == MemoryKind::kCudaDevice
            ? holomq::BufferMemoryType::Device
            : (memory_kind_ == MemoryKind::kPinnedHost ? holomq::BufferMemoryType::HostPinned
                                                       : holomq::BufferMemoryType::HostPageable);
    if (!pool_ || pool_.buffer_bytes() < max_byte_span_ || pool_.buffer_count() < slot_count_ ||
        pool_.memory_type() != expected_pool_type) {
      return make_unexpected(Error{ErrorCode::kNotSupported,
                                   "external Tensor pool does not match the catalog contract"});
    }
    if (memory_kind_ == MemoryKind::kCudaDevice) {
      const auto ordinal = pool_.cuda_device_ordinal();
      if (!ordinal || *ordinal < 0 || !device_id_.valid() || *ordinal != device_id_.value) {
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "external Tensor device pool has no CUDA device"});
      }
    } else if (memory_kind_ == MemoryKind::kPinnedHost && !device_id_.valid()) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported,
                "external pinned Tensor pool has no private CUDA synchronization device"});
    } else if (memory_kind_ == MemoryKind::kHost && device_id_.valid()) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "external host Tensor pool has a CUDA device"});
    }
    slots_.resize(slot_count_);
    buffered_free_slots_.reserve(slot_count_);
    bufferless_free_slots_.reserve(slot_count_);
    const bool needs_copy_events =
        memory_kind_ == MemoryKind::kCudaDevice || memory_kind_ == MemoryKind::kPinnedHost;
    std::optional<ScopedCUDADevice> selected;
    if (needs_copy_events) {
      selected.emplace(device_id_);
      if (!selected->active()) {
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "external Tensor CUDA device is unavailable"});
      }
    }
    for (std::size_t index = 0U; index < slot_count_; ++index) {
      auto& slot = slots_[index];
      slot.owner = this;
      slot.index = index;
      slot.storage = nullptr;
      slot.writer_stream = nullptr;
      slot.shape.reserve(max_rank_);
      slot.byte_strides.reserve(max_rank_);
      auto memory_reference = std::make_shared<CatalogTensorSlotMemoryReference>(this, index);
      slot.memory_reference = memory_reference.get();
      slot.context = std::make_shared<DLManagedTensorContext>();
      slot.context->memory_reference = std::move(memory_reference);
      slot.context->dl_shape.reserve(max_rank_);
      slot.context->dl_strides.reserve(max_rank_);
      auto& tensor = slot.context->tensor;
      tensor.version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
      tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
      tensor.manager_ctx = nullptr;
      tensor.deleter = nullptr;
      tensor.dl_tensor = DLTensor{
          .data = nullptr,
          .device = tensor_dl_device(),
          .ndim = 0,
          .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
          .shape = nullptr,
          .strides = nullptr,
          .byte_offset = 0U,
      };
      slot.template_tensor =
          detail::TensorAccess::from_context(slot.context, detail::TensorAuthority::kReadOnly);
      if (needs_copy_events && cudaEventCreateWithFlags(&slot.producer_ready_event,
                                                        cudaEventDisableTiming) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted,
                                     "external Tensor copy-order event allocation failed"});
      }
      bufferless_free_slots_.push_back(slot_count_ - index - 1U);
    }
    return {};
  }

  void retain() noexcept { references_.fetch_add(1U, std::memory_order_relaxed); }

  void release() noexcept {
    if (references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      delete this;
    }
  }

  void destroy() noexcept {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    release();
  }

  [[nodiscard]] expected<void, Error> close_stream() noexcept {
    std::uint64_t proposed_sequence{};
    bool fail_pre_submit_for_test{};
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      proposed_sequence = operation_sequence_ + 1U;
      fail_pre_submit_for_test = std::exchange(fail_next_terminal_pre_submit_for_test_, false);
    }
    const SampleMetadata metadata{
        .sequence = proposed_sequence,
        .byte_size = 0U,
        .payload_kind = PayloadKind::kDescriptorBacked,
    };
    auto result =
        fail_pre_submit_for_test
            ? expected<std::uint32_t, Error>{make_unexpected(Error{ErrorCode::kResourceExhausted})}
            : channel_internal::TensorPublishLoanAccess::emit_end_of_stream(publisher_, metadata);
    // Permanent operation-sequence exhaustion returned above. Resource exhaustion from the
    // terminal publisher here is therefore a loan rejection before backend submission and keeps
    // this catalog endpoint and sequence available for retry.
    if (!result && result.error().code == ErrorCode::kResourceExhausted) {
      auto error = std::move(result).error();
      error.code = ErrorCode::kBackpressured;
      return make_unexpected(std::move(error));
    }
    inject_post_submit_result_for_test(result);
    if (result ||
        channel_internal::terminal_publication_error_consumes_authority(result.error().code)) {
      const std::lock_guard<std::mutex> lock(mutex_);
      operation_sequence_ = proposed_sequence;
      closed_ = true;
    }
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
    return {};
  }

  [[nodiscard]] expected<TensorPublishLoan, Error> loan_tensor(
      const TensorLoanRequest& request) noexcept {
    NormalizedTensorLoanRequest normalized{};
    if (auto valid = validate_request(request, normalized); !valid) {
      return make_unexpected(std::move(valid).error());
    }

    bool reclaim_transport_samples = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (active_loan_count_ >= max_loaned_samples_) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      reclaim_transport_samples = buffered_free_slots_.empty() && bufferless_free_slots_.empty();
    }
    if (reclaim_transport_samples) {
      // Do not hold the catalog mutex here: the HoloMQ release callback retires the logical slot
      // through retire_published(), which takes the same mutex. The physical BufferPool claim may
      // still keep acquire() exhausted until every imported Tensor/read lease also retires.
      auto refreshed = publisher_.update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
    }

    const std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (active_loan_count_ >= max_loaned_samples_) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    if (buffered_free_slots_.empty() && bufferless_free_slots_.empty()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const bool buffered = !buffered_free_slots_.empty();
    auto& selected_free_slots = buffered ? buffered_free_slots_ : bufferless_free_slots_;
    const std::size_t index = selected_free_slots.back();
    selected_free_slots.pop_back();
    auto& slot = slots_[index];
    if (slot.phase != TensorLoanSlotPhase::kFree || slot.loan_handle_live ||
        slot.memory_reference->has_buffer() != buffered) {
      slot.phase = TensorLoanSlotPhase::kQuarantined;
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (!buffered) {
      auto acquired = pool_.acquire();
      if (!acquired) {
        bufferless_free_slots_.push_back(index);
        return make_unexpected(
            Error{ErrorCode::kResourceExhausted, "external Tensor export pool is exhausted"});
      }
      auto bound = slot.memory_reference->bind(std::move(acquired).value());
      if (!bound) {
        slot.phase = TensorLoanSlotPhase::kQuarantined;
        return make_unexpected(std::move(bound).error());
      }
    }
    slot.storage = static_cast<std::byte*>(slot.memory_reference->data());
    slot.shape.assign(request.shape.begin(), request.shape.end());
    slot.byte_strides.clear();
    if (request.byte_strides) {
      slot.byte_strides.assign(request.byte_strides.begin(), request.byte_strides.end());
    }
    slot.dtype = request.dtype;
    slot.byte_size = normalized.byte_size;
    configure_tensor_context(slot, normalized.item_size);
    slot.phase = TensorLoanSlotPhase::kLoaned;
    slot.loan_handle_live = true;
    ++active_loan_count_;
    retain();
    return channel_internal::TensorPublishLoanAccess::loan(&slot, &kLoanOps);
  }

  void retire_published(const std::size_t slot_index, const std::uint64_t generation) noexcept {
    Tensor retired_copy_source;
    bool release_slot_reference = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (slot_index >= slot_count_) {
        return;
      }
      auto& slot = slots_[slot_index];
      if (slot.phase != TensorLoanSlotPhase::kPublished || slot.generation != generation ||
          slot.loan_handle_live) {
        return;
      }
      retired_copy_source = std::move(slot.retained_copy_source);
      recycle_slot_locked(slot, true);
      release_slot_reference = true;
    }
    if (release_slot_reference) {
      release();
    }
    // Keep destruction outside mutex_: a foreign Tensor deleter may enter unrelated runtime code.
    static_cast<void>(retired_copy_source);
  }

  [[nodiscard]] expected<std::uint32_t, Error> emit(TensorPublishLoan loan,
                                                    EmitOptions options) noexcept {
    SampleMetadata metadata{};
    metadata.frame_id = options.frame_id.value_or(0U);
    metadata.trace_id = options.trace ? options.trace->trace_id : 0U;
    metadata.flags = options.flags;
    if (options.capture_time) {
      metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
      metadata.source_clock_id = options.capture_time->clock_id;
    }
    return emit_sample_impl(std::move(loan), metadata);
  }

  [[nodiscard]] expected<std::uint32_t, Error> emit_captured(TensorPublishLoan loan,
                                                             const std::int64_t timestamp_ns,
                                                             EmitOptions options) noexcept {
    if (options.capture_time) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto capture = make_capture_time(timestamp_ns);
    if (!capture) {
      return make_unexpected(std::move(capture).error());
    }
    options.capture_time = *capture;
    return emit(std::move(loan), std::move(options));
  }

  [[nodiscard]] expected<std::uint32_t, Error> emit_sample_impl(TensorPublishLoan loan,
                                                                SampleMetadata metadata) noexcept {
    auto taken = channel_internal::TensorPublishLoanAccess::take(loan);
    if (taken.state == nullptr || taken.ops == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (taken.ops != &kLoanOps) {
      if (taken.ops->abandon != nullptr) {
        taken.ops->abandon(taken.state);
      }
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(taken.state);
    if (slot.owner != this) {
      kLoanOps.abandon(&slot);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    // Publication time is framework-owned, and this is the caller-facing loan publication.
    // Erasing a supplied coordinate before validation rather than after keeps the coherence
    // check in valid_metadata answerable: any coordinate it still sees came from Runtime.
    metadata.publish_timestamp_ns = 0;
    metadata.publish_clock_id = {};

    Tensor tensor;
    std::uint64_t proposed_sequence{};
    ErrorCode rejection = ErrorCode::kSuccess;
    bool release_slot_reference = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!consume_loan_handle_locked(slot)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (slot.phase == TensorLoanSlotPhase::kWriting) {
        slot.phase = TensorLoanSlotPhase::kCancelPending;
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (closed_) {
        rejection = ErrorCode::kClosed;
      } else if (slot.phase != TensorLoanSlotPhase::kReady || !valid_metadata(metadata)) {
        rejection = ErrorCode::kInvalidArgument;
      } else if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        rejection = ErrorCode::kResourceExhausted;
      } else {
        proposed_sequence = operation_sequence_ + 1U;
        metadata.sequence = proposed_sequence;
        metadata.byte_size = slot.byte_size;
        metadata.payload_kind = PayloadKind::kDescriptorBacked;
        slot.phase = TensorLoanSlotPhase::kPublished;
        slot.memory_reference->arm(slot.generation);
        tensor = slot.template_tensor;
      }
      if (rejection != ErrorCode::kSuccess) {
        recycle_slot_locked(slot);
        release_slot_reference = true;
      }
    }
    if (release_slot_reference) {
      release();
      return make_unexpected(Error{rejection});
    }

    auto result = publisher_.emit_sample(Sample<Tensor>(std::move(tensor), metadata));
    inject_post_submit_result_for_test(result);
    if (result || catalog_publication_consumes_authority(result.error().code)) {
      const std::lock_guard<std::mutex> lock(mutex_);
      operation_sequence_ = proposed_sequence;
    }
    return result;
  }

  [[nodiscard]] expected<std::uint32_t, Error> emit_copy(const Tensor& source,
                                                         const ::cuda::stream_ref producer_stream,
                                                         EmitOptions options) noexcept {
    if (foreign_emit_policy_ == InterfaceTensorEmitPolicy::kRequireExportable) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto staged = stage_foreign_tensor(source, producer_stream.get(), nullptr);
    if (!staged) {
      return make_unexpected(std::move(staged).error());
    }
    SampleMetadata metadata{};
    metadata.frame_id = options.frame_id.value_or(0U);
    metadata.trace_id = options.trace ? options.trace->trace_id : 0U;
    metadata.flags = options.flags;
    if (options.capture_time) {
      metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
      metadata.source_clock_id = options.capture_time->clock_id;
    }
    return emit_prevalidated_exportable_tensor(staged->tensor, std::move(metadata));
  }

  [[nodiscard]] expected<channel_internal::ForeignTensorDisposition, Error> classify_foreign_tensor(
      const Tensor& source) const noexcept {
    if (source.data() == nullptr || source.nbytes() < 0 || !source.is_contiguous()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const TensorLoanRequest request{
        .shape = source.shape_span(),
        .dtype = source.dtype(),
        .byte_strides = std::nullopt,
    };
    NormalizedTensorLoanRequest normalized{};
    if (auto valid = validate_request(request, normalized); !valid) {
      return make_unexpected(std::move(valid).error());
    }
    if (normalized.byte_size != static_cast<std::size_t>(source.nbytes())) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    const DLDevice source_device = source.device();
    const bool placement_matches =
        (memory_kind_ == MemoryKind::kHost && source_device.device_type == kDLCPU &&
         source_device.device_id == 0) ||
        (memory_kind_ == MemoryKind::kPinnedHost && source_device.device_type == kDLCUDAHost &&
         source_device.device_id == 0) ||
        (memory_kind_ == MemoryKind::kCudaDevice && source_device.device_type == kDLCUDA &&
         source_device.device_id == device_id_.value);
    const auto& context = detail::TensorAccess::context(source);
    const bool exportable = source.byte_offset() == 0U && placement_matches && context &&
                            context->memory_reference &&
                            dynamic_cast<channel_internal::HoloMQExportableTensorMemoryReference*>(
                                context->memory_reference.get()) != nullptr;
    if (exportable) {
      return channel_internal::ForeignTensorDisposition::kExportable;
    }
    if (foreign_emit_policy_ == InterfaceTensorEmitPolicy::kRequireExportable) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return channel_internal::ForeignTensorDisposition::kCopyRequired;
  }

  // Keep validation, stream ordering, and the staged-loan ownership transfer auditable together.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<channel_internal::StagedTensorCopy, Error> stage_foreign_tensor(
      const Tensor& source, cudaStream_t producer_stream, cudaStream_t copy_stream) noexcept {
    if (source.data() == nullptr || source.nbytes() < 0 || !source.is_contiguous()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const DLDevice source_device = source.device();
    if (memory_kind_ == MemoryKind::kCudaDevice) {
      if (source_device.device_type != kDLCUDA || source_device.device_id != device_id_.value ||
          producer_stream == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
    } else if (memory_kind_ == MemoryKind::kPinnedHost) {
      if (source_device.device_type == kDLCPU) {
        if (source_device.device_id != 0 || producer_stream != nullptr) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else if (source_device.device_type == kDLCUDAHost) {
        if (source_device.device_id != 0) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else if (source_device.device_type == kDLCUDA) {
        if (source_device.device_id != device_id_.value || producer_stream == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else if (source_device.device_type == kDLCUDAManaged) {
        if (source_device.device_id != 0 || producer_stream == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    } else {
      if (source_device.device_type == kDLCPU) {
        if (source_device.device_id != 0 || producer_stream != nullptr) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else if (source_device.device_type == kDLCUDA) {
        if (source_device.device_id < 0 || producer_stream == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else if (source_device.device_type == kDLCUDAManaged) {
        if (source_device.device_id != 0 || producer_stream == nullptr) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      } else {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    }
    const TensorLoanRequest request{
        .shape = source.shape_span(),
        .dtype = source.dtype(),
        .byte_strides = std::nullopt,
    };
    auto loan = loan_tensor(request);
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto* const copy_slot = static_cast<CatalogTensorLoanSlot*>(
        channel_internal::TensorPublishLoanAccess::state(*loan));
    if (copy_slot == nullptr || copy_slot->owner != this) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool cuda_source =
        source_device.device_type == kDLCUDA || source_device.device_type == kDLCUDAManaged;
    // Pageable-host staging is synchronous, so execute it on the explicit producer stream rather
    // than introducing a cross-stream event for a pool with no declared CUDA device.
    auto* const execution_stream = memory_kind_ == MemoryKind::kHost && cuda_source
                                       ? producer_stream
                                       : (copy_stream != nullptr ? copy_stream : producer_stream);
    const bool asynchronous_copy =
        execution_stream != nullptr && (memory_kind_ == MemoryKind::kCudaDevice ||
                                        (memory_kind_ == MemoryKind::kPinnedHost && cuda_source));
    auto writer =
        asynchronous_copy ? loan->write(::cuda::stream_ref{execution_stream}) : loan->write_host();
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    if (writer->byte_size() != static_cast<std::size_t>(source.nbytes())) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    Tensor retained_copy_source;
    bool source_release_stream_registered = false;
    bool device_copy_enqueued = false;
    if (cuda_source && producer_stream != execution_stream) {
      if (copy_slot->producer_ready_event == nullptr ||
          cudaEventRecord(copy_slot->producer_ready_event, producer_stream) != cudaSuccess ||
          cudaStreamWaitEvent(execution_stream, copy_slot->producer_ready_event, 0U) !=
              cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
    if (memory_kind_ == MemoryKind::kCudaDevice) {
      retained_copy_source = source;
      if (cudaMemcpyAsync(writer->data(),
                          source.data(),
                          writer->byte_size(),
                          cudaMemcpyDeviceToDevice,
                          execution_stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      device_copy_enqueued = true;
      source_release_stream_registered =
          retained_copy_source.set_deallocation_stream(execution_stream);
    } else if (memory_kind_ == MemoryKind::kPinnedHost && cuda_source) {
      retained_copy_source = source;
      if (cudaMemcpyAsync(writer->data(),
                          source.data(),
                          writer->byte_size(),
                          cudaMemcpyDeviceToHost,
                          execution_stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      device_copy_enqueued = true;
      source_release_stream_registered =
          retained_copy_source.set_deallocation_stream(execution_stream);
    } else if (memory_kind_ == MemoryKind::kHost && cuda_source) {
      if (cudaMemcpyAsync(writer->data(),
                          source.data(),
                          writer->byte_size(),
                          cudaMemcpyDeviceToHost,
                          execution_stream) != cudaSuccess ||
          cudaStreamSynchronize(execution_stream) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    } else {
      auto copied = source.copy_to_host(writer->data(), writer->byte_size(), execution_stream);
      if (!copied) {
        return make_unexpected(std::move(copied).error());
      }
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      if (device_copy_enqueued && !source_release_stream_registered) {
        static_cast<void>(cudaStreamSynchronize(execution_stream));
      }
      return make_unexpected(std::move(committed).error());
    }
    auto taken = channel_internal::TensorPublishLoanAccess::take(*loan);
    if (taken.state == nullptr || taken.ops == nullptr) {
      if (device_copy_enqueued && !source_release_stream_registered) {
        static_cast<void>(cudaStreamSynchronize(execution_stream));
      }
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (taken.ops != &kLoanOps) {
      if (taken.ops->abandon != nullptr) {
        taken.ops->abandon(taken.state);
      }
      if (device_copy_enqueued && !source_release_stream_registered) {
        static_cast<void>(cudaStreamSynchronize(execution_stream));
      }
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(taken.state);
    if (slot.owner != this) {
      kLoanOps.abandon(&slot);
      if (device_copy_enqueued && !source_release_stream_registered) {
        static_cast<void>(cudaStreamSynchronize(execution_stream));
      }
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    Tensor staged_tensor;
    bool release_slot_reference = false;
    bool recycle_rejected_slot = false;
    ErrorCode rejection = ErrorCode::kSuccess;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!consume_loan_handle_locked(slot)) {
        rejection = ErrorCode::kInvalidArgument;
      } else {
        recycle_rejected_slot = true;
        if (closed_) {
          rejection = ErrorCode::kClosed;
        } else if (slot.phase != TensorLoanSlotPhase::kReady) {
          rejection = ErrorCode::kInvalidArgument;
        } else {
          // Transfer the loan's retained control reference to an armed backing claim before any
          // endpoint descriptor can escape. The local Tensor and every exported HoloMQ callback
          // then share the existing handle-count retirement path independently of which endpoint
          // supplied the physical pool.
          slot.phase = TensorLoanSlotPhase::kPublished;
          if (device_copy_enqueued && !source_release_stream_registered) {
            slot.retained_copy_source = std::move(retained_copy_source);
          }
          slot.memory_reference->arm(slot.generation);
          staged_tensor = slot.template_tensor;
        }
      }
      if (rejection != ErrorCode::kSuccess) {
        if (recycle_rejected_slot) {
          recycle_slot_locked(slot);
        }
        release_slot_reference = true;
      }
    }
    if (release_slot_reference) {
      release();
      if (device_copy_enqueued && !source_release_stream_registered) {
        static_cast<void>(cudaStreamSynchronize(execution_stream));
      }
      return make_unexpected(Error{rejection});
    }
    return channel_internal::StagedTensorCopy{std::move(staged_tensor)};
  }

  [[nodiscard]] expected<std::uint32_t, Error> emit_prevalidated_exportable_tensor(
      const Tensor& source, SampleMetadata metadata) noexcept {
    std::uint64_t proposed_sequence{};
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (!valid_metadata(metadata)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      proposed_sequence = operation_sequence_ + 1U;
      metadata.sequence = proposed_sequence;
      metadata.byte_size = static_cast<std::uint64_t>(source.nbytes());
      metadata.payload_kind = PayloadKind::kDescriptorBacked;
    }

    auto result = publisher_.emit_sample(Sample<Tensor>{source, metadata});
    inject_post_submit_result_for_test(result);
    if (result || catalog_publication_consumes_authority(result.error().code)) {
      const std::lock_guard<std::mutex> lock(mutex_);
      operation_sequence_ = proposed_sequence;
    }
    return result;
  }

  [[nodiscard]] expected<TimePoint, Error> make_capture_time(
      const std::int64_t timestamp_ns) const noexcept {
    if (!capture_clock_.valid()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return TimePoint{.timestamp_ns = timestamp_ns, .clock_id = capture_clock_};
  }

  [[nodiscard]] expected<void, Error> report_clock_health(const ClockHealthReport report) noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (clock_health_.capability != ClockHealthCapability::kStatusReports || health_ == nullptr ||
        !channel_internal::valid_catalog_clock_health_report(report)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max() ||
        report_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const std::uint64_t proposed_operation = operation_sequence_ + 1U;
    const std::uint64_t proposed_report = report_sequence_ + 1U;
    const auto wire = channel_internal::encode_tensor_catalog_clock_health(
        producer_generation_, proposed_operation, proposed_report, report);
    const SampleMetadata metadata{
        .sequence = proposed_operation,
        .byte_size = wire.size(),
        .payload_kind = PayloadKind::kInlineTyped,
    };
    auto result = health_->emit_phase(wire, metadata);
    if (result.phase == channel_internal::InlinePublicationPhase::kAccepted &&
        next_post_submit_result_for_test_) {
      result.phase = channel_internal::InlinePublicationPhase::kPostSubmitUncertain;
      result.error = Error{*next_post_submit_result_for_test_};
      next_post_submit_result_for_test_.reset();
    }
    if (result.phase == channel_internal::InlinePublicationPhase::kAccepted &&
        result.recipient_count == 1U) {
      operation_sequence_ = proposed_operation;
      report_sequence_ = proposed_report;
      channel_internal::TensorPublishLoanAccess::set_health_report_sequence(publisher_,
                                                                            proposed_report);
      return {};
    }
    if (result.phase == channel_internal::InlinePublicationPhase::kAccepted ||
        result.phase == channel_internal::InlinePublicationPhase::kPostSubmitUncertain) {
      operation_sequence_ = proposed_operation;
      report_sequence_ = proposed_report;
      closed_ = true;
      channel_internal::TensorPublishLoanAccess::set_health_report_sequence(publisher_,
                                                                            proposed_report);
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
    return make_unexpected(std::move(result.error));
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept {
    if (health_ != nullptr) {
      auto updated_health = health_->update_connections();
      if (!updated_health) {
        return updated_health;
      }
    }
    return publisher_.update_connections();
  }

  [[nodiscard]] bool has_subscribers() const noexcept {
    return publisher_.has_subscribers() && (health_ == nullptr || health_->has_subscribers());
  }

  [[nodiscard]] bool fail_next_terminal_pre_submit_for_test() noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || fail_next_terminal_pre_submit_for_test_) {
      return false;
    }
    fail_next_terminal_pre_submit_for_test_ = true;
    return true;
  }

  [[nodiscard]] bool exhaust_operation_sequence_for_test() noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
    operation_sequence_ = std::numeric_limits<std::uint64_t>::max();
    return true;
  }

  void fail_next_post_submit_for_test(const ErrorCode code) noexcept {
    next_post_submit_result_for_test_ = code;
  }

  void fail_next_readiness_registration_for_test() noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    fail_next_readiness_registration_for_test_ = true;
  }

  void corrupt_next_free_slot_for_test() noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto* const available =
        !buffered_free_slots_.empty()
            ? &buffered_free_slots_
            : (!bufferless_free_slots_.empty() ? &bufferless_free_slots_ : nullptr);
    if (available != nullptr) {
      slots_[available->back()].phase = TensorLoanSlotPhase::kQuarantined;
    }
  }

  [[nodiscard]] std::size_t preallocated_copy_event_count_for_test() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(slots_.begin(), slots_.end(), [](const auto& slot) {
          return slot.producer_ready_event != nullptr;
        }));
  }

  void inject_post_submit_result_for_test(expected<std::uint32_t, Error>& result) noexcept {
    if (!result || !next_post_submit_result_for_test_) {
      return;
    }
    // This source-tree-only seam runs after the real backend send succeeds. Preserve every
    // transport/backing claim and expose only the selected result to the caller.
    result = make_unexpected(Error{*next_post_submit_result_for_test_});
    next_post_submit_result_for_test_.reset();
  }

  [[nodiscard]] expected<TensorLoanWriteGuard, Error> begin_write(
      CatalogTensorLoanSlot& slot, const std::optional<::cuda::stream_ref> stream) noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (!slot.loan_handle_live || slot.phase != TensorLoanSlotPhase::kLoaned) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const bool has_stream = stream.has_value() && stream->get() != nullptr;
    if ((memory_kind_ == MemoryKind::kCudaDevice && !has_stream) ||
        (memory_kind_ == MemoryKind::kHost && has_stream) ||
        (stream.has_value() && stream->get() == nullptr)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    slot.writer_stream = stream ? stream->get() : nullptr;
    slot.phase = TensorLoanSlotPhase::kWriting;
    return channel_internal::TensorPublishLoanAccess::write_guard(&slot, &kWriterOps);
  }

  [[nodiscard]] expected<void, Error> commit_write(CatalogTensorLoanSlot& slot) noexcept {
    bool release_slot_reference = false;
    ErrorCode result = ErrorCode::kSuccess;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (slot.phase == TensorLoanSlotPhase::kCancelPending && !slot.loan_handle_live) {
        recycle_slot_locked(slot);
        release_slot_reference = true;
        result = ErrorCode::kInvalidArgument;
      } else if (slot.phase != TensorLoanSlotPhase::kWriting || !slot.loan_handle_live) {
        result = ErrorCode::kInvalidArgument;
      } else if (closed_) {
        slot.phase = TensorLoanSlotPhase::kCancelPending;
        result = ErrorCode::kStaleEndpoint;
      } else if (slot.writer_stream != nullptr &&
                 (std::exchange(fail_next_readiness_registration_for_test_, false) ||
                  !slot.memory_reference->set_readiness_stream(slot.writer_stream))) {
        slot.phase = TensorLoanSlotPhase::kCancelPending;
        result = ErrorCode::kFailure;
      } else {
        slot.phase = TensorLoanSlotPhase::kReady;
      }
    }
    if (release_slot_reference) {
      release();
    }
    if (result != ErrorCode::kSuccess) {
      return make_unexpected(Error{result});
    }
    return {};
  }

  void abandon_write(CatalogTensorLoanSlot& slot) noexcept {
    bool release_slot_reference = false;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (slot.phase == TensorLoanSlotPhase::kWriting) {
        slot.phase = TensorLoanSlotPhase::kCancelPending;
      }
      if (slot.phase == TensorLoanSlotPhase::kCancelPending && !slot.loan_handle_live) {
        recycle_slot_locked(slot);
        release_slot_reference = true;
      }
    }
    if (release_slot_reference) {
      release();
    }
  }

  [[nodiscard]] expected<void, Error> consume_loan(CatalogTensorLoanSlot& slot) noexcept {
    bool release_slot_reference = false;
    ErrorCode result = ErrorCode::kSuccess;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!consume_loan_handle_locked(slot)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (slot.phase == TensorLoanSlotPhase::kWriting) {
        slot.phase = TensorLoanSlotPhase::kCancelPending;
        result = ErrorCode::kInvalidArgument;
      } else {
        recycle_slot_locked(slot);
        release_slot_reference = true;
      }
    }
    if (release_slot_reference) {
      release();
    }
    if (result != ErrorCode::kSuccess) {
      return make_unexpected(Error{result});
    }
    return {};
  }

  [[nodiscard]] bool loan_view_live(const CatalogTensorLoanSlot& slot) const noexcept {
    return slot.loan_handle_live && slot.phase != TensorLoanSlotPhase::kFree &&
           slot.phase != TensorLoanSlotPhase::kQuarantined;
  }

 private:
  [[nodiscard]] bool consume_loan_handle_locked(CatalogTensorLoanSlot& slot) noexcept {
    if (!slot.loan_handle_live || active_loan_count_ == 0U) {
      return false;
    }
    slot.loan_handle_live = false;
    --active_loan_count_;
    return true;
  }

  [[nodiscard]] bool valid_request_header(const TensorLoanRequest& request,
                                          std::uint64_t& item_size) const noexcept {
    if (!detail::valid_tensor_dtype(request.dtype) ||
        (required_dtype_ && !dtype_equal(request.dtype, *required_dtype_)) ||
        request.shape.size() > max_rank_ || (exact_rank_ && request.shape.size() != *exact_rank_) ||
        (request.byte_strides && request.byte_strides.size() != request.shape.size())) {
      return false;
    }
    item_size = dlpack_dtype_item_size_bytes(request.dtype);
    return item_size != 0U && item_size <= std::numeric_limits<std::size_t>::max();
  }

  [[nodiscard]] bool valid_request_axes(const TensorLoanRequest& request) const noexcept {
    for (std::size_t index = 0U; index < request.shape.size(); ++index) {
      const std::int64_t extent = request.shape[index];
      if (extent < 0 || (index < axis_bounds_.size() &&
                         static_cast<std::uint64_t>(extent) > axis_bounds_[index].max_extent)) {
        return false;
      }
      if (request.byte_strides &&
          (request.byte_strides[index] < 0 ||
           (index < axis_bounds_.size() && static_cast<std::uint64_t>(request.byte_strides[index]) >
                                               axis_bounds_[index].max_abs_stride_bytes))) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool valid_contiguous_strides(const TensorLoanRequest& request,
                                              const std::uint64_t item_size) const noexcept {
    if (!request.byte_strides || stride_policy_ != TensorStridePolicy::kContiguousOnly) {
      return true;
    }
    std::uint64_t contiguous_stride = item_size;
    for (std::size_t reverse = request.shape.size(); reverse > 0U; --reverse) {
      const std::size_t index = reverse - 1U;
      if (static_cast<std::uint64_t>(request.byte_strides[index]) != contiguous_stride) {
        return false;
      }
      const auto extent =
          std::max<std::uint64_t>(static_cast<std::uint64_t>(request.shape[index]), 1U);
      if (contiguous_stride > std::numeric_limits<std::uint64_t>::max() / extent) {
        return false;
      }
      contiguous_stride *= extent;
    }
    return true;
  }

  [[nodiscard]] expected<void, Error> validate_request(
      const TensorLoanRequest& request, NormalizedTensorLoanRequest& result) const noexcept {
    std::uint64_t item_size{};
    if (!valid_request_header(request, item_size) || !valid_request_axes(request) ||
        !valid_contiguous_strides(request, item_size)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    const std::optional<std::span<const std::int64_t>> strides =
        request.byte_strides
            ? std::optional<std::span<const std::int64_t>>{request.byte_strides.value()}
            : std::nullopt;
    const auto layout = detail::validate_tensor_layout(
        request.shape, request.dtype, strides, detail::TensorStrideUnit::kBytes);
    if (!layout || layout->byte_span > max_byte_span_ ||
        layout->byte_span > std::numeric_limits<std::size_t>::max()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    result.byte_size = static_cast<std::size_t>(layout->byte_span);
    result.item_size = static_cast<std::size_t>(item_size);
    return {};
  }

  [[nodiscard]] DLDevice tensor_dl_device() const noexcept {
    switch (memory_kind_) {
      case MemoryKind::kPinnedHost:
        return DLDevice{kDLCUDAHost, 0};
      case MemoryKind::kCudaDevice:
        return DLDevice{kDLCUDA, device_id_.value};
      case MemoryKind::kHost:
      case MemoryKind::kUnknown:
      case MemoryKind::kCudaManaged:
      default:
        return DLDevice{kDLCPU, 0};
    }
  }

  void configure_tensor_context(CatalogTensorLoanSlot& slot, const std::size_t item_size) noexcept {
    auto& context = *slot.context;
    context.dl_shape.assign(slot.shape.begin(), slot.shape.end());
    context.dl_strides.clear();
    if (!slot.byte_strides.empty()) {
      for (const std::int64_t stride : slot.byte_strides) {
        context.dl_strides.push_back(stride / static_cast<std::int64_t>(item_size));
      }
    }
    auto& tensor = context.tensor.dl_tensor;
    tensor.data = slot.storage;
    tensor.device = tensor_dl_device();
    tensor.ndim = static_cast<std::int32_t>(context.dl_shape.size());
    tensor.dtype = slot.dtype;
    tensor.shape = context.dl_shape.empty() ? nullptr : context.dl_shape.data();
    tensor.strides = context.dl_strides.empty() ? nullptr : context.dl_strides.data();
    tensor.byte_offset = 0U;
  }

  [[nodiscard]] bool valid_metadata(const SampleMetadata& metadata) const noexcept {
    if (metadata.version.major != 1U || metadata.version.minor != 0U ||
        (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U) {
      return false;
    }
    // Only Runtime's prevalidated export path can still present a coordinate here. The
    // caller-facing loan publication erases the pair before validating, and emit_copy
    // builds its own metadata. That path forwards what it stamped rather than erasing it,
    // so the pair must name a clock this catalog declares. A bare timestamp with no clock
    // names no domain, and an identity past the declared range resolves to no record at
    // all; either one reaches a peer as a plausible number it cannot place on a timeline.
    if (!channel_internal::publication_coordinate_declared(metadata, catalog_clock_count_)) {
      return false;
    }
    if (capture_clock_.valid()) {
      return metadata.source_clock_id == capture_clock_;
    }
    return !metadata.source_clock_id.valid() && metadata.capture_timestamp_ns == 0;
  }

  void recycle_slot_locked(CatalogTensorLoanSlot& slot,
                           const bool release_physical_buffer = false) noexcept {
    slot.memory_reference->disarm();
    if (release_physical_buffer) {
      slot.memory_reference->release_buffer();
      slot.storage = nullptr;
    } else {
      // An unpublished/cancelled loan never exported its BufferHandle, so retaining this already
      // acquired lease avoids a per-loan HoloMQ control-object allocation on the warmed local path.
      slot.memory_reference->prepare_reuse();
    }
    slot.loan_handle_live = false;
    slot.byte_size = 0U;
    slot.dtype = {};
    slot.writer_stream = nullptr;
    slot.shape.clear();
    slot.byte_strides.clear();
    if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
      slot.phase = TensorLoanSlotPhase::kQuarantined;
      return;
    }
    ++slot.generation;
    slot.phase = TensorLoanSlotPhase::kFree;
    if (release_physical_buffer) {
      bufferless_free_slots_.push_back(slot.index);
    } else {
      buffered_free_slots_.push_back(slot.index);
    }
  }

  static void abandon_writer(void* state) noexcept {
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(state);
    slot.owner->abandon_write(slot);
  }

  static void* writer_data(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.phase == TensorLoanSlotPhase::kWriting ? slot.storage : nullptr;
  }

  static std::size_t writer_byte_size(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.phase == TensorLoanSlotPhase::kWriting ? slot.byte_size : 0U;
  }

  static std::optional<::cuda::stream_ref> writer_stream(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    if (slot.phase != TensorLoanSlotPhase::kWriting || slot.writer_stream == nullptr) {
      return std::nullopt;
    }
    return ::cuda::stream_ref{slot.writer_stream};
  }

  static expected<void, Error> commit_writer(void* state) noexcept {
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(state);
    return slot.owner->commit_write(slot);
  }

  static void abandon_loan(void* state) noexcept {
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(state);
    (void)slot.owner->consume_loan(slot);
  }

  static TensorShapeView loan_shape(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.owner->loan_view_live(slot) ? TensorShapeView{slot.shape} : TensorShapeView{};
  }

  static TensorByteStridesView loan_byte_strides(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    if (!slot.owner->loan_view_live(slot) || slot.byte_strides.empty()) {
      return std::nullopt;
    }
    return TensorByteStridesView{slot.byte_strides};
  }

  static DLDataType loan_dtype(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.owner->loan_view_live(slot) ? slot.dtype : DLDataType{};
  }

  static DeviceId loan_device_id(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.owner->loan_view_live(slot) ? slot.owner->device_id_ : DeviceId{};
  }

  static MemoryKind loan_memory_kind(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.owner->loan_view_live(slot) ? slot.owner->memory_kind_ : MemoryKind::kUnknown;
  }

  static std::size_t loan_byte_size(const void* state) noexcept {
    const auto& slot = *static_cast<const CatalogTensorLoanSlot*>(state);
    return slot.owner->loan_view_live(slot) ? slot.byte_size : 0U;
  }

  static expected<TensorLoanWriteGuard, Error> begin_cuda_write(
      void* state, const ::cuda::stream_ref stream) noexcept {
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(state);
    return slot.owner->begin_write(slot, stream);
  }

  static expected<TensorLoanWriteGuard, Error> begin_host_write(void* state) noexcept {
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(state);
    return slot.owner->begin_write(slot, std::nullopt);
  }

  static expected<void, Error> cancel_loan(void* state) noexcept {
    auto& slot = *static_cast<CatalogTensorLoanSlot*>(state);
    return slot.owner->consume_loan(slot);
  }

  static constexpr channel_internal::TensorLoanWriteGuardOps kWriterOps{
      .abandon = abandon_writer,
      .data = writer_data,
      .byte_size = writer_byte_size,
      .stream = writer_stream,
      .commit = commit_writer,
  };

  static constexpr channel_internal::TensorPublishLoanOps kLoanOps{
      .abandon = abandon_loan,
      .shape = loan_shape,
      .byte_strides = loan_byte_strides,
      .dtype = loan_dtype,
      .device_id = loan_device_id,
      .memory_kind = loan_memory_kind,
      .byte_size = loan_byte_size,
      .write = begin_cuda_write,
      .write_host = begin_host_write,
      .cancel = cancel_loan,
  };

  ManualTensorPublisher publisher_;
  std::unique_ptr<channel_internal::InlinePayloadPublisher> health_;
  holomq::BufferPool pool_;
  ClockId capture_clock_{};
  // Declared clock count of the catalog owning this endpoint, so a forwarded publication coordinate
  // can be checked against the identities a peer reading that catalog is able to resolve.
  std::uint64_t catalog_clock_count_{};
  MemoryKind memory_kind_{MemoryKind::kUnknown};
  DeviceId device_id_{};
  std::optional<DLDataType> required_dtype_;
  std::optional<std::uint32_t> exact_rank_;
  std::uint32_t max_rank_{};
  std::uint64_t max_byte_span_{};
  std::vector<TensorAxisBounds> axis_bounds_;
  TensorStridePolicy stride_policy_{TensorStridePolicy::kContiguousOnly};
  InterfaceTensorEmitPolicy foreign_emit_policy_{InterfaceTensorEmitPolicy::kRequireExportable};
  ClockHealthOptions clock_health_{};
  std::uint32_t max_loaned_samples_{};
  std::uint32_t active_loan_count_{};
  std::size_t slot_count_{};
  std::vector<CatalogTensorLoanSlot> slots_;
  std::vector<std::size_t> buffered_free_slots_;
  std::vector<std::size_t> bufferless_free_slots_;
  mutable std::mutex mutex_;
  std::atomic<std::uint32_t> references_{1U};
  std::uint64_t producer_generation_{};
  std::uint64_t operation_sequence_{};
  std::uint64_t report_sequence_{};
  bool closed_{};
  bool fail_next_terminal_pre_submit_for_test_{};
  std::optional<ErrorCode> next_post_submit_result_for_test_;
  bool fail_next_readiness_registration_for_test_{};
};

void CatalogTensorSlotMemoryReference::release_tensor_handle() noexcept {
  if (overflowed_.load(std::memory_order_acquire)) {
    return;
  }
  auto previous = handle_count_.load(std::memory_order_relaxed);
  while (previous != 0U) {
    if (handle_count_.compare_exchange_weak(
            previous, previous - 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      if (previous == 2U && armed_.exchange(false, std::memory_order_acq_rel)) {
        owner_->retire_published(slot_index_, generation_.load(std::memory_order_relaxed));
      }
      return;
    }
  }
  overflowed_.store(true, std::memory_order_release);
}

void destroy_catalog_tensor_publisher(void* state) noexcept {
  static_cast<CatalogTensorPublisherControl*>(state)->destroy();
}

expected<void, Error> close_catalog_tensor_publisher(void* state) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->close_stream();
}

expected<TensorPublishLoan, Error> catalog_loan_tensor(void* state,
                                                       TensorLoanRequest request) noexcept {
  try {
    return static_cast<CatalogTensorPublisherControl*>(state)->loan_tensor(request);
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

expected<std::uint32_t, Error> catalog_emit(void* state, TensorPublishLoan loan,
                                            EmitOptions options) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->emit(std::move(loan),
                                                                  std::move(options));
}

expected<std::uint32_t, Error> catalog_emit_captured(void* state, TensorPublishLoan loan,
                                                     const std::int64_t timestamp_ns,
                                                     EmitOptions options) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->emit_captured(
      std::move(loan), timestamp_ns, std::move(options));
}

expected<std::uint32_t, Error> catalog_emit_sample(void* state, TensorPublishLoan loan,
                                                   SampleMetadata metadata) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->emit_sample_impl(std::move(loan),
                                                                              metadata);
}

expected<std::uint32_t, Error> catalog_emit_copy(void* state, const Tensor& source,
                                                 const ::cuda::stream_ref stream,
                                                 EmitOptions options) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->emit_copy(
      source, stream, std::move(options));
}

expected<channel_internal::ForeignTensorDisposition, Error> catalog_classify_foreign_tensor(
    const void* state, const Tensor& source) noexcept {
  return static_cast<const CatalogTensorPublisherControl*>(state)->classify_foreign_tensor(source);
}

expected<channel_internal::StagedTensorCopy, Error> catalog_stage_foreign_tensor(
    void* state, const Tensor& source, cudaStream_t producer_stream,
    cudaStream_t copy_stream) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->stage_foreign_tensor(
      source, producer_stream, copy_stream);
}

expected<std::uint32_t, Error> catalog_emit_prevalidated_exportable_tensor(
    void* state, const Tensor& source, SampleMetadata metadata) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->emit_prevalidated_exportable_tensor(
      source, std::move(metadata));
}

expected<TimePoint, Error> catalog_make_capture_time(const void* state,
                                                     const std::int64_t timestamp_ns) noexcept {
  return static_cast<const CatalogTensorPublisherControl*>(state)->make_capture_time(timestamp_ns);
}

expected<void, Error> catalog_report_clock_health(void* state,
                                                  const ClockHealthReport report) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->report_clock_health(report);
}

expected<void, Error> catalog_update_connections(void* state) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->update_connections();
}

bool catalog_has_subscribers(const void* state) noexcept {
  return static_cast<const CatalogTensorPublisherControl*>(state)->has_subscribers();
}

bool catalog_exhaust_operation_sequence_for_test(void* state) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)->exhaust_operation_sequence_for_test();
}

bool catalog_fail_next_terminal_pre_submit_for_test(void* state) noexcept {
  return static_cast<CatalogTensorPublisherControl*>(state)
      ->fail_next_terminal_pre_submit_for_test();
}

void catalog_fail_next_post_submit_for_test(void* state, const ErrorCode code) noexcept {
  static_cast<CatalogTensorPublisherControl*>(state)->fail_next_post_submit_for_test(code);
}

void catalog_fail_next_readiness_registration_for_test(void* state) noexcept {
  static_cast<CatalogTensorPublisherControl*>(state)->fail_next_readiness_registration_for_test();
}

void catalog_corrupt_next_free_slot_for_test(void* state) noexcept {
  static_cast<CatalogTensorPublisherControl*>(state)->corrupt_next_free_slot_for_test();
}

std::size_t catalog_preallocated_copy_event_count_for_test(const void* state) noexcept {
  return static_cast<const CatalogTensorPublisherControl*>(state)
      ->preallocated_copy_event_count_for_test();
}

constexpr channel_internal::TensorPublisherOps kCatalogTensorPublisherOps{
    .destroy = destroy_catalog_tensor_publisher,
    .close = close_catalog_tensor_publisher,
    .loan_tensor = catalog_loan_tensor,
    .emit = catalog_emit,
    .emit_captured = catalog_emit_captured,
    .emit_sample = catalog_emit_sample,
    .emit_copy = catalog_emit_copy,
    .classify_foreign_tensor = catalog_classify_foreign_tensor,
    .stage_foreign_tensor = catalog_stage_foreign_tensor,
    .emit_prevalidated_exportable_tensor = catalog_emit_prevalidated_exportable_tensor,
    .make_capture_time = catalog_make_capture_time,
    .report_clock_health = catalog_report_clock_health,
    .update_connections = catalog_update_connections,
    .has_subscribers = catalog_has_subscribers,
    .exhaust_operation_sequence_for_test = catalog_exhaust_operation_sequence_for_test,
    .fail_next_terminal_pre_submit_for_test = catalog_fail_next_terminal_pre_submit_for_test,
    .fail_next_post_submit_for_test = catalog_fail_next_post_submit_for_test,
    .fail_next_readiness_registration_for_test = catalog_fail_next_readiness_registration_for_test,
    .corrupt_next_free_slot_for_test = catalog_corrupt_next_free_slot_for_test,
    .preallocated_copy_event_count_for_test = catalog_preallocated_copy_event_count_for_test,
};

[[nodiscard]] ChannelQueuePolicy channel_queue_policy(const QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return ChannelQueuePolicy::kBlocking;
    case QueuePolicy::kDropOldest:
      return ChannelQueuePolicy::kDropOldest;
    default:
      return ChannelQueuePolicy::kBlocking;
  }
}

[[nodiscard]] bool supported_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking || policy == QueuePolicy::kDropOldest;
}

[[nodiscard]] ChannelOptions base_tensor_channel_options(
    const detail::ChannelEndpointRecord& endpoint) noexcept {
  return ChannelOptions{
      .queue_policy = channel_queue_policy(endpoint.service_static.queue_policy),
      .max_subscriber_queue_depth = endpoint.service_static.max_subscriber_depth,
      .subscriber_queue_depth = endpoint.service_static.max_subscriber_depth,
      .max_subscriber_borrowed_samples = endpoint.service_static.max_subscriber_borrowed,
      .max_borrowed_samples = endpoint.service_static.max_subscriber_borrowed,
      .publisher_history_depth = endpoint.service_static.publisher_history_depth,
      .max_loaned_samples = 1U,
      .expected_recipient_count = 1U,
      .enable_event_wake = endpoint.service_static.event_wake,
      .replacement_attribution_required =
          endpoint.service_static.queue_policy != QueuePolicy::kBlocking,
      .tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle,
  };
}

[[nodiscard]] ChannelOptions tensor_channel_options(
    const detail::ChannelEndpointRecord& endpoint,
    const ValidatedTensorEndpoint& validated) noexcept {
  ChannelOptions options = base_tensor_channel_options(endpoint);
  options.max_loaned_samples = validated.max_loaned_samples;
  return options;
}

[[nodiscard]] ChannelOptions tensor_subscriber_channel_options(
    const detail::ChannelEndpointRecord& endpoint) noexcept {
  ChannelOptions options = base_tensor_channel_options(endpoint);
  options.replacement_attribution_required = false;
  if (endpoint.subscriber_request) {
    const auto& subscriber_request = endpoint.subscriber_request.value();
    options.subscriber_queue_depth = subscriber_request.queue_depth;
    options.max_borrowed_samples = subscriber_request.max_borrowed_samples;
  }
  return options;
}

[[nodiscard]] ChannelOptions tensor_health_channel_options(ChannelOptions options) noexcept {
  options.queue_policy = ChannelQueuePolicy::kBlocking;
  options.replacement_attribution_required = false;
  return options;
}

[[nodiscard]] std::chrono::nanoseconds effective_attach_bound(
    const ValidatedTensorEndpoint& endpoint, const ChannelAttachOptions& options) noexcept {
  return std::min(options.timeout.value_or(endpoint.max_wait), endpoint.max_wait);
}

[[nodiscard]] expected<void, Error> wait_for_required_tensor_subscriber(
    const detail::ChannelEndpointRecord& endpoint, const ValidatedTensorEndpoint& validated,
    const ChannelAttachOptions& options, ManualTensorPublisher& data,
    channel_internal::InlinePayloadPublisher* health) noexcept {
  if (endpoint.attach.peer == PeerRequirement::kOptional) {
    return {};
  }

  constexpr std::chrono::nanoseconds kPollInterval = std::chrono::milliseconds{1};
  const std::chrono::nanoseconds bound = effective_attach_bound(validated, options);
  const auto started = std::chrono::steady_clock::now();
  while (true) {
    if (health != nullptr) {
      auto refreshed_health = health->update_connections();
      if (!refreshed_health) {
        return make_unexpected(std::move(refreshed_health).error());
      }
    }
    auto refreshed_data = data.update_connections();
    if (!refreshed_data) {
      return make_unexpected(std::move(refreshed_data).error());
    }
    if (data.has_subscribers() && (health == nullptr || health->has_subscribers())) {
      return {};
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    if (elapsed >= bound) {
      return make_unexpected(Error{ErrorCode::kTimeout, "required Tensor subscriber timed out"});
    }
    std::this_thread::sleep_for(std::min(kPollInterval, bound - elapsed));
  }
}

[[nodiscard]] expected<ValidatedTensorEndpoint, Error> validate_tensor_endpoint_common(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options,
    const TopicDirection expected_direction) noexcept {
  if (!endpoint.service_static.tensor) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto& tensor = endpoint.service_static.tensor.value();
  const bool has_memory_kind = tensor.memory_kind.has_value();
  const MemoryKind memory_kind = tensor.memory_kind.value_or(MemoryKind::kUnknown);
  const std::uint32_t max_rank = tensor.max_rank.value_or(0U);
  const std::uint64_t max_byte_span = tensor.max_byte_span.value_or(0U);
  const std::chrono::nanoseconds max_wait =
      endpoint.attach.max_wait.value_or(std::chrono::nanoseconds::zero());
  const std::chrono::nanoseconds requested_timeout = options.timeout.value_or(max_wait);

  if (endpoint.direction != expected_direction ||
      endpoint.service_static.payload_identity.kind != PayloadIdentityKind::kTensor ||
      endpoint.service_static.payload_identity.payload_kind != PayloadKind::kDescriptorBacked ||
      !has_memory_kind || max_rank == 0U || max_byte_span == 0U ||
      endpoint.service_static.id.empty() || endpoint.service_static.max_subscriber_depth == 0U ||
      endpoint.service_static.max_subscriber_borrowed == 0U ||
      !supported_queue_policy(endpoint.service_static.queue_policy) ||
      endpoint.service_static.publisher_history_depth != 0U || max_wait.count() <= 0 ||
      (endpoint.attach.peer != PeerRequirement::kOptional &&
       endpoint.attach.peer != PeerRequirement::kRequired) ||
      (options.timeout && (requested_timeout.count() <= 0 || requested_timeout > max_wait))) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (memory_kind != MemoryKind::kHost && memory_kind != MemoryKind::kPinnedHost &&
      memory_kind != MemoryKind::kCudaDevice) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return ValidatedTensorEndpoint{
      .tensor = &tensor,
      .memory_kind = memory_kind,
      .max_rank = max_rank,
      .max_byte_span = max_byte_span,
      .max_wait = max_wait,
  };
}

[[nodiscard]] expected<ValidatedTensorEndpoint, Error> validate_tensor_endpoint(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options,
    const std::optional<std::uint32_t> publisher_loan_bound = std::nullopt,
    const TopicDirection publisher_direction = TopicDirection::kImport) noexcept {
  const std::optional<std::uint32_t> catalog_loan_bound =
      endpoint.publisher_request ? endpoint.publisher_request->max_loaned_samples : std::nullopt;
  const std::uint32_t effective_loan_bound =
      publisher_loan_bound.value_or(catalog_loan_bound.value_or(0U));
  if (effective_loan_bound == 0U || (publisher_loan_bound && catalog_loan_bound &&
                                     publisher_loan_bound.value() != catalog_loan_bound.value())) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto validated = validate_tensor_endpoint_common(endpoint, options, publisher_direction);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  const std::uint64_t effective_delivery_depth = endpoint.service_static.max_subscriber_depth;
  const std::uint64_t publisher_pool_slots = static_cast<std::uint64_t>(effective_loan_bound) +
                                             effective_delivery_depth +
                                             endpoint.service_static.max_subscriber_borrowed;
  if (publisher_pool_slots > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted,
              "external Tensor pool capacity exceeds the supported slot-count range"});
  }
  validated->max_loaned_samples = effective_loan_bound;
  validated->publisher_pool_slots = static_cast<std::uint32_t>(publisher_pool_slots);
  validated->clock_health =
      endpoint.publisher_request ? endpoint.publisher_request->clock_health : endpoint.clock_health;
  return validated;
}

[[nodiscard]] expected<ValidatedTensorEndpoint, Error> validate_tensor_subscriber_endpoint(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options,
    const TopicDirection subscriber_direction) noexcept {
  auto validated = validate_tensor_endpoint_common(endpoint, options, subscriber_direction);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  if (subscriber_direction == TopicDirection::kExport) {
    if (!endpoint.subscriber_request || endpoint.subscriber_request->queue_depth == 0U ||
        endpoint.subscriber_request->max_borrowed_samples == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    validated->max_borrowed_samples = endpoint.subscriber_request->max_borrowed_samples;
  } else {
    if (!endpoint.publisher_request || !endpoint.publisher_request->max_loaned_samples ||
        *endpoint.publisher_request->max_loaned_samples == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    validated->max_borrowed_samples = endpoint.service_static.max_subscriber_borrowed;
  }
  validated->clock_health = endpoint.clock_health;
  return validated;
}

[[nodiscard]] expected<void, Error> wait_for_required_tensor_publisher(
    const detail::ChannelEndpointRecord& endpoint, const ValidatedTensorEndpoint& validated,
    const ChannelAttachOptions& options, ChannelSubscriber<Tensor>& subscriber) {
  if (endpoint.attach.peer == PeerRequirement::kOptional) {
    return {};
  }
  auto* barrier = g_required_publisher_wait_barrier.exchange(nullptr, std::memory_order_acq_rel);
  if (barrier != nullptr) {
    barrier->reached.release();
    barrier->resume.acquire();
  }
  constexpr std::chrono::nanoseconds kPollInterval = std::chrono::milliseconds{1};
  const std::chrono::nanoseconds bound = effective_attach_bound(validated, options);
  const auto started = std::chrono::steady_clock::now();
  while (true) {
    auto refreshed = subscriber.refresh_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (barrier != nullptr && barrier->pause_after_refresh) {
      barrier->refreshed.release();
      barrier->resume_after_refresh.acquire();
      barrier = nullptr;
    }
    auto ready = subscriber.has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    // A short-lived graph publisher may enqueue terminal control and retire before this required
    // attachment observes the refreshed peer count. Non-consuming readiness proves that the peer
    // existed and left usable work; do not turn retained EOS into a spurious attach timeout.
    if (subscriber.has_publishers() || *ready) {
      return {};
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    if (elapsed >= bound) {
      return make_unexpected(Error{ErrorCode::kTimeout, "required Tensor publisher timed out"});
    }
    std::this_thread::sleep_for(std::min(kPollInterval, bound - elapsed));
  }
}

struct CatalogTensorPool {
  holomq::BufferPool pool;
  DeviceId synchronization_device{};
};

[[nodiscard]] expected<CatalogTensorPool, Error> create_catalog_tensor_pool(
    const ValidatedTensorEndpoint& endpoint, const std::optional<DeviceId> planned_device) {
  if (endpoint.max_byte_span > std::numeric_limits<std::size_t>::max()) {
    return channel_internal::make_unexpected_nothrow<CatalogTensorPool>(
        ErrorCode::kResourceExhausted, "external Tensor pool byte span exceeds address space");
  }

  // Host export still uses a CUDA VMM generation sidecar. Device export additionally creates its
  // VMM payload allocation on the compiler-selected device (or the standalone caller's current
  // device when no plan authority exists).
  const ScopedCUDADevice selected{planned_device};
  if (!selected.active()) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "external Tensor export device is unavailable"});
  }
  if (cudaFree(nullptr) != cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "external Tensor export requires CUDA VMM support"});
  }

  // FD-export infrastructure is private and lazy: constructing a catalog Tensor publisher is the
  // first operation that needs an exportable allocation. BufferPool retains the ContextImpl (and
  // its fd-pass server), so the temporary Context shell can be destroyed after pool creation.
  auto context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
  if (!context) {
    return make_unexpected(channel_internal::from_holomq_error(
        std::move(context).error(), "failed to create external Tensor export context"));
  }
  auto pool =
      endpoint.memory_kind == MemoryKind::kCudaDevice
          ? context->create_device_buffer_pool(static_cast<std::size_t>(endpoint.max_byte_span),
                                               endpoint.publisher_pool_slots)
      : endpoint.memory_kind == MemoryKind::kPinnedHost
          ? context->create_host_pinned_buffer_pool(
                static_cast<std::size_t>(endpoint.max_byte_span),
                endpoint.publisher_pool_slots,
                CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP)
          : context->create_host_buffer_pool(static_cast<std::size_t>(endpoint.max_byte_span),
                                             endpoint.publisher_pool_slots);
  if (!pool) {
    return make_unexpected(channel_internal::from_holomq_error(
        std::move(pool).error(), "failed to create external Tensor export pool"));
  }
  return CatalogTensorPool{
      .pool = std::move(pool).value(),
      .synchronization_device =
          endpoint.memory_kind == MemoryKind::kHost ? DeviceId{} : selected.selected_device(),
  };
}

}  // namespace

expected<ChannelSubscriber<Tensor>, Error>
channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(
    ChannelRuntime& runtime, const ChannelEndpointCatalog::EndpointView endpoint) {
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto validated = validate_tensor_endpoint(*record, ChannelAttachOptions{});
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  return runtime.create_tensor_subscriber(
      detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint),
      tensor_channel_options(*record, *validated));
}

void channel_internal::TensorPublisherTestAccess::pause_next_required_publisher_wait_for_test(
    RequiredPublisherWaitBarrier& barrier) noexcept {
  g_required_publisher_wait_barrier.store(&barrier, std::memory_order_release);
}

expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
channel_internal::TensorPublisherTestAccess::create_catalog_clock_health_subscriber(
    ChannelRuntime& runtime, const ChannelEndpointCatalog::EndpointView endpoint) {
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (runtime.impl_ == nullptr || record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto validated = validate_tensor_endpoint(*record, ChannelAttachOptions{});
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  if (validated->clock_health.capability != ClockHealthCapability::kStatusReports) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  const ChannelOptions data_options = tensor_channel_options(*record, *validated);
  return channel_internal::create_inline_payload_subscriber(
      runtime.impl_->context_,
      detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint) + ".clock_health.v1",
      channel_internal::kTensorCatalogClockHealthControlBytes,
      tensor_health_channel_options(data_options),
      runtime.impl_->context_.pubsub_capabilities());
}

expected<ChannelPublisher<Tensor>, Error> ChannelRuntime::create_catalog_tensor_publisher(
    const ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options,
    const std::optional<std::uint32_t> publisher_loan_bound,
    const TopicDirection publisher_direction, const bool enforce_peer_requirement,
    const std::optional<DeviceId> planned_device) {
  return create_catalog_tensor_publisher(endpoint,
                                         std::move(options),
                                         publisher_loan_bound,
                                         publisher_direction,
                                         enforce_peer_requirement,
                                         planned_device,
                                         channel_internal::PublisherPressureOptions{});
}

expected<ChannelPublisher<Tensor>, Error> ChannelRuntime::create_catalog_tensor_publisher(
    const ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options,
    const std::optional<std::uint32_t> publisher_loan_bound,
    const TopicDirection publisher_direction, const bool enforce_peer_requirement,
    const std::optional<DeviceId> planned_device,
    const channel_internal::PublisherPressureOptions& pressure) {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto validated =
      validate_tensor_endpoint(*record, options, publisher_loan_bound, publisher_direction);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }

  try {
    const std::string binding =
        detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint);
    const ChannelOptions data_options = tensor_channel_options(*record, *validated);
    if (validated->memory_kind == MemoryKind::kHost && planned_device && planned_device->valid()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError,
          fmt::format("Tensor catalog publisher '{}' has planned_device={} for host-pageable "
                      "storage",
                      binding,
                      planned_device->value)));
    }
    if ((validated->memory_kind == MemoryKind::kPinnedHost ||
         validated->memory_kind == MemoryKind::kCudaDevice) &&
        planned_device && !planned_device->valid()) {
      return make_unexpected(
          make_channel_error(ErrorCode::kProtocolError,
                             fmt::format("Tensor catalog publisher '{}' has "
                                         "planned_device=invalid for device-bound storage",
                                         binding)));
    }
    auto pool = create_catalog_tensor_pool(*validated, planned_device);
    if (!pool) {
      return make_unexpected(std::move(pool).error());
    }
    const channel_internal::PublisherPressureOptions data_pressure =
        pressure.for_queue(data_options.queue_policy);
    auto data = create_tensor_publisher(binding, data_options, data_pressure);
    if (!data) {
      return make_unexpected(std::move(data).error());
    }
    const std::uint64_t producer_generation = data->producer_generation();
    if (producer_generation == 0U) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }

    std::unique_ptr<channel_internal::InlinePayloadPublisher> health;
    if (validated->clock_health.capability == ClockHealthCapability::kStatusReports) {
      const ChannelOptions health_options = tensor_health_channel_options(data_options);
      const channel_internal::PublisherPressureOptions health_pressure =
          pressure.for_queue(health_options.queue_policy);
      auto created_health = channel_internal::create_inline_payload_publisher(
          impl_->context_,
          binding + ".clock_health.v1",
          channel_internal::kTensorCatalogClockHealthControlBytes,
          health_options,
          impl_->context_.pubsub_capabilities(),
          health_pressure);
      if (!created_health) {
        return make_unexpected(std::move(created_health).error());
      }
      health = std::move(created_health).value();
    }

    if (enforce_peer_requirement) {
      auto attached =
          wait_for_required_tensor_subscriber(*record, *validated, options, *data, health.get());
      if (!attached) {
        return make_unexpected(std::move(attached).error());
      }
    }

    const DeviceId synchronization_device = pool->synchronization_device;
    auto control = std::make_unique<CatalogTensorPublisherControl>(
        std::move(data).value(),
        std::move(health),
        std::move(pool->pool),
        *validated,
        synchronization_device,
        detail::ChannelEndpointCatalogAccess::capture_clock_id(endpoint),
        detail::ChannelEndpointCatalogAccess::clock_count(endpoint),
        producer_generation);
    auto initialized = control->initialize();
    if (!initialized) {
      return make_unexpected(std::move(initialized).error());
    }
    return channel_internal::TensorPublishLoanAccess::publisher(control.release(),
                                                                &kCatalogTensorPublisherOps);
  } catch (const std::length_error&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

expected<ChannelSubscriber<Tensor>, Error> ChannelRuntime::create_catalog_tensor_subscriber(
    const ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options,
    const TopicDirection subscriber_direction, const bool enforce_peer_requirement) {
  return create_catalog_tensor_subscriber(
      endpoint, std::move(options), subscriber_direction, enforce_peer_requirement, 0U);
}

expected<ChannelSubscriber<Tensor>, Error> ChannelRuntime::create_catalog_tensor_subscriber(
    const ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options,
    const TopicDirection subscriber_direction, const bool enforce_peer_requirement,
    const std::uint32_t receive_control_capacity) {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto validated = validate_tensor_subscriber_endpoint(*record, options, subscriber_direction);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  if (receive_control_capacity != 0U &&
      receive_control_capacity < validated->max_borrowed_samples) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  try {
    const std::string binding =
        detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint);
    const ChannelOptions data_options = tensor_subscriber_channel_options(*record);
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> health;
    if (validated->clock_health.capability == ClockHealthCapability::kStatusReports) {
      auto created_health = channel_internal::create_inline_payload_subscriber(
          impl_->context_,
          binding + ".clock_health.v1",
          channel_internal::kTensorCatalogClockHealthControlBytes,
          tensor_health_channel_options(data_options),
          impl_->context_.pubsub_capabilities());
      if (!created_health) {
        return make_unexpected(std::move(created_health).error());
      }
      health = std::move(*created_health);
    }
    auto subscriber = create_tensor_subscriber(
        binding,
        data_options,
        *validated->tensor,
        receive_control_capacity == 0U ? validated->max_borrowed_samples : receive_control_capacity,
        std::move(health),
        detail::ChannelEndpointCatalogAccess::clock_count(endpoint));
    if (!subscriber) {
      return make_unexpected(std::move(subscriber).error());
    }
    if (enforce_peer_requirement) {
      auto attached = wait_for_required_tensor_publisher(*record, *validated, options, *subscriber);
      if (!attached) {
        return make_unexpected(std::move(attached).error());
      }
    }
    return subscriber;
  } catch (const std::length_error&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

TensorLoanWriteGuard::~TensorLoanWriteGuard() noexcept {
  if (state_ != nullptr && ops_ != nullptr && ops_->abandon != nullptr) {
    ops_->abandon(state_);
  }
}

void* TensorLoanWriteGuard::data() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->data != nullptr ? ops_->data(state_)
                                                                       : nullptr;
}

std::size_t TensorLoanWriteGuard::byte_size() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->byte_size != nullptr
             ? ops_->byte_size(state_)
             : 0U;
}

std::optional<::cuda::stream_ref> TensorLoanWriteGuard::stream() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->stream != nullptr ? ops_->stream(state_)
                                                                         : std::nullopt;
}

expected<void, Error> TensorLoanWriteGuard::commit() && {
  if (state_ == nullptr || ops_ == nullptr || ops_->commit == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  void* const state = std::exchange(state_, nullptr);
  const auto* const ops = std::exchange(ops_, nullptr);
  return ops->commit(state);
}

TensorPublishLoan::~TensorPublishLoan() noexcept {
  if (state_ != nullptr && ops_ != nullptr && ops_->abandon != nullptr) {
    ops_->abandon(state_);
  }
}

TensorShapeView TensorPublishLoan::shape() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->shape != nullptr ? ops_->shape(state_)
                                                                        : TensorShapeView{};
}

TensorByteStridesView TensorPublishLoan::byte_strides() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->byte_strides != nullptr
             ? ops_->byte_strides(state_)
             : TensorByteStridesView{std::nullopt};
}

DLDataType TensorPublishLoan::dtype() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->dtype != nullptr ? ops_->dtype(state_)
                                                                        : DLDataType{};
}

DeviceId TensorPublishLoan::device_id() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->device_id != nullptr
             ? ops_->device_id(state_)
             : DeviceId{};
}

MemoryKind TensorPublishLoan::memory_kind() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->memory_kind != nullptr
             ? ops_->memory_kind(state_)
             : MemoryKind::kUnknown;
}

std::size_t TensorPublishLoan::byte_size() const noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->byte_size != nullptr
             ? ops_->byte_size(state_)
             : 0U;
}

expected<TensorLoanWriteGuard, Error> TensorPublishLoan::write(
    const ::cuda::stream_ref producer_stream) & {
  if (state_ == nullptr || ops_ == nullptr || ops_->write == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return ops_->write(state_, producer_stream);
}

expected<TensorLoanWriteGuard, Error> TensorPublishLoan::write_host() & {
  if (state_ == nullptr || ops_ == nullptr || ops_->write_host == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return ops_->write_host(state_);
}

expected<void, Error> TensorPublishLoan::cancel() && {
  if (state_ == nullptr || ops_ == nullptr || ops_->cancel == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  void* const state = std::exchange(state_, nullptr);
  const auto* const ops = std::exchange(ops_, nullptr);
  return ops->cancel(state);
}

ChannelPublisher<Tensor>::ChannelPublisher(ChannelPublisher&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)), ops_(std::exchange(other.ops_, nullptr)) {}

ChannelPublisher<Tensor>::~ChannelPublisher() noexcept {
  if (state_ != nullptr && ops_ != nullptr && ops_->destroy != nullptr) {
    ops_->destroy(state_);
  }
}

expected<void, Error> ChannelPublisher<Tensor>::close() {
  if (state_ == nullptr || ops_ == nullptr || ops_->close == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto result = ops_->close(state_);
  if (result ||
      channel_internal::terminal_publication_error_consumes_authority(result.error().code)) {
    void* const state = std::exchange(state_, nullptr);
    const auto* const ops = std::exchange(ops_, nullptr);
    if (ops->destroy != nullptr) {
      ops->destroy(state);
    }
  }
  return result;
}

expected<TensorPublishLoan, Error> ChannelPublisher<Tensor>::loan_tensor(
    const TensorLoanRequest request) {
  if (state_ == nullptr || ops_ == nullptr || ops_->loan_tensor == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->loan_tensor(state_, request);
}

expected<std::uint32_t, Error> ChannelPublisher<Tensor>::emit(TensorPublishLoan loan,
                                                              EmitOptions options) {
  if (state_ == nullptr || ops_ == nullptr || ops_->emit == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->emit(state_, std::move(loan), std::move(options));
}

expected<std::uint32_t, Error> ChannelPublisher<Tensor>::emit_captured(
    TensorPublishLoan loan, const std::int64_t capture_timestamp_ns, EmitOptions options) {
  if (state_ == nullptr || ops_ == nullptr || ops_->emit_captured == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->emit_captured(state_, std::move(loan), capture_timestamp_ns, std::move(options));
}

expected<std::uint32_t, Error> ChannelPublisher<Tensor>::emit_sample(TensorPublishLoan loan,
                                                                     SampleMetadata metadata) {
  if (state_ == nullptr || ops_ == nullptr || ops_->emit_sample == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->emit_sample(state_, std::move(loan), metadata);
}

expected<std::uint32_t, Error> ChannelPublisher<Tensor>::emit_copy(
    const Tensor& source, const ::cuda::stream_ref producer_stream, EmitOptions options) {
  if (state_ == nullptr || ops_ == nullptr || ops_->emit_copy == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->emit_copy(state_, source, producer_stream, std::move(options));
}

expected<TimePoint, Error> ChannelPublisher<Tensor>::make_capture_time(
    const std::int64_t timestamp_ns) const noexcept {
  if (state_ == nullptr || ops_ == nullptr || ops_->make_capture_time == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->make_capture_time(state_, timestamp_ns);
}

expected<void, Error> ChannelPublisher<Tensor>::report_clock_health(ClockHealthReport report) {
  if (state_ == nullptr || ops_ == nullptr || ops_->report_clock_health == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->report_clock_health(state_, report);
}

expected<void, Error> ChannelPublisher<Tensor>::update_connections() noexcept {
  if (state_ == nullptr || ops_ == nullptr || ops_->update_connections == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return ops_->update_connections(state_);
}

bool ChannelPublisher<Tensor>::has_subscribers() noexcept {
  return state_ != nullptr && ops_ != nullptr && ops_->has_subscribers != nullptr &&
         ops_->has_subscribers(state_);
}

}  // namespace holoscan
