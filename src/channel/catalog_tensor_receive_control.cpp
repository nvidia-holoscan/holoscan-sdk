/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "catalog_tensor_receive_control.hpp"

#include <holomq/const_buffer.hpp>
#include <holomq/context.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>

#include "channel_common.hpp"
#include "core/domain/tensor_access.hpp"
#include "holomq_tensor.hpp"

namespace holoscan::channel_internal {

namespace {

constexpr std::size_t kBitsPerFreeWord = std::numeric_limits<std::uint64_t>::digits;

class ScopedCatalogCUDADevice {
 public:
  explicit ScopedCatalogCUDADevice(const std::optional<std::int32_t> cuda_device_ordinal) noexcept
      : cuda_device_ordinal_(cuda_device_ordinal) {
    if (!cuda_device_ordinal_.has_value()) {
      active_ = true;
      return;
    }
    if (cuda_device_ordinal_.value() < 0 || cudaGetDevice(&previous_device_) != cudaSuccess) {
      return;
    }
    if (previous_device_ != cuda_device_ordinal_.value() &&
        cudaSetDevice(cuda_device_ordinal_.value()) != cudaSuccess) {
      return;
    }
    active_ = true;
  }
  ScopedCatalogCUDADevice(const ScopedCatalogCUDADevice&) = delete;
  ScopedCatalogCUDADevice& operator=(const ScopedCatalogCUDADevice&) = delete;
  ~ScopedCatalogCUDADevice() {
    if (active_ && cuda_device_ordinal_.has_value() &&
        previous_device_ != cuda_device_ordinal_.value()) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }
  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  std::optional<std::int32_t> cuda_device_ordinal_;
  int previous_device_{};
  bool active_{};
};

class CatalogTensorHostReadinessStream {
 public:
  CatalogTensorHostReadinessStream() = default;
  CatalogTensorHostReadinessStream(const CatalogTensorHostReadinessStream&) = delete;
  CatalogTensorHostReadinessStream& operator=(const CatalogTensorHostReadinessStream&) = delete;
  ~CatalogTensorHostReadinessStream() {
    if (stream_ != nullptr) {
      const ScopedCatalogCUDADevice selected{cuda_device_ordinal_};
      if (selected.active()) {
        static_cast<void>(cudaStreamDestroy(stream_));
      }
    }
  }

  void configure(const std::optional<std::int32_t> cuda_device_ordinal) noexcept {
    if (stream_ == nullptr) {
      cuda_device_ordinal_ = cuda_device_ordinal;
    }
  }

  [[nodiscard]] expected<cudaStream_t, Error> get_or_create() noexcept {
    const ScopedCatalogCUDADevice selected{cuda_device_ordinal_};
    if (!selected.active()) {
      return make_unexpected_nothrow<cudaStream_t>(
          ErrorCode::kFailure, "failed to select catalog Tensor host-readiness device");
    }
    if (stream_ == nullptr &&
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      return make_unexpected_nothrow<cudaStream_t>(
          ErrorCode::kFailure, "failed to create catalog Tensor host-readiness stream");
    }
    return stream_;
  }

 private:
  std::optional<std::int32_t> cuda_device_ordinal_;
  cudaStream_t stream_{nullptr};
};

[[nodiscard]] holomq::BufferMemoryType descriptor_memory_type(
    const HoloMQTensorDescriptorMemoryKind memory_kind) noexcept {
  switch (memory_kind) {
    case HoloMQTensorDescriptorMemoryKind::kDevice:
      return holomq::BufferMemoryType::Device;
    case HoloMQTensorDescriptorMemoryKind::kHostPinned:
      return holomq::BufferMemoryType::HostPinned;
    case HoloMQTensorDescriptorMemoryKind::kHostPageable:
      return holomq::BufferMemoryType::HostPageable;
    case HoloMQTensorDescriptorMemoryKind::kNone:
    default:
      return holomq::BufferMemoryType::None;
  }
}

[[nodiscard]] DLDevice descriptor_device(const HoloMQTensorDescriptor& descriptor) noexcept {
  switch (descriptor.memory_kind) {
    case HoloMQTensorDescriptorMemoryKind::kDevice:
      return DLDevice{kDLCUDA, descriptor.cuda_device_ordinal};
    case HoloMQTensorDescriptorMemoryKind::kHostPinned:
      return DLDevice{kDLCUDAHost, 0};
    case HoloMQTensorDescriptorMemoryKind::kHostPageable:
    case HoloMQTensorDescriptorMemoryKind::kNone:
    default:
      return DLDevice{kDLCPU, 0};
  }
}

}  // namespace

class BufferImportArray {
 public:
  explicit BufferImportArray(const std::size_t count)
      // The admitted borrow bound fixes this runtime-sized storage for the control's lifetime.
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      : values_(std::make_unique<holomq::ConstBuffer[]>(count)) {}

  [[nodiscard]] holomq::ConstBuffer& operator[](const std::size_t index) noexcept {
    return values_[index];
  }

 private:
  // The admitted borrow bound fixes this runtime-sized storage for the control's lifetime.
  std::unique_ptr<holomq::ConstBuffer[]> values_;  // NOLINT(cppcoreguidelines-avoid-c-arrays)
};

class CatalogTensorReceiveMemoryReference final : public HoloMQExportableTensorMemoryReference {
 public:
  CatalogTensorReceiveMemoryReference(CatalogTensorReceiveControl& owner,
                                      holomq::ConstBuffer& imported,
                                      const std::size_t slot_index) noexcept
      : owner_(&owner), imported_(&imported), slot_index_(slot_index) {}

  void retain_tensor_handle() noexcept override {
    auto count = handle_count_.load(std::memory_order_relaxed);
    while (count != std::numeric_limits<std::uint32_t>::max()) {
      if (handle_count_.compare_exchange_weak(
              count, count + 1U, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return;
      }
    }
    overflowed_.store(true, std::memory_order_release);
  }

  void release_tensor_handle() noexcept override;

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept override {
    if (!static_cast<bool>(*imported_)) {
      return make_unexpected_nothrow<holomq::BufferHandle>(ErrorCode::kInvalidArgument,
                                                           "catalog Tensor import is inactive");
    }
    auto handle = imported_->export_handle();
    if (!handle) {
      return make_unexpected(from_holomq_error(std::move(handle).error(),
                                               "failed to re-export catalog Tensor import"));
    }
    return std::move(handle).value();
  }

  bool set_deallocation_stream(cudaStream_t stream) override {
    if (!static_cast<bool>(*imported_)) {
      return false;
    }
    return stream == nullptr || imported_->enqueue_ready_wait(stream);
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    if (!static_cast<bool>(*imported_)) {
      return false;
    }
    if (stream != nullptr && !imported_->enqueue_ready_wait(stream)) {
      return false;
    }
    readiness_stream_.store(stream, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override {
    return static_cast<bool>(*imported_);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    cudaStream_t readiness_stream = readiness_stream_.load(std::memory_order_acquire);
    if (readiness_stream == nullptr) {
      if (consumer_stream == nullptr) {
        return {};
      }
      if (!imported_->enqueue_ready_wait(consumer_stream)) {
        return make_unexpected(Error{ErrorCode::kFailure, "catalog Tensor readiness wait failed"});
      }
      readiness_stream_.store(consumer_stream, std::memory_order_release);
      return {};
    }
    return synchronize_cuda_streams(readiness_stream, consumer_stream);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    cudaStream_t readiness_stream = readiness_stream_.load(std::memory_order_acquire);
    if (readiness_stream == nullptr) {
      auto stream = host_wait_stream_.get_or_create();
      if (!stream) {
        return make_unexpected(std::move(stream).error());
      }
      readiness_stream = stream.value();
      if (!imported_->enqueue_ready_wait(readiness_stream)) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "catalog Tensor host-readiness wait failed"});
      }
      readiness_stream_.store(readiness_stream, std::memory_order_release);
    }
    return synchronize_cuda_stream_for_host(readiness_stream);
  }

  void prepare_import(const std::optional<std::int32_t> cuda_device_ordinal) noexcept {
    host_wait_stream_.configure(cuda_device_ordinal);
    readiness_stream_.store(nullptr, std::memory_order_release);
  }

  void arm() noexcept {
    owner_->retain();
    armed_.store(true, std::memory_order_release);
  }

 private:
  CatalogTensorReceiveControl* owner_{};
  holomq::ConstBuffer* imported_{};
  std::size_t slot_index_{};
  std::atomic<std::uint32_t> handle_count_{0U};
  std::atomic<bool> armed_{false};
  std::atomic<bool> overflowed_{false};
  CatalogTensorHostReadinessStream host_wait_stream_;
  std::atomic<cudaStream_t> readiness_stream_{nullptr};
};

struct CatalogTensorReceiveControl::Slot {
  std::shared_ptr<DLManagedTensorContext> context;
  CatalogTensorReceiveMemoryReference* memory_reference{};
  Tensor template_tensor;
};

CatalogTensorReceiveControl::CatalogTensorReceiveControl(
    std::shared_ptr<void> runtime_lifetime, holomq::Context& context,
    const std::uint32_t borrow_capacity) noexcept
    : runtime_lifetime_(std::move(runtime_lifetime)),
      context_(&context),
      slot_count_(borrow_capacity),
      free_word_count_((slot_count_ + kBitsPerFreeWord - 1U) / kBitsPerFreeWord) {}

CatalogTensorReceiveControl::~CatalogTensorReceiveControl() = default;

expected<std::unique_ptr<CatalogTensorReceiveControl>, Error> CatalogTensorReceiveControl::create(
    std::shared_ptr<void> runtime_lifetime, holomq::Context& context,
    const LogicalTensorRequirementsRecord& contract, const std::uint32_t borrow_capacity) {
  if (!runtime_lifetime || borrow_capacity == 0U || !contract.max_rank) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::uint32_t max_rank = contract.max_rank.value();
  if (max_rank == 0U || max_rank > kHoloMQTensorDescriptorMaxRank) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto control = std::unique_ptr<CatalogTensorReceiveControl>(
      new CatalogTensorReceiveControl(std::move(runtime_lifetime), context, borrow_capacity));
  auto initialized = control->initialize(contract);
  if (!initialized) {
    return make_unexpected(std::move(initialized).error());
  }
  return control;
}

expected<void, Error> CatalogTensorReceiveControl::initialize(
    const LogicalTensorRequirementsRecord& contract) {
  if (!contract.max_rank) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::uint32_t max_rank = contract.max_rank.value();
  imports_ = std::make_unique<BufferImportArray>(slot_count_);
  // The admitted borrow bound fixes both runtime-sized arrays for the control's lifetime.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  slots_ = std::make_unique<Slot[]>(slot_count_);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): see the fixed-capacity invariant above.
  free_words_ = std::make_unique<std::atomic<std::uint64_t>[]>(free_word_count_);

  for (std::size_t word = 0U; word < free_word_count_; ++word) {
    const std::size_t remaining = slot_count_ - word * kBitsPerFreeWord;
    const std::uint64_t free_bits = remaining >= kBitsPerFreeWord
                                        ? std::numeric_limits<std::uint64_t>::max()
                                        : (std::uint64_t{1U} << remaining) - 1U;
    free_words_[word].store(free_bits, std::memory_order_relaxed);
  }

  for (std::size_t index = 0U; index < slot_count_; ++index) {
    auto& slot = slots_[index];
    auto memory_reference =
        std::make_shared<CatalogTensorReceiveMemoryReference>(*this, (*imports_)[index], index);
    slot.memory_reference = memory_reference.get();
    slot.context = std::make_shared<DLManagedTensorContext>();
    slot.context->memory_reference = std::move(memory_reference);
    slot.context->dl_shape.reserve(max_rank);
    slot.context->dl_strides.reserve(max_rank);
    auto& tensor = slot.context->tensor;
    tensor.version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    tensor.manager_ctx = nullptr;
    tensor.deleter = nullptr;
    tensor.dl_tensor = DLTensor{
        .data = nullptr,
        .device = DLDevice{kDLCPU, 0},
        .ndim = 0,
        .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
        .shape = nullptr,
        .strides = nullptr,
        .byte_offset = 0U,
    };
    slot.template_tensor =
        detail::TensorAccess::from_context(slot.context, detail::TensorAuthority::kReadOnly);
  }
  return {};
}

expected<std::size_t, Error> CatalogTensorReceiveControl::acquire_slot() noexcept {
  for (std::size_t word = 0U; word < free_word_count_; ++word) {
    std::uint64_t available = free_words_[word].load(std::memory_order_acquire);
    while (available != 0U) {
      const auto bit = static_cast<std::size_t>(std::countr_zero(available));
      const std::uint64_t claimed = std::uint64_t{1U} << bit;
      if (free_words_[word].compare_exchange_weak(available,
                                                  available & ~claimed,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) {
        return word * kBitsPerFreeWord + bit;
      }
    }
  }
  return make_unexpected_nothrow<std::size_t>(
      ErrorCode::kResourceExhausted, "catalog Tensor receive borrow capacity is exhausted");
}

void CatalogTensorReceiveControl::return_unmaterialized_slot(
    const std::size_t slot_index) noexcept {
  (*imports_)[slot_index].release();
  const std::size_t word = slot_index / kBitsPerFreeWord;
  const std::size_t bit = slot_index % kBitsPerFreeWord;
  free_words_[word].fetch_or(std::uint64_t{1U} << bit, std::memory_order_release);
}

void CatalogTensorReceiveControl::retire_slot(const std::size_t slot_index) noexcept {
  return_unmaterialized_slot(slot_index);
  release();
}

void CatalogTensorReceiveControl::retain() noexcept {
  references_.fetch_add(1U, std::memory_order_relaxed);
}

void CatalogTensorReceiveControl::release() noexcept {
  if (references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    delete this;
  }
}

expected<Tensor, Error> CatalogTensorReceiveControl::materialize(
    const HoloMQTensorDescriptor& descriptor) {
  auto acquired = acquire_slot();
  if (!acquired) {
    return make_unexpected(std::move(acquired).error());
  }
  const std::size_t slot_index = acquired.value();
  auto& imported = (*imports_)[slot_index];
  auto& slot = slots_[slot_index];
  const auto cuda_device_ordinal =
      descriptor.memory_kind == HoloMQTensorDescriptorMemoryKind::kDevice
          ? std::optional<std::int32_t>{descriptor.cuda_device_ordinal}
          : std::nullopt;
  slot.memory_reference->prepare_import(cuda_device_ordinal);

  auto rebound = context_->rebind_buffer_unsynchronized(descriptor.buffer_handle, imported);
  if (!rebound) {
    return_unmaterialized_slot(slot_index);
    return make_unexpected(
        from_holomq_error(std::move(rebound).error(), "catalog Tensor BufferHandle import failed"));
  }
  if (imported.memory_type() != descriptor_memory_type(descriptor.memory_kind) ||
      imported.ptr() == nullptr || descriptor.byte_size > imported.size_bytes() ||
      (descriptor.memory_kind == HoloMQTensorDescriptorMemoryKind::kDevice &&
       imported.cuda_device_ordinal() != cuda_device_ordinal)) {
    return_unmaterialized_slot(slot_index);
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "catalog Tensor import does not match its descriptor"});
  }

  auto& context = *slot.context;
  context.dl_shape.assign(descriptor.shape.begin(), descriptor.shape.begin() + descriptor.rank);
  context.dl_strides.clear();
  if ((descriptor.flags &
       static_cast<std::uint32_t>(HoloMQTensorDescriptorFlags::kHasByteStrides)) != 0U) {
    const std::uint64_t item_size = dlpack_dtype_item_size_bytes(descriptor.dtype);
    for (std::size_t index = 0U; index < descriptor.rank; ++index) {
      context.dl_strides.push_back(descriptor.byte_strides[index] /
                                   static_cast<std::int64_t>(item_size));
    }
  }
  auto& tensor = context.tensor.dl_tensor;
  tensor.data = const_cast<void*>(imported.ptr());
  tensor.device = descriptor_device(descriptor);
  tensor.ndim = static_cast<std::int32_t>(descriptor.rank);
  tensor.dtype = descriptor.dtype;
  tensor.shape = context.dl_shape.empty() ? nullptr : context.dl_shape.data();
  tensor.strides = context.dl_strides.empty() ? nullptr : context.dl_strides.data();
  tensor.byte_offset = 0U;

  Tensor result = slot.template_tensor;
  slot.memory_reference->arm();
  return result;
}

void CatalogTensorReceiveMemoryReference::release_tensor_handle() noexcept {
  if (overflowed_.load(std::memory_order_acquire)) {
    return;
  }
  auto previous = handle_count_.load(std::memory_order_relaxed);
  while (previous != 0U) {
    if (handle_count_.compare_exchange_weak(
            previous, previous - 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      if (previous == 2U && armed_.exchange(false, std::memory_order_acq_rel)) {
        readiness_stream_.store(nullptr, std::memory_order_release);
        owner_->retire_slot(slot_index_);
      }
      return;
    }
  }
  overflowed_.store(true, std::memory_order_release);
}

}  // namespace holoscan::channel_internal
