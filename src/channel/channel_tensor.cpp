/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>

#include "catalog_tensor_receive_control.hpp"
#include "channel_common.hpp"
#include "channel_runtime_impl.hpp"
#include "channel_tensor_sidecar_store.hpp"
#include "channel_tensor_transport_payload.hpp"
#include "core/domain/tensor_layout_validation.hpp"
#include "holomq_tensor.hpp"
#include "process_identity.hpp"

#include <holomq/delivery_attribution.hpp>
#include <holomq/inbound_message.hpp>
#include <holomq/outbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/subscriber.hpp>
#include <holomq/wait_set.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {

using channel_internal::from_holomq_error;
using channel_internal::from_holomq_publication_result;
using channel_internal::from_holomq_result;
using channel_internal::make_channel_error;
using channel_internal::replaces_oldest_unread_sample;
using channel_internal::TensorTransportPayload;
using channel_internal::to_holomq_publisher_options;
using channel_internal::to_holomq_service_options;
using channel_internal::to_holomq_subscriber_options;
using channel_internal::validate_capabilities;
using channel_internal::validate_tensor_channel_options;

namespace {

template <typename T>
[[nodiscard]] T* optional_value_pointer(std::optional<T>& value) noexcept {
  if (!value.has_value()) {
    return nullptr;
  }
  return std::addressof(value.value());
}

template <typename T>
[[nodiscard]] const T* optional_value_pointer(const std::optional<T>& value) noexcept {
  if (!value.has_value()) {
    return nullptr;
  }
  return std::addressof(value.value());
}

[[nodiscard]] bool tensor_dtype_equal(const DLDataType lhs, const DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

[[nodiscard]] bool valid_catalog_tensor_envelope(
    const TensorTransportPayload& payload,
    const LogicalTensorRequirementsRecord& contract) noexcept {
  const auto& descriptor = payload.reference.holomq;
  const bool memory_kind_matches =
      !contract.memory_kind ||
      ((*contract.memory_kind == MemoryKind::kHost &&
        descriptor.memory_kind ==
            channel_internal::HoloMQTensorDescriptorMemoryKind::kHostPageable) ||
       (*contract.memory_kind == MemoryKind::kPinnedHost &&
        descriptor.memory_kind ==
            channel_internal::HoloMQTensorDescriptorMemoryKind::kHostPinned) ||
       (*contract.memory_kind == MemoryKind::kCudaDevice &&
        descriptor.memory_kind == channel_internal::HoloMQTensorDescriptorMemoryKind::kDevice));
  const bool descriptor_valid =
      descriptor.has_supported_abi() &&
      descriptor.rank <= channel_internal::kHoloMQTensorDescriptorMaxRank &&
      static_cast<bool>(descriptor.buffer_handle) && descriptor.dtype.bits != 0U &&
      descriptor.dtype.lanes != 0U && memory_kind_matches;
  const bool contract_valid =
      (!contract.dtype || tensor_dtype_equal(descriptor.dtype, *contract.dtype)) &&
      (!contract.rank || descriptor.rank == *contract.rank) &&
      (!contract.max_rank || descriptor.rank <= *contract.max_rank) &&
      (!contract.max_byte_span || descriptor.byte_size <= *contract.max_byte_span);
  const auto& metadata = payload.metadata;
  const bool metadata_valid = metadata.version.major == 1U && metadata.version.minor == 0U &&
                              (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) == 0U &&
                              metadata.sequence != 0U &&
                              metadata.byte_size == descriptor.byte_size &&
                              metadata.payload_kind == PayloadKind::kDescriptorBacked;
  return payload.operation_kind == channel_internal::TensorTransportOperationKind::kData &&
         payload.producer_generation != 0U && descriptor_valid && contract_valid && metadata_valid;
}

[[nodiscard]] bool catalog_tensor_axis_within_bounds(
    const channel_internal::HoloMQTensorDescriptor& descriptor,
    const LogicalTensorRequirementsRecord& contract, const std::size_t index,
    const bool has_byte_strides) noexcept {
  const std::int64_t extent = descriptor.shape[index];
  if (extent < 0 || (index < contract.axes.size() &&
                     static_cast<std::uint64_t>(extent) > contract.axes[index].max_extent)) {
    return false;
  }
  if (!has_byte_strides) {
    return true;
  }
  const std::int64_t stride = descriptor.byte_strides[index];
  return stride >= 0 &&
         (index >= contract.axes.size() ||
          static_cast<std::uint64_t>(stride) <= contract.axes[index].max_abs_stride_bytes);
}

[[nodiscard]] expected<void, Error> validate_catalog_tensor_axes(
    const channel_internal::HoloMQTensorDescriptor& descriptor,
    const LogicalTensorRequirementsRecord& contract, const bool has_byte_strides) noexcept {
  for (std::size_t index = 0U; index < descriptor.rank; ++index) {
    if (!catalog_tensor_axis_within_bounds(descriptor, contract, index, has_byte_strides)) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError, "Tensor channel layout exceeds its catalog axis bounds"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_catalog_tensor_contiguity(
    const channel_internal::HoloMQTensorDescriptor& descriptor,
    const LogicalTensorRequirementsRecord& contract, const bool has_byte_strides) noexcept {
  if (!has_byte_strides || contract.strides != TensorStridePolicy::kContiguousOnly) {
    return {};
  }
  std::uint64_t contiguous_stride = dlpack_dtype_item_size_bytes(descriptor.dtype);
  for (std::size_t reverse = descriptor.rank; reverse > 0U; --reverse) {
    const std::size_t index = reverse - 1U;
    if (contiguous_stride == 0U ||
        static_cast<std::uint64_t>(descriptor.byte_strides[index]) != contiguous_stride) {
      return make_unexpected(make_channel_error(ErrorCode::kProtocolError,
                                                "Tensor channel layout is not catalog-contiguous"));
    }
    const std::uint64_t extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(descriptor.shape[index]), 1U);
    if (contiguous_stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return make_unexpected(make_channel_error(ErrorCode::kProtocolError,
                                                "Tensor channel contiguous layout overflows"));
    }
    contiguous_stride *= extent;
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_catalog_tensor_payload(
    const TensorTransportPayload& payload,
    const LogicalTensorRequirementsRecord& contract) noexcept {
  const auto& descriptor = payload.reference.holomq;
  if (!valid_catalog_tensor_envelope(payload, contract)) {
    return make_unexpected(make_channel_error(
        ErrorCode::kProtocolError, "Tensor channel payload violates its catalog contract"));
  }

  const std::span<const std::int64_t> shape{descriptor.shape.data(), descriptor.rank};
  const bool has_byte_strides =
      (descriptor.flags & static_cast<std::uint32_t>(
                              channel_internal::HoloMQTensorDescriptorFlags::kHasByteStrides)) !=
      0U;
  const std::optional<std::span<const std::int64_t>> byte_strides =
      has_byte_strides ? std::optional<std::span<const std::int64_t>>{std::span<const std::int64_t>{
                             descriptor.byte_strides.data(), descriptor.rank}}
                       : std::nullopt;
  if (auto valid = validate_catalog_tensor_axes(descriptor, contract, has_byte_strides); !valid) {
    return valid;
  }
  if (auto valid = validate_catalog_tensor_contiguity(descriptor, contract, has_byte_strides);
      !valid) {
    return valid;
  }

  const auto layout = detail::validate_tensor_layout(
      shape, descriptor.dtype, byte_strides, detail::TensorStrideUnit::kBytes);
  if (!layout || layout->byte_span != descriptor.byte_size ||
      (contract.max_byte_span && layout->byte_span > *contract.max_byte_span)) {
    return make_unexpected(make_channel_error(
        ErrorCode::kProtocolError, "Tensor channel byte span violates its catalog contract"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_catalog_tensor_end_of_stream(
    const TensorTransportPayload& payload) noexcept {
  const auto& metadata = payload.metadata;
  if (payload.operation_kind != channel_internal::TensorTransportOperationKind::kEndOfStream ||
      payload.producer_generation == 0U ||
      payload.payload_kind != ChannelPayloadKind::kBufferHandle || metadata.version.major != 1U ||
      metadata.version.minor != 0U || metadata.sequence == 0U ||
      metadata.capture_timestamp_ns != 0 || metadata.source_clock_id.valid() ||
      metadata.publish_timestamp_ns != 0 || metadata.publish_clock_id.valid() ||
      metadata.trace_id != 0U || metadata.frame_id != 0U || metadata.byte_size != 0U ||
      metadata.payload_kind != PayloadKind::kDescriptorBacked ||
      metadata.flags != SampleFlags::kNone) {
    return make_unexpected(make_channel_error(ErrorCode::kProtocolError,
                                              "Tensor channel end-of-stream record is invalid"));
  }
  return {};
}

class SidecarPublicationRollback {
 public:
  SidecarPublicationRollback() = default;
  SidecarPublicationRollback(const SidecarPublicationRollback&) = delete;
  SidecarPublicationRollback& operator=(const SidecarPublicationRollback&) = delete;

  ~SidecarPublicationRollback() noexcept {
    if (store_ != nullptr) {
      store_->release_all(descriptor_);
    }
  }

  void arm(TensorSidecarStore& store, const ChannelSampleDescriptor& descriptor) noexcept {
    store_ = &store;
    descriptor_ = descriptor;
  }

  void release_to_transport_callback() noexcept { store_ = nullptr; }

 private:
  TensorSidecarStore* store_{};
  ChannelSampleDescriptor descriptor_{};
};

[[nodiscard]] expected<SampleMetadata, Error> tensor_metadata_for_transport(
    const SampleMetadata& metadata, const ChannelPayloadKind payload_kind,
    const std::uint64_t byte_size) noexcept {
  auto mapped_kind = core_payload_kind(payload_kind);
  if (!mapped_kind) {
    return make_unexpected(std::move(mapped_kind).error());
  }
  SampleMetadata result = metadata;
  result.byte_size = byte_size;
  result.payload_kind = mapped_kind.value();
  return result;
}

}  // namespace

class ManualChannelBinding::Impl {
 public:
  Impl(std::shared_ptr<ChannelRuntime::Impl> runtime, std::string service_name,
       const ManualChannelRole role) noexcept
      : runtime_(std::move(runtime)), service_name_(std::move(service_name)), role_(role) {}

  std::shared_ptr<ChannelRuntime::Impl> runtime_;
  std::string service_name_;
  ManualChannelRole role_;
};

class ManualTensorPublisher::Impl {
 public:
  Impl(holomq::Publisher<TensorTransportPayload>&& publisher,
       std::optional<holomq::Publisher<TensorTransportPayload>> terminal_publisher,
       std::shared_ptr<TensorSidecarStore> sidecar_store, ChannelOptions options,
       std::shared_ptr<ChannelRuntime::Impl> runtime,
       std::unique_ptr<ManualChannelBinding::Impl> binding = nullptr,
       const bool graph_retry_boundary = false) noexcept
      : runtime_(std::move(runtime)),
        binding_(std::move(binding)),
        publisher_(std::move(publisher)),
        terminal_publisher_(std::move(terminal_publisher)),
        sidecar_store_(std::move(sidecar_store)),
        options_(options),
        graph_retry_boundary_(graph_retry_boundary),
        producer_generation_(channel_internal::mint_producer_generation()) {}

  ~Impl() {
    if (binding_ && runtime_) {
      runtime_->retire_manual_tensor_sidecar_store(binding_->service_name_, sidecar_store_);
    }
  }

  std::shared_ptr<ChannelRuntime::Impl> runtime_;
  std::unique_ptr<ManualChannelBinding::Impl> binding_;
  holomq::Publisher<TensorTransportPayload> publisher_;
  std::optional<holomq::Publisher<TensorTransportPayload>> terminal_publisher_;
  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  ChannelOptions options_;
  bool graph_retry_boundary_{};
  std::uint64_t producer_generation_{};
};

class ChannelSubscriber<Tensor>::Impl
    : public channel_internal::WaitSetSubscriber,
      public channel_internal::ProtectedTerminalWaitSetSubscriber {
 public:
  Impl(std::shared_ptr<void> runtime_lifetime, holomq::Context& context,
       holomq::Subscriber<TensorTransportPayload>&& subscriber,
       std::optional<holomq::Subscriber<TensorTransportPayload>> terminal_subscriber,
       std::shared_ptr<TensorSidecarStore> sidecar_store, std::size_t recipient_index,
       std::optional<LogicalTensorRequirementsRecord> receive_contract,
       std::unique_ptr<channel_internal::CatalogTensorReceiveControl> receive_control) noexcept
      : runtime_lifetime_(std::move(runtime_lifetime)),
        context_(&context),
        subscriber_(std::move(subscriber)),
        terminal_subscriber_(std::move(terminal_subscriber)),
        sidecar_store_(std::move(sidecar_store)),
        recipient_index_(recipient_index),
        has_receive_contract_(receive_contract.has_value()),
        receive_contract_(std::move(receive_contract).value_or(LogicalTensorRequirementsRecord{})),
        receive_control_(receive_control.release()) {}

  ~Impl() {
    if (receive_control_ != nullptr) {
      receive_control_->release();
    }
  }

  [[nodiscard]] expected<void, Error> validate_catalog_operation_sequence(
      const TensorTransportPayload& payload) const noexcept {
    if (latched_producer_generation_ != 0U &&
        latched_producer_generation_ != payload.producer_generation) {
      return make_unexpected(make_channel_error(ErrorCode::kStaleEndpoint,
                                                "Tensor channel producer incarnation changed"));
    }
    if (payload.metadata.sequence <= last_operation_sequence_) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError, "Tensor channel operation sequence did not advance"));
    }
    return {};
  }

  [[nodiscard]] expected<Sample<Tensor>, Error> receive_end_of_stream(
      const TensorTransportPayload& payload) noexcept {
    if (auto valid = validate_catalog_tensor_end_of_stream(payload); !valid) {
      return make_unexpected(std::move(valid).error());
    }
    if (auto valid = validate_catalog_operation_sequence(payload); !valid) {
      return make_unexpected(std::move(valid).error());
    }
    latched_producer_generation_ = payload.producer_generation;
    last_operation_sequence_ = payload.metadata.sequence;
    stream_closed_ = true;
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  [[nodiscard]] expected<Tensor, Error> materialize_buffer_handle(
      const channel_internal::HoloMQTensorDescriptor& descriptor) {
    if (receive_control_ != nullptr) {
      return receive_control_->materialize(descriptor);
    }
    auto imported = context_->import_buffer_unsynchronized(descriptor.buffer_handle);
    if (!imported) {
      return make_unexpected(from_holomq_error(std::move(imported).error(),
                                               "Tensor channel BufferHandle import failed"));
    }
    return channel_internal::make_tensor_from_holomq_const_buffer(descriptor,
                                                                  std::move(imported).value());
  }

  [[nodiscard]] expected<Sample<Tensor>, Error> receive_buffer_handle(
      const TensorTransportPayload& payload) {
    if (has_receive_contract_) {
      if (auto valid = validate_catalog_tensor_payload(payload, receive_contract_); !valid) {
        return make_unexpected(std::move(valid).error());
      }
      if (auto valid = validate_catalog_operation_sequence(payload); !valid) {
        return make_unexpected(std::move(valid).error());
      }
    }
    auto tensor = materialize_buffer_handle(payload.reference.holomq);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    if (has_receive_contract_) {
      if (latched_producer_generation_ == 0U) {
        latched_producer_generation_ = payload.producer_generation;
      }
      last_operation_sequence_ = payload.metadata.sequence;
    }
    return Sample<Tensor>(std::move(tensor).value(), payload.metadata);
  }

  [[nodiscard]] expected<Sample<Tensor>, Error> receive_local_sidecar(
      const TensorTransportPayload& payload) {
    if (payload.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return make_unexpected(make_channel_error(
          ErrorCode::kReceiveError, "Tensor channel received an unsupported payload descriptor"));
    }
    if (!sidecar_store_) {
      return make_unexpected(
          make_channel_error(ErrorCode::kReceiveError,
                             "Tensor channel received a local descriptor without a sidecar"));
    }
    auto sample = sidecar_store_->acquire_sample(payload.reference.local, recipient_index_);
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return std::move(sample).value();
  }

  [[nodiscard]] expected<Sample<Tensor>, Error> receive_payload(
      const TensorTransportPayload& payload) {
    if (payload.operation_kind == channel_internal::TensorTransportOperationKind::kEndOfStream) {
      return receive_end_of_stream(payload);
    }
    if (payload.operation_kind != channel_internal::TensorTransportOperationKind::kData) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError, "Tensor channel received an unsupported operation"));
    }
    if (payload.payload_kind == ChannelPayloadKind::kBufferHandle) {
      return receive_buffer_handle(payload);
    }
    return receive_local_sidecar(payload);
  }

  // Data and protected-terminal arbitration intentionally remain one ordered receive state machine.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<Sample<Tensor>, Error> receive_sample(const bool accept_data) {
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto* const terminal_subscriber = optional_value_pointer(terminal_subscriber_);
    if (!pending_inbound_.has_value()) {
      auto data_ready = from_holomq_result(subscriber_.has_samples());
      if (!data_ready) {
        return make_unexpected(std::move(data_ready).error());
      }
      std::optional<bool> terminal_ready;
      if (terminal_subscriber != nullptr) {
        auto ready = from_holomq_result(terminal_subscriber->has_samples());
        if (!ready) {
          return make_unexpected(std::move(ready).error());
        }
        terminal_ready = *ready;
      }
      if (*data_ready) {
        auto inbound = subscriber_.try_take();
        if (!inbound) {
          return make_unexpected(from_holomq_error(std::move(inbound).error()));
        }
        pending_inbound_.emplace(std::move(inbound).value());
      } else if (!pending_terminal_inbound_.has_value() && terminal_subscriber != nullptr) {
        if (!terminal_ready.value_or(false)) {
          return make_unexpected(Error{ErrorCode::kNotReady});
        }
        auto inbound = terminal_subscriber->try_take();
        if (!inbound) {
          return make_unexpected(from_holomq_error(std::move(inbound).error()));
        }
        pending_terminal_inbound_.emplace(std::move(inbound).value());
      }
      if (!pending_inbound_.has_value() && !pending_terminal_inbound_.has_value()) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
    }
    if (pending_inbound_.has_value() &&
        pending_inbound_->payload().operation_kind ==
            channel_internal::TensorTransportOperationKind::kData &&
        !accept_data) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (pending_inbound_.has_value()) {
      auto received = receive_payload(pending_inbound_->payload());
      pending_inbound_.reset();
      return received;
    }

    const auto* const terminal_inbound = optional_value_pointer(pending_terminal_inbound_);
    if (terminal_inbound == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    const auto& terminal = terminal_inbound->payload();
    if (terminal.operation_kind != channel_internal::TensorTransportOperationKind::kEndOfStream) {
      stream_closed_ = true;
      pending_terminal_inbound_.reset();
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto received = receive_payload(terminal);
    pending_terminal_inbound_.reset();
    return received;
  }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void* wait_set) override {
    if (wait_set == nullptr || attachment_.has_value() || terminal_attachment_.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto* const typed_wait_set = static_cast<holomq::WaitSet*>(wait_set);
    auto attached = typed_wait_set->attach(subscriber_);
    if (!attached) {
      return make_unexpected(from_holomq_error(std::move(attached).error()));
    }
    if (terminal_subscriber_) {
      auto terminal_attached = typed_wait_set->attach(*terminal_subscriber_);
      if (!terminal_attached) {
        return make_unexpected(from_holomq_error(std::move(terminal_attached).error()));
      }
      terminal_wait_handle_ = terminal_attached->handle().id();
      terminal_attachment_.emplace(std::move(*terminal_attached));
    }
    const std::uint64_t handle = attached->handle().id();
    attachment_.emplace(std::move(*attached));
    return handle;
  }

  [[nodiscard]] std::optional<std::uint64_t> protected_terminal_wait_handle()
      const noexcept override {
    return terminal_wait_handle_ == 0U ? std::nullopt
                                       : std::optional<std::uint64_t>{terminal_wait_handle_};
  }

  std::shared_ptr<void> runtime_lifetime_;
  holomq::Context* context_{};
  holomq::Subscriber<TensorTransportPayload> subscriber_;
  std::optional<holomq::Subscriber<TensorTransportPayload>> terminal_subscriber_;
  std::optional<holomq::WaitSetAttachment> attachment_;
  std::optional<holomq::WaitSetAttachment> terminal_attachment_;
  std::uint64_t terminal_wait_handle_{};
  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  std::size_t recipient_index_{};
  bool has_receive_contract_{};
  LogicalTensorRequirementsRecord receive_contract_;
  channel_internal::CatalogTensorReceiveControl* receive_control_{};
  std::uint64_t latched_producer_generation_{};
  std::uint64_t last_operation_sequence_{};
  bool stream_closed_{};
  std::optional<holomq::InboundMessage<TensorTransportPayload>> pending_inbound_;
  std::optional<holomq::InboundMessage<TensorTransportPayload>> pending_terminal_inbound_;
};

ManualChannelBinding::ManualChannelBinding(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
ManualChannelBinding::ManualChannelBinding(ManualChannelBinding&& other) noexcept = default;
ManualChannelBinding::~ManualChannelBinding() noexcept = default;

ManualTensorPublisher::ManualTensorPublisher(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ManualTensorPublisher::ManualTensorPublisher(ManualTensorPublisher&& other) noexcept = default;

ManualTensorPublisher::~ManualTensorPublisher() noexcept = default;

std::uint64_t ManualTensorPublisher::producer_generation() const noexcept {
  return impl_ ? impl_->producer_generation_ : 0U;
}

expected<std::uint32_t, Error> ManualTensorPublisher::emit_end_of_stream(
    SampleMetadata metadata) noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (impl_->options_.tensor_transfer_mode != ChannelTensorTransferMode::kBufferHandle) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  auto* const terminal_publisher = optional_value_pointer(impl_->terminal_publisher_);
  if (!impl_->publisher_.has_subscribers() ||
      (terminal_publisher != nullptr && !terminal_publisher->has_subscribers())) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  auto& terminal_lane = terminal_publisher != nullptr ? *terminal_publisher : impl_->publisher_;
  auto message = terminal_lane.loan();
  if (!message) {
    return make_unexpected(from_holomq_error(std::move(message).error()));
  }
  message->payload() = TensorTransportPayload{
      .metadata = metadata,
      .producer_generation = impl_->producer_generation_,
      .operation_kind = channel_internal::TensorTransportOperationKind::kEndOfStream,
      .payload_kind = ChannelPayloadKind::kBufferHandle,
  };
  // HoloMQ send synchronously enqueues each accepted data descriptor before returning. Close can
  // therefore submit this terminal record only after the data publication is visible, and the
  // receiver's data-first probe preserves that order independently for every fanout arm.
  auto sent = terminal_lane.send(std::move(message).value());
  return from_holomq_publication_result(
      std::move(sent), impl_->options_.expected_recipient_count, impl_->graph_retry_boundary_);
}

expected<std::uint32_t, Error> ManualTensorPublisher::emit(Tensor tensor, EmitOptions options) {
  if (!impl_) {
    return make_unexpected(
        make_channel_error(ErrorCode::kClosed, "Tensor channel publisher is closed"));
  }
  if (options.capture_time.has_value()) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "manual Tensor publisher has no catalog capture-clock authority"));
  }
  SampleMetadata metadata{};
  metadata.frame_id = options.frame_id.value_or(0U);
  metadata.trace_id = options.trace.has_value() ? options.trace->trace_id : 0U;
  metadata.flags = options.flags;
  return emit_sample(Sample<Tensor>(std::move(tensor), metadata));
}

expected<std::uint32_t, Error> ManualTensorPublisher::emit_sample(Sample<Tensor> sample) {
  if (!impl_) {
    return make_unexpected(
        make_channel_error(ErrorCode::kClosed, "Tensor channel publisher is closed"));
  }
  if (impl_->binding_ && sample.metadata.capture_time().valid()) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "manual Tensor publisher has no catalog capture-clock authority"));
  }
  auto message = impl_->publisher_.loan();
  if (!message) {
    return make_unexpected(from_holomq_error(std::move(message).error()));
  }

  TensorTransportPayload payload{};
  payload.producer_generation = impl_->producer_generation_;
  SidecarPublicationRollback sidecar_rollback;
  if (impl_->options_.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle) {
    auto descriptor = channel_internal::export_holomq_tensor_descriptor(sample.data);
    if (!descriptor) {
      return make_unexpected(std::move(descriptor).error());
    }
    payload.payload_kind = ChannelPayloadKind::kBufferHandle;
    payload.reference.holomq = descriptor.value();
    auto metadata =
        tensor_metadata_for_transport(sample.metadata, payload.payload_kind, descriptor->byte_size);
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    payload.metadata = metadata.value();
  } else {
    auto descriptor = impl_->sidecar_store_->insert(std::move(sample.data), sample.metadata);
    if (!descriptor) {
      return make_unexpected(std::move(descriptor).error());
    }
    sidecar_rollback.arm(*impl_->sidecar_store_, descriptor.value());
    payload.payload_kind = ChannelPayloadKind::kLocalBufferDescriptor;
    payload.reference.local = descriptor.value();
    auto metadata =
        tensor_metadata_for_transport(sample.metadata, payload.payload_kind, descriptor->byte_size);
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    payload.metadata = metadata.value();
  }

  message->payload() = payload;
  auto send = [&]() {
    if (payload.payload_kind == ChannelPayloadKind::kBufferHandle) {
      return impl_->publisher_.send(std::move(message).value(),
                                    [tensor = std::move(sample.data)]() noexcept { (void)tensor; });
    }
    // HoloMQ retains this callback after a send error because some recipients may already hold
    // the transport sample. Release unresolved sidecar claims only when it is actually freed.
    return impl_->publisher_.send(
        std::move(message).value(),
        [sidecar_store = impl_->sidecar_store_, descriptor = payload.reference.local]() noexcept {
          sidecar_store->release_all(descriptor);
        });
  };
  expected<int, holomq::Error> sent;
  try {
    sent = send();
    // Tensor publishers always enable release callbacks and each freshly loaned transport sample
    // has a unique payload address. Therefore a normal return, including a transport send error,
    // means HoloMQ retained the callback and now owns sidecar cleanup. An exception can only occur
    // while registering that callback, before the registry marks it pending, so the local rollback
    // remains armed on the exceptional path.
    sidecar_rollback.release_to_transport_callback();
  } catch (const std::length_error&) {
    return channel_internal::make_unexpected_nothrow<std::uint32_t>(
        ErrorCode::kResourceExhausted,
        "Tensor release-callback registration exceeds allocator limits");
  } catch (const std::bad_alloc&) {
    return channel_internal::make_unexpected_nothrow<std::uint32_t>(
        ErrorCode::kResourceExhausted, "Tensor release-callback registration allocation failed");
  } catch (...) {
    return channel_internal::make_unexpected_nothrow<std::uint32_t>(
        ErrorCode::kFailure, "Tensor release-callback registration failed");
  }
  return from_holomq_publication_result(
      std::move(sent), impl_->options_.expected_recipient_count, impl_->graph_retry_boundary_);
}

expected<void, Error> ManualTensorPublisher::update_connections() {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto updated = impl_->publisher_.update_connections();
  if (!updated) {
    return make_unexpected(from_holomq_error(std::move(updated).error()));
  }
  if (auto* const terminal_publisher = optional_value_pointer(impl_->terminal_publisher_);
      terminal_publisher != nullptr) {
    auto terminal_updated = terminal_publisher->update_connections();
    if (!terminal_updated) {
      return make_unexpected(from_holomq_error(std::move(terminal_updated).error()));
    }
  }
  if (impl_->options_.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle) {
    // iceoryx2 returns released samples to the publisher while entering an allocation/send path.
    // Borrowing and immediately abandoning one transport payload therefore dispatches pending
    // release callbacks without publishing data. This keeps a one-slot external Tensor pool from
    // waiting forever for a second application Tensor that it cannot yet loan.
    static_cast<void>(impl_->publisher_.loan());
  }
  return {};
}

bool ManualTensorPublisher::has_subscribers() const noexcept {
  if (!impl_ || !impl_->publisher_.has_subscribers()) {
    return false;
  }
  const auto* const terminal_publisher = optional_value_pointer(impl_->terminal_publisher_);
  return terminal_publisher == nullptr || terminal_publisher->has_subscribers();
}

expected<ManualChannelBinding, Error> ChannelRuntime::bind_manual_channel(
    const ManualChannelBindingOptions options) {
  try {
    if (!impl_) {
      return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
    }
    if (options.service_name.empty()) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument, "manual channel service name is empty"));
    }
    if (options.role != ManualChannelRole::kPublisher) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument, "unsupported manual channel role"));
    }
    return ManualChannelBinding(std::make_unique<ManualChannelBinding::Impl>(
        impl_, std::string(options.service_name), options.role));
  } catch (const std::length_error&) {
    return channel_internal::make_unexpected_nothrow<ManualChannelBinding>(
        ErrorCode::kResourceExhausted, "manual channel service identity exceeds allocator limits");
  } catch (const std::bad_alloc&) {
    return channel_internal::make_unexpected_nothrow<ManualChannelBinding>(
        ErrorCode::kResourceExhausted, "manual channel binding allocation failed");
  }
}

expected<ManualTensorPublisher, Error> make_manual_tensor_publisher(
    ManualChannelBinding binding, const ManualTensorPublisherOptions options) {
  std::shared_ptr<ChannelRuntime::Impl> runtime_impl;
  std::string service_name;
  std::shared_ptr<TensorSidecarStore> registered_store;
  std::unique_lock<std::recursive_mutex> setup_lock;
  try {
    if (!binding.impl_ || !binding.impl_->runtime_) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument, "manual channel binding is empty"));
    }
    if (binding.impl_->role_ != ManualChannelRole::kPublisher) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "manual binding does not authorize publish"));
    }
    if (options.max_inflight_samples == 0U ||
        options.max_inflight_samples == std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "manual Tensor publisher max_inflight_samples must be finite and positive"));
    }

    runtime_impl = binding.impl_->runtime_;
    service_name = binding.impl_->service_name_;
    // Keep provisional sidecar state unobservable until backend publisher construction either
    // commits or rolls back. create_tensor_publisher() re-enters this setup lock on the same
    // thread; competing publisher/subscriber factories wait at their store lookup.
    setup_lock = std::unique_lock<std::recursive_mutex>(runtime_impl->tensor_sidecar_store_mutex_);
    auto sidecar_store = runtime_impl->register_manual_tensor_sidecar_store(
        service_name, options.max_inflight_samples);
    if (!sidecar_store) {
      return make_unexpected(std::move(sidecar_store).error());
    }
    registered_store = sidecar_store.value();
    runtime_impl->throw_if_manual_backend_failure_requested_for_test();

    ChannelOptions channel_options{};
    channel_options.max_loaned_samples = options.max_inflight_samples;
    channel_options.expected_recipient_count = 1U;
    channel_options.tensor_transfer_mode = ChannelTensorTransferMode::kSameProcessSidecar;

    ChannelRuntime retained_runtime(runtime_impl);
    auto publisher = retained_runtime.create_tensor_publisher(service_name, channel_options);
    if (!publisher) {
      // Publisher construction never made the sidecar identity observable, so this is a rollback,
      // not retirement. Retaining the tombstone here would permanently consume the service name
      // after a failed cold setup attempt.
      runtime_impl->rollback_manual_tensor_sidecar_store(service_name, registered_store);
      registered_store.reset();
      return make_unexpected(std::move(publisher).error());
    }
    publisher->impl_->binding_ = std::move(binding.impl_);
    registered_store.reset();
    return publisher;
  } catch (const std::length_error&) {
    if (registered_store) {
      runtime_impl->rollback_manual_tensor_sidecar_store(service_name, registered_store);
    }
    return channel_internal::make_unexpected_nothrow<ManualTensorPublisher>(
        ErrorCode::kResourceExhausted, "manual Tensor publisher setup exceeds allocator limits");
  } catch (const std::bad_alloc&) {
    if (registered_store) {
      runtime_impl->rollback_manual_tensor_sidecar_store(service_name, registered_store);
    }
    return channel_internal::make_unexpected_nothrow<ManualTensorPublisher>(
        ErrorCode::kResourceExhausted, "manual Tensor publisher setup allocation failed");
  } catch (...) {
    if (registered_store) {
      runtime_impl->rollback_manual_tensor_sidecar_store(service_name, registered_store);
    }
    return channel_internal::make_unexpected_nothrow<ManualTensorPublisher>(
        ErrorCode::kFailure, "manual Tensor publisher backend setup failed");
  }
}

ChannelSubscriber<Tensor>::ChannelSubscriber(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ChannelSubscriber<Tensor>::ChannelSubscriber(ChannelSubscriber&& other) noexcept = default;

ChannelSubscriber<Tensor>& ChannelSubscriber<Tensor>::operator=(
    ChannelSubscriber&& other) noexcept = default;

ChannelSubscriber<Tensor>::~ChannelSubscriber() = default;

channel_internal::WaitSetSubscriber* ChannelSubscriber<Tensor>::wait_set_subscriber() noexcept {
  return impl_.get();
}

std::uint64_t ChannelSubscriber<Tensor>::accepted_end_of_stream_sequence() const noexcept {
  return impl_ != nullptr && impl_->stream_closed_ ? impl_->last_operation_sequence_ : 0U;
}

expected<bool, Error> ChannelSubscriber<Tensor>::has_samples() const {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (impl_->stream_closed_) {
    return false;
  }
  auto data = from_holomq_result(impl_->subscriber_.has_samples());
  if (!data) {
    return data;
  }
  const auto* const terminal_subscriber = optional_value_pointer(impl_->terminal_subscriber_);
  if (terminal_subscriber == nullptr) {
    return impl_->pending_inbound_.has_value() || *data;
  }
  // Refresh both receiver mappings before returning the OR result. Tensor replacement can keep
  // the data lane continuously ready; probing only that lane would let the private terminal
  // transport mapping lapse while its registry-level peer count still appears connected.
  auto terminal = from_holomq_result(terminal_subscriber->has_samples());
  if (!terminal) {
    return terminal;
  }
  return impl_->pending_inbound_.has_value() || *data ||
         impl_->pending_terminal_inbound_.has_value() || *terminal;
}

expected<Tensor, Error> ChannelSubscriber<Tensor>::receive() {
  auto sample = receive_sample();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  return std::move(sample).value().data;
}

expected<Sample<Tensor>, Error> ChannelSubscriber<Tensor>::receive_sample() {
  if (!impl_) {
    return make_unexpected(
        make_channel_error(ErrorCode::kClosed, "Tensor channel subscriber is closed"));
  }
  return impl_->receive_sample(true);
}

expected<Sample<Tensor>, Error> ChannelSubscriber<Tensor>::receive_sample_if_data_accepted(
    const bool accept_data) {
  if (!impl_) {
    return make_unexpected(
        make_channel_error(ErrorCode::kClosed, "Tensor channel subscriber is closed"));
  }
  return impl_->receive_sample(accept_data);
}

expected<void, Error> ChannelSubscriber<Tensor>::refresh_connections() {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (impl_->stream_closed_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto refreshed = impl_->subscriber_.has_samples();
  if (!refreshed) {
    return make_unexpected(from_holomq_error(std::move(refreshed).error()));
  }
  if (auto* const terminal_subscriber = optional_value_pointer(impl_->terminal_subscriber_);
      terminal_subscriber != nullptr) {
    auto terminal_refreshed = terminal_subscriber->has_samples();
    if (!terminal_refreshed) {
      return make_unexpected(from_holomq_error(std::move(terminal_refreshed).error()));
    }
  }
  return {};
}

bool ChannelSubscriber<Tensor>::has_publishers() const noexcept {
  if (!impl_ || impl_->stream_closed_ || !impl_->subscriber_.has_publishers()) {
    return false;
  }
  const auto* const terminal_subscriber = optional_value_pointer(impl_->terminal_subscriber_);
  return terminal_subscriber == nullptr || terminal_subscriber->has_publishers();
}

expected<ManualTensorPublisher, Error> ChannelRuntime::create_tensor_publisher(
    const std::string_view channel_name, const ChannelOptions& options) {
  return create_tensor_publisher(
      channel_name, options, channel_internal::PublisherPressureOptions{});
}

expected<ManualTensorPublisher, Error> ChannelRuntime::create_tensor_publisher(
    const std::string_view channel_name, const ChannelOptions& options,
    const channel_internal::PublisherPressureOptions& pressure) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  ChannelOptions effective_options = options;
  if (replaces_oldest_unread_sample(effective_options.queue_policy)) {
    effective_options.replacement_attribution_required = true;
  }
  if (auto valid = validate_tensor_channel_options(effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (channel_internal::is_ordered_terminal_service_id(channel_name)) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor channel name occupies the reserved terminal namespace"));
  }
  std::string terminal_service_id;
  if (effective_options.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle &&
      replaces_oldest_unread_sample(effective_options.queue_policy)) {
    terminal_service_id = channel_internal::ordered_terminal_service_id(channel_name);
    if (terminal_service_id.empty()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor channel name cannot derive a protected terminal service identity"));
    }
  }
  const auto capabilities = impl_->context_.pubsub_capabilities();
  if (auto valid = validate_capabilities(capabilities, effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  auto sidecar_store = impl_->tensor_sidecar_store(channel_name, effective_options);
  if (!sidecar_store) {
    return make_unexpected(std::move(sidecar_store).error());
  }

  const auto holomq_options = to_holomq_publisher_options(effective_options, pressure);
  const bool use_setup_replacement_callback =
      replaces_oldest_unread_sample(effective_options.queue_policy) &&
      effective_options.replacement_attribution_required;

  // For lossy channels, attribution is requested authoritatively by on_sample_replaced() below
  // (which sets PublisherPubSubOptions::replacement_attribution_required on the builder).
  // Holoscan releases skipped sidecar claims directly from the replaced descriptor payload via
  // that setup-time callback, not through HoloMQ's per-send release registry.
  auto publisher_builder = impl_->context_.publisher_builder(std::string(channel_name))
                               .service_options(to_holomq_service_options(effective_options))
                               .publisher_options(holomq_options)
                               .payload_type<TensorTransportPayload>();
  if (use_setup_replacement_callback) {
    publisher_builder =
        std::move(publisher_builder)
            .on_sample_replaced([sidecar_store = sidecar_store.value()](
                                    const TensorTransportPayload& payload,
                                    const holomq::SampleReplacedInfo& info) noexcept {
              if (payload.payload_kind == ChannelPayloadKind::kLocalBufferDescriptor) {
                sidecar_store->release_replaced(payload.reference.local, info.receiver_id);
              }
            });
  }

  auto publisher = std::move(publisher_builder).create();
  if (!publisher) {
    return make_unexpected(from_holomq_error(std::move(publisher).error(),
                                             "failed to create Tensor channel publisher"));
  }
  std::optional<holomq::Publisher<TensorTransportPayload>> terminal_publisher;
  if (effective_options.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle &&
      replaces_oldest_unread_sample(effective_options.queue_policy)) {
    auto created =
        impl_->context_.publisher_builder(terminal_service_id)
            .service_options(channel_internal::ordered_terminal_service_options(effective_options))
            .publisher_options(
                channel_internal::ordered_terminal_publisher_options(effective_options))
            .payload_type<TensorTransportPayload>()
            .create();
    if (!created) {
      return make_unexpected(from_holomq_error(
          std::move(created).error(), "failed to create protected Tensor terminal publisher"));
    }
    terminal_publisher.emplace(std::move(*created));
  }
  return ManualTensorPublisher(std::make_unique<ManualTensorPublisher::Impl>(
      std::move(publisher).value(),
      std::move(terminal_publisher),
      std::move(sidecar_store).value(),
      effective_options,
      impl_,
      nullptr,
      pressure.enables_graph_retry_boundary(effective_options.publisher_history_depth)));
}

expected<ChannelSubscriber<Tensor>, Error> ChannelRuntime::create_tensor_subscriber(
    const std::string_view channel_name, const ChannelOptions& options,
    std::optional<LogicalTensorRequirementsRecord> receive_contract,
    const std::uint32_t receive_control_capacity) {
  if (!impl_) {
    return make_unexpected(make_channel_error(ErrorCode::kClosed, "channel runtime is closed"));
  }
  ChannelOptions effective_options = options;
  if (replaces_oldest_unread_sample(effective_options.queue_policy)) {
    effective_options.replacement_attribution_required = true;
  }
  if (auto valid = validate_tensor_channel_options(effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (channel_internal::is_ordered_terminal_service_id(channel_name)) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor channel name occupies the reserved terminal namespace"));
  }
  std::string terminal_service_id;
  if (effective_options.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle &&
      replaces_oldest_unread_sample(effective_options.queue_policy)) {
    terminal_service_id = channel_internal::ordered_terminal_service_id(channel_name);
    if (terminal_service_id.empty()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor channel name cannot derive a protected terminal service identity"));
    }
  }
  const auto capabilities = impl_->context_.pubsub_capabilities();
  if (auto valid = validate_capabilities(capabilities, effective_options); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  std::shared_ptr<TensorSidecarStore> sidecar_store;
  if (effective_options.tensor_transfer_mode != ChannelTensorTransferMode::kBufferHandle) {
    auto created_sidecar_store = impl_->tensor_sidecar_store(channel_name, effective_options);
    if (!created_sidecar_store) {
      return make_unexpected(std::move(created_sidecar_store).error());
    }
    sidecar_store = std::move(created_sidecar_store).value();
  }

  auto subscriber = impl_->context_.subscriber_builder(std::string(channel_name))
                        .service_options(to_holomq_service_options(effective_options))
                        .subscriber_options(to_holomq_subscriber_options(effective_options))
                        .payload_type<TensorTransportPayload>()
                        .create();
  if (!subscriber) {
    return make_unexpected(from_holomq_error(std::move(subscriber).error(),
                                             "failed to create Tensor channel subscriber"));
  }
  std::optional<holomq::Subscriber<TensorTransportPayload>> terminal_subscriber;
  if (effective_options.tensor_transfer_mode == ChannelTensorTransferMode::kBufferHandle &&
      replaces_oldest_unread_sample(effective_options.queue_policy)) {
    auto created =
        impl_->context_.subscriber_builder(terminal_service_id)
            .service_options(channel_internal::ordered_terminal_service_options(effective_options))
            .subscriber_options(channel_internal::ordered_terminal_subscriber_options())
            .payload_type<TensorTransportPayload>()
            .create();
    if (!created) {
      return make_unexpected(from_holomq_error(
          std::move(created).error(), "failed to create protected Tensor terminal subscriber"));
    }
    terminal_subscriber.emplace(std::move(*created));
  }
  std::size_t recipient_index = 0U;
  if (sidecar_store) {
    auto registered_recipient = sidecar_store->register_recipient(subscriber->recipient_id());
    if (!registered_recipient) {
      return make_unexpected(std::move(registered_recipient).error());
    }
    recipient_index = registered_recipient.value();
  }
  std::unique_ptr<channel_internal::CatalogTensorReceiveControl> receive_control;
  if (receive_contract && receive_contract->memory_kind.has_value() &&
      (receive_contract->memory_kind == MemoryKind::kHost ||
       receive_contract->memory_kind == MemoryKind::kPinnedHost ||
       receive_contract->memory_kind == MemoryKind::kCudaDevice)) {
    auto created = channel_internal::CatalogTensorReceiveControl::create(
        impl_, impl_->context_, *receive_contract, receive_control_capacity);
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    receive_control = std::move(created).value();
  }
  auto subscriber_impl =
      std::make_unique<ChannelSubscriber<Tensor>::Impl>(impl_,
                                                        impl_->context_,
                                                        std::move(subscriber).value(),
                                                        std::move(terminal_subscriber),
                                                        std::move(sidecar_store),
                                                        recipient_index,
                                                        std::move(receive_contract),
                                                        std::move(receive_control));
  return ChannelSubscriber<Tensor>(std::move(subscriber_impl));
}
}  // namespace holoscan
