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
#include "catalog_attach_wait.hpp"
#include "catalog_tensor_clock_health.hpp"
#include "catalog_tensor_receive_control.hpp"
#include "channel_common.hpp"
#include "channel_payload_limits.hpp"
#include "channel_tensor_schema_payload.hpp"
#include "channel_tensor_sidecar_store.hpp"
#include "channel_tensor_test_access.hpp"
#include "endpoint_catalog_state.hpp"
#include "holomq_tensor.hpp"
#include "payload_descriptor_wire.hpp"
#include "process_identity.hpp"

namespace holoscan {

using channel_internal::from_holomq_error;
using channel_internal::from_holomq_publication_result;
using channel_internal::from_holomq_result;
using channel_internal::InlinePayloadBucket;
using channel_internal::make_channel_error;
using channel_internal::PublisherPressureOptions;
using channel_internal::replaces_oldest_unread_sample;
using channel_internal::TensorSchemaPayloadBucket;
using channel_internal::to_holomq_publisher_options;
using channel_internal::to_holomq_service_options;
using channel_internal::to_holomq_subscriber_options;

namespace {

using ReceiveProbeBarrier = channel_internal::TensorPublisherTestAccess::ReceiveProbeBarrier;

template <typename Bucket>
void set_tensor_schema_catalog_fields(Bucket& bucket, const std::uint64_t producer_generation,
                                      const std::uint64_t health_report_sequence) noexcept {
  if constexpr (requires {
                  bucket.producer_generation;
                  bucket.health_report_sequence;
                }) {
    bucket.producer_generation = producer_generation;
    bucket.health_report_sequence = health_report_sequence;
  }
}

template <typename Bucket>
[[nodiscard]] std::uint64_t tensor_schema_producer_generation(const Bucket& bucket) noexcept {
  if constexpr (requires { bucket.producer_generation; }) {
    return bucket.producer_generation;
  }
  return 0U;
}

template <typename Bucket>
[[nodiscard]] std::uint64_t tensor_schema_health_report_sequence(const Bucket& bucket) noexcept {
  if constexpr (requires { bucket.health_report_sequence; }) {
    return bucket.health_report_sequence;
  }
  return 0U;
}

[[nodiscard]] bool dtype_equal(const DLDataType lhs, const DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

template <typename Bucket>
[[nodiscard]] expected<void, Error> validate_manual_tensor_schema_end_of_stream(
    const Bucket& bucket) noexcept {
  const SampleMetadata& metadata = bucket.metadata;
  if (bucket.operation != channel_internal::InlinePayloadOperation::kEndOfStream ||
      bucket.payload_size != 0U || metadata.version.major != 1U || metadata.version.minor != 0U ||
      metadata.sequence == 0U || metadata.capture_timestamp_ns != 0 ||
      metadata.source_clock_id.valid() || metadata.publish_timestamp_ns != 0 ||
      metadata.publish_clock_id.valid() || metadata.trace_id != 0U || metadata.frame_id != 0U ||
      metadata.byte_size != 0U || metadata.payload_kind != PayloadKind::kDescriptorBacked ||
      metadata.flags != SampleFlags::kNone) {
    return make_unexpected(make_channel_error(ErrorCode::kProtocolError,
                                              "Tensor schema end-of-stream record is invalid"));
  }
  return {};
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

[[nodiscard]] bool schema_identity_equal(const SchemaIdentityRecord& actual,
                                         const SchemaIdentity expected) noexcept {
  return actual.canonical_name == expected.canonical_name && actual.version == expected.version &&
         actual.schema_sha256 == expected.schema_sha256 && actual.id == expected.id;
}

[[nodiscard]] bool supported_catalog_tensor_memory(const MemoryKind memory_kind) noexcept {
  return memory_kind == MemoryKind::kHost || memory_kind == MemoryKind::kPinnedHost ||
         memory_kind == MemoryKind::kCudaDevice;
}

[[nodiscard]] bool tensor_memory_matches_contract(const Tensor& tensor,
                                                  const MemoryKind memory_kind,
                                                  const DeviceId device) noexcept {
  const DLDevice actual = tensor.device();
  switch (memory_kind) {
    case MemoryKind::kHost:
      return actual.device_type == kDLCPU && actual.device_id == 0;
    case MemoryKind::kPinnedHost:
      return actual.device_type == kDLCUDAHost && actual.device_id == 0;
    case MemoryKind::kCudaDevice:
      return actual.device_type == kDLCUDA && (!device.valid() || actual.device_id == device.value);
    default:
      return false;
  }
}

[[nodiscard]] bool tensor_memory_matches_contract(const TensorViewMetadata& metadata,
                                                  const MemoryKind memory_kind,
                                                  const DeviceId device) noexcept {
  switch (memory_kind) {
    case MemoryKind::kHost:
      return metadata.device.device_type == kDLCPU && metadata.device.device_id == 0;
    case MemoryKind::kPinnedHost:
      return metadata.device.device_type == kDLCUDAHost && metadata.device.device_id == 0;
    case MemoryKind::kCudaDevice:
      return metadata.device.device_type == kDLCUDA &&
             (!device.valid() || metadata.device.device_id == device.value);
    default:
      return false;
  }
}

[[nodiscard]] std::optional<std::uint64_t> tensor_axis_byte_stride(
    const Tensor& tensor, const std::size_t axis, const std::uint64_t item_bytes) noexcept {
  const auto shape = tensor.shape_span();
  if (axis >= shape.size()) {
    return std::nullopt;
  }
  const auto strides = tensor.dlpack_strides_span();
  if (!strides.empty()) {
    if (strides.size() != shape.size() || strides[axis] < 0 ||
        static_cast<std::uint64_t>(strides[axis]) >
            std::numeric_limits<std::uint64_t>::max() / item_bytes) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(strides[axis]) * item_bytes;
  }
  std::uint64_t stride = item_bytes;
  for (std::size_t reverse = shape.size(); reverse > axis + 1U; --reverse) {
    if (shape[reverse - 1U] < 0) {
      return std::nullopt;
    }
    const std::uint64_t extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(shape[reverse - 1U]), 1U);
    if (stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return std::nullopt;
    }
    stride *= extent;
  }
  return stride;
}

[[nodiscard]] expected<void, Error> validate_tensor_schema_field_contract(
    const Tensor& tensor, const channel_internal::TensorSchemaStagingOptions& contract) {
  const auto shape = tensor.shape_span();
  const std::int64_t byte_size = tensor.nbytes();
  if (tensor.data() == nullptr || byte_size <= 0 ||
      static_cast<std::uint64_t>(byte_size) > contract.max_byte_span ||
      !tensor_memory_matches_contract(tensor, contract.memory_kind, contract.device) ||
      (contract.dtype && !dtype_equal(tensor.dtype(), *contract.dtype)) ||
      (contract.rank && shape.size() != *contract.rank) ||
      (!contract.axes.empty() && shape.size() != contract.axes.size()) ||
      (contract.strides == TensorStridePolicy::kContiguousOnly && !tensor.is_contiguous())) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema source does not match its immutable field contract"));
  }
  const std::uint64_t item_bytes = dlpack_dtype_item_size_bytes(tensor.dtype());
  if (item_bytes == 0U) {
    return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                              "Tensor schema source has an invalid DLPack dtype"));
  }
  for (std::size_t index = 0U; index < contract.axes.size(); ++index) {
    const auto stride = tensor_axis_byte_stride(tensor, index, item_bytes);
    if (shape[index] < 0 ||
        static_cast<std::uint64_t>(shape[index]) > contract.axes[index].max_extent || !stride ||
        (contract.axes[index].max_abs_stride_bytes != 0U &&
         *stride > contract.axes[index].max_abs_stride_bytes)) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument,
                             "Tensor schema source exceeds its immutable field extent contract"));
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_tensor_schema_field_contract(
    const TensorViewMetadata& metadata, const PayloadDescriptor& descriptor,
    const channel_internal::TensorSchemaStagingOptions& contract) noexcept {
  if (metadata.rank > TensorViewMetadata::kMaxRank || metadata.byte_size == 0U ||
      descriptor.byte_size != metadata.byte_size || metadata.byte_size > contract.max_byte_span ||
      !tensor_memory_matches_contract(metadata, contract.memory_kind, contract.device) ||
      (contract.dtype && !dtype_equal(metadata.dtype, *contract.dtype)) ||
      (contract.rank && metadata.rank != *contract.rank) ||
      (!contract.axes.empty() && metadata.rank != contract.axes.size())) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema descriptor violates its immutable field contract"));
  }

  const std::uint64_t item_bytes = dlpack_dtype_item_size_bytes(metadata.dtype);
  if (item_bytes == 0U) {
    return make_unexpected(make_channel_error(
        ErrorCode::kInvalidArgument, "Tensor schema descriptor has an invalid DLPack dtype"));
  }
  std::uint64_t contiguous_stride = item_bytes;
  for (std::size_t reverse = metadata.rank; reverse > 0U; --reverse) {
    const std::size_t index = reverse - 1U;
    if (metadata.shape[index] < 0 ||
        (metadata.has_explicit_byte_strides && metadata.byte_strides[index] < 0)) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema descriptor has an invalid axis"));
    }
    const std::uint64_t stride = metadata.has_explicit_byte_strides
                                     ? static_cast<std::uint64_t>(metadata.byte_strides[index])
                                     : contiguous_stride;
    if ((contract.strides == TensorStridePolicy::kContiguousOnly && stride != contiguous_stride) ||
        (!contract.axes.empty() &&
         (static_cast<std::uint64_t>(metadata.shape[index]) > contract.axes[index].max_extent ||
          (contract.axes[index].max_abs_stride_bytes != 0U &&
           stride > contract.axes[index].max_abs_stride_bytes)))) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument,
                             "Tensor schema descriptor exceeds its immutable field axis contract"));
    }
    const std::uint64_t extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(metadata.shape[index]), 1U);
    if (contiguous_stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema descriptor stride overflows"));
    }
    contiguous_stride *= extent;
  }
  return {};
}

[[nodiscard]] expected<const detail::ChannelEndpointRecord*, Error>
validate_catalog_tensor_schema_endpoint(const ChannelEndpointCatalog::EndpointView endpoint,
                                        const SchemaIdentity expected_schema,
                                        const std::size_t payload_size,
                                        const std::size_t tensor_field_count,
                                        const TopicDirection expected_direction,
                                        const ChannelAttachOptions& options) noexcept {
  const auto* const record = detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_TYPE_MISMATCH"});
  }
  if (record->direction != expected_direction) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_CLIENT_ROLE_MISMATCH"});
  }
  if (payload_size == 0U || tensor_field_count == 0U ||
      tensor_field_count > channel_internal::kMaximumDescriptorTensorFields ||
      record->service_static.payload_identity.kind != PayloadIdentityKind::kSchema ||
      record->service_static.payload_identity.payload_kind != PayloadKind::kDescriptorBacked ||
      !record->service_static.schema_identity ||
      !schema_identity_equal(*record->service_static.schema_identity, expected_schema) ||
      record->service_static.serialized_payload_capacity_bytes != payload_size) {
    return make_unexpected(Error{ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"});
  }
  if (record->service_static.id.empty() ||
      record->service_static.tensor_fields.size() != tensor_field_count ||
      record->tensor_fields.size() != tensor_field_count ||
      record->service_static.max_subscriber_depth == 0U ||
      record->service_static.max_subscriber_borrowed == 0U ||
      record->service_static.publisher_history_depth != 0U || !record->service_static.event_wake ||
      (record->service_static.queue_policy != QueuePolicy::kBlocking &&
       record->service_static.queue_policy != QueuePolicy::kDropOldest)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_CONTRACT_INVALID"});
  }
  if ((record->attach.peer != PeerRequirement::kOptional &&
       record->attach.peer != PeerRequirement::kRequired) ||
      !record->attach.max_wait || record->attach.max_wait->count() <= 0 ||
      (options.timeout &&
       (options.timeout->count() <= 0 || *options.timeout > *record->attach.max_wait))) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_BOUND_INVALID"});
  }
  if ((expected_direction == TopicDirection::kImport &&
       (!record->publisher_request || !record->publisher_request->max_loaned_samples ||
        *record->publisher_request->max_loaned_samples == 0U)) ||
      (expected_direction == TopicDirection::kExport &&
       (!record->subscriber_request || record->subscriber_request->queue_depth == 0U ||
        record->subscriber_request->max_borrowed_samples == 0U))) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_BOUND_INVALID"});
  }
  // Descriptor mapping bounds use packed TensorViewMetadata::byte_size. Non-contiguous strides
  // require the addressed byte span instead, so keep bounded-strided fields rejected until that
  // span is validated through the mapping path.
  for (std::size_t index = 0U; index < record->service_static.tensor_fields.size(); ++index) {
    const auto& field = record->service_static.tensor_fields[index];
    const auto& endpoint_field = record->tensor_fields[index];
    if (field.tensor_index != index ||
        (index != 0U &&
         record->service_static.tensor_fields[index - 1U].field_id >= field.field_id) ||
        field.field_id != endpoint_field.field_id ||
        field.tensor_index != endpoint_field.tensor_index || !field.tensor.memory_kind ||
        !supported_catalog_tensor_memory(*field.tensor.memory_kind) || !field.tensor.dtype ||
        !field.tensor.rank || !field.tensor.max_rank || !field.tensor.max_byte_span ||
        *field.tensor.max_rank == 0U || *field.tensor.max_byte_span == 0U ||
        field.tensor.strides == TensorStridePolicy::kBoundedStrided ||
        (field.tensor.foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable &&
         field.tensor.foreign_emit_policy != InterfaceTensorEmitPolicy::kCopyIfNeeded)) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_CONTRACT_INVALID"});
    }
  }
  return record;
}

[[nodiscard]] ChannelOptions catalog_tensor_schema_options(
    const detail::ChannelEndpointRecord& record) noexcept {
  const auto queue_policy = record.service_static.queue_policy == QueuePolicy::kDropOldest
                                ? ChannelQueuePolicy::kDropOldest
                                : ChannelQueuePolicy::kBlocking;
  return ChannelOptions{
      .queue_policy = queue_policy,
      .max_subscriber_queue_depth = record.service_static.max_subscriber_depth,
      .subscriber_queue_depth = record.subscriber_request
                                    ? record.subscriber_request->queue_depth
                                    : record.service_static.max_subscriber_depth,
      .max_subscriber_borrowed_samples = record.service_static.max_subscriber_borrowed,
      .max_borrowed_samples = record.subscriber_request
                                  ? record.subscriber_request->max_borrowed_samples
                                  : record.service_static.max_subscriber_borrowed,
      .publisher_history_depth = record.service_static.publisher_history_depth,
      .max_loaned_samples = record.publisher_request && record.publisher_request->max_loaned_samples
                                ? *record.publisher_request->max_loaned_samples
                                : 1U,
      .expected_recipient_count = 1U,
      .enable_event_wake = record.service_static.event_wake,
      .replacement_attribution_required = queue_policy == ChannelQueuePolicy::kDropOldest,
      .tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle,
  };
}

[[nodiscard]] ChannelOptions catalog_tensor_schema_health_options(ChannelOptions options) noexcept {
  options.queue_policy = ChannelQueuePolicy::kBlocking;
  options.replacement_attribution_required = false;
  return options;
}

template <bool Publisher, typename Endpoint>
[[nodiscard]] expected<void, Error> wait_for_catalog_tensor_schema_peer(
    Endpoint& endpoint, const detail::ChannelEndpointRecord& record,
    const ChannelAttachOptions& options) {
  return channel_internal::wait_for_catalog_required_peer(
      record, options, [&]() -> expected<bool, Error> {
        if constexpr (Publisher) {
          auto refreshed = endpoint.update_connections();
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
          return endpoint.has_subscribers();
        } else {
          auto refreshed = endpoint.has_samples();
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
          return endpoint.has_publishers();
        }
      });
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
    cudaEvent_t producer_ready{};
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
        if (result->stages_device_memory()) {
          if (cudaEventCreateWithFlags(&result->slots_[index].producer_ready,
                                       cudaEventDisableTiming) != cudaSuccess ||
              cudaEventCreateWithFlags(&result->slots_[index].copy_complete,
                                       cudaEventDisableTiming) != cudaSuccess) {
            return make_unexpected(make_channel_error(
                ErrorCode::kResourceExhausted, "Tensor schema staging event allocation failed"));
          }
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
      // CUDA is already unavailable, so no valid event wait can be issued. Continue best-effort
      // destruction; normal Runtime teardown keeps the device available through this pool wait.
      return;
    }
    for (Slot& slot : slots_) {
      if (slot.copy_complete != nullptr) {
        // A borrowed Runtime stream can outlive this publisher, but its staging buffers cannot.
        // Wait for the slot-local copy before BufferPool releases the backing allocation.
        if (slot.copy_pending) {
          static_cast<void>(cudaEventSynchronize(slot.copy_complete));
        }
        static_cast<void>(cudaEventDestroy(slot.copy_complete));
      }
      if (slot.producer_ready != nullptr) {
        static_cast<void>(cudaEventDestroy(slot.producer_ready));
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

    const bool cuda_source = device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged;
    cudaStream_t producer_stream =
        cuda_source ? detail::TensorAccess::readiness_stream(tensor) : nullptr;
    cudaStream_t copy_stream = joined_stream != nullptr ? joined_stream : producer_stream;
    if (cuda_source && copy_stream == nullptr) {
      return make_unexpected(make_channel_error(
          ErrorCode::kNotSupported, "CUDA Tensor schema staging requires a producer stream"));
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
      if (!selected.active()) {
        copied = make_unexpected(
            make_channel_error(ErrorCode::kFailure, "Tensor schema staging device is unavailable"));
      } else if (joined_stream != nullptr && producer_stream != joined_stream) {
        if (!tensor.has_readiness_stream_contract()) {
          copied = make_unexpected(
              make_channel_error(ErrorCode::kNotImplemented,
                                 "Tensor schema source has no producer-readiness stream contract"));
        } else if (producer_stream == nullptr) {
          // Null means that the reference cannot expose a process-local stream identity; it does
          // not identify CUDA's default stream. Delegate so imported references can enqueue their
          // transport readiness wait on the joined service stream.
          copied = tensor.synchronize_for_consumer_stream(joined_stream);
        } else if (cudaEventRecord(slot->producer_ready, producer_stream) != cudaSuccess ||
                   cudaStreamWaitEvent(joined_stream, slot->producer_ready, 0U) != cudaSuccess) {
          copied = make_unexpected(make_channel_error(
              ErrorCode::kFailure, "Tensor schema producer-to-service ordering failed"));
        }
      }
      if (copied && cudaMemcpyAsync(slot->buffer.ptr(),
                                    tensor.data(),
                                    static_cast<std::size_t>(tensor.nbytes()),
                                    cudaMemcpyDeviceToDevice,
                                    copy_stream) != cudaSuccess) {
        copied = make_unexpected(
            make_channel_error(ErrorCode::kFailure, "Tensor schema device staging copy failed"));
      } else if (copied) {
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
      // Host-resident staging completes before publication. A future cross-placement CUDA-to-host
      // contract must admit that source independently rather than hiding it behind pool placement.
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
      return std::make_shared<TensorSchemaJoinedReadinessStream>(device, stream, true);
    } catch (const std::bad_alloc&) {
      static_cast<void>(cudaStreamDestroy(stream));
      return make_unexpected(make_channel_error(
          ErrorCode::kResourceExhausted, "Tensor schema joined-readiness owner allocation failed"));
    }
  }

  [[nodiscard]] static expected<std::shared_ptr<TensorSchemaJoinedReadinessStream>, Error> borrow(
      const DeviceId device, cudaStream_t stream) {
    if (!device.valid() || stream == nullptr) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema staging stream binding is invalid"));
    }
    try {
      return std::make_shared<TensorSchemaJoinedReadinessStream>(device, stream, false);
    } catch (const std::bad_alloc&) {
      return make_unexpected(make_channel_error(
          ErrorCode::kResourceExhausted, "Tensor schema staging stream owner allocation failed"));
    }
  }

  TensorSchemaJoinedReadinessStream(const DeviceId device, cudaStream_t stream, bool owned) noexcept
      : device_(device), stream_(stream), owned_(owned) {}

  TensorSchemaJoinedReadinessStream(const TensorSchemaJoinedReadinessStream&) = delete;
  TensorSchemaJoinedReadinessStream& operator=(const TensorSchemaJoinedReadinessStream&) = delete;

  ~TensorSchemaJoinedReadinessStream() {
    if (!owned_) {
      // Each staging pool waits for its pending slot events before releasing backing storage. The
      // Runtime CUDA execution owner separately retains and eventually destroys this stream.
      return;
    }
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
  bool owned_{};
};

void clear_tensor_schema_field_reservation(
    channel_internal::TensorSchemaPayloadFieldReservation& field) noexcept {
  // Match reverse declaration-order destruction: backing and Tensor references must retire before
  // the readiness resource that makes their final release safe.
  field.retained_state.reset();
  field.retained_tensor.reset();
  field.readiness_owner.reset();
  field.descriptor = {};
}

void clear_tensor_schema_payload_reservation(
    channel_internal::TensorSchemaPayloadReservation& reservation) noexcept {
  for (std::size_t index = 0U; index < reservation.field_count; ++index) {
    if (auto* const field = reservation.field_at(index); field != nullptr) {
      clear_tensor_schema_field_reservation(*field);
    }
  }
  reservation.field_count = 0U;
  reservation.additional_fields = nullptr;
  reservation.field_storage.reset();
}

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
      clear_tensor_schema_field_reservation(field);
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
    clear_tensor_schema_payload_reservation(reservation_);
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

template <std::size_t Size, typename Bucket>
class TensorSchemaPayloadPublisherImpl final
    : public channel_internal::TensorSchemaPayloadPublisher {
 public:
  TensorSchemaPayloadPublisherImpl(
      holomq::Publisher<Bucket>&& publisher,
      std::optional<holomq::Publisher<Bucket>> terminal_publisher,
      std::unique_ptr<channel_internal::InlinePayloadPublisher> health_publisher,
      std::shared_ptr<TensorSidecarStore> sidecar_store,
      std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts,
      std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools,
      std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream,
      std::shared_ptr<TensorSchemaReservationPool> reservation_pool, ChannelOptions options,
      const bool graph_retry_boundary, const ClockId capture_clock,
      const bool catalog_control) noexcept
      : reservation_pool_(std::move(reservation_pool)),
        joined_stream_(std::move(joined_stream)),
        field_contracts_(std::move(field_contracts)),
        staging_pools_(std::move(staging_pools)),
        sidecar_store_(std::move(sidecar_store)),
        publisher_(std::move(publisher)),
        terminal_publisher_(std::move(terminal_publisher)),
        health_publisher_(std::move(health_publisher)),
        options_(options),
        graph_retry_boundary_(graph_retry_boundary),
        capture_clock_(capture_clock),
        producer_generation_(catalog_control ? channel_internal::mint_producer_generation() : 0U) {}

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error> reserve_tensors(
      const std::span<const Tensor> tensors, const bool require_direct_export) override {
    if (tensors.empty() || tensors.size() > channel_internal::kMaximumDescriptorTensorFields ||
        (!field_contracts_.empty() && field_contracts_.size() != tensors.size()) ||
        (!staging_pools_.empty() && staging_pools_.size() != tensors.size())) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument,
                             "Tensor schema reservation field count violates its endpoint"));
    }
    for (std::size_t index = 0U; index < field_contracts_.size(); ++index) {
      auto valid = validate_tensor_schema_field_contract(tensors[index], field_contracts_[index]);
      if (!valid) {
        return make_unexpected(std::move(valid).error());
      }
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
      std::array<channel_internal::HoloMQExportableTensorMemoryReference*,
                 channel_internal::kMaximumDescriptorTensorFields>
          exportable_memory{};
      std::span<channel_internal::HoloMQExportableTensorMemoryReference* const>
          resolved_exportable_memory;
      if (!require_direct_export && !staging_pools_.empty()) {
        resolved_exportable_memory =
            std::span<channel_internal::HoloMQExportableTensorMemoryReference* const>{
                exportable_memory.data(), tensors.size()};
        for (std::size_t index = 0U; index < tensors.size(); ++index) {
          exportable_memory[index] =
              channel_internal::holomq_exportable_tensor_memory_reference(tensors[index]);
        }
        if (std::any_of(exportable_memory.begin(),
                        exportable_memory.begin() + tensors.size(),
                        [](const auto* memory) { return memory == nullptr; })) {
          return reserve_mixed_tensors(tensors, resolved_exportable_memory, std::move(reservation));
        }
      }
      auto direct_export = reserve_direct_tensors(tensors, resolved_exportable_memory, reservation);
      if (direct_export) {
        return reservation;
      }
      clear_field_reservations(reservation);
      if (require_direct_export || staging_pools_.empty()) {
        return make_unexpected(std::move(direct_export).error());
      }
      return reserve_mixed_tensors(tensors, resolved_exportable_memory, std::move(reservation));
    }
    return reserve_local_tensors(tensors, std::move(reservation));
  }

 private:
  [[nodiscard]] static expected<void, Error> reserve_direct_tensors(
      const std::span<const Tensor> tensors,
      const std::span<channel_internal::HoloMQExportableTensorMemoryReference* const>
          exportable_memory,
      channel_internal::TensorSchemaPayloadReservation& reservation) {
    if (!exportable_memory.empty() && exportable_memory.size() != tensors.size()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError, "Tensor schema exportable backing count is inconsistent"));
    }
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      auto descriptor = exportable_memory.empty()
                            ? channel_internal::export_holomq_tensor_descriptor(tensors[index])
                            : channel_internal::export_holomq_tensor_descriptor(
                                  tensors[index], *exportable_memory[index]);
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
      clear_tensor_schema_field_reservation(*reservation.field_at(index));
    }
  }

  static void synchronize_stream(cudaStream_t stream) noexcept {
    if (stream != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream));
    }
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReservation, Error>
  reserve_mixed_tensors(
      const std::span<const Tensor> tensors,
      const std::span<channel_internal::HoloMQExportableTensorMemoryReference* const>
          exportable_memory,
      channel_internal::TensorSchemaPayloadReservation reservation) {
    if (exportable_memory.size() != tensors.size() || staging_pools_.size() != tensors.size()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kProtocolError, "Tensor schema mixed staging inventory is inconsistent"));
    }
    cudaStream_t joined_stream = joined_stream_ != nullptr ? joined_stream_->get() : nullptr;
    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      if (exportable_memory[index] != nullptr) {
        auto descriptor = channel_internal::export_holomq_tensor_descriptor(
            tensors[index], *exportable_memory[index]);
        if (!descriptor) {
          synchronize_stream(joined_stream);
          return make_unexpected(std::move(descriptor).error());
        }
        *reservation.field_at(index) = channel_internal::TensorSchemaPayloadFieldReservation{
            .descriptor =
                channel_internal::payload_descriptor_from_holomq_tensor_descriptor(*descriptor),
            .retained_tensor = tensors[index],
        };
        continue;
      }
      if (staging_pools_[index] == nullptr) {
        synchronize_stream(joined_stream);
        return make_unexpected(make_channel_error(
            ErrorCode::kNotSupported,
            "Tensor schema field requires exportable backing and has no staging pool"));
      }
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
      channel_internal::TensorSchemaPayloadReservation reservation,
      const SampleMetadata transport_metadata) override {
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
    if (producer_generation_ != 0U &&
        (operation_sequence_ == std::numeric_limits<std::uint64_t>::max() ||
         transport_metadata.sequence != operation_sequence_ + 1U)) {
      release_reservation(reservation);
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "catalog Tensor schema operation sequence is invalid"));
    }
    bucket.operation = channel_internal::InlinePayloadOperation::kData;
    bucket.metadata = transport_metadata;
    set_tensor_schema_catalog_fields(bucket, producer_generation_, report_sequence_);
    std::memcpy(bucket.bytes.data(), payload.data(), payload.size());
    bucket.payload_size = static_cast<std::uint32_t>(payload.size());
    auto published = send_message(std::move(*message), std::move(reservation));
    if (producer_generation_ != 0U &&
        (published || publication_error_consumes_authority(published.error().code))) {
      operation_sequence_ = transport_metadata.sequence;
    }
    return published;
  }

  [[nodiscard]] expected<void, Error> validate_sample_metadata(
      const SampleMetadata& metadata) const noexcept override {
    if (producer_generation_ == 0U) {
      return channel_internal::TensorSchemaPayloadPublisher::validate_sample_metadata(metadata);
    }
    // Runtime forwards the publication coordinate it stamped, so only an incoherent pair is
    // noncanonical here. A timestamp with no clock names no domain and cannot be resolved by a
    // peer, while a clock with a zero coordinate remains the canonical absent value.
    if (metadata.version.major != 1U || metadata.version.minor != 0U ||
        (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U ||
        (!metadata.publish_clock_id.valid() && metadata.publish_timestamp_ns != 0)) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor schema sample metadata is noncanonical"));
    }
    if (capture_clock_.valid()) {
      if (metadata.source_clock_id != capture_clock_) {
        return make_unexpected(make_channel_error(
            ErrorCode::kInvalidArgument,
            "Tensor schema sample capture clock does not match its catalog endpoint"));
      }
    } else if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor schema sample has capture metadata for an uncaptured catalog endpoint"));
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> validate_field_descriptor(
      const std::uint32_t field_index, const TensorViewMetadata& metadata,
      const PayloadDescriptor& descriptor) const noexcept override {
    if (field_contracts_.empty()) {
      return {};
    }
    if (field_index >= field_contracts_.size()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "Tensor schema field index exceeds endpoint contract"));
    }
    return validate_tensor_schema_field_contract(
        metadata, descriptor, field_contracts_[field_index]);
  }

  [[nodiscard]] expected<TimePoint, Error> make_capture_time(
      const std::int64_t timestamp_ns) const noexcept override {
    if (!capture_clock_.valid()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return TimePoint{.timestamp_ns = timestamp_ns, .clock_id = capture_clock_};
  }

  [[nodiscard]] channel_internal::InlinePublicationResult close_stream(
      const SampleMetadata metadata) override {
    if (closed_) {
      return {
          .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
          .error = Error{ErrorCode::kClosed},
      };
    }
    if (metadata.sequence == 0U ||
        (producer_generation_ != 0U &&
         (operation_sequence_ == std::numeric_limits<std::uint64_t>::max() ||
          metadata.sequence != operation_sequence_ + 1U))) {
      return {
          .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
          .error = Error{ErrorCode::kInvalidArgument},
      };
    }
    if (!publisher_.has_subscribers() ||
        (terminal_publisher_ && !terminal_publisher_->has_subscribers())) {
      return {
          .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
          .error = Error{ErrorCode::kNotReady},
      };
    }
    auto& terminal_lane = terminal_publisher_ ? *terminal_publisher_ : publisher_;
    auto message = terminal_lane.loan();
    if (!message) {
      return {
          .phase = channel_internal::InlinePublicationPhase::kPreSubmitRejected,
          .error = from_holomq_error(std::move(message).error()),
      };
    }
    auto& bucket = message->payload();
    bucket = Bucket{};
    bucket.operation = channel_internal::InlinePayloadOperation::kEndOfStream;
    bucket.metadata = metadata;
    bucket.metadata.byte_size = 0U;
    bucket.metadata.payload_kind = PayloadKind::kDescriptorBacked;
    set_tensor_schema_catalog_fields(bucket, producer_generation_, report_sequence_);
    auto sent = terminal_lane.send(std::move(*message));
    auto published = from_holomq_publication_result(
        std::move(sent), options_.expected_recipient_count, graph_retry_boundary_);
    if (published) {
      if (producer_generation_ != 0U) {
        operation_sequence_ = metadata.sequence;
      }
      closed_ = true;
      return {
          .phase = channel_internal::InlinePublicationPhase::kAccepted,
          .recipient_count = *published,
      };
    }
    Error error = std::move(published).error();
    const bool consumes_authority =
        channel_internal::terminal_publication_error_consumes_authority(error.code);
    if (consumes_authority) {
      if (producer_generation_ != 0U) {
        operation_sequence_ = metadata.sequence;
      }
      closed_ = true;
    }
    return {
        .phase = consumes_authority ? channel_internal::InlinePublicationPhase::kPostSubmitUncertain
                                    : channel_internal::InlinePublicationPhase::kPreSubmitRejected,
        .error = std::move(error),
    };
  }

  [[nodiscard]] std::uint64_t producer_generation() const noexcept override {
    return producer_generation_;
  }

  [[nodiscard]] expected<std::uint64_t, Error> next_operation_sequence() const noexcept override {
    if (producer_generation_ == 0U) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    return operation_sequence_ + 1U;
  }

  [[nodiscard]] expected<void, Error> report_clock_health(const ClockHealthReport report) override {
    if (producer_generation_ == 0U || health_publisher_ == nullptr || closed_ ||
        !channel_internal::valid_catalog_clock_health_report(report)) {
      return make_unexpected(Error{closed_ ? ErrorCode::kClosed : ErrorCode::kInvalidArgument});
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
    auto result = health_publisher_->emit_phase(wire, metadata);
    if (result.phase == channel_internal::InlinePublicationPhase::kAccepted &&
        result.recipient_count == options_.expected_recipient_count) {
      operation_sequence_ = proposed_operation;
      report_sequence_ = proposed_report;
      return {};
    }
    if (result.phase == channel_internal::InlinePublicationPhase::kAccepted ||
        result.phase == channel_internal::InlinePublicationPhase::kPostSubmitUncertain) {
      operation_sequence_ = proposed_operation;
      report_sequence_ = proposed_report;
      closed_ = true;
      return make_unexpected(result.error.code == ErrorCode::kSuccess
                                 ? Error{ErrorCode::kPublicationIndeterminate}
                                 : std::move(result.error));
    }
    return make_unexpected(std::move(result.error));
  }

  [[nodiscard]] expected<std::uint32_t, Error> publish_encoded_payload(
      const channel_internal::TensorSchemaPayloadEncoder& encoder,
      channel_internal::TensorSchemaPayloadReservation reservation,
      const SampleMetadata transport_metadata) override {
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
    if (producer_generation_ != 0U &&
        (operation_sequence_ == std::numeric_limits<std::uint64_t>::max() ||
         transport_metadata.sequence != operation_sequence_ + 1U)) {
      release_reservation(reservation);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    bucket.operation = channel_internal::InlinePayloadOperation::kData;
    bucket.metadata = transport_metadata;
    set_tensor_schema_catalog_fields(bucket, producer_generation_, report_sequence_);
    bucket.payload_size = static_cast<std::uint32_t>(*encoded);
    auto published = send_message(std::move(*message), std::move(reservation));
    if (producer_generation_ != 0U &&
        (published || publication_error_consumes_authority(published.error().code))) {
      operation_sequence_ = transport_metadata.sequence;
    }
    return published;
  }

 private:
  [[nodiscard]] expected<std::uint32_t, Error> send_message(
      holomq::OutboundMessage<Bucket> message,
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
    if (health_publisher_ != nullptr) {
      auto health = health_publisher_->update_connections();
      if (!health) {
        return health;
      }
    }
    auto data = from_holomq_result(publisher_.update_connections());
    if (!data || !terminal_publisher_) {
      return data;
    }
    return from_holomq_result(terminal_publisher_->update_connections());
  }

  [[nodiscard]] bool has_subscribers() const noexcept override {
    return !closed_ && publisher_.has_subscribers() &&
           (!terminal_publisher_ || terminal_publisher_->has_subscribers()) &&
           (health_publisher_ == nullptr || health_publisher_->has_subscribers());
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
    clear_tensor_schema_payload_reservation(reservation);
  }

  std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream_;
  std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts_;
  std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools_;
  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  holomq::Publisher<Bucket> publisher_;
  std::optional<holomq::Publisher<Bucket>> terminal_publisher_;
  std::unique_ptr<channel_internal::InlinePayloadPublisher> health_publisher_;
  ChannelOptions options_;
  bool graph_retry_boundary_{};
  ClockId capture_clock_{};
  std::uint64_t producer_generation_{};
  std::uint64_t operation_sequence_{};
  std::uint64_t report_sequence_{};
  bool closed_{};
};

template <std::size_t Size, typename Bucket>
class TensorSchemaPayloadSubscriberImpl final
    : public channel_internal::CatalogTensorSchemaPayloadSubscriber,
      public channel_internal::ProtectedTerminalWaitSetSubscriber,
      public channel_internal::StagedDataWaitSetSubscriber {
 public:
  TensorSchemaPayloadSubscriberImpl(
      holomq::Context& context, holomq::Subscriber<Bucket>&& subscriber,
      std::optional<holomq::Subscriber<Bucket>> terminal_subscriber,
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber,
      std::shared_ptr<TensorSidecarStore> sidecar_store, std::size_t recipient_index,
      std::vector<TensorSchemaReceiveControlPtr> receive_controls,
      std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts,
      const bool catalog_control, ReceiveProbeBarrier* receive_probe_barrier,
      const std::uint64_t catalog_clock_count = 0U) noexcept
      : subscriber_(std::move(subscriber)),
        context_(&context),
        terminal_subscriber_(std::move(terminal_subscriber)),
        health_subscriber_(std::move(health_subscriber)),
        sidecar_store_(std::move(sidecar_store)),
        recipient_index_(recipient_index),
        receive_controls_(std::move(receive_controls)),
        field_contracts_(std::move(field_contracts)),
        replacement_gaps_allowed_(catalog_control && terminal_subscriber_.has_value()),
        catalog_control_(catalog_control),
        receive_probe_barrier_(receive_probe_barrier),
        catalog_clock_count_(catalog_clock_count) {}

  ~TensorSchemaPayloadSubscriberImpl() override {
    sidecar_store_->unregister_recipient(recipient_index_, subscriber_.recipient_id());
  }

  [[nodiscard]] expected<bool, Error> has_samples() const override {
    if (stream_closed_) {
      return false;
    }
    if (inbound_ || pending_terminal_ || pending_health_) {
      return true;
    }
    auto health =
        health_subscriber_ ? health_subscriber_->has_samples() : expected<bool, Error>{false};
    if (!health) {
      return make_unexpected(std::move(health).error());
    }
    auto data = from_holomq_result(subscriber_.has_samples());
    if (!data) {
      return data;
    }
    if (!terminal_subscriber_) {
      return *data || *health;
    }
    auto terminal = from_holomq_result(terminal_subscriber_->has_samples());
    if (!terminal) {
      return terminal;
    }
    return *data || *terminal || *health;
  }

  [[nodiscard]] bool has_publishers() const noexcept override {
    return !stream_closed_ && subscriber_.has_publishers() &&
           (!terminal_subscriber_ || terminal_subscriber_->has_publishers()) &&
           (!health_subscriber_ || health_subscriber_->has_publishers());
  }

  // Read from the port rather than a cached flag so this can never disagree with the transport.
  [[nodiscard]] bool event_wake_enabled() const noexcept override {
    return subscriber_.is_event_wake_enabled();
  }

  // The parent Channel runtime owns this transport context and outlives every endpoint created
  // from it, so this borrow stays valid for the subscriber's lifetime.
  [[nodiscard]] void* wait_context_runtime() noexcept override { return context_; }

  // Known asymmetry with has_samples(): the data and terminal lanes are attached here, but the
  // clock-health lane is not, even though has_samples() ORs its readiness into the answer. A
  // health record therefore cannot wake a thread blocked in WaitSetSubscriber::wait_for_readiness;
  // it is discovered on that loop's bounded fallback pass instead (kMaxIterationWait in
  // channel.cpp), so health-only readiness is reported with up to that much latency rather than
  // promptly. Attaching a third lane here would also need this function's rollback story extended,
  // which is why it is documented rather than done in passing.
  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void* wait_set) override {
    if (wait_set == nullptr || attachment_.has_value() || terminal_attachment_.has_value()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto attached = static_cast<holomq::WaitSet*>(wait_set)->attach(subscriber_);
    if (!attached) {
      return make_unexpected(from_holomq_error(std::move(attached).error()));
    }
    if (terminal_subscriber_) {
      auto terminal_attached =
          static_cast<holomq::WaitSet*>(wait_set)->attach(*terminal_subscriber_);
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
    return borrow_payload_if_data_accepted(true);
  }

  [[nodiscard]] expected<channel_internal::TensorSchemaPayloadReceiveGuard, Error>
  borrow_payload_if_data_accepted(const bool accept_data) override {
    if (payload_loan_active_) {
      return make_unexpected(make_channel_error(
          ErrorCode::kFailure, "Tensor schema payload already has an active transport loan"));
    }
    // Checked before the catalog branch, not after it. That branch returns on every path, so a
    // latch consulted below it is unreachable for a catalog-attached subscriber: the first call
    // would report end of stream and every later one would fall through to kNotReady, making a
    // terminal stream read as merely idle. The manual family already latches here.
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (catalog_control_) {
      auto operation = receive_catalog_operation(accept_data);
      if (!operation) {
        return make_unexpected(std::move(operation).error());
      }
      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kData) {
        if (operation->schema_payload.empty() || !payload_loan_active_) {
          release_payload_loan();
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        return channel_internal::TensorSchemaPayloadReceiveGuard{*this, operation->schema_payload};
      }
      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kEndOfStream) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (operation->kind ==
          channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (operation->kind == channel_internal::CatalogReceivedOperationKind::kDataPending) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    auto staged = stage_next();
    if (!staged) {
      return make_unexpected(std::move(staged).error());
    }
    if (!inbound_ && !pending_terminal_) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    const auto& bucket = inbound_ ? inbound_->payload() : *pending_terminal_;
    if (bucket.operation == channel_internal::InlinePayloadOperation::kEndOfStream) {
      if (auto valid = validate_manual_tensor_schema_end_of_stream(bucket); !valid) {
        if (inbound_) {
          inbound_.reset();
        } else {
          pending_terminal_.reset();
        }
        return make_unexpected(std::move(valid).error());
      }
      end_of_stream_sequence_ = bucket.metadata.sequence;
      stream_closed_ = true;
      inbound_.reset();
      pending_terminal_.reset();
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (!accept_data) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (bucket.payload_size == 0U || bucket.payload_size > Size) {
      // No payload loan exists until the length is validated, so discard the staged transport
      // record directly instead of routing through release_payload_loan().  Keeping a malformed
      // record staged would make every later valid terminal record observe the same protocol
      // failure forever.
      if (inbound_) {
        inbound_.reset();
      } else {
        pending_terminal_.reset();
      }
      return make_unexpected(
          make_channel_error(ErrorCode::kProtocolError, "Tensor schema payload length is invalid"));
    }
    borrowed_terminal_ = !inbound_.has_value();
    payload_loan_active_ = true;
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

  [[nodiscard]] expected<void, Error> validate_field_descriptor(
      const std::uint32_t field_index, const TensorViewMetadata& metadata,
      const PayloadDescriptor& descriptor) const noexcept override {
    if (field_contracts_.empty()) {
      return {};
    }
    if (field_index >= field_contracts_.size()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "Tensor schema field index exceeds endpoint contract"));
    }
    return validate_tensor_schema_field_contract(
        metadata, descriptor, field_contracts_[field_index]);
  }

  void release_recipient_claim(const PayloadDescriptor& descriptor) noexcept override {
    if (descriptor.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return;
    }
    sidecar_store_->release_recipient_claim(channel_descriptor_from_payload_descriptor(descriptor),
                                            recipient_index_);
  }

  void release_payload_loan() noexcept override {
    if (!payload_loan_active_) {
      return;
    }
    inbound_.reset();
    if (borrowed_terminal_) {
      pending_terminal_.reset();
      borrowed_terminal_ = false;
    }
    payload_loan_active_ = false;
  }

  [[nodiscard]] std::optional<std::uint64_t> end_of_stream_sequence() const noexcept override {
    return end_of_stream_sequence_;
  }

  [[nodiscard]] expected<void, Error> detach_catalog_wait_set() noexcept override {
    terminal_attachment_.reset();
    terminal_wait_handle_ = 0U;
    attachment_.reset();
    return {};
  }

  [[nodiscard]] std::optional<std::uint64_t> protected_terminal_wait_handle()
      const noexcept override {
    return terminal_wait_handle_ == 0U ? std::nullopt
                                       : std::optional<std::uint64_t>{terminal_wait_handle_};
  }

  [[nodiscard]] std::optional<channel_internal::StagedDataState> staged_data_state()
      const noexcept override {
    if (!inbound_ ||
        inbound_->payload().operation != channel_internal::InlinePayloadOperation::kData ||
        inbound_->payload().payload_size == 0U || inbound_->payload().payload_size > Size ||
        inbound_->payload().metadata.sequence == 0U) {
      return std::nullopt;
    }
    return channel_internal::StagedDataState{
        .sequence = inbound_->payload().metadata.sequence,
        .capture_time = inbound_->payload().metadata.capture_time()};
  }

  [[nodiscard]] std::optional<std::uint64_t> staged_terminal_sequence() const noexcept override {
    if (!pending_terminal_ ||
        pending_terminal_->operation != channel_internal::InlinePayloadOperation::kEndOfStream ||
        tensor_schema_producer_generation(*pending_terminal_) == 0U ||
        foreign_producer_generation(tensor_schema_producer_generation(*pending_terminal_)) ||
        pending_terminal_->metadata.sequence == 0U) {
      return std::nullopt;
    }
    return pending_terminal_->metadata.sequence;
  }

  [[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
  receive_catalog_operation(const bool accept_data) noexcept override {
    if (!catalog_control_ || payload_loan_active_) {
      return make_unexpected(
          Error{catalog_control_ ? ErrorCode::kFailure : ErrorCode::kNotSupported});
    }
    auto staged = stage_next();
    if (!staged) {
      return make_unexpected(std::move(staged).error());
    }
    auto fenced = fence_pending_incarnations();
    if (!fenced) {
      return make_unexpected(std::move(fenced).error());
    }
    if (*fenced) {
      return channel_internal::CatalogReceivedOperation{
          .kind = channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged};
    }

    auto payload_lane = next_payload_lane();
    if (!payload_lane) {
      return make_unexpected(std::move(payload_lane).error());
    }
    const auto* const bucket = payload_bucket(*payload_lane);
    const std::uint64_t payload_sequence =
        bucket != nullptr ? bucket->metadata.sequence : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t health_sequence = pending_health_
                                              ? pending_health_state_.publisher_operation_sequence
                                              : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t next_sequence = std::min(payload_sequence, health_sequence);
    bool initial_attachment_gap = false;
    auto aligned = align_expected_operation_sequence(
        next_sequence, payload_sequence, health_sequence, bucket, initial_attachment_gap);
    if (!aligned) {
      return make_unexpected(std::move(aligned).error());
    }
    if (!*aligned) {
      return channel_internal::CatalogReceivedOperation{};
    }
    if (health_sequence == expected_operation_sequence_) {
      return accept_pending_health(initial_attachment_gap);
    }
    if (bucket == nullptr || payload_sequence != expected_operation_sequence_) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return accept_pending_payload(*payload_lane, *bucket, accept_data, initial_attachment_gap);
  }

  [[nodiscard]] expected<bool, Error> has_pending_catalog_operation() const noexcept override {
    if (!catalog_control_) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (inbound_ || pending_terminal_ || pending_health_) {
      return true;
    }
    auto data = from_holomq_result(subscriber_.has_samples());
    if (!data) {
      return data;
    }
    auto terminal = terminal_subscriber_ ? from_holomq_result(terminal_subscriber_->has_samples())
                                         : expected<bool, Error>{false};
    if (!terminal) {
      return terminal;
    }
    auto health =
        health_subscriber_ ? health_subscriber_->has_samples() : expected<bool, Error>{false};
    if (!health) {
      return health;
    }
    return *data || *terminal || *health;
  }

  [[nodiscard]] std::uint64_t latched_producer_generation() const noexcept override {
    return latched_producer_generation_;
  }

  [[nodiscard]] expected<void, Error> supersede_producer_generation(
      const std::uint64_t generation) noexcept override {
    if (!catalog_control_ || generation == 0U || latched_producer_generation_ != 0U || inbound_ ||
        pending_terminal_ || pending_health_ || expected_operation_sequence_ != 1U ||
        stream_closed_ || payload_loan_active_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    superseded_producer_generation_ = generation;
    return {};
  }

  [[nodiscard]] expected<channel_internal::CatalogImportQuiescence, Error>
  import_quiescence() noexcept override {
    if (!catalog_control_) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    auto staged = stage_next();
    if (!staged) {
      return make_unexpected(std::move(staged).error());
    }
    channel_internal::CatalogImportQuiescence result{
        .latched_producer_generation = latched_producer_generation_,
    };
    const auto classify = [&](const Bucket& bucket) {
      const bool foreign = foreign_producer_generation(tensor_schema_producer_generation(bucket));
      result.pending_foreign = result.pending_foreign || foreign;
      result.pending_current = result.pending_current || !foreign;
    };
    if (inbound_) {
      classify(inbound_->payload());
    }
    if (pending_terminal_) {
      classify(*pending_terminal_);
    }
    if (pending_health_) {
      const bool foreign = foreign_producer_generation(pending_health_state_.producer_epoch);
      result.pending_foreign = result.pending_foreign || foreign;
      result.pending_current = result.pending_current || !foreign;
    }
    return result;
  }

 private:
  enum class PendingPayloadLane : std::uint8_t {
    kNone,
    kData,
    kTerminal,
  };

  [[nodiscard]] const Bucket* payload_bucket(const PendingPayloadLane lane) const noexcept {
    if (lane == PendingPayloadLane::kData && inbound_) {
      return &inbound_->payload();
    }
    if (lane == PendingPayloadLane::kTerminal && pending_terminal_) {
      return &*pending_terminal_;
    }
    return nullptr;
  }

  [[nodiscard]] expected<PendingPayloadLane, Error> next_payload_lane() const noexcept {
    const std::uint64_t data_sequence = inbound_ ? inbound_->payload().metadata.sequence
                                                 : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t terminal_sequence = pending_terminal_
                                                ? pending_terminal_->metadata.sequence
                                                : std::numeric_limits<std::uint64_t>::max();
    if (data_sequence == terminal_sequence &&
        data_sequence != std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (data_sequence < terminal_sequence) {
      return PendingPayloadLane::kData;
    }
    if (terminal_sequence < data_sequence) {
      return PendingPayloadLane::kTerminal;
    }
    return PendingPayloadLane::kNone;
  }

  void consume_payload(const PendingPayloadLane lane) noexcept {
    if (lane == PendingPayloadLane::kData) {
      inbound_.reset();
    } else if (lane == PendingPayloadLane::kTerminal) {
      pending_terminal_.reset();
    }
  }

  [[nodiscard]] bool discard_pending_generation(const std::uint64_t generation) noexcept {
    bool discarded = false;
    if (inbound_ && tensor_schema_producer_generation(inbound_->payload()) == generation) {
      inbound_.reset();
      discarded = true;
    }
    if (pending_terminal_ && tensor_schema_producer_generation(*pending_terminal_) == generation) {
      pending_terminal_.reset();
      discarded = true;
    }
    if (pending_health_ && pending_health_state_.producer_epoch == generation) {
      pending_health_ = false;
      discarded = true;
    }
    return discarded;
  }

  [[nodiscard]] expected<bool, Error> fence_pending_incarnations() noexcept {
    if (superseded_producer_generation_ != 0U &&
        discard_pending_generation(superseded_producer_generation_)) {
      return true;
    }
    struct Candidate {
      std::uint64_t sequence{};
      std::uint64_t generation{};
      PendingPayloadLane lane{PendingPayloadLane::kNone};
      bool health{};
    };
    std::array<Candidate, 3U> candidates{};
    std::size_t count = 0U;
    if (inbound_) {
      candidates[count++] = Candidate{inbound_->payload().metadata.sequence,
                                      tensor_schema_producer_generation(inbound_->payload()),
                                      PendingPayloadLane::kData,
                                      false};
    }
    if (pending_terminal_) {
      candidates[count++] = Candidate{pending_terminal_->metadata.sequence,
                                      tensor_schema_producer_generation(*pending_terminal_),
                                      PendingPayloadLane::kTerminal,
                                      false};
    }
    if (pending_health_) {
      candidates[count++] = Candidate{pending_health_state_.publisher_operation_sequence,
                                      pending_health_state_.producer_epoch,
                                      PendingPayloadLane::kNone,
                                      true};
    }
    // Generation zero is malformed wire data, not a foreign incarnation to discard silently.
    for (std::size_t index = 0U; index < count; ++index) {
      if (candidates[index].generation == 0U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    if (latched_producer_generation_ == 0U && count > 1U) {
      const Candidate* earliest = &candidates[0];
      for (std::size_t index = 1U; index < count; ++index) {
        if (candidates[index].sequence < earliest->sequence) {
          earliest = &candidates[index];
        }
      }
      bool changed = false;
      for (std::size_t index = 0U; index < count; ++index) {
        const auto& candidate = candidates[index];
        if (candidate.generation == earliest->generation) {
          continue;
        }
        if (candidate.sequence == earliest->sequence) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        if (candidate.health) {
          pending_health_ = false;
        } else {
          consume_payload(candidate.lane);
        }
        changed = true;
      }
      if (changed) {
        return true;
      }
    }

    bool changed = false;
    if (inbound_ &&
        foreign_producer_generation(tensor_schema_producer_generation(inbound_->payload()))) {
      inbound_.reset();
      changed = true;
    }
    if (pending_terminal_ &&
        foreign_producer_generation(tensor_schema_producer_generation(*pending_terminal_))) {
      pending_terminal_.reset();
      changed = true;
    }
    if (pending_health_ && foreign_producer_generation(pending_health_state_.producer_epoch)) {
      pending_health_ = false;
      changed = true;
    }
    return changed;
  }

  [[nodiscard]] expected<bool, Error> align_expected_operation_sequence(
      const std::uint64_t next_sequence, const std::uint64_t payload_sequence,
      const std::uint64_t health_sequence, const Bucket* const bucket,
      bool& initial_attachment_gap) noexcept {
    if (next_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
    if (next_sequence < expected_operation_sequence_ ||
        (payload_sequence == health_sequence &&
         payload_sequence != std::numeric_limits<std::uint64_t>::max())) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (next_sequence == expected_operation_sequence_) {
      return true;
    }
    initial_attachment_gap = latched_producer_generation_ == 0U &&
                             expected_operation_sequence_ == 1U && !health_state_.has_report;
    if (!initial_attachment_gap && !replacement_gaps_allowed_) {
      return false;
    }
    if (!initial_attachment_gap && health_sequence == next_sequence) {
      if (health_state_.report_sequence == std::numeric_limits<std::uint64_t>::max() ||
          pending_health_state_.report_sequence != health_state_.report_sequence + 1U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    } else if (!initial_attachment_gap &&
               (bucket == nullptr ||
                tensor_schema_health_report_sequence(*bucket) > health_state_.report_sequence)) {
      return false;
    } else if (!initial_attachment_gap &&
               tensor_schema_health_report_sequence(*bucket) < health_state_.report_sequence) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    expected_operation_sequence_ = next_sequence;
    return true;
  }

  [[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error> accept_pending_health(
      const bool initial_attachment_gap) noexcept {
    const std::uint64_t operation_sequence = pending_health_state_.publisher_operation_sequence;
    if ((!initial_attachment_gap &&
         (health_state_.report_sequence == std::numeric_limits<std::uint64_t>::max() ||
          pending_health_state_.report_sequence != health_state_.report_sequence + 1U)) ||
        (initial_attachment_gap && pending_health_state_.report_sequence == 0U) ||
        pending_health_metadata_.sequence != operation_sequence ||
        pending_health_metadata_.byte_size !=
            channel_internal::kTensorCatalogClockHealthControlBytes ||
        pending_health_metadata_.payload_kind != PayloadKind::kInlineTyped) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    latched_producer_generation_ = pending_health_state_.producer_epoch;
    health_state_ = pending_health_state_;
    pending_health_ = false;
    advance_expected_operation_sequence();
    return channel_internal::CatalogReceivedOperation{
        .kind = channel_internal::CatalogReceivedOperationKind::kClockHealth,
        .clock_health = health_state_,
    };
  }

  [[nodiscard]] expected<void, Error> validate_end_of_stream(const PendingPayloadLane lane,
                                                             const Bucket& bucket) const noexcept {
    const SampleMetadata& metadata = bucket.metadata;
    const PendingPayloadLane expected_lane =
        terminal_subscriber_ ? PendingPayloadLane::kTerminal : PendingPayloadLane::kData;
    if (bucket.operation != channel_internal::InlinePayloadOperation::kEndOfStream ||
        tensor_schema_producer_generation(bucket) == 0U || lane != expected_lane ||
        bucket.payload_size != 0U || metadata.version.major != 1U || metadata.version.minor != 0U ||
        metadata.sequence == 0U || metadata.capture_timestamp_ns != 0 ||
        metadata.source_clock_id.valid() || metadata.publish_timestamp_ns != 0 ||
        metadata.publish_clock_id.valid() || metadata.trace_id != 0U || metadata.frame_id != 0U ||
        metadata.byte_size != 0U || metadata.payload_kind != PayloadKind::kDescriptorBacked ||
        metadata.flags != SampleFlags::kNone) {
      return make_unexpected(make_channel_error(ErrorCode::kProtocolError,
                                                "Tensor schema end-of-stream record is invalid"));
    }
    return {};
  }

  [[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error> accept_pending_payload(
      const PendingPayloadLane lane, const Bucket& bucket, const bool accept_data,
      const bool initial_attachment_gap) noexcept {
    const std::uint64_t producer_generation = tensor_schema_producer_generation(bucket);
    if (bucket.operation == channel_internal::InlinePayloadOperation::kEndOfStream) {
      if (auto valid = validate_end_of_stream(lane, bucket); !valid) {
        return make_unexpected(std::move(valid).error());
      }
      if (!initial_attachment_gap &&
          tensor_schema_health_report_sequence(bucket) != health_state_.report_sequence) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const SampleMetadata metadata = bucket.metadata;
      if (initial_attachment_gap) {
        health_state_.report_sequence = tensor_schema_health_report_sequence(bucket);
      }
      latched_producer_generation_ = producer_generation;
      consume_payload(lane);
      stream_closed_ = true;
      advance_expected_operation_sequence();
      return channel_internal::CatalogReceivedOperation{
          .kind = channel_internal::CatalogReceivedOperationKind::kEndOfStream,
          .metadata = metadata,
      };
    }
    // A coordinate naming a clock outside the declared range resolves to no record, so exposing it
    // would hand a subscriber a plausible publish_time() it cannot place on any timeline. Runtime
    // always projects a declared identity, so this rejects only a malformed or stale peer. A manual
    // subscriber declares no catalog and never reaches this branch with a populated coordinate.
    if (producer_generation == 0U || bucket.metadata.sequence == 0U ||
        (!initial_attachment_gap &&
         tensor_schema_health_report_sequence(bucket) != health_state_.report_sequence) ||
        bucket.operation != channel_internal::InlinePayloadOperation::kData ||
        bucket.payload_size == 0U || bucket.payload_size > Size ||
        lane != PendingPayloadLane::kData ||
        (catalog_control_ && !channel_internal::publication_coordinate_declared(
                                 bucket.metadata, catalog_clock_count_))) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (!accept_data) {
      return channel_internal::CatalogReceivedOperation{
          .kind = channel_internal::CatalogReceivedOperationKind::kDataPending};
    }
    if (initial_attachment_gap) {
      health_state_.report_sequence = tensor_schema_health_report_sequence(bucket);
    }
    latched_producer_generation_ = producer_generation;
    payload_loan_active_ = true;
    advance_expected_operation_sequence();
    return channel_internal::CatalogReceivedOperation{
        .kind = channel_internal::CatalogReceivedOperationKind::kData,
        .metadata = bucket.metadata,
        .schema_payload = std::span<const std::byte>{bucket.bytes}.first(bucket.payload_size),
    };
  }

  void advance_expected_operation_sequence() noexcept {
    if (expected_operation_sequence_ != std::numeric_limits<std::uint64_t>::max()) {
      ++expected_operation_sequence_;
    }
  }

  [[nodiscard]] expected<void, Error> stage_next() noexcept {
    const auto stage_data = [&]() -> expected<bool, Error> {
      if (inbound_) {
        return true;
      }
      auto inbound = subscriber_.try_take();
      if (inbound) {
        inbound_.emplace(std::move(*inbound));
      } else {
        Error error = from_holomq_error(std::move(inbound).error());
        if (error.code != ErrorCode::kNotReady) {
          return make_unexpected(std::move(error));
        }
      }
      return inbound_.has_value();
    };
    const auto stage_terminal = [&]() -> expected<bool, Error> {
      if (!pending_terminal_ && terminal_subscriber_) {
        auto terminal = terminal_subscriber_->try_take();
        if (terminal) {
          pending_terminal_.emplace(terminal->payload());
        } else {
          Error error = from_holomq_error(std::move(terminal).error());
          if (error.code != ErrorCode::kNotReady) {
            return make_unexpected(std::move(error));
          }
        }
      }
      return pending_terminal_.has_value();
    };
    auto staged = channel_internal::stage_protected_terminal_transaction(
        stage_data, stage_terminal, [&]() noexcept {
          if (receive_probe_barrier_ != nullptr &&
              receive_probe_barrier_->pause_after_empty_data_probe()) {
            receive_probe_barrier_ = nullptr;
          }
        });
    if (!staged) {
      return make_unexpected(std::move(staged).error());
    }
    if (!pending_health_ && health_subscriber_) {
      auto ready = health_subscriber_->has_samples();
      if (!ready) {
        return make_unexpected(std::move(ready).error());
      }
      if (*ready) {
        std::array<std::byte, channel_internal::kTensorCatalogClockHealthControlBytes> wire{};
        auto received = health_subscriber_->receive(wire);
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        if (received->payload_size != channel_internal::kTensorCatalogClockHealthControlBytes) {
          return make_unexpected(make_channel_error(
              ErrorCode::kProtocolError, "Tensor schema clock health record is invalid"));
        }
        auto health = channel_internal::decode_tensor_catalog_clock_health(wire);
        if (!health) {
          return make_unexpected(std::move(health).error());
        }
        pending_health_metadata_ = received->metadata;
        pending_health_state_ = *health;
        pending_health_ = true;
      }
    }
    return {};
  }

  [[nodiscard]] bool foreign_producer_generation(const std::uint64_t generation) const noexcept {
    return generation == 0U || generation == superseded_producer_generation_ ||
           (latched_producer_generation_ != 0U && generation != latched_producer_generation_);
  }

  holomq::Subscriber<Bucket> subscriber_;
  holomq::Context* context_{};
  std::optional<holomq::Subscriber<Bucket>> terminal_subscriber_;
  std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber_;
  std::optional<holomq::WaitSetAttachment> attachment_;
  std::optional<holomq::WaitSetAttachment> terminal_attachment_;
  std::optional<holomq::InboundMessage<Bucket>> inbound_;
  std::optional<Bucket> pending_terminal_;
  std::shared_ptr<TensorSidecarStore> sidecar_store_;
  std::size_t recipient_index_{};
  std::vector<TensorSchemaReceiveControlPtr> receive_controls_;
  std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts_;
  std::uint64_t terminal_wait_handle_{};
  std::uint64_t latched_producer_generation_{};
  std::uint64_t superseded_producer_generation_{};
  std::uint64_t expected_operation_sequence_{1U};
  channel_internal::CatalogClockHealthState health_state_{};
  channel_internal::CatalogClockHealthState pending_health_state_{};
  SampleMetadata pending_health_metadata_{};
  bool pending_health_{};
  bool replacement_gaps_allowed_{};
  bool catalog_control_{};
  // Declared clock count of the catalog behind this endpoint. Only the catalog receive path
  // consults it, so a manual subscriber leaves it zero and never range-checks a coordinate.
  std::uint64_t catalog_clock_count_{};
  bool borrowed_terminal_{};
  bool payload_loan_active_{};
  bool stream_closed_{};
  std::optional<std::uint64_t> end_of_stream_sequence_;
  ReceiveProbeBarrier* receive_probe_barrier_{};
};

template <std::size_t Size, typename Bucket>
[[nodiscard]] PayloadDescriptor payload_descriptor_from_bucket(
    const Bucket& bucket, const std::size_t descriptor_offset) noexcept {
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

template <std::size_t Size, typename Bucket>
[[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
create_tensor_schema_payload_publisher_impl(
    holomq::Context& context, const std::string_view channel_name, const ChannelOptions& options,
    std::shared_ptr<TensorSidecarStore> sidecar_store, const std::size_t payload_size,
    const std::vector<std::size_t>& descriptor_offsets,
    std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts,
    std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools,
    std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream,
    const PublisherPressureOptions pressure, const ClockId capture_clock,
    std::unique_ptr<channel_internal::InlinePayloadPublisher> health_publisher,
    const bool catalog_control) {
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
                               .template payload_type<Bucket>();
  if (replaces_oldest_unread_sample(options.queue_policy) &&
      options.replacement_attribution_required) {
    publisher_builder =
        std::move(publisher_builder)
            .on_sample_replaced([sidecar_store, descriptor_offsets](
                                    const Bucket& bucket,
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
  std::optional<holomq::Publisher<Bucket>> terminal_publisher;
  if (replaces_oldest_unread_sample(options.queue_policy)) {
    const std::string terminal_service_id =
        channel_internal::ordered_terminal_service_id(channel_name);
    if (terminal_service_id.empty()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor schema channel cannot derive a protected terminal service identity"));
    }
    auto created =
        context.publisher_builder(terminal_service_id)
            .service_options(channel_internal::ordered_terminal_service_options(options))
            .publisher_options(channel_internal::ordered_terminal_publisher_options(options))
            .template payload_type<Bucket>()
            .create();
    if (!created) {
      return make_unexpected(
          from_holomq_error(std::move(created).error(),
                            "failed to create protected Tensor schema terminal publisher"));
    }
    terminal_publisher.emplace(std::move(*created));
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
  return std::make_unique<TensorSchemaPayloadPublisherImpl<Size, Bucket>>(
      std::move(publisher).value(),
      std::move(terminal_publisher),
      std::move(health_publisher),
      std::move(sidecar_store),
      std::move(field_contracts),
      std::move(staging_pools),
      std::move(joined_stream),
      std::move(reservation_pool),
      options,
      pressure.enables_graph_retry_boundary(options.publisher_history_depth),
      capture_clock,
      catalog_control);
}

template <std::size_t Size, typename Bucket>
[[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
create_tensor_schema_payload_subscriber_impl(
    holomq::Context& context, const std::string_view channel_name, const ChannelOptions& options,
    std::shared_ptr<TensorSidecarStore> sidecar_store, std::shared_ptr<void> runtime_lifetime,
    const std::uint32_t tensor_field_count,
    std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts,
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber,
    const bool catalog_control, ReceiveProbeBarrier* receive_probe_barrier,
    const std::uint64_t catalog_clock_count) {
  auto subscriber = context.subscriber_builder(std::string(channel_name))
                        .service_options(to_holomq_service_options(options))
                        .subscriber_options(to_holomq_subscriber_options(options))
                        .template payload_type<Bucket>()
                        .create();
  if (!subscriber) {
    return make_unexpected(from_holomq_error(std::move(subscriber).error(),
                                             "failed to create Tensor schema channel subscriber"));
  }
  std::optional<holomq::Subscriber<Bucket>> terminal_subscriber;
  if (replaces_oldest_unread_sample(options.queue_policy)) {
    const std::string terminal_service_id =
        channel_internal::ordered_terminal_service_id(channel_name);
    if (terminal_service_id.empty()) {
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor schema channel cannot derive a protected terminal service identity"));
    }
    auto created = context.subscriber_builder(terminal_service_id)
                       .service_options(channel_internal::ordered_terminal_service_options(options))
                       .subscriber_options(channel_internal::ordered_terminal_subscriber_options())
                       .template payload_type<Bucket>()
                       .create();
    if (!created) {
      return make_unexpected(
          from_holomq_error(std::move(created).error(),
                            "failed to create protected Tensor schema terminal subscriber"));
    }
    terminal_subscriber.emplace(std::move(*created));
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
  TensorBoundaryContractRecord receive_contract{};
  receive_contract.max_rank = TensorViewMetadata::kMaxRank;
  std::vector<TensorSchemaReceiveControlPtr> receive_controls;
  receive_controls.reserve(tensor_field_count);
  for (std::uint32_t index = 0U; index < tensor_field_count; ++index) {
    auto control = channel_internal::CatalogTensorReceiveControl::create(
        runtime_lifetime, context, receive_contract, static_cast<std::uint32_t>(*receive_capacity));
    if (!control) {
      sidecar_store->unregister_recipient(recipient_index.value(), subscriber->recipient_id());
      return make_unexpected(std::move(control).error());
    }
    receive_controls.emplace_back(control->release());
  }
  return std::make_unique<TensorSchemaPayloadSubscriberImpl<Size, Bucket>>(
      context,
      std::move(subscriber).value(),
      std::move(terminal_subscriber),
      std::move(health_subscriber),
      std::move(sidecar_store),
      recipient_index.value(),
      std::move(receive_controls),
      std::move(field_contracts),
      catalog_control,
      receive_probe_barrier,
      catalog_clock_count);
}
struct TensorSchemaPayloadPublisherFactory {
  holomq::Context& context;
  std::string_view channel_name;
  const ChannelOptions& options;
  std::shared_ptr<TensorSidecarStore> sidecar_store;
  std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts;
  std::vector<std::shared_ptr<TensorSchemaStagingPool>> staging_pools;
  std::shared_ptr<TensorSchemaJoinedReadinessStream> joined_stream;
  std::size_t payload_size;
  std::vector<std::size_t> descriptor_offsets;
  PublisherPressureOptions pressure;
  ClockId capture_clock;
  std::unique_ptr<channel_internal::InlinePayloadPublisher> health_publisher;
  bool catalog_control;

  template <std::size_t Size>
  [[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
  operator()() {
    if (catalog_control) {
      return create_tensor_schema_payload_publisher_impl<Size, TensorSchemaPayloadBucket<Size>>(
          context,
          channel_name,
          options,
          sidecar_store,
          payload_size,
          descriptor_offsets,
          field_contracts,
          staging_pools,
          joined_stream,
          pressure,
          capture_clock,
          std::move(health_publisher),
          true);
    }
    return create_tensor_schema_payload_publisher_impl<Size, InlinePayloadBucket<Size>>(
        context,
        channel_name,
        options,
        sidecar_store,
        payload_size,
        descriptor_offsets,
        field_contracts,
        staging_pools,
        joined_stream,
        pressure,
        capture_clock,
        std::move(health_publisher),
        false);
  }
};

struct TensorSchemaPayloadSubscriberFactory {
  holomq::Context& context;
  std::string_view channel_name;
  const ChannelOptions& options;
  std::shared_ptr<TensorSidecarStore> sidecar_store;
  std::shared_ptr<void> runtime_lifetime;
  std::uint32_t tensor_field_count;
  std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts;
  std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber;
  bool catalog_control;
  ReceiveProbeBarrier* receive_probe_barrier;
  std::uint64_t catalog_clock_count;

  template <std::size_t Size>
  [[nodiscard]] expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
  operator()() {
    if (catalog_control) {
      return create_tensor_schema_payload_subscriber_impl<Size, TensorSchemaPayloadBucket<Size>>(
          context,
          channel_name,
          options,
          sidecar_store,
          runtime_lifetime,
          tensor_field_count,
          field_contracts,
          std::move(health_subscriber),
          true,
          receive_probe_barrier,
          catalog_clock_count);
    }
    return create_tensor_schema_payload_subscriber_impl<Size, InlinePayloadBucket<Size>>(
        context,
        channel_name,
        options,
        sidecar_store,
        runtime_lifetime,
        tensor_field_count,
        field_contracts,
        std::move(health_subscriber),
        false,
        receive_probe_barrier,
        catalog_clock_count);
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
    case channel_internal::kMaximumInlinePayloadBytes:
      return factory.template operator()<channel_internal::kMaximumInlinePayloadBytes>();
    default:
      return make_unexpected(make_channel_error(
          ErrorCode::kInvalidArgument, "unsupported Tensor schema channel payload bucket size"));
  }
}

}  // namespace

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_catalog_tensor_schema_payload_publisher(
    const ChannelEndpointCatalog::EndpointView endpoint, const SchemaIdentity expected_schema,
    const std::size_t payload_size, const std::span<const std::size_t> descriptor_offsets,
    ChannelAttachOptions options) {
  return create_catalog_tensor_schema_payload_publisher(
      endpoint,
      expected_schema,
      payload_size,
      descriptor_offsets,
      std::move(options),
      {},
      nullptr,
      channel_internal::PublisherPressureOptions{},
      TopicDirection::kImport,
      true);
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
ChannelRuntime::create_catalog_tensor_schema_payload_publisher(
    const ChannelEndpointCatalog::EndpointView endpoint, const SchemaIdentity expected_schema,
    const std::size_t payload_size, const std::span<const std::size_t> descriptor_offsets,
    ChannelAttachOptions options,
    const std::span<const channel_internal::TensorSchemaStagingOptions> staging_override,
    cudaStream_t staging_stream, const channel_internal::PublisherPressureOptions& pressure,
    const TopicDirection publisher_direction, const bool enforce_peer_requirement) {
  auto validated = validate_catalog_tensor_schema_endpoint(endpoint,
                                                           expected_schema,
                                                           payload_size,
                                                           descriptor_offsets.size(),
                                                           publisher_direction,
                                                           options);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  const auto& record = **validated;
  if (!descriptor_offsets_unique(descriptor_offsets) ||
      std::any_of(descriptor_offsets.begin(), descriptor_offsets.end(), [&](const auto offset) {
        return offset > payload_size || sizeof(PayloadDescriptor) > payload_size - offset;
      })) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "descriptor Tensor schema offsets are invalid"});
  }

  const std::uint64_t staging_slots =
      record.publisher_request && record.publisher_request->max_loaned_samples
          ? static_cast<std::uint64_t>(*record.publisher_request->max_loaned_samples) +
                record.service_static.max_subscriber_depth +
                record.service_static.max_subscriber_borrowed
          : 0U;
  if (staging_slots > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted,
                                 "descriptor Tensor schema staging capacity is too large"});
  }
  std::vector<channel_internal::TensorSchemaStagingOptions> staging;
  if (!staging_override.empty()) {
    if (staging_override.size() != record.service_static.tensor_fields.size()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "descriptor Tensor schema staging override is incomplete"});
    }
    for (std::size_t index = 0U; index < staging_override.size(); ++index) {
      const auto& override = staging_override[index];
      const auto& tensor = record.service_static.tensor_fields[index].tensor;
      const bool dtype_matches = override.dtype.has_value() == tensor.dtype.has_value() &&
                                 (!override.dtype || dtype_equal(*override.dtype, *tensor.dtype));
      if (override.max_byte_span != tensor.max_byte_span.value_or(0U) || !dtype_matches ||
          override.rank != tensor.rank || override.axes != tensor.axes ||
          override.strides != tensor.strides || override.memory_kind != tensor.memory_kind ||
          override.foreign_emit_policy != tensor.foreign_emit_policy) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument,
                  "descriptor Tensor schema staging override changes an immutable contract field"});
      }
    }
    staging.assign(staging_override.begin(), staging_override.end());
  } else {
    staging.reserve(record.service_static.tensor_fields.size());
    for (const auto& field : record.service_static.tensor_fields) {
      DeviceId device{};
      // Catalog authoring does not yet carry a resolved CUDA device. Select the attaching
      // thread's current device only for fields that actually require a staging pool; a
      // require-exportable field retains its source buffer and must not depend on ambient CUDA.
      if (field.tensor.foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable &&
          *field.tensor.memory_kind != MemoryKind::kHost) {
        int current_device = -1;
        if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0) {
          return make_unexpected(Error{ErrorCode::kNotSupported,
                                       "descriptor Tensor schema staging device is unavailable"});
        }
        device = DeviceId{current_device};
      }
      staging.push_back(channel_internal::TensorSchemaStagingOptions{
          .max_byte_span = *field.tensor.max_byte_span,
          .slot_count = static_cast<std::uint32_t>(staging_slots),
          .dtype = field.tensor.dtype,
          .rank = field.tensor.rank,
          .axes = field.tensor.axes,
          .strides = field.tensor.strides,
          .memory_kind = *field.tensor.memory_kind,
          .device = device,
          .foreign_emit_policy = field.tensor.foreign_emit_policy,
      });
    }
  }

  const std::string binding =
      detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint);
  const ChannelOptions data_options = catalog_tensor_schema_options(record);
  std::unique_ptr<channel_internal::InlinePayloadPublisher> health;
  if (record.clock_health.capability == ClockHealthCapability::kStatusReports) {
    auto created =
        create_inline_payload_publisher(binding + ".clock_health.v1",
                                        channel_internal::kTensorCatalogClockHealthControlBytes,
                                        catalog_tensor_schema_health_options(data_options));
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    health = std::move(*created);
  }
  auto publisher = create_tensor_schema_payload_publisher(
      binding,
      payload_size,
      descriptor_offsets,
      data_options,
      pressure,
      staging,
      staging_stream,
      detail::ChannelEndpointCatalogAccess::capture_clock_id(endpoint),
      std::move(health),
      true);
  if (!publisher) {
    return make_unexpected(std::move(publisher).error());
  }
  if (enforce_peer_requirement) {
    auto attached = wait_for_catalog_tensor_schema_peer<true>(**publisher, record, options);
    if (!attached) {
      return make_unexpected(std::move(attached).error());
    }
  }
  return publisher;
}

expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
ChannelRuntime::create_catalog_tensor_schema_payload_subscriber(
    const ChannelEndpointCatalog::EndpointView endpoint, const SchemaIdentity expected_schema,
    const std::size_t payload_size, const std::uint32_t tensor_field_count,
    ChannelAttachOptions options, const TopicDirection subscriber_direction,
    const bool enforce_peer_requirement) {
  auto validated = validate_catalog_tensor_schema_endpoint(
      endpoint, expected_schema, payload_size, tensor_field_count, subscriber_direction, options);
  if (!validated) {
    return make_unexpected(std::move(validated).error());
  }
  const auto& record = **validated;
  const std::string binding =
      detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint);
  const ChannelOptions data_options = catalog_tensor_schema_options(record);
  std::unique_ptr<channel_internal::InlinePayloadSubscriber> health;
  if (record.clock_health.capability == ClockHealthCapability::kStatusReports) {
    auto created =
        create_inline_payload_subscriber(binding + ".clock_health.v1",
                                         channel_internal::kTensorCatalogClockHealthControlBytes,
                                         catalog_tensor_schema_health_options(data_options));
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    health = std::move(*created);
  }
  std::vector<channel_internal::TensorSchemaStagingOptions> field_contracts;
  field_contracts.reserve(record.service_static.tensor_fields.size());
  for (const auto& field : record.service_static.tensor_fields) {
    field_contracts.push_back(channel_internal::TensorSchemaStagingOptions{
        .max_byte_span = *field.tensor.max_byte_span,
        .dtype = field.tensor.dtype,
        .rank = field.tensor.rank,
        .axes = field.tensor.axes,
        .strides = field.tensor.strides,
        .memory_kind = *field.tensor.memory_kind,
        .foreign_emit_policy = field.tensor.foreign_emit_policy,
    });
  }
  auto subscriber = create_tensor_schema_payload_subscriber(
      binding,
      payload_size,
      tensor_field_count,
      data_options,
      std::move(health),
      true,
      field_contracts,
      detail::ChannelEndpointCatalogAccess::clock_count(endpoint));
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  if (enforce_peer_requirement) {
    auto attached = wait_for_catalog_tensor_schema_peer<false>(**subscriber, record, options);
    if (!attached) {
      return make_unexpected(std::move(attached).error());
    }
  }
  return subscriber;
}

namespace channel_internal {

expected<std::unique_ptr<TensorSchemaPayloadPublisher>, Error>
create_tensor_schema_payload_publisher(
    holomq::Context& context, const std::string_view channel_name, const std::size_t payload_size,
    const std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
    std::shared_ptr<TensorSidecarStore> sidecar_store,
    const std::span<const TensorSchemaStagingOptions> staging, cudaStream_t staging_stream,
    const PublisherPressureOptions pressure, const ClockId capture_clock,
    std::unique_ptr<InlinePayloadPublisher> health_publisher, const bool catalog_control) {
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
    if (field.foreign_emit_policy == InterfaceTensorEmitPolicy::kRequireExportable) {
      staging_pools.push_back(nullptr);
      continue;
    }
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
  if (staging_stream != nullptr && !joined_device) {
    return make_unexpected(
        make_channel_error(ErrorCode::kInvalidArgument,
                           "Tensor schema staging stream requires device-ordered staging storage"));
  }
  if (joined_device && (staging_stream != nullptr || staging.size() > 1U)) {
    auto created = staging_stream != nullptr
                       ? TensorSchemaJoinedReadinessStream::borrow(*joined_device, staging_stream)
                       : TensorSchemaJoinedReadinessStream::create(*joined_device);
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
      {staging.begin(), staging.end()},
      std::move(staging_pools),
      std::move(joined_stream),
      payload_size,
      {descriptor_offsets.begin(), descriptor_offsets.end()},
      pressure,
      capture_clock,
      std::move(health_publisher),
      catalog_control};
  return dispatch_tensor_schema_payload_bucket(payload_size, factory);
}

expected<std::unique_ptr<TensorSchemaPayloadSubscriber>, Error>
create_tensor_schema_payload_subscriber(
    holomq::Context& context, const std::string_view channel_name, const std::size_t payload_size,
    const ChannelOptions& options, std::shared_ptr<TensorSidecarStore> sidecar_store,
    std::shared_ptr<void> runtime_lifetime, const std::uint32_t tensor_field_count,
    std::unique_ptr<InlinePayloadSubscriber> health_subscriber, const bool catalog_control,
    const std::span<const TensorSchemaStagingOptions> field_contracts,
    const std::uint64_t catalog_clock_count) {
  // Consume the source-tree-only attachment on the cold construction path so a failed
  // construction cannot leak a borrowed barrier into a later subscriber.
  auto* const receive_probe_barrier = take_next_tensor_subscriber_receive_probe_barrier_for_test();
  if (!runtime_lifetime || tensor_field_count == 0U ||
      tensor_field_count > channel_internal::kMaximumDescriptorTensorFields ||
      (!field_contracts.empty() && field_contracts.size() != tensor_field_count)) {
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
                                               tensor_field_count,
                                               {field_contracts.begin(), field_contracts.end()},
                                               std::move(health_subscriber),
                                               catalog_control,
                                               receive_probe_barrier,
                                               catalog_clock_count};
  return dispatch_tensor_schema_payload_bucket(payload_size, factory);
}

}  // namespace channel_internal
}  // namespace holoscan
