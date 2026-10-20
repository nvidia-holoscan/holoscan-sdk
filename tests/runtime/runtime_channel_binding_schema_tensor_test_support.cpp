/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/operator_spec_compiler_access.hpp"
#include "runtime/runtime_channel_binding.hpp"

#include "runtime_channel_binding_schema_tensor_test_support.hpp"

namespace holoscan::detail::runtime_channel_binding_schema_tensor_test {

using namespace std::chrono_literals;

void ContractTensorSourceOperator::setup(OperatorSpec& spec) {
  spec.output(output).max_emits_per_compute(8U).produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U},
      .bounds = tensor_bounds(16U),
      .storage = TensorOutputStorage::kRuntimePool,
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  });
}

expected<void, Error> ContractTensorSourceOperator::compute(ExecutionContext&) {
  return {};
}

void DeviceContractTensorSourceOperator::setup(OperatorSpec& spec) {
  spec.output(output).max_emits_per_compute(8U).produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kCudaDevice,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U},
      .bounds = tensor_bounds(16U),
      .storage = TensorOutputStorage::kRuntimePool,
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  });
}

expected<void, Error> DeviceContractTensorSourceOperator::compute(ExecutionContext&) {
  return {};
}

void RequireExportableTensorSourceOperator::setup(OperatorSpec& spec) {
  spec.output(output).max_emits_per_compute(8U).produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U},
      .bounds = tensor_bounds(16U),
      .storage = TensorOutputStorage::kRuntimePool,
  });
}

expected<void, Error> RequireExportableTensorSourceOperator::compute(ExecutionContext&) {
  return {};
}

[[nodiscard]] const PortPlanRecord* planned_operator_port(const ExecutionPlanState& plan,
                                                          PortDirection direction) {
  const auto record =
      std::find_if(plan.ports.begin(), plan.ports.end(), [direction](const auto& candidate) {
        return candidate.owner_kind == PortOwnerKind::kOperator && candidate.direction == direction;
      });
  if (record == plan.ports.end()) {
    ADD_FAILURE() << "missing planned Operator "
                  << (direction == PortDirection::kInput ? "input" : "output") << " port";
    return nullptr;
  }
  return &*record;
}
GeneratedSchemaSessionFixture::GeneratedSchemaSessionFixture() {
  source.setup(source_spec);
  sink.setup(sink_spec);
  second_sink.setup(second_sink_spec);
  const auto output = OperatorSpecCompilerAccess::port(source_spec, 0U);
  const auto input = OperatorSpecCompilerAccess::port(sink_spec, 0U);
  const auto second_input = OperatorSpecCompilerAccess::port(second_sink_spec, 0U);

  auto& plan = base.plan;
  plan.ports[0].payload_identity = output.payload_identity;
  plan.ports[0].serialized_payload_capacity_bytes = output.serialized_payload_capacity_bytes;
  plan.ports[1].payload_identity = input.payload_identity;
  plan.ports[1].serialized_payload_capacity_bytes = input.serialized_payload_capacity_bytes;
  plan.process_local_channel_services[0].payload_identity = output.payload_identity;
  plan.process_local_channel_services[0].payload_size_bytes =
      output.serialized_payload_capacity_bytes + kSchemaEnvelopeHeaderSize;
  plan.payload_edges[0].payload_identity = output.payload_identity;
  plan.generic_output_loans[0].slot_size_bytes = sizeof(schema::StringT);
  plan.generic_output_loans[0].aggregate_byte_capacity =
      plan.generic_output_loans[0].slot_count * sizeof(schema::StringT);

  replay = {
      RuntimeChannelPortReplayRecord{
          .port = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .declaration = const_cast<void*>(output.declaration),
          .type_ops = output.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(input.declaration),
          .type_ops = input.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{3U},
          .owner = VertexId{3U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(second_input.declaration),
          .type_ops = second_input.runtime_type_ops,
      },
  };
}

void GeneratedSchemaSessionFixture::add_blocking_fanout() {
  base.add_blocking_fanout();
}

void GeneratedSchemaSessionFixture::use_cross_process_edge() {
  base.use_cross_process_edge();
  auto& service = base.plan.cross_process_channel_services.front();
  service.payload_identity = base.plan.ports.front().payload_identity;
  service.payload_size_bytes = replay.front().type_ops->wire_capacity;
}
ImageSessionFixture::ImageSessionFixture() {
  source.setup(source_spec);
  sink.setup(sink_spec);
  second_sink.setup(second_sink_spec);
  const auto output = OperatorSpecCompilerAccess::port(source_spec, 0U);
  const auto input = OperatorSpecCompilerAccess::port(sink_spec, 0U);
  const auto second_input = OperatorSpecCompilerAccess::port(second_sink_spec, 0U);

  auto& plan = base.plan;
  plan.ports[0].payload_identity = output.payload_identity;
  plan.ports[0].serialized_payload_capacity_bytes = output.serialized_payload_capacity_bytes;
  plan.ports[1].payload_identity = input.payload_identity;
  plan.ports[1].serialized_payload_capacity_bytes = input.serialized_payload_capacity_bytes;
  plan.process_local_channel_services[0].payload_identity = output.payload_identity;
  plan.process_local_channel_services[0].payload_size_bytes =
      output.serialized_payload_capacity_bytes;
  plan.process_local_channel_services[0].tensor_payload = true;
  plan.process_local_channel_services[0].tensor_pool_id.clear();
  plan.process_local_channel_services[0].tensor_max_byte_span = 0U;
  plan.process_local_channel_services[0].tensor_memory_kind = MemoryKind::kUnknown;
  plan.process_local_channel_services[0].tensor_device = DeviceId{};
  plan.payload_edges[0].payload_identity = output.payload_identity;
  plan.generic_output_loans.clear();

  replay = {
      RuntimeChannelPortReplayRecord{
          .port = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .declaration = const_cast<void*>(output.declaration),
          .type_ops = output.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(input.declaration),
          .type_ops = input.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{3U},
          .owner = VertexId{3U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(second_input.declaration),
          .type_ops = second_input.runtime_type_ops,
      },
  };
}

void ImageSessionFixture::enable_reuse_last() {
  base.enable_reuse_last();
}
void ImageSessionFixture::enable_bracketed_window(std::uint32_t max_selected_samples) {
  base.enable_bracketed_window(max_selected_samples);
}
void ImageSessionFixture::use_direct_retained_delivery() {
  base.use_direct_retained_delivery();
}
void ImageSessionFixture::add_blocking_fanout() {
  base.add_blocking_fanout();
}
void ImageSessionFixture::use_one_slot_blocking_transport() {
  base.use_one_slot_blocking_transport();
}
void ImageSessionFixture::use_cross_process_staging(std::uint64_t max_byte_span,
                                                    MemoryKind memory_kind, DeviceId device) {
  base.use_cross_process_edge();
  const TensorRepresentation layout{
      .memory_kind = memory_kind,
      .dtype = DLDataType{kDLUInt, 64U, 1U},
      .rank = 1U,
  };
  base.plan.ports[0].tensor_layout = layout;
  base.plan.ports[1].tensor_layout = layout;
  auto& service = base.plan.cross_process_channel_services.front();
  service.payload_identity = base.plan.ports.front().payload_identity;
  service.payload_size_bytes = replay.front().type_ops->wire_capacity;
  service.tensor_payload = true;
  service.tensor_max_byte_span = max_byte_span;
  service.tensor_pool_id = "runtime-image-staging." + std::to_string(base.plan.plan_generation);
  service.tensor_memory_kind = memory_kind;
  service.tensor_device = device;
  service.tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
}
TensorPacketSessionFixture::TensorPacketSessionFixture() {
  source.setup(source_spec);
  sink.setup(sink_spec);
  const auto output = OperatorSpecCompilerAccess::port(source_spec, 0U);
  const auto input = OperatorSpecCompilerAccess::port(sink_spec, 0U);

  auto& plan = base.plan;
  plan.ports[0].payload_identity = output.payload_identity;
  plan.ports[0].serialized_payload_capacity_bytes = output.serialized_payload_capacity_bytes;
  plan.ports[1].payload_identity = input.payload_identity;
  plan.ports[1].serialized_payload_capacity_bytes = input.serialized_payload_capacity_bytes;
  plan.process_local_channel_services[0].payload_identity = output.payload_identity;
  plan.process_local_channel_services[0].payload_size_bytes =
      output.serialized_payload_capacity_bytes;
  plan.process_local_channel_services[0].tensor_payload = true;
  plan.process_local_channel_services[0].tensor_pool_id.clear();
  plan.process_local_channel_services[0].tensor_max_byte_span = 0U;
  plan.process_local_channel_services[0].tensor_memory_kind = MemoryKind::kUnknown;
  plan.process_local_channel_services[0].tensor_device = DeviceId{};
  plan.payload_edges[0].payload_identity = output.payload_identity;
  plan.generic_output_loans.clear();

  replay = {
      RuntimeChannelPortReplayRecord{
          .port = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .declaration = const_cast<void*>(output.declaration),
          .type_ops = output.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(input.declaration),
          .type_ops = input.runtime_type_ops,
      },
  };
}

void TensorPacketSessionFixture::use_cross_process_staging(std::uint64_t max_byte_span) {
  base.use_cross_process_edge();
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLUInt, 64U, 1U},
      .rank = 1U,
  };
  base.plan.ports[0].tensor_layout = layout;
  base.plan.ports[1].tensor_layout = layout;
  auto& service = base.plan.cross_process_channel_services.front();
  service.payload_identity = base.plan.ports.front().payload_identity;
  service.payload_size_bytes = replay.front().type_ops->wire_capacity;
  service.tensor_payload = true;
  service.tensor_max_byte_span = max_byte_span;
  service.tensor_pool_id =
      "runtime-tensor-packet-staging." + std::to_string(base.plan.plan_generation);
  service.tensor_memory_kind = MemoryKind::kHost;
  service.tensor_device = DeviceId{};
  service.tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
}
TensorSessionFixture::TensorSessionFixture() {
  source.setup(source_spec);
  sink.setup(sink_spec);
  second_sink.setup(second_sink_spec);
  const auto output = OperatorSpecCompilerAccess::port(source_spec, 0U);
  const auto input = OperatorSpecCompilerAccess::port(sink_spec, 0U);
  const auto second_input = OperatorSpecCompilerAccess::port(second_sink_spec, 0U);

  auto& plan = base.plan;
  plan.ports[0].payload_identity = output.payload_identity;
  plan.ports[0].serialized_payload_capacity_bytes = 0U;
  plan.ports[1].payload_identity = input.payload_identity;
  plan.ports[1].serialized_payload_capacity_bytes = 0U;
  plan.process_local_channel_services[0].payload_identity = output.payload_identity;
  plan.process_local_channel_services[0].payload_size_bytes = 0U;
  plan.process_local_channel_services[0].tensor_payload = true;
  plan.process_local_channel_services[0].tensor_pool_id.clear();
  plan.process_local_channel_services[0].tensor_max_byte_span = 0U;
  plan.process_local_channel_services[0].tensor_memory_kind = MemoryKind::kUnknown;
  plan.process_local_channel_services[0].tensor_device = DeviceId{};
  plan.payload_edges[0].payload_identity = output.payload_identity;
  plan.generic_output_loans.clear();

  replay = {
      RuntimeChannelPortReplayRecord{
          .port = PortId{1U},
          .owner = VertexId{1U},
          .direction = PortDirection::kOutput,
          .declaration = const_cast<void*>(output.declaration),
          .type_ops = output.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{2U},
          .owner = VertexId{2U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(input.declaration),
          .type_ops = input.runtime_type_ops,
      },
      RuntimeChannelPortReplayRecord{
          .port = PortId{3U},
          .owner = VertexId{3U},
          .direction = PortDirection::kInput,
          .declaration = const_cast<void*>(second_input.declaration),
          .type_ops = second_input.runtime_type_ops,
      },
  };
}

void TensorSessionFixture::enable_latest_within() {
  base.enable_latest_within();
}
void TensorSessionFixture::enable_reuse_last() {
  base.enable_reuse_last();
}
void TensorSessionFixture::enable_drain_all(std::uint32_t max_selected_samples) {
  base.enable_drain_all(max_selected_samples);
}
void TensorSessionFixture::enable_bracketed_window(std::uint32_t max_selected_samples) {
  base.enable_bracketed_window(max_selected_samples);
}
void TensorSessionFixture::use_direct_retained_delivery() {
  base.use_direct_retained_delivery();
}
void TensorSessionFixture::add_blocking_fanout() {
  base.add_blocking_fanout();
}
void TensorSessionFixture::use_one_slot_blocking_fanout() {
  base.use_one_slot_blocking_fanout();
}
void TensorSessionFixture::use_one_slot_blocking_transport() {
  base.use_one_slot_blocking_transport();
}
void TensorSessionFixture::use_one_slot_drop_oldest_queue() {
  base.use_one_slot_drop_oldest_queue();
}
void TensorSessionFixture::use_direct_drop_oldest_fanout() {
  base.use_direct_drop_oldest_fanout();
}

void TensorSessionFixture::use_cross_process_edge() {
  base.use_cross_process_edge();
  auto& service = base.plan.cross_process_channel_services.front();
  service.payload_size_bytes = 0U;
  service.tensor_payload = true;
  service.tensor_max_byte_span = 0U;
  service.tensor_pool_id.clear();
  service.tensor_memory_kind = MemoryKind::kUnknown;
  service.tensor_device = DeviceId{};
}
[[nodiscard]] expected<OwnedTensorSample, Error> make_owned_tensor_sample(std::uint64_t value) {
  auto releases = std::make_shared<std::atomic<std::uint32_t>>(0U);
  auto* storage = new (std::nothrow) std::uint64_t{value};
  if (storage == nullptr) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const std::array<std::int64_t, 1U> shape{1};
  auto tensor = make_tensor(shape,
                            DLDataType{kDLUInt, 64U, 1U},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage,
                            [releases](void* pointer) -> expected<void, Error> {
                              delete static_cast<std::uint64_t*>(pointer);
                              releases->fetch_add(1U, std::memory_order_relaxed);
                              return {};
                            });
  if (!tensor) {
    delete storage;
    return make_unexpected(std::move(tensor).error());
  }
  return OwnedTensorSample{.tensor = std::move(*tensor), .data = storage, .releases = releases};
}

[[nodiscard]] expected<Tensor, Error> make_contract_test_tensor(
    const std::span<const std::int64_t> shape, const DLDataType dtype,
    const std::span<const std::int64_t> byte_strides, const MemoryStorageType storage_type) {
  constexpr std::size_t kStorageBytes = 128U;
  auto* storage = new (std::nothrow) std::byte[kStorageBytes];
  if (storage == nullptr) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const TensorByteStridesView strides =
      byte_strides.empty() ? TensorByteStridesView{std::nullopt}
                           : TensorByteStridesView{TensorByteStridesSpan{byte_strides}};
  auto tensor = make_tensor(
      shape, dtype, strides, storage_type, storage, [](void* pointer) -> expected<void, Error> {
        delete[] static_cast<std::byte*>(pointer);
        return {};
      });
  if (!tensor) {
    delete[] storage;
    return make_unexpected(std::move(tensor).error());
  }
  return tensor;
}
[[nodiscard]] expected<PublishedImageScript, Error> publish_image_script(
    RuntimeChannelSession& session, PortBinding& binding, ImageSourceOperator& source,
    std::span<const std::uint64_t> values, std::span<const std::int64_t> capture_times) {
  if (values.empty() || values.size() != capture_times.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto prepared = session.prepare_activation(VertexId{1U}, 1U, {});
  if (!prepared) {
    return make_unexpected(std::move(prepared).error());
  }

  PublishedImageScript result;
  result.data.reserve(values.size());
  result.releases.reserve(values.size());
  for (std::size_t index = 0U; index < values.size(); ++index) {
    auto owned = make_owned_tensor_sample(values[index]);
    if (!owned) {
      return make_unexpected(std::move(owned).error());
    }

    schema::ImageT image{};
    image.height = 1;
    image.width = 1;
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.header = std::make_shared<schema::HeaderT>();
    image.header->stamp = std::make_shared<schema::TimeT>();
    image.header->stamp->time_ns = 123'456;
    image.header->stamp->acqtime_ns = 123'450;
    image.header->frame_id = "runtime_image";
    image.data = std::make_shared<Tensor>(std::move(owned->tensor));
    result.data.push_back(owned->data);
    result.releases.push_back(owned->releases);
    const SampleMetadata metadata{
        .capture_timestamp_ns = capture_times[index],
        .source_clock_id = ClockId{1U},
        .trace_id = 0xABCDU,
        .frame_id = 73U,
    };
    auto emitted = source.output.emit(Sample<schema::ImageT>{std::move(image), metadata});
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
  }
  auto retired = session.retire_activation(VertexId{1U}, 1U, true);
  if (!retired) {
    return make_unexpected(std::move(retired).error());
  }

  result.events.reserve(values.size());
  result.records.reserve(values.size());
  for (std::size_t index = 0U; index < values.size(); ++index) {
    auto event = wait_for_event(session, binding);
    if (!event) {
      return make_unexpected(std::move(event).error());
    }
    auto record = session.temporal_sample(event->port, event->source_sequence);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    result.events.push_back(*event);
    result.records.push_back(*record);
  }
  return result;
}

[[nodiscard]] expected<PublishedImageSample, Error> publish_image_sample(
    RuntimeChannelSession& session, PortBinding& binding, ImageSourceOperator& source,
    std::uint64_t value) {
  const std::array values{value};
  constexpr std::array<std::int64_t, 1U> capture_times{10};
  auto script = publish_image_script(session, binding, source, values, capture_times);
  if (!script) {
    return make_unexpected(std::move(script).error());
  }
  return PublishedImageSample{
      .event = script->events.front(),
      .record = script->records.front(),
      .data = script->data.front(),
      .releases = std::move(script->releases.front()),
  };
}

[[nodiscard]] expected<PublishedImageTensor, Error> publish_image_tensor(
    RuntimeChannelSession& session, PortBinding& binding, ImageSourceOperator& source,
    Tensor tensor) {
  schema::ImageT image{};
  image.height = 1;
  image.width = 1;
  image.encoding = schema::ImageEncoding_CUSTOM;
  image.header = std::make_shared<schema::HeaderT>();
  image.header->stamp = std::make_shared<schema::TimeT>();
  image.header->stamp->time_ns = 123'456;
  image.header->stamp->acqtime_ns = 123'450;
  image.header->frame_id = "runtime_image";
  image.data = std::make_shared<Tensor>(std::move(tensor));

  if (auto prepared = session.prepare_activation(VertexId{1U}, 1U, {}); !prepared) {
    return make_unexpected(std::move(prepared).error());
  }
  const SampleMetadata metadata{
      .capture_timestamp_ns = 10,
      .source_clock_id = ClockId{1U},
      .trace_id = 0xABCDU,
      .frame_id = 73U,
  };
  if (auto emitted = source.output.emit(Sample<schema::ImageT>{std::move(image), metadata});
      !emitted) {
    return make_unexpected(std::move(emitted).error());
  }
  if (auto retired = session.retire_activation(VertexId{1U}, 1U, true); !retired) {
    return make_unexpected(std::move(retired).error());
  }
  auto event = wait_for_event(session, binding);
  if (!event) {
    return make_unexpected(std::move(event).error());
  }
  auto record = session.temporal_sample(event->port, event->source_sequence);
  if (!record) {
    return make_unexpected(std::move(record).error());
  }
  return PublishedImageTensor{.event = *event, .record = *record};
}

[[nodiscard]] expected<PublishedImageTensor, Error> publish_tensor_packet_tensor(
    RuntimeChannelSession& session, PortBinding& binding, TensorPacketSourceOperator& source,
    Tensor tensor) {
  schema::TensorPacketT packet{};
  packet.sequence = 99U;
  packet.source = "external-sensor";
  packet.scale = 1.25;
  packet.data = std::make_shared<Tensor>(std::move(tensor));

  if (auto prepared = session.prepare_activation(VertexId{1U}, 1U, {}); !prepared) {
    return make_unexpected(std::move(prepared).error());
  }
  const SampleMetadata metadata{
      .capture_timestamp_ns = 20,
      .source_clock_id = ClockId{1U},
      .trace_id = 0xFACEU,
      .frame_id = 74U,
  };
  if (auto emitted = source.output.emit(Sample<schema::TensorPacketT>{std::move(packet), metadata});
      !emitted) {
    return make_unexpected(std::move(emitted).error());
  }
  if (auto retired = session.retire_activation(VertexId{1U}, 1U, true); !retired) {
    return make_unexpected(std::move(retired).error());
  }
  auto event = wait_for_event(session, binding);
  if (!event) {
    return make_unexpected(std::move(event).error());
  }
  auto record = session.temporal_sample(event->port, event->source_sequence);
  if (!record) {
    return make_unexpected(std::move(record).error());
  }
  return PublishedImageTensor{.event = *event, .record = *record};
}
StreamTrackingTensorStorage::StreamTrackingTensorStorage(std::shared_ptr<StreamTrackingState> state)
    : state_(std::move(state)) {}

StreamTrackingTensorStorage::~StreamTrackingTensorStorage() {
  state_->destroyed.fetch_add(1U, std::memory_order_relaxed);
}

void StreamTrackingTensorStorage::retain_tensor_handle() noexcept {
  state_->live_handles.fetch_add(1U, std::memory_order_relaxed);
}

void StreamTrackingTensorStorage::release_tensor_handle() noexcept {
  state_->live_handles.fetch_sub(1U, std::memory_order_relaxed);
}

bool StreamTrackingTensorStorage::has_readiness_stream_contract() const noexcept {
  return true;
}

expected<void, Error> StreamTrackingTensorStorage::synchronize_for_consumer(cudaStream_t stream) {
  state_->last_consumer_stream = stream;
  state_->consumer_syncs.fetch_add(1U, std::memory_order_relaxed);
  return {};
}

bool StreamTrackingTensorStorage::set_deallocation_stream(cudaStream_t stream) {
  state_->last_deallocation_stream = stream;
  state_->deallocation_streams.fetch_add(1U, std::memory_order_relaxed);
  return true;
}

HostCopyTrackingTensorStorage::HostCopyTrackingTensorStorage(
    std::shared_ptr<HostCopyTrackingState> state)
    : state_(std::move(state)) {}

expected<void, Error> HostCopyTrackingTensorStorage::synchronize_for_host() {
  state_->host_synchronizations.fetch_add(1U, std::memory_order_relaxed);
  return {};
}

void* HostCopyTrackingTensorStorage::data() const noexcept {
  return state_->data.data();
}

[[nodiscard]] HostCopyTrackingTensor make_host_copy_tracking_tensor(const std::size_t byte_offset) {
  auto state = std::make_shared<HostCopyTrackingState>();
  if (byte_offset >= state->data.size()) {
    return {};
  }
  for (std::size_t index = 0U; index < state->data.size(); ++index) {
    state->data[index] = static_cast<std::byte>(index + 1U);
  }
  auto storage = std::make_shared<HostCopyTrackingTensorStorage>(state);
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = storage;
  context->dl_shape = {static_cast<std::int64_t>(state->data.size() - byte_offset)};
  context->tensor.dl_tensor.data = storage->data();
  context->tensor.dl_tensor.device = DLDevice{kDLCPU, 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 8U, 1U};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = byte_offset;
  return HostCopyTrackingTensor{.tensor = Tensor{std::move(context)}, .state = std::move(state)};
}

[[nodiscard]] expected<Sample<Tensor>, Error> receive_catalog_tensor(
    ChannelSubscriber<Tensor>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto sample = subscriber.receive_sample();
    if (sample || sample.error().code != ErrorCode::kNotReady) {
      return sample;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}
[[nodiscard]] expected<void, Error> wait_for_catalog_input(
    RuntimeChannelSession& session, const PortId input, const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = RuntimeChannelSessionTestAccess::receive_one(session, input);
    if (received || received.error().code != ErrorCode::kNotReady) {
      return received;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<void, Error> publish_catalog_host_tensor(ChannelPublisher<Tensor>& publisher,
                                                                const std::byte value) {
  constexpr std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher.loan_tensor(
      TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write_host();
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), value);
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  auto emitted = publisher.emit(std::move(*loan));
  return emitted ? expected<void, Error>{}
                 : expected<void, Error>{make_unexpected(std::move(emitted).error())};
}
[[nodiscard]] StreamTrackingTensor make_stream_tracking_tensor() {
  auto state = std::make_shared<StreamTrackingState>();
  auto storage = std::make_shared<StreamTrackingTensorStorage>(state);
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = storage;
  context->dl_shape = {1};
  context->tensor.dl_tensor.data = storage->data.data();
  context->tensor.dl_tensor.device = DLDevice{kDLCUDA, 0};
  context->tensor.dl_tensor.ndim = 1;
  context->tensor.dl_tensor.dtype = DLDataType{kDLUInt, 64U, 1U};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0U;
  const void* const data = storage->data.data();
  return StreamTrackingTensor{
      .tensor = Tensor{std::move(context)}, .data = data, .state = std::move(state)};
}

[[nodiscard]] expected<PublishedTensorScript, Error> publish_tensor_script(
    RuntimeChannelSession& session, PortBinding& binding, TensorSourceOperator& source,
    std::span<const std::uint64_t> values, std::span<const std::int64_t> capture_times) {
  if (values.empty() || values.size() != capture_times.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto prepared = session.prepare_activation(VertexId{1U}, 1U, {});
  if (!prepared) {
    return make_unexpected(std::move(prepared).error());
  }

  PublishedTensorScript result;
  result.data.reserve(values.size());
  result.releases.reserve(values.size());
  for (std::size_t index = 0U; index < values.size(); ++index) {
    auto owned = make_owned_tensor_sample(values[index]);
    if (!owned) {
      return make_unexpected(std::move(owned).error());
    }
    result.data.push_back(owned->data);
    result.releases.push_back(owned->releases);
    const SampleMetadata metadata{
        .capture_timestamp_ns = capture_times[index],
        .source_clock_id = ClockId{1U},
    };
    auto emitted = source.output.emit(Sample<Tensor>{std::move(owned->tensor), metadata});
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
  }
  auto retired = session.retire_activation(VertexId{1U}, 1U, true);
  if (!retired) {
    return make_unexpected(std::move(retired).error());
  }

  result.events.reserve(values.size());
  result.records.reserve(values.size());
  for (std::size_t index = 0U; index < values.size(); ++index) {
    auto event = wait_for_event(session, binding);
    if (!event) {
      return make_unexpected(std::move(event).error());
    }
    auto record = session.temporal_sample(event->port, event->source_sequence);
    if (!record) {
      return make_unexpected(std::move(record).error());
    }
    result.events.push_back(*event);
    result.records.push_back(*record);
  }
  return result;
}

}  // namespace holoscan::detail::runtime_channel_binding_schema_tensor_test
