/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <vector>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>

#include "../compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// Describe the allocation and layouts admitted for one direct descriptor-Tensor edge.
struct InlineDescriptorTensorContract {
  PortId source{};
  PortId target{};
  std::uint64_t max_byte_span{};
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
  std::optional<TensorRepresentation> output_layout;
  std::optional<TensorRepresentation> input_layout;
};

/// Pair the catalog and plan clock identities validated for one endpoint.
struct CatalogCaptureClockBinding {
  ClockId catalog{};
  ClockId plan{};
};

[[nodiscard]] bool unique_valid_vertices(std::span<const VertexId> vertices) noexcept;

[[nodiscard]] expected<void, Error> validate_import_generation_capacities(
    std::span<const ImportedGenerationCapacityPlanRecord> records);

[[nodiscard]] expected<ChannelOptions, Error> process_local_service_options(
    const ProcessLocalChannelServicePlanRecord& service) noexcept;

[[nodiscard]] expected<ChannelOptions, Error> external_service_options(
    const ChannelServiceStaticPlanRecord& service,
    const ChannelPublisherEndpointPlanRecord& publisher,
    const ChannelSubscriberEndpointPlanRecord& subscriber) noexcept;

[[nodiscard]] expected<std::vector<TensorFieldServiceStorageContract>, Error>
external_descriptor_tensor_storage_contracts(
    const ExecutionPlanState& plan, ExternalTopicId topic, PortId graph_port,
    std::span<const DescriptorTensorTopicFieldRecord> logical_fields);

[[nodiscard]] bool singular_descriptor_storage_contract(
    const ProcessLocalChannelServicePlanRecord& service) noexcept;

[[nodiscard]] expected<std::optional<InlineDescriptorTensorContract>, Error>
find_inline_descriptor_tensor_contract(
    const ExecutionPlanState& plan, PortId source, std::optional<PortId> target,
    const ProcessLocalChannelServicePlanRecord& service) noexcept;

[[nodiscard]] bool runtime_codec_matches_plan(const RuntimePortTypeOps* ops,
                                              const PortPlanRecord& port,
                                              std::uint64_t service_payload_capacity,
                                              bool process_local_schema_envelope) noexcept;

/// @param projected True when this endpoint narrows the producer's schema sample to one Tensor
/// member, which pairs a Tensor port with the producer's descriptor-schema service.
[[nodiscard]] bool runtime_tensor_matches_plan(const RuntimePortTypeOps* ops,
                                               const PortPlanRecord& port,
                                               const ProcessLocalChannelServicePlanRecord& service,
                                               bool projected = false) noexcept;

[[nodiscard]] bool runtime_external_tensor_matches_plan(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept;

[[nodiscard]] bool runtime_external_tensor_uses_unsupported_strides(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept;

[[nodiscard]] const EndpointClockHealthPlanRecord* find_endpoint_clock_health(
    const ExecutionPlanState& plan, PortId endpoint) noexcept;

[[nodiscard]] expected<ChannelEndpointCatalog::EndpointView, Error>
find_validated_catalog_endpoint_by_boundary_port(const ExecutionPlanState& plan,
                                                 PortId boundary) noexcept;

[[nodiscard]] expected<CatalogCaptureClockBinding, Error> bind_catalog_capture_clock(
    const ExecutionPlanState& plan, ChannelEndpointCatalog::EndpointView endpoint,
    ClockId plan_capture_clock) noexcept;

[[nodiscard]] expected<PortId, Error> external_internal_port(const ExecutionPlanState& plan,
                                                             PortId boundary,
                                                             bool imported) noexcept;

}  // namespace holoscan::detail
