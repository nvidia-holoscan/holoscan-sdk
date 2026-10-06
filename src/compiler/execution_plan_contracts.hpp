/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "execution_plan_state.hpp"

namespace holoscan::detail {

/// Return the policy-defined semantic rank for one closed-set execution profile.
///
/// The result deliberately does not depend on the enum's underlying value. Unknown values fail
/// closed so callers cannot accidentally admit or combine a profile outside the active policy.
[[nodiscard]] constexpr std::optional<std::uint8_t> profile_semantic_rank(
    Profile profile) noexcept {
  switch (profile) {
    case Profile::kResearch:
      return 0U;
    case Profile::kStandard:
      return 1U;
    case Profile::kRobotics:
      return 2U;
    case Profile::kRealTime:
      return 3U;
    case Profile::kRegulated:
      return 4U;
  }
  return std::nullopt;
}

/// Non-owning projection of facts common to all private Tensor Channel service families.
struct TensorServiceContractView {
  const PayloadIdentityRecord& payload_identity;
  std::uint64_t payload_size_bytes{};
  std::uint64_t selected_inline_bucket_bytes{};
  bool tensor_payload{};
  std::uint64_t tensor_max_byte_span{};
  std::string_view tensor_pool_id;
  MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
  DeviceId tensor_device{};
  std::span<const TensorFieldServiceStorageContract> tensor_fields;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};
  std::uint32_t max_subscribers{};
  std::uint32_t max_subscriber_queue_depth{};
  std::uint32_t max_subscriber_borrowed_samples{};
  std::uint32_t publisher_history_depth{};
  bool event_wake{};
};

/// Non-owning projection of one scalar or field-specific Tensor storage obligation.
struct TensorServiceStorageView {
  std::uint64_t max_byte_span{};
  std::string_view pool_id;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
};

/// Resolved publisher and subscriber bounds for one external topic.
struct ExternalTopicBoundsResolution {
  std::uint32_t publisher_max_loaned_samples{};
  std::uint32_t subscriber_queue_depth{};
  std::uint32_t subscriber_max_borrowed_samples{};
};

/// Return one provider's validated contiguous sizing-dependency range.
///
/// Provider dependency identifiers are one-based indexes into the canonical
/// plan collection. The returned span borrows from \p dependencies and is
/// unavailable when the provider range is malformed.
[[nodiscard]] inline std::optional<std::span<const PoolCapacityDependencyPlanRecord>>
provider_sizing_dependencies(std::span<const PoolCapacityDependencyPlanRecord> dependencies,
                             const ProviderAllocationPlanRecord& provider) noexcept {
  if (!provider.first_sizing_dependency || provider.sizing_dependency_count == 0U) {
    return std::nullopt;
  }
  const std::size_t first = static_cast<std::size_t>(provider.first_sizing_dependency.value - 1U);
  const std::size_t count = static_cast<std::size_t>(provider.sizing_dependency_count);
  if (first > dependencies.size() || count > dependencies.size() - first) {
    return std::nullopt;
  }
  const auto result = dependencies.subspan(first, count);
  for (std::size_t offset = 0U; offset < result.size(); ++offset) {
    const auto expected_id = static_cast<std::uint64_t>(first) + offset + 1U;
    if (result[offset].id.value != expected_id || result[offset].pool_id != provider.pool_id) {
      return std::nullopt;
    }
  }
  return result;
}

/// Resolve authored endpoint bounds over already-derived Graph edge fallbacks.
[[nodiscard]] inline ExternalTopicBoundsResolution resolve_external_topic_bounds(
    const ExternalTopicRecord& topic, std::uint32_t graph_queue_depth,
    std::uint32_t graph_max_emits_per_compute) noexcept {
  const std::uint32_t publisher_max_loaned_samples =
      topic.imported && topic.bounds.publisher && topic.bounds.publisher->max_loaned_samples
          ? *topic.bounds.publisher->max_loaned_samples
          : graph_max_emits_per_compute;
  const std::uint32_t subscriber_queue_depth =
      !topic.imported && topic.bounds.subscriber && topic.bounds.subscriber->queue_depth
          ? *topic.bounds.subscriber->queue_depth
          : graph_queue_depth;
  const std::uint32_t subscriber_max_borrowed_samples =
      !topic.imported && topic.bounds.subscriber && topic.bounds.subscriber->max_borrowed_samples
          ? *topic.bounds.subscriber->max_borrowed_samples
          : subscriber_queue_depth;
  return ExternalTopicBoundsResolution{
      .publisher_max_loaned_samples = publisher_max_loaned_samples,
      .subscriber_queue_depth = subscriber_queue_depth,
      .subscriber_max_borrowed_samples = subscriber_max_borrowed_samples,
  };
}

/// Project the common facts from one owning Tensor service contract.
[[nodiscard]] inline TensorServiceContractView tensor_service_contract_view(
    const TensorServiceContract& service) noexcept {
  return TensorServiceContractView{
      .payload_identity = service.payload_identity,
      .payload_size_bytes = service.payload_size_bytes,
      .selected_inline_bucket_bytes = service.selected_inline_bucket_bytes,
      .tensor_payload = service.tensor_payload,
      .tensor_max_byte_span = service.tensor_max_byte_span,
      .tensor_pool_id = service.tensor_pool_id,
      .tensor_memory_kind = service.tensor_memory_kind,
      .tensor_device = service.tensor_device,
      .tensor_fields = service.tensor_fields,
      .queue_policy = service.queue_policy,
      .max_subscriber_queue_depth = service.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples,
      .publisher_history_depth = service.publisher_history_depth,
      .event_wake = service.event_wake,
  };
}

/// Project the common facts from one process-local Tensor Channel service.
[[nodiscard]] inline TensorServiceContractView tensor_service_contract_view(
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  return TensorServiceContractView{
      .payload_identity = service.payload_identity,
      .payload_size_bytes = service.payload_size_bytes,
      .selected_inline_bucket_bytes = service.selected_inline_bucket_bytes,
      .tensor_payload = service.tensor_payload,
      .tensor_max_byte_span = service.tensor_max_byte_span,
      .tensor_pool_id = service.tensor_pool_id,
      .tensor_memory_kind = service.tensor_memory_kind,
      .tensor_device = service.tensor_device,
      .tensor_fields = service.tensor_fields,
      .queue_policy = service.queue_policy,
      .max_subscribers = service.max_subscribers,
      .max_subscriber_queue_depth = service.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples,
      .publisher_history_depth = service.publisher_history_depth,
      .event_wake = service.event_wake,
  };
}

/// Project the common facts from one cross-process Tensor Channel service.
[[nodiscard]] inline TensorServiceContractView tensor_service_contract_view(
    const CrossProcessChannelServicePlanRecord& service) noexcept {
  return TensorServiceContractView{
      .payload_identity = service.payload_identity,
      .payload_size_bytes = service.payload_size_bytes,
      .selected_inline_bucket_bytes = service.selected_inline_bucket_bytes,
      .tensor_payload = service.tensor_payload,
      .tensor_max_byte_span = service.tensor_max_byte_span,
      .tensor_pool_id = service.tensor_pool_id,
      .tensor_memory_kind = service.tensor_memory_kind,
      .tensor_device = service.tensor_device,
      .tensor_fields = service.tensor_fields,
      .queue_policy = service.queue_policy,
      .max_subscribers = service.max_subscribers,
      .max_subscriber_queue_depth = service.max_subscriber_queue_depth,
      .max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples,
      .publisher_history_depth = service.publisher_history_depth,
      .event_wake = service.event_wake,
  };
}

/// Visit each effective scalar or field-specific Tensor storage obligation.
template <typename Function>
inline void for_each_tensor_service_storage(TensorServiceContractView service, Function function) {
  if (!service.tensor_payload) {
    return;
  }
  if (service.tensor_fields.empty()) {
    function(TensorServiceStorageView{
        .max_byte_span = service.tensor_max_byte_span,
        .pool_id = service.tensor_pool_id,
        .memory_kind = service.tensor_memory_kind,
        .device = service.tensor_device,
    });
    return;
  }
  for (const auto& field : service.tensor_fields) {
    function(TensorServiceStorageView{
        .max_byte_span = field.max_byte_span,
        .pool_id = field.pool_id,
        .memory_kind = field.memory_kind,
        .device = field.device,
    });
  }
}

/// Return whether two services have identical payload and effective storage facts.
[[nodiscard]] inline bool tensor_service_payload_storage_exactly_equal(
    TensorServiceContractView left, TensorServiceContractView right) noexcept {
  return payload_identity_exactly_equal(left.payload_identity, right.payload_identity) &&
         left.payload_size_bytes == right.payload_size_bytes &&
         left.selected_inline_bucket_bytes == right.selected_inline_bucket_bytes &&
         left.tensor_payload == right.tensor_payload &&
         left.tensor_max_byte_span == right.tensor_max_byte_span &&
         left.tensor_pool_id == right.tensor_pool_id &&
         left.tensor_memory_kind == right.tensor_memory_kind &&
         left.tensor_device == right.tensor_device &&
         left.tensor_fields.size() == right.tensor_fields.size() &&
         std::equal(
             left.tensor_fields.begin(), left.tensor_fields.end(), right.tensor_fields.begin());
}

/// Return whether a common service view carries a descriptor-backed Tensor schema.
[[nodiscard]] inline bool descriptor_tensor_schema_service(
    TensorServiceContractView service) noexcept {
  return service.tensor_payload && service.payload_identity.kind == PayloadIdentityKind::kSchema &&
         service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
}

/// Return whether a private service record carries a descriptor-backed Tensor schema.
template <typename Service>
[[nodiscard]] inline bool descriptor_tensor_schema_service(const Service& service) noexcept {
  return descriptor_tensor_schema_service(tensor_service_contract_view(service));
}

/// Return whether service transfer ordering requires a compiler-owned CUDA stream.
[[nodiscard]] inline bool tensor_service_storage_requires_stream(TensorServiceContractView service,
                                                                 MemoryKind memory_kind) noexcept {
  return memory_kind == MemoryKind::kCudaDevice ||
         (!descriptor_tensor_schema_service(service) && memory_kind == MemoryKind::kPinnedHost);
}

/// Return whether private service storage requires a compiler-owned CUDA transfer stream.
template <typename Service>
[[nodiscard]] inline bool tensor_service_storage_requires_stream(const Service& service,
                                                                 MemoryKind memory_kind) noexcept {
  return tensor_service_storage_requires_stream(tensor_service_contract_view(service), memory_kind);
}

/// Return whether Operator access to the service storage requires a planned CUDA stream.
[[nodiscard]] inline bool tensor_service_storage_requires_operator_cuda_stream(
    TensorServiceContractView service, MemoryKind memory_kind) noexcept {
  // Keep Operator access independent from service transfer ordering. These predicates agree for
  // today's pinned-host and device services, but they answer different ownership questions and
  // must not silently narrow together when a transport contract changes.
  switch (memory_kind) {
    case MemoryKind::kPinnedHost:
      return !descriptor_tensor_schema_service(service);
    case MemoryKind::kCudaDevice:
    case MemoryKind::kCudaManaged:
      return true;
    case MemoryKind::kHost:
    case MemoryKind::kUnknown:
    default:
      return false;
  }
}

/// Return whether Operator access to private service storage requires a planned CUDA stream.
template <typename Service>
[[nodiscard]] inline bool tensor_service_storage_requires_operator_cuda_stream(
    const Service& service, MemoryKind memory_kind) noexcept {
  return tensor_service_storage_requires_operator_cuda_stream(tensor_service_contract_view(service),
                                                              memory_kind);
}

/// Return whether a process-local service needs a transfer stream on one exact device.
[[nodiscard]] inline bool process_local_tensor_service_requires_stream_for_device(
    const ProcessLocalChannelServicePlanRecord& service, DeviceId device) noexcept {
  const auto view = tensor_service_contract_view(service);
  if (!view.tensor_payload || !device.valid()) {
    return false;
  }
  bool required = false;
  for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
    required = required || (tensor_service_storage_requires_stream(view, storage.memory_kind) &&
                            storage.device == device);
  });
  return required;
}

/// Return whether a cross-process service needs any transfer stream in one process group.
[[nodiscard]] inline bool cross_process_tensor_service_requires_stream_in_group(
    const CrossProcessChannelServicePlanRecord& service, ProcessGroupId process_group) noexcept {
  const auto view = tensor_service_contract_view(service);
  const bool target_group = std::find(service.target_process_groups.begin(),
                                      service.target_process_groups.end(),
                                      process_group) != service.target_process_groups.end();
  const bool participating_group = target_group || (descriptor_tensor_schema_service(view) &&
                                                    service.source_process_group == process_group);
  if (!view.tensor_payload || !participating_group) {
    return false;
  }
  bool required = false;
  for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
    required = required || (tensor_service_storage_requires_stream(view, storage.memory_kind) &&
                            storage.device.valid());
  });
  return required;
}

/// Return whether a cross-process service needs a transfer stream on one group and device.
[[nodiscard]] inline bool cross_process_tensor_service_requires_stream_for_device_in_group(
    const CrossProcessChannelServicePlanRecord& service, ProcessGroupId process_group,
    DeviceId device) noexcept {
  if (!device.valid() ||
      !cross_process_tensor_service_requires_stream_in_group(service, process_group)) {
    return false;
  }
  const auto view = tensor_service_contract_view(service);
  bool required = false;
  for_each_tensor_service_storage(view, [&](TensorServiceStorageView storage) {
    required = required || (tensor_service_storage_requires_stream(view, storage.memory_kind) &&
                            storage.device == device);
  });
  return required;
}

/// Derive the bounded number of distinct release streams retained by one output-pool slot.
///
/// CUDA backing can be forwarded by arbitrary Operator callbacks, so immediate fanout is not a
/// sound lifetime bound. For CUDA-addressable storage, reserve the producer stream plus one
/// runtime and one application-selected stream for every other CUDA-active Operator in the
/// allocator's final process/device binding domain, after any Green Context replacement. Also
/// reserve every compatible descriptor-staging stream in that domain. Host-only storage retains
/// the direct-service bound because it cannot accumulate CUDA release streams through callback
/// forwarding.
[[nodiscard]] inline std::optional<std::uint32_t> tensor_output_release_stream_capacity(
    std::span<const ProcessLocalChannelServicePlanRecord> process_local_services,
    std::span<const CrossProcessChannelServicePlanRecord> cross_process_services,
    std::span<const OperatorCudaStreamBindingPlanRecord> final_operator_bindings,
    VertexId allocating_producer, PortId output, std::optional<std::uint16_t> tensor_index,
    ProcessGroupId allocator_process_group, DeviceId device) noexcept {
  constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint32_t>::max();

  if (allocator_process_group && device.valid()) {
    std::vector<VertexId> operators;
    for (const auto& binding : final_operator_bindings) {
      if (!binding.operator_vertex || binding.process_group != allocator_process_group ||
          binding.device != device) {
        continue;
      }
      if (std::ranges::find(operators, binding.operator_vertex) == operators.end()) {
        operators.push_back(binding.operator_vertex);
      }
    }

    std::uint64_t staging_streams{};
    for (const auto& service : cross_process_services) {
      if (!descriptor_tensor_schema_service(service)) {
        continue;
      }
      const bool retains_forwarded_backing =
          cross_process_tensor_service_requires_stream_for_device_in_group(
              service, allocator_process_group, device);
      if (retains_forwarded_backing && staging_streams == kMaximum) {
        return std::nullopt;
      }
      staging_streams += static_cast<std::uint64_t>(retains_forwarded_backing);
    }

    const bool producer_counted =
        allocating_producer && std::ranges::find(operators, allocating_producer) != operators.end();
    const std::uint64_t possible_consumers =
        operators.size() - static_cast<std::size_t>(producer_counted);
    if (staging_streams + 1U > kMaximum ||
        possible_consumers > (kMaximum - staging_streams - 1U) / 2U) {
      return std::nullopt;
    }
    return static_cast<std::uint32_t>(2U * possible_consumers + staging_streams + 1U);
  }

  std::uint64_t subscribers{};
  for (const auto& service : process_local_services) {
    if (service.source != output || !service.tensor_payload) {
      continue;
    }
    if (subscribers > kMaximum - service.max_subscribers) {
      return std::nullopt;
    }
    subscribers += service.max_subscribers;
  }

  std::uint64_t staging_streams{};
  for (const auto& service : cross_process_services) {
    if (service.source != output || !descriptor_tensor_schema_service(service)) {
      continue;
    }
    const auto view = tensor_service_contract_view(service);
    bool retains_pool_on_service_stream = !tensor_index && view.tensor_fields.empty() &&
                                          view.tensor_memory_kind == MemoryKind::kCudaDevice;
    if (tensor_index) {
      retains_pool_on_service_stream = std::ranges::any_of(
          view.tensor_fields, [&](const TensorFieldServiceStorageContract& field) {
            return field.tensor_index == *tensor_index &&
                   field.memory_kind == MemoryKind::kCudaDevice;
          });
    }
    if (retains_pool_on_service_stream && staging_streams == kMaximum) {
      return std::nullopt;
    }
    staging_streams += static_cast<std::uint64_t>(retains_pool_on_service_stream);
  }

  if (staging_streams + 1U > kMaximum || subscribers > (kMaximum - staging_streams - 1U) / 2U) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(2U * subscribers + staging_streams + 1U);
}

}  // namespace holoscan::detail
