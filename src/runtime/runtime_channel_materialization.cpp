/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/profiler/nvtx3.hpp>

#include "../channel/channel_catalog_binding.hpp"
#include "../channel/channel_tensor_loan_internal.hpp"
#include "../channel/payload_descriptor_wire.hpp"
#include "../channel/pod_codec.hpp"
#include "../channel/schema_envelope_codec.hpp"
#include "../compiler/cuda_service_stream_plan.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../compiler/process_local_channel_plan_validation.hpp"
#include "../core/domain/tensor_access.hpp"
#include "../core/plan_record_contracts.hpp"
#include "../core/tensor_allocation_record_validation.hpp"
#include "channel_runtime_binding_access.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "operator_profile_labels.hpp"
#include "runtime_channel_binding.hpp"
#include "runtime_channel_plan_binding.hpp"
#include "runtime_channel_session_impl.hpp"
#include "runtime_channel_storage.hpp"
#include "runtime_cuda_execution_resources.hpp"
#include "runtime_phase_counters.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_tensor_resources.hpp"
#include "runtime_worker.hpp"
#include "temporal_preselection.hpp"

namespace holoscan::detail {

// Ordered construction and unpublished-owner rollback for one Channel session.

namespace {

[[nodiscard]] const RuntimeChannelPortReplayRecord* find_replay(
    std::span<const RuntimeChannelPortReplayRecord> replay, PortId port,
    PortDirection direction) noexcept {
  const auto found = std::find_if(replay.begin(), replay.end(), [&](const auto& candidate) {
    return candidate.port == port && candidate.direction == direction;
  });
  return found == replay.end() ? nullptr : &*found;
}

template <typename OutputState>
[[nodiscard]] expected<void, Error> initialize_output_nvtx(OutputState& output,
                                                           const ExecutionPlanState& plan,
                                                           const PortPlanRecord& port_plan) {
  const VertexRecord* const owner_vertex = find_vertex_by_id(plan, output.owner);
  if (owner_vertex == nullptr || owner_vertex->name.empty() || port_plan.local_name.empty()) {
    HOLOSCAN_LOG_WARN(
        "OCL NVTX emit annotations disabled: missing output operator or port metadata.");
    return {};
  }

  const std::string source_identity_label = operator_profile_port_label(
      plan, owner_vertex->graph, owner_vertex->name, port_plan.local_name);
  output.source_identity_nvtx_string = nvtxDomainRegisterStringA(
      holoscan::profiler::operator_profile_domain(), source_identity_label.c_str());

  std::string emit_label = source_identity_label;
  emit_label += ":emit";
  output.emit_nvtx_message =
      nvtxDomainRegisterStringA(holoscan::profiler::operator_profile_domain(), emit_label.c_str());
  return {};
}

template <typename SubscriberState>
[[nodiscard]] expected<void, Error> initialize_subscriber_nvtx(SubscriberState& subscriber,
                                                               const ExecutionPlanState& plan,
                                                               const PortPlanRecord& source_port,
                                                               const PortPlanRecord& target_port) {
  const VertexRecord* const source_vertex = find_vertex_by_id(plan, subscriber.source_owner);
  const VertexRecord* const target_vertex = find_vertex_by_id(plan, subscriber.owner);
  if (source_vertex == nullptr || target_vertex == nullptr || source_vertex->name.empty() ||
      target_vertex->name.empty() || source_port.local_name.empty() ||
      target_port.local_name.empty()) {
    HOLOSCAN_LOG_WARN(
        "OCL NVTX receive annotations disabled: missing source/target operator or port metadata.");
    return {};
  }

  const std::string source_identity_label = operator_profile_port_label(
      plan, source_vertex->graph, source_vertex->name, source_port.local_name);
  subscriber.source_identity_nvtx_string = nvtxDomainRegisterStringA(
      holoscan::profiler::operator_profile_domain(), source_identity_label.c_str());

  const std::string target_identity_label = operator_profile_port_label(
      plan, target_vertex->graph, target_vertex->name, target_port.local_name);
  std::string receive_label = target_identity_label;
  receive_label += ":receive";
  subscriber.receive_nvtx_message = nvtxDomainRegisterStringA(
      holoscan::profiler::operator_profile_domain(), receive_label.c_str());
  return {};
}

/// Resolve the catalog-local identity of the clock this plan stamps publications from.
///
/// Runtime stamps a publication in the plan clock namespace, but an exported sample is read by a
/// client that only knows the endpoint catalog namespace. The shared clock key is the durable join
/// between the independently numbered namespaces.
[[nodiscard]] ClockId resolve_catalog_publish_clock(const ExecutionPlanState& plan) noexcept {
  const auto* catalog = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  const auto* plan_clock = find_clock_by_id(plan, plan.root_default_clock);
  if (catalog == nullptr || plan_clock == nullptr) {
    return {};
  }
  const auto found =
      std::find_if(catalog->clocks.begin(), catalog->clocks.end(), [&](const auto& candidate) {
        return candidate.key == plan_clock->key;
      });
  return found == catalog->clocks.end() ? ClockId{} : ClockId{found->id};
}

[[nodiscard]] bool green_context_device_opted_in(const ExecutionPlanState& plan,
                                                 const ProcessGroupId process_group,
                                                 const DeviceId device) noexcept {
  return std::any_of(plan.cuda_green_context_pools.begin(),
                     plan.cuda_green_context_pools.end(),
                     [&](const auto& pool) {
                       return pool.process_group == process_group && pool.device == device;
                     });
}

[[nodiscard]] expected<cudaStream_t, Error> planned_service_stream(
    const ExecutionPlanState& plan, const RuntimeCudaExecutionResources* cuda_resources,
    const ProcessGroupId process_group, const DeviceId device, const std::string_view owner_id) {
  if (!green_context_device_opted_in(plan, process_group, device)) {
    return nullptr;
  }
  if (cuda_resources == nullptr || !process_group || !device.valid() || owner_id.empty()) {
    return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GREEN_CONTEXT_STREAM_MISMATCH"});
  }
  auto* const stream = cuda_resources->service_stream(process_group, device, owner_id);
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kFailure, "CUDA_GREEN_CONTEXT_STREAM_MISMATCH"});
  }
  return stream;
}

[[nodiscard]] expected<std::string_view, Error> tensor_transfer_service_owner(
    const ExecutionPlanState& plan, const ExternalTopicId topic, const ProcessGroupId process_group,
    const DeviceId device) noexcept {
  std::string_view owner;
  for (const auto& transfer : plan.tensor_transfer_plans) {
    if (transfer.topic != topic || transfer.disposition != PlanDisposition::kAccepted ||
        !transfer.requires_exportable_copy_pool ||
        transfer.interface_process_group != process_group || transfer.interface_device != device) {
      continue;
    }
    if (transfer.service_owner_id.empty() ||
        (!owner.empty() && owner != transfer.service_owner_id)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    owner = transfer.service_owner_id;
  }
  if (owner.empty()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return owner;
}

}  // namespace

/// Own the cold, unpublished construction of one Channel session.
///
/// Every endpoint, binding, retained slot, and construction-only projection remains reachable
/// from this object until materialize() succeeds. Destruction before that commit point releases the
/// partially built Impl and its children in their existing reverse ownership order.
class RuntimeChannelSession::Impl::MaterializationTransaction {
 public:
  MaterializationTransaction(const ExecutionPlanState& plan,
                             const std::span<const RuntimeChannelPortReplayRecord> replay_ports,
                             const ProcessGroupId process_group,
                             const RuntimeChannelReadinessWake readiness_wake,
                             const RuntimeCudaExecutionResources* cuda_resources,
                             const RuntimeChannelSizingOverride sizing_override,
                             const RunGenerationId run_generation)
      : plan_(plan),
        replay_ports_(replay_ports),
        process_group_(process_group),
        readiness_wake_(readiness_wake),
        cuda_resources_(cuda_resources),
        sizing_override_(sizing_override),
        run_generation_(run_generation) {}

  MaterializationTransaction(const MaterializationTransaction&) = delete;
  MaterializationTransaction& operator=(const MaterializationTransaction&) = delete;

  [[nodiscard]] expected<std::unique_ptr<Impl>, Error> materialize(ChannelRuntime runtime);

 private:
  struct PendingCatalogTensorPublisher {
    Impl::OutputState* output{};
    ChannelEndpointCatalog::EndpointView endpoint;
    std::uint32_t max_loaned_samples{};
    std::optional<DeviceId> planned_device;
    cudaStream_t staging_stream{};
    ClockId plan_capture_clock{};
    ClockId catalog_capture_clock{};
    bool peer_required{true};
  };

  struct PendingCatalogTensorSchemaPublisher {
    Impl::OutputState* output{};
    ChannelEndpointCatalog::EndpointView endpoint;
    SchemaIdentityRecord schema_identity;
    std::size_t payload_size{};
    std::vector<std::size_t> descriptor_offsets;
    std::vector<channel_internal::TensorSchemaStagingOptions> staging;
    cudaStream_t staging_stream{};
    ClockId plan_capture_clock{};
    ClockId catalog_capture_clock{};
    std::uint32_t expected_recipient_count{};
    bool peer_required{true};
  };

  [[nodiscard]] expected<void, Error> initialize_common_owner(ChannelRuntime runtime);
  void project_cross_process_records();
  [[nodiscard]] const ProcessLocalChannelServicePlanRecord* locate_service(
      std::string_view service_id) const noexcept;
  [[nodiscard]] const ProcessLocalChannelPublisherPlanRecord* locate_publisher(
      std::string_view service_id) const noexcept;
  [[nodiscard]] expected<std::optional<std::uint32_t>, Error> resolved_queue_depth(
      EdgeId edge, std::uint32_t planned) const;
  [[nodiscard]] expected<void, Error> materialize_publishers();
  [[nodiscard]] expected<void, Error> validate_external_exports();
  [[nodiscard]] expected<void, Error> prepare_output_storage();
  [[nodiscard]] expected<void, Error> materialize_subscribers();
  [[nodiscard]] expected<void, Error> materialize_external_imports();
  [[nodiscard]] expected<void, Error> publish_input_bindings_and_indexes();

  const ExecutionPlanState& plan_;
  std::span<const RuntimeChannelPortReplayRecord> replay_ports_;
  ProcessGroupId process_group_{};
  RuntimeChannelReadinessWake readiness_wake_{};
  const RuntimeCudaExecutionResources* cuda_resources_{};
  RuntimeChannelSizingOverride sizing_override_{};
  RunGenerationId run_generation_{};
  std::unique_ptr<Impl> impl_;
  std::vector<ProcessLocalChannelServicePlanRecord> cross_process_services_;
  std::vector<ProcessLocalChannelPublisherPlanRecord> cross_process_publishers_;
  std::vector<ProcessLocalChannelSubscriberPlanRecord> cross_process_subscribers_;
  std::vector<PendingCatalogTensorPublisher> pending_catalog_tensor_publishers_;
  std::vector<PendingCatalogTensorSchemaPublisher> pending_catalog_tensor_schema_publishers_;
};

// Reconstruction validates and binds every compiler-frozen service, endpoint, replay record,
// capacity, and readiness edge as one fail-closed transaction before publishing the session.
expected<std::unique_ptr<RuntimeChannelSession>, Error> RuntimeChannelSession::create(
    const ExecutionPlanState& plan,
    const std::span<const RuntimeChannelPortReplayRecord> replay_ports,
    ProcessGroupId process_group, RuntimeChannelReadinessWake readiness_wake,
    const RuntimeCudaExecutionResources* cuda_resources,
    RuntimeChannelSizingOverride sizing_override, RunGenerationId run_generation) {
  if (!plan.admitted || plan.plan_generation == 0U ||
      ((readiness_wake.state == nullptr) != (readiness_wake.signal == nullptr)) ||
      (readiness_wake.signal_owner_progress != nullptr && readiness_wake.state == nullptr) ||
      ((sizing_override.state == nullptr) != (sizing_override.queue_depth == nullptr)) ||
      plan.process_local_channel_publishers.size() != plan.process_local_channel_services.size() ||
      plan.cross_process_channel_publishers.size() != plan.cross_process_channel_services.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!valid_process_local_channel_plan(plan) || !valid_external_channel_plan(plan) ||
      !valid_cross_process_channel_plan(plan)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto generation_capacities_valid =
      validate_import_generation_capacities(plan.imported_generation_capacities);
  if (!generation_capacities_valid) {
    return make_unexpected(std::move(generation_capacities_valid).error());
  }

  auto runtime = ChannelRuntime::create();
  if (!runtime) {
    return make_unexpected(std::move(runtime).error());
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    Impl::MaterializationTransaction transaction{plan,
                                                 replay_ports,
                                                 process_group,
                                                 readiness_wake,
                                                 cuda_resources,
                                                 sizing_override,
                                                 run_generation};
    auto impl = transaction.materialize(std::move(*runtime));
    if (!impl) {
      return make_unexpected(std::move(impl).error());
    }
    return std::unique_ptr<RuntimeChannelSession>{new RuntimeChannelSession{std::move(*impl)}};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

expected<std::unique_ptr<RuntimeChannelSession::Impl>, Error>
RuntimeChannelSession::Impl::MaterializationTransaction::materialize(ChannelRuntime runtime) {
  auto initialized = initialize_common_owner(std::move(runtime));
  if (!initialized) {
    return make_unexpected(std::move(initialized).error());
  }
  project_cross_process_records();
  auto publishers_materialized = materialize_publishers();
  if (!publishers_materialized) {
    return make_unexpected(std::move(publishers_materialized).error());
  }
  auto external_exports_validated = validate_external_exports();
  if (!external_exports_validated) {
    return make_unexpected(std::move(external_exports_validated).error());
  }
  auto output_storage_prepared = prepare_output_storage();
  if (!output_storage_prepared) {
    return make_unexpected(std::move(output_storage_prepared).error());
  }
  auto subscribers_materialized = materialize_subscribers();
  if (!subscribers_materialized) {
    return make_unexpected(std::move(subscribers_materialized).error());
  }
  auto external_imports_materialized = materialize_external_imports();
  if (!external_imports_materialized) {
    return make_unexpected(std::move(external_imports_materialized).error());
  }
  impl_->clock_health_reset_requests_.reserve(impl_->subscribers_.size());
  auto input_bindings_published = publish_input_bindings_and_indexes();
  if (!input_bindings_published) {
    return make_unexpected(std::move(input_bindings_published).error());
  }

  // Successful publication is the only operation that disarms transaction rollback.
  return std::move(impl_);
}

expected<void, Error>
RuntimeChannelSession::Impl::MaterializationTransaction::initialize_common_owner(
    ChannelRuntime runtime) {
  const std::size_t local_partition_runners = static_cast<std::size_t>(
      std::count_if(plan_.partition_runners.begin(),
                    plan_.partition_runners.end(),
                    [&](const PartitionRunnerPlanRecord& runner) {
                      return !process_group_ || runner.process_group == process_group_;
                    }));
  // Accepted production plans name every local runner. Hand-authored private test plans omit
  // that inventory, so they conservatively retain the concurrent-publication guard.
  const bool track_concurrent_process_local_publications = local_partition_runners != 1U;
  auto wait_set = RuntimeChannelBindingAccess::create_wait_set(runtime);
  if (!wait_set) {
    return make_unexpected(std::move(wait_set).error());
  }
  impl_ = std::make_unique<Impl>(std::move(runtime),
                                 plan_.plan_generation,
                                 run_generation_,
                                 plan_.clock_conversions,
                                 plan_.executable_factories.support_thread_control,
                                 plan_.executable_factories.test_channel_incomplete_poll_control,
                                 plan_.executable_factories.test_force_process_local_channel_pump,
                                 track_concurrent_process_local_publications,
                                 plan_.executable_factories.phase_counters.get(),
                                 readiness_wake_);
  impl_->catalog_publish_clock_ = resolve_catalog_publish_clock(plan_);
  impl_->plan_publish_clock_ = plan_.root_default_clock;
  impl_->wait_set_ = std::move(*wait_set);
  return {};
}

void RuntimeChannelSession::Impl::MaterializationTransaction::project_cross_process_records() {
  const auto& plan = plan_;
  const ProcessGroupId process_group = process_group_;
  auto& cross_process_services = cross_process_services_;
  auto& cross_process_publishers = cross_process_publishers_;
  auto& cross_process_subscribers = cross_process_subscribers_;

  cross_process_services.reserve(plan.cross_process_channel_services.size());
  cross_process_publishers.reserve(plan.cross_process_channel_publishers.size());
  cross_process_subscribers.reserve(plan.cross_process_channel_subscribers.size());

  // Project the worker-visible cross-process records into the process-local construction
  // vocabulary. These vectors remain owned by this transaction and only provide stable views to
  // the later publisher and subscriber phases.
  // Compatible fanout arms share one plan-frozen cross-process service while
  // retaining one subscriber record per edge. Reuse the process-local
  // construction path after selecting records owned by this worker.
  const auto service_reaches_process_group =
      [&process_group](const CrossProcessChannelServicePlanRecord& service) noexcept {
        return !process_group || service.source_process_group == process_group ||
               std::find(service.target_process_groups.begin(),
                         service.target_process_groups.end(),
                         process_group) != service.target_process_groups.end();
      };
  for (const auto& service : plan.cross_process_channel_services) {
    if (!service_reaches_process_group(service)) {
      continue;
    }
    cross_process_services.push_back(
        make_process_local_service_record(service.source,
                                          service.service_id,
                                          make_tensor_service_contract(service),
                                          service.max_subscribers,
                                          true));
  }
  for (const auto& publisher : plan.cross_process_channel_publishers) {
    const auto* service = find_cross_process_channel_service_by_id(plan, publisher.service_id);
    if (service == nullptr || !service_reaches_process_group(*service)) {
      continue;
    }
    cross_process_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
        .source = publisher.source,
        .service_id = publisher.service_id,
        .max_loaned_samples = publisher.max_loaned_samples,
        .replacement_attribution_required = publisher.replacement_attribution_required,
    });
  }
  for (const auto& subscriber : plan.cross_process_channel_subscribers) {
    const auto* service = find_cross_process_channel_service_by_id(plan, subscriber.service_id);
    const auto* target =
        find_port_by_id_and_direction(plan, subscriber.target, PortDirection::kInput);
    if (service == nullptr || target == nullptr ||
        (process_group && target->owner_kind == PortOwnerKind::kOperator &&
         !worker_owns_vertex(plan, process_group, target->owner))) {
      continue;
    }
    cross_process_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
        .edge = subscriber.edge,
        .source = subscriber.source,
        .target = subscriber.target,
        .service_id = subscriber.service_id,
        .requested_queue_depth = subscriber.requested_queue_depth,
        .max_borrowed_samples = subscriber.max_borrowed_samples,
        .cache_capacity = subscriber.cache_capacity,
        .readiness_capacity = subscriber.readiness_capacity,
        .event_wake = subscriber.event_wake,
    });
  }
}

const ProcessLocalChannelServicePlanRecord*
RuntimeChannelSession::Impl::MaterializationTransaction::locate_service(
    const std::string_view service_id) const noexcept {
  if (const auto* found = find_process_local_channel_service_by_id(plan_, service_id);
      found != nullptr) {
    return found;
  }
  const auto cross_found = std::find_if(
      cross_process_services_.begin(), cross_process_services_.end(), [&](const auto& candidate) {
        return candidate.service_id == service_id;
      });
  return cross_found == cross_process_services_.end() ? nullptr : &*cross_found;
}

const ProcessLocalChannelPublisherPlanRecord*
RuntimeChannelSession::Impl::MaterializationTransaction::locate_publisher(
    const std::string_view service_id) const noexcept {
  if (const auto* found = find_process_local_channel_publisher_by_service_id(plan_, service_id);
      found != nullptr) {
    return found;
  }
  const auto cross_found =
      std::find_if(cross_process_publishers_.begin(),
                   cross_process_publishers_.end(),
                   [&](const auto& candidate) { return candidate.service_id == service_id; });
  return cross_found == cross_process_publishers_.end() ? nullptr : &*cross_found;
}

expected<std::optional<std::uint32_t>, Error>
RuntimeChannelSession::Impl::MaterializationTransaction::resolved_queue_depth(
    const EdgeId edge, const std::uint32_t planned) const {
  if (sizing_override_.queue_depth == nullptr) {
    return std::optional<std::uint32_t>{};
  }
  auto selected = sizing_override_.queue_depth(sizing_override_.state, edge);
  if (!selected) {
    return make_unexpected(std::move(selected.error()));
  }
  if (*selected && (**selected == 0U || planned == 0U)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  return *selected;
}

// Phase 1: materialize all participating process-local and projected cross-process outputs.
expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::MaterializationTransaction::materialize_publishers() {
  const auto& plan = plan_;
  const auto replay_ports = replay_ports_;
  const ProcessGroupId process_group = process_group_;
  const RuntimeCudaExecutionResources* cuda_resources = cuda_resources_;
  auto& cross_process_publishers = cross_process_publishers_;
  auto& impl = impl_;

  const std::array<const std::vector<ProcessLocalChannelPublisherPlanRecord>*, 2U>
      publisher_families{&plan.process_local_channel_publishers, &cross_process_publishers};
  for (const auto* publisher_records : publisher_families) {
    for (const auto& publisher_plan : *publisher_records) {
      const auto* service = locate_service(publisher_plan.service_id);
      const auto* replay = find_replay(replay_ports, publisher_plan.source, PortDirection::kOutput);
      const auto* port_plan =
          find_port_by_id_and_direction(plan, publisher_plan.source, PortDirection::kOutput);
      if (service == nullptr || port_plan == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (process_group && port_plan->owner_kind == PortOwnerKind::kOperator &&
          !worker_owns_vertex(plan, process_group, port_plan->owner)) {
        continue;
      }
      if (replay == nullptr && port_plan->owner_kind != PortOwnerKind::kOperator) {
        continue;
      }
      const bool tensor_service = service->tensor_payload;
      auto inline_descriptor_contract = find_inline_descriptor_tensor_contract(
          plan, publisher_plan.source, std::nullopt, *service);
      if (!inline_descriptor_contract) {
        return make_unexpected(std::move(inline_descriptor_contract).error());
      }
      const std::uint64_t effective_tensor_max_byte_span =
          *inline_descriptor_contract ? (*inline_descriptor_contract)->max_byte_span
                                      : service->tensor_max_byte_span;
      const MemoryKind effective_tensor_memory_kind =
          *inline_descriptor_contract ? (*inline_descriptor_contract)->memory_kind
                                      : service->tensor_memory_kind;
      const DeviceId effective_tensor_device = *inline_descriptor_contract
                                                   ? (*inline_descriptor_contract)->device
                                                   : service->tensor_device;
      const std::optional<TensorRepresentation> effective_tensor_layout =
          *inline_descriptor_contract
              ? (*inline_descriptor_contract)->output_layout
              : (singular_descriptor_storage_contract(*service) ? port_plan->tensor_layout
                                                                : std::nullopt);
      const bool runtime_type_matches =
          tensor_service
              ? runtime_tensor_matches_plan(
                    replay == nullptr ? nullptr : replay->type_ops, *port_plan, *service)
              : runtime_codec_matches_plan(replay == nullptr ? nullptr : replay->type_ops,
                                           *port_plan,
                                           service->payload_size_bytes,
                                           !service->cross_process_transport);
      if (replay == nullptr || !replay->owner || replay->declaration == nullptr ||
          !runtime_type_matches || replay->type_ops->sample_payload == nullptr ||
          replay->type_ops->sample_metadata == nullptr ||
          replay->type_ops->bind_output == nullptr || replay->type_ops->unbind_output == nullptr ||
          publisher_plan.max_loaned_samples == 0U || port_plan->owner != replay->owner ||
          port_plan->max_emits_per_compute == 0U) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      const bool direct_retained =
          service->delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained;
      ChannelOptions options;
      if (!direct_retained) {
        auto resolved_options = process_local_service_options(*service);
        if (!resolved_options) {
          return make_unexpected(std::move(resolved_options).error());
        }
        options = *resolved_options;
        options.max_loaned_samples = publisher_plan.max_loaned_samples;
        options.replacement_attribution_required = publisher_plan.replacement_attribution_required;
      }
      auto ensured_output = impl->ensure_output_state(
          *replay,
          *port_plan,
          plan.executable_factories.test_tensor_output_control.get(),
          [&](Impl::OutputState& created) -> expected<void, Error> {
            auto nvtx_initialized = initialize_output_nvtx(created, plan, *port_plan);
            if (!nvtx_initialized) {
              return make_unexpected(std::move(nvtx_initialized).error());
            }
            if (tensor_service) {
              created.tensor_max_byte_span = effective_tensor_max_byte_span;
              created.tensor_memory_kind = effective_tensor_memory_kind;
              created.tensor_device = effective_tensor_device;
              created.tensor_layout = effective_tensor_layout;
              created.tensor_contract = port_plan->tensor_contract;
              created.tensor_fields = service->tensor_fields;
              if (!*inline_descriptor_contract) {
                auto field_device =
                    uniform_tensor_field_device(created.tensor_device, created.tensor_fields);
                if (!field_device) {
                  return make_unexpected(std::move(field_device).error());
                }
                created.tensor_device = *field_device;
              }
              if (replay->type_ops->codec_family ==
                  RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
                if (!port_plan->payload_identity.schema) {
                  return make_unexpected(Error{ErrorCode::kProtocolError});
                }
                created.tensor_schema_id = port_plan->payload_identity.schema->id;
                created.tensor_schema_scratch.resize(
                    replay->type_ops->descriptor_tensor_static_count);
              }
            }
            if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
                !service->cross_process_transport) {
              if (!port_plan->payload_identity.schema) {
                return make_unexpected(Error{ErrorCode::kProtocolError});
              }
              created.process_local_schema_identity = port_plan->payload_identity.schema;
            }
            return {};
          });
      if (!ensured_output) {
        return make_unexpected(std::move(ensured_output).error());
      }
      auto [output, output_created] = *ensured_output;
      if (!output_created &&
          (output->max_emits_per_activation != port_plan->max_emits_per_compute ||
           (tensor_service &&
            (output->tensor_max_byte_span != effective_tensor_max_byte_span ||
             output->tensor_memory_kind != effective_tensor_memory_kind ||
             output->tensor_fields != service->tensor_fields ||
             (service->tensor_fields.empty() && output->tensor_device != effective_tensor_device) ||
             !tensor_port_layout_exactly_equal(output->tensor_layout, effective_tensor_layout) ||
             output->tensor_contract.has_value() != port_plan->tensor_contract.has_value() ||
             (output->tensor_contract &&
              !tensor_contract_equal(*output->tensor_contract, *port_plan->tensor_contract)))))) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (direct_retained) {
        if (service->cross_process_transport || service->event_wake ||
            output->direct_subscriber != nullptr ||
            (tensor_service && !output->tensor_payload() && !output->tensor_schema_payload())) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      } else if (tensor_service && replay->type_ops->codec_family ==
                                       RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
        std::vector<channel_internal::TensorSchemaStagingOptions> staging;
        if (service->cross_process_transport) {
          const std::uint64_t staging_slots =
              static_cast<std::uint64_t>(options.max_loaned_samples) +
              static_cast<std::uint64_t>(options.max_subscriber_queue_depth) +
              static_cast<std::uint64_t>(options.max_subscriber_borrowed_samples);
          if (staging_slots == 0U || staging_slots > std::numeric_limits<std::uint32_t>::max()) {
            return make_unexpected(Error{ErrorCode::kResourceExhausted});
          }
          if (!service->tensor_fields.empty()) {
            staging.reserve(service->tensor_fields.size());
            for (const auto& field : service->tensor_fields) {
              staging.push_back(channel_internal::TensorSchemaStagingOptions{
                  .max_byte_span = field.max_byte_span,
                  .slot_count = static_cast<std::uint32_t>(staging_slots),
                  .dtype = field.dtype,
                  .rank = field.rank,
                  .memory_kind = field.memory_kind,
                  .device = field.device,
                  .foreign_emit_policy = field.tensor_emit_policy,
              });
            }
          } else {
            if (!port_plan->tensor_layout || !port_plan->tensor_layout->dtype ||
                !port_plan->tensor_layout->rank || service->tensor_max_byte_span == 0U ||
                (service->tensor_memory_kind != MemoryKind::kHost &&
                 service->tensor_memory_kind != MemoryKind::kPinnedHost &&
                 service->tensor_memory_kind != MemoryKind::kCudaDevice)) {
              return make_unexpected(Error{ErrorCode::kProtocolError});
            }
            staging.push_back(channel_internal::TensorSchemaStagingOptions{
                .max_byte_span = service->tensor_max_byte_span,
                .slot_count = static_cast<std::uint32_t>(staging_slots),
                .dtype = port_plan->tensor_layout->dtype,
                .rank = port_plan->tensor_layout->rank,
                .memory_kind = service->tensor_memory_kind,
                .device = service->tensor_device,
                .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
            });
          }
        }
        std::vector<std::size_t> descriptor_offsets;
        descriptor_offsets.reserve(replay->type_ops->descriptor_tensor_static_count);
        for (std::uint32_t index = 0U; index < replay->type_ops->descriptor_tensor_static_count;
             ++index) {
          descriptor_offsets.push_back(replay->type_ops->descriptor_offset +
                                       static_cast<std::size_t>(index) * sizeof(PayloadDescriptor));
        }
        cudaStream_t staging_stream = nullptr;
        std::optional<DeviceId> staging_device;
        for (const auto& field : staging) {
          if (field.memory_kind != MemoryKind::kCudaDevice) {
            continue;
          }
          if (!field.device.valid() || (staging_device && *staging_device != field.device)) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          staging_device = field.device;
        }
        if (staging_device && service->cross_process_transport) {
          const auto* cross_service =
              find_cross_process_channel_service_by_id(plan, service->service_id);
          if (cross_service == nullptr) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          const std::string owner_id = cuda_channel_service_stream_owner(
              *cross_service, cross_service->source_process_group, *staging_device);
          auto resolved_stream = planned_service_stream(
              plan, cuda_resources, cross_service->source_process_group, *staging_device, owner_id);
          if (!resolved_stream) {
            return make_unexpected(std::move(resolved_stream).error());
          }
          staging_stream = *resolved_stream;
        }
        auto endpoint = RuntimeChannelBindingAccess::create_tensor_schema_publisher(
            *impl->runtime_,
            service->service_id,
            replay->type_ops->wire_capacity,
            descriptor_offsets,
            options,
            staging,
            staging_stream);
        if (!endpoint) {
          return make_unexpected(std::move(endpoint).error());
        }
        output->tensor_schema_endpoints.push_back(Impl::OutputState::TensorSchemaEndpointState{
            .publisher = std::move(*endpoint),
            .process_local_service = service->service_id,
            .expected_recipient_count = options.expected_recipient_count,
            .process_local_transport = !service->cross_process_transport,
        });
      } else if (tensor_service) {
        auto endpoint = RuntimeChannelBindingAccess::create_tensor_publisher(
            *impl->runtime_, service->service_id, options);
        if (!endpoint) {
          return make_unexpected(std::move(endpoint).error());
        }
        output->tensor_endpoints.push_back(Impl::OutputState::TensorEndpointState{
            .publisher = std::move(*endpoint),
            .process_local_service = service->service_id,
            .expected_recipient_count = options.expected_recipient_count,
            .process_local_transport = !service->cross_process_transport,
        });
      } else {
        auto endpoint = RuntimeChannelBindingAccess::create_publisher(
            *impl->runtime_, service->service_id, service->payload_size_bytes, options);
        if (!endpoint) {
          return make_unexpected(std::move(endpoint).error());
        }
        output->endpoints.push_back(Impl::OutputState::EndpointState{
            .publisher = std::move(*endpoint),
            .process_local_service = service->service_id,
            .expected_recipient_count = options.expected_recipient_count,
            .process_local_transport = !service->cross_process_transport,
            .process_local_schema_envelope =
                replay->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
                !service->cross_process_transport,
        });
      }
    }
  }
  return {};
}

// Phase 2: validate external exports before deferred Tensor endpoint materialization.
expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::MaterializationTransaction::validate_external_exports() {
  const auto& plan = plan_;
  const auto replay_ports = replay_ports_;
  const ProcessGroupId process_group = process_group_;
  const RuntimeCudaExecutionResources* cuda_resources = cuda_resources_;
  auto& impl = impl_;
  auto& pending_catalog_tensor_publishers = pending_catalog_tensor_publishers_;
  auto& pending_catalog_tensor_schema_publishers = pending_catalog_tensor_schema_publishers_;

  pending_catalog_tensor_publishers.reserve(plan.graph.external_topics().size());
  pending_catalog_tensor_schema_publishers.reserve(plan.graph.external_topics().size());

  for (const auto& topic : plan.graph.external_topics()) {
    if (topic.imported) {
      continue;
    }
    auto internal_port = external_internal_port(plan, PortId{topic.graph_port.value}, false);
    if (!internal_port) {
      if (internal_port.error().code == ErrorCode::kNotFound) {
        continue;
      }
      return make_unexpected(std::move(internal_port).error());
    }
    const auto* replay = find_replay(replay_ports, *internal_port, PortDirection::kOutput);
    const auto* port_plan =
        find_port_by_id_and_direction(plan, *internal_port, PortDirection::kOutput);
    const auto* service = find_external_channel_service_by_topic(plan, topic.id);
    const auto* publisher = find_external_channel_publisher_by_topic(plan, topic.id);
    const auto* subscriber = find_external_channel_subscriber_by_topic(plan, topic.id);
    if (process_group && port_plan != nullptr &&
        port_plan->owner_kind == PortOwnerKind::kOperator &&
        !worker_owns_vertex(plan, process_group, port_plan->owner)) {
      continue;
    }
    const bool tensor_service = service != nullptr && service->tensor_payload;
    const bool descriptor_tensor_service =
        tensor_service && service->payload_identity.kind == PayloadIdentityKind::kSchema &&
        service->payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const bool runtime_type_matches =
        tensor_service ? replay != nullptr && port_plan != nullptr &&
                             ((replay->type_ops == runtime_port_type_ops<Tensor>() &&
                               port_plan->payload_identity.kind == PayloadIdentityKind::kTensor) ||
                              (descriptor_tensor_service && replay->type_ops != nullptr &&
                               replay->type_ops->codec_family ==
                                   RuntimePayloadCodecFamily::kDescriptorTensorSchema))
                       : replay != nullptr && port_plan != nullptr &&
                             runtime_codec_matches_plan(
                                 replay->type_ops,
                                 *port_plan,
                                 replay->type_ops == nullptr ? 0U : replay->type_ops->wire_capacity,
                                 false);
    if (replay == nullptr || port_plan == nullptr || service == nullptr || publisher == nullptr ||
        subscriber == nullptr || replay->owner != port_plan->owner ||
        replay->declaration == nullptr || !runtime_type_matches ||
        ((!tensor_service || descriptor_tensor_service) &&
         (replay->type_ops->sample_payload == nullptr ||
          replay->type_ops->sample_metadata == nullptr)) ||
        replay->type_ops->bind_output == nullptr || replay->type_ops->unbind_output == nullptr ||
        port_plan->max_emits_per_compute == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto options = external_service_options(*service, *publisher, *subscriber);
    if (!options) {
      return make_unexpected(std::move(options).error());
    }
    auto catalog_endpoint =
        find_validated_catalog_endpoint_by_boundary_port(plan, PortId{topic.graph_port.value});
    const auto* const catalog_record =
        catalog_endpoint ? ChannelEndpointCatalogAccess::record(*catalog_endpoint) : nullptr;
    if (catalog_record == nullptr || catalog_record->service != service->service_id ||
        catalog_record->service_static.id != service->service_id ||
        ChannelEndpointCatalogAccess::backend_service_binding(*catalog_endpoint).empty()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (tensor_service && runtime_external_tensor_uses_unsupported_strides(
                              replay->type_ops, *port_plan, *service, *catalog_record)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (tensor_service && !runtime_external_tensor_matches_plan(
                              replay->type_ops, *port_plan, *service, *catalog_record)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const PortId boundary_port{topic.graph_port.value};
    const EndpointClockHealthPlanRecord* const health_plan =
        find_endpoint_clock_health(plan, boundary_port);
    if (health_plan == nullptr || health_plan->endpoint != boundary_port ||
        health_plan->plan_generation != plan.plan_generation ||
        catalog_record->direction != TopicDirection::kExport ||
        catalog_record->boundary_port != boundary_port ||
        catalog_record->clock_health.capability != health_plan->capability ||
        catalog_record->clock_health.requirement != health_plan->requirement ||
        catalog_record->clock_health.max_report_age != health_plan->max_report_age) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto capture_clock_binding =
        bind_catalog_capture_clock(plan, *catalog_endpoint, health_plan->capture_clock);
    if (!capture_clock_binding) {
      return make_unexpected(std::move(capture_clock_binding).error());
    }
    std::unique_ptr<channel_internal::InlinePayloadPublisher> endpoint;
    channel_internal::CatalogCanonicalPodPublisherBinding canonical_pod_publisher;
    if (!tensor_service) {
      auto created = channel_internal::CatalogChannelBindingAccess::create_payload_publisher(
          *impl->runtime_,
          *catalog_endpoint,
          replay->type_ops->wire_capacity,
          {},
          RuntimeChannelBindingAccess::graph_blocking_pressure_options());
      if (!created) {
        return make_unexpected(std::move(created).error());
      }
      endpoint = std::move(*created);
      if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
        canonical_pod_publisher =
            channel_internal::CatalogChannelBindingAccess::bind_canonical_pod_publisher(*endpoint);
        if (!canonical_pod_publisher) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
    }
    std::vector<TensorFieldServiceStorageContract> descriptor_storage_fields;
    if (descriptor_tensor_service) {
      auto fields = external_descriptor_tensor_storage_contracts(
          plan, topic.id, boundary_port, service->tensor_fields);
      if (!fields || !port_plan->payload_identity.schema ||
          replay->type_ops->descriptor_tensor_static_count != service->tensor_fields.size()) {
        return make_unexpected(fields ? Error{ErrorCode::kProtocolError}
                                      : std::move(fields).error());
      }
      descriptor_storage_fields = std::move(*fields);
    }
    auto ensured_output = impl->ensure_output_state(
        *replay,
        *port_plan,
        plan.executable_factories.test_tensor_output_control.get(),
        [&](Impl::OutputState& created) -> expected<void, Error> {
          auto nvtx_initialized = initialize_output_nvtx(created, plan, *port_plan);
          if (!nvtx_initialized) {
            return make_unexpected(std::move(nvtx_initialized).error());
          }
          if (!descriptor_tensor_service) {
            return {};
          }
          created.tensor_schema_id = port_plan->payload_identity.schema->id;
          created.tensor_fields = descriptor_storage_fields;
          created.tensor_logical_fields = service->tensor_fields;
          created.tensor_schema_scratch.resize(replay->type_ops->descriptor_tensor_static_count);
          return {};
        });
    if (!ensured_output) {
      return make_unexpected(std::move(ensured_output).error());
    }
    auto [output, output_created] = *ensured_output;
    if (descriptor_tensor_service && !output_created) {
      if ((!output->tensor_logical_fields.empty() &&
           !descriptor_tensor_fields_equal(output->tensor_logical_fields,
                                           service->tensor_fields)) ||
          (!output->tensor_fields.empty() && output->tensor_fields != descriptor_storage_fields)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (output->tensor_logical_fields.empty()) {
        if (output->tensor_fields.empty()) {
          output->tensor_fields = descriptor_storage_fields;
        }
        output->tensor_logical_fields = service->tensor_fields;
        output->tensor_schema_id = port_plan->payload_identity.schema->id;
        if (output->tensor_schema_scratch.empty()) {
          output->tensor_schema_scratch.resize(replay->type_ops->descriptor_tensor_static_count);
        }
      }
    }
    if (descriptor_tensor_service) {
      if (!catalog_record->service_static.schema_identity) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      std::vector<channel_internal::TensorSchemaStagingOptions> staging;
      const bool copy_allowed = std::any_of(
          service->tensor_fields.begin(), service->tensor_fields.end(), [](const auto& field) {
            return field.tensor.foreign_emit_policy !=
                   InterfaceTensorEmitPolicy::kRequireExportable;
          });
      std::optional<DeviceId> staging_device;
      ProcessGroupId staging_process_group{};
      if (copy_allowed) {
        staging.reserve(service->tensor_fields.size());
        for (const auto& field : service->tensor_fields) {
          const bool field_copy_allowed =
              field.tensor.foreign_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable;
          const auto* const pool = field_copy_allowed
                                       ? find_tensor_interface_pool_by_topic_field(
                                             plan, topic.id, field.field_id, field.tensor_index)
                                       : nullptr;
          if (!field.tensor.memory_kind || !field.tensor.dtype || !field.tensor.rank ||
              !field.tensor.max_byte_span || (field_copy_allowed && pool == nullptr)) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          if (pool != nullptr && pool->device.valid()) {
            if ((staging_device && *staging_device != pool->device) ||
                (staging_process_group && staging_process_group != pool->allocator_process_group)) {
              return make_unexpected(Error{ErrorCode::kNotSupported});
            }
            staging_device = pool->device;
            staging_process_group = pool->allocator_process_group;
          }
          const auto storage = std::find_if(
              output->tensor_fields.begin(), output->tensor_fields.end(), [&](const auto& row) {
                return row.field_id == field.field_id && row.tensor_index == field.tensor_index;
              });
          if (storage == output->tensor_fields.end()) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          staging.push_back(channel_internal::TensorSchemaStagingOptions{
              .max_byte_span = *field.tensor.max_byte_span,
              .slot_count = pool != nullptr ? pool->slot_count : 0U,
              .dtype = field.tensor.dtype,
              .rank = field.tensor.rank,
              .axes = field.tensor.axes,
              .strides = field.tensor.strides,
              .memory_kind = *field.tensor.memory_kind,
              .device = storage->device,
              .foreign_emit_policy = field.tensor.foreign_emit_policy,
          });
        }
      }
      cudaStream_t staging_stream = nullptr;
      if (staging_device) {
        auto owner_id =
            tensor_transfer_service_owner(plan, topic.id, staging_process_group, *staging_device);
        if (!owner_id) {
          return make_unexpected(std::move(owner_id).error());
        }
        auto resolved_stream = planned_service_stream(
            plan, cuda_resources, staging_process_group, *staging_device, *owner_id);
        if (!resolved_stream) {
          return make_unexpected(std::move(resolved_stream).error());
        }
        staging_stream = *resolved_stream;
      }
      std::vector<std::size_t> descriptor_offsets;
      descriptor_offsets.reserve(replay->type_ops->descriptor_tensor_static_count);
      for (std::uint32_t index = 0U; index < replay->type_ops->descriptor_tensor_static_count;
           ++index) {
        descriptor_offsets.push_back(replay->type_ops->descriptor_offset +
                                     static_cast<std::size_t>(index) * sizeof(PayloadDescriptor));
      }
      // Defer backend construction until every catalog descriptor-schema fanout arm has passed
      // protocol validation, matching the raw-Tensor catalog path below.
      pending_catalog_tensor_schema_publishers.push_back(PendingCatalogTensorSchemaPublisher{
          .output = output,
          .endpoint = *catalog_endpoint,
          .schema_identity = *catalog_record->service_static.schema_identity,
          .payload_size = replay->type_ops->wire_capacity,
          .descriptor_offsets = std::move(descriptor_offsets),
          .staging = std::move(staging),
          .staging_stream = staging_stream,
          .plan_capture_clock = capture_clock_binding->plan,
          .catalog_capture_clock = capture_clock_binding->catalog,
          .expected_recipient_count = options->expected_recipient_count,
          .peer_required = catalog_record->attach.peer == PeerRequirement::kRequired,
      });
    } else if (tensor_service) {
      const auto& catalog_requirements = *catalog_record->service_static.tensor;
      const auto* const interface_pool = find_tensor_interface_pool_by_topic(plan, topic.id);
      if (service->tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable &&
          (!catalog_requirements.memory_kind || interface_pool == nullptr ||
           interface_pool->topic != topic.id || interface_pool->graph_port != boundary_port ||
           interface_pool->memory_kind != *catalog_requirements.memory_kind ||
           ((interface_pool->memory_kind == MemoryKind::kPinnedHost ||
             interface_pool->memory_kind == MemoryKind::kCudaDevice) &&
            !interface_pool->device.valid()))) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (!output->catalog_tensor_requirements) {
        output->catalog_tensor_requirements = catalog_requirements;
      } else if (!logical_tensor_requirements_exactly_equal(*output->catalog_tensor_requirements,
                                                            catalog_requirements)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      cudaStream_t staging_stream = nullptr;
      // Ordinary execution intentionally lets the catalog publisher select its existing copy
      // stream. Only a Green Context plan needs the separately frozen service-stream owner.
      if (interface_pool != nullptr && interface_pool->device.valid() &&
          green_context_device_opted_in(
              plan, interface_pool->allocator_process_group, interface_pool->device)) {
        auto owner_id = tensor_transfer_service_owner(
            plan, topic.id, interface_pool->allocator_process_group, interface_pool->device);
        if (!owner_id) {
          return make_unexpected(std::move(owner_id).error());
        }
        auto resolved_stream = planned_service_stream(plan,
                                                      cuda_resources,
                                                      interface_pool->allocator_process_group,
                                                      interface_pool->device,
                                                      *owner_id);
        if (!resolved_stream) {
          return make_unexpected(std::move(resolved_stream).error());
        }
        staging_stream = *resolved_stream;
      }
      // Defer materialization until every catalog Tensor fanout arm has agreed on the one
      // Output-wide contract. This gives protocol validation deterministic precedence over
      // backend availability and avoids constructing a partially validated fanout.
      pending_catalog_tensor_publishers.push_back(PendingCatalogTensorPublisher{
          .output = output,
          .endpoint = *catalog_endpoint,
          .max_loaned_samples = publisher->max_loaned_samples,
          .planned_device = interface_pool != nullptr && interface_pool->device.valid()
                                ? std::optional<DeviceId>{interface_pool->device}
                                : std::nullopt,
          .staging_stream = staging_stream,
          .plan_capture_clock = capture_clock_binding->plan,
          .catalog_capture_clock = capture_clock_binding->catalog,
          .peer_required = catalog_record->attach.peer == PeerRequirement::kRequired,
      });
    } else {
      output->endpoints.push_back(Impl::OutputState::EndpointState{
          .publisher = std::move(endpoint),
          .plan_capture_clock = capture_clock_binding->plan,
          .catalog_capture_clock = capture_clock_binding->catalog,
          .expected_recipient_count = options->expected_recipient_count,
          .refresh_before_emit = true,
          .peer_required = catalog_record->attach.peer == PeerRequirement::kRequired,
          .catalog_encodes_custom_pod =
              replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod,
          .canonical_pod_publisher = canonical_pod_publisher,
      });
    }
  }
  return {};
}

// Phase 3: materialize deferred Tensor endpoints, output scratch, and loan state before publishing
// any output declaration binding.
expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::MaterializationTransaction::prepare_output_storage() {
  const auto& plan = plan_;
  auto& impl = impl_;
  auto& pending_catalog_tensor_publishers = pending_catalog_tensor_publishers_;
  auto& pending_catalog_tensor_schema_publishers = pending_catalog_tensor_schema_publishers_;

  for (auto& pending : pending_catalog_tensor_schema_publishers) {
    auto schema_endpoint = RuntimeChannelBindingAccess::create_catalog_tensor_schema_publisher(
        *impl->runtime_,
        pending.endpoint,
        pending.schema_identity,
        pending.payload_size,
        pending.descriptor_offsets,
        pending.staging,
        pending.staging_stream);
    if (!schema_endpoint) {
      return make_unexpected(std::move(schema_endpoint).error());
    }
    pending.output->tensor_schema_endpoints.push_back(Impl::OutputState::TensorSchemaEndpointState{
        .publisher = std::move(*schema_endpoint),
        .plan_capture_clock = pending.plan_capture_clock,
        .catalog_capture_clock = pending.catalog_capture_clock,
        .expected_recipient_count = pending.expected_recipient_count,
        .refresh_before_emit = true,
        .peer_required = pending.peer_required,
        .process_local_transport = false,
    });
  }

  for (auto& pending : pending_catalog_tensor_publishers) {
    auto tensor_endpoint = RuntimeChannelBindingAccess::create_catalog_tensor_publisher(
        *impl->runtime_, pending.endpoint, pending.max_loaned_samples, pending.planned_device);
    if (!tensor_endpoint) {
      HOLOSCAN_LOG_ERROR("External Tensor publisher materialization failed with {}",
                         runtime_error_message(tensor_endpoint.error()));
      return make_unexpected(std::move(tensor_endpoint).error());
    }
    pending.output->catalog_tensor_endpoints.push_back(
        Impl::OutputState::CatalogTensorEndpointState{
            .publisher = std::move(*tensor_endpoint),
            .staging_stream = pending.staging_stream,
            .plan_capture_clock = pending.plan_capture_clock,
            .catalog_capture_clock = pending.catalog_capture_clock,
            .peer_required = pending.peer_required,
            .active_for_current_fanout = false,
        });
  }

  for (const auto& output : impl->outputs_) {
    // Phase 3: allocate output scratch and loan state, then publish declaration bindings.
    if (output->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema ||
        output->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod ||
        output->type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
      auto scratch =
          AlignedBlock::create(output->type_ops->wire_capacity, alignof(std::max_align_t));
      if (!scratch) {
        return make_unexpected(std::move(scratch).error());
      }
      output->encode_scratch = std::move(*scratch);
    }
    if (output->process_local_schema_identity) {
      auto scratch = AlignedBlock::create(
          kSchemaEnvelopeHeaderSize + output->type_ops->wire_capacity, alignof(std::max_align_t));
      if (!scratch) {
        return make_unexpected(std::move(scratch).error());
      }
      output->process_local_schema_scratch = std::move(*scratch);
    }
    output->binding =
        RuntimeOutputBinding{.state = output.get(), .ops = &Impl::OutputState::kBindingOps};
    const auto loan_plan = std::find_if(
        plan.generic_output_loans.begin(), plan.generic_output_loans.end(), [&](const auto& loan) {
          return loan.output == output->port;
        });
    if (loan_plan != plan.generic_output_loans.end()) {
      if (loan_plan->disposition != PlanDisposition::kAccepted ||
          loan_plan->plan_generation != plan.plan_generation || loan_plan->run_generation != 0U ||
          loan_plan->slot_count == 0U || loan_plan->max_live_loans == 0U ||
          loan_plan->max_live_loans > loan_plan->slot_count ||
          loan_plan->slot_size_bytes != output->type_ops->payload_size ||
          output->type_ops->construct_value == nullptr ||
          output->type_ops->destroy_value == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      output->max_live_loans = loan_plan->max_live_loans;
      output->loans.reserve(loan_plan->slot_count);
      for (std::uint32_t index = 0U; index < loan_plan->slot_count; ++index) {
        auto value = AlignedBlock::create(output->type_ops->payload_size,
                                          output->type_ops->payload_alignment);
        if (!value) {
          return make_unexpected(std::move(value).error());
        }
        auto* control = new Impl::LoanControl;
        control->owner = output.get();
        control->live_credit = &output->live_loan_count;
        control->type_ops = output->type_ops;
        control->value = std::move(*value);
        output->type_ops->construct_value(control->value.data());
        control->binding =
            RuntimeLoanBinding{.state = control, .cancel = &Impl::LoanControl::cancel};
        output->loans.push_back(control);
      }
    }
    output->type_ops->bind_output(output->declaration, &output->binding);
  }
  return {};
}

// Phase 4: materialize process-local and projected cross-process subscriber state.
expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::MaterializationTransaction::materialize_subscribers() {
  const auto& plan = plan_;
  const auto replay_ports = replay_ports_;
  const ProcessGroupId process_group = process_group_;
  auto& cross_process_subscribers = cross_process_subscribers_;
  auto& impl = impl_;

  const std::array<const std::vector<ProcessLocalChannelSubscriberPlanRecord>*, 2U>
      subscriber_families{&plan.process_local_channel_subscribers, &cross_process_subscribers};
  for (const auto* subscriber_records : subscriber_families) {
    for (const auto& subscriber_plan : *subscriber_records) {
      const auto* service = locate_service(subscriber_plan.service_id);
      const auto* replay = find_replay(replay_ports, subscriber_plan.target, PortDirection::kInput);
      const auto* port_plan =
          find_port_by_id_and_direction(plan, subscriber_plan.target, PortDirection::kInput);
      const auto* publisher_plan = locate_publisher(subscriber_plan.service_id);
      if (service == nullptr || port_plan == nullptr || publisher_plan == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (process_group && port_plan->owner_kind == PortOwnerKind::kOperator &&
          !worker_owns_vertex(plan, process_group, port_plan->owner)) {
        continue;
      }
      if (replay == nullptr && port_plan->owner_kind != PortOwnerKind::kOperator) {
        continue;
      }
      const bool tensor_service = service->tensor_payload;
      const bool runtime_type_matches =
          tensor_service
              ? runtime_tensor_matches_plan(
                    replay == nullptr ? nullptr : replay->type_ops, *port_plan, *service)
              : runtime_codec_matches_plan(replay == nullptr ? nullptr : replay->type_ops,
                                           *port_plan,
                                           service->payload_size_bytes,
                                           !service->cross_process_transport);
      if (replay == nullptr || !replay->owner || replay->declaration == nullptr ||
          !runtime_type_matches ||
          (!tensor_service && replay->type_ops->construct_sample == nullptr) ||
          replay->type_ops->destroy_sample == nullptr || replay->type_ops->bind_input == nullptr ||
          replay->type_ops->unbind_input == nullptr || subscriber_plan.readiness_capacity == 0U ||
          subscriber_plan.requested_queue_depth == 0U ||
          subscriber_plan.max_borrowed_samples == 0U ||
          subscriber_plan.event_wake != service->event_wake || port_plan->owner != replay->owner ||
          (service->queue_policy != QueuePolicy::kBlocking &&
           !publisher_plan->replacement_attribution_required)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }

      auto ensured_input = impl->ensure_input_state(plan, *replay);
      if (!ensured_input) {
        return make_unexpected(std::move(ensured_input).error());
      }
      Impl::InputPortState* const input = *ensured_input;

      const bool direct_retained =
          service->delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained;
      auto selected_queue_depth =
          resolved_queue_depth(subscriber_plan.edge, subscriber_plan.requested_queue_depth);
      if (!selected_queue_depth) {
        return make_unexpected(std::move(selected_queue_depth.error()));
      }
      if (*selected_queue_depth &&
          (!direct_retained || service->cross_process_transport || service->event_wake ||
           subscriber_plan.event_wake || service->queue_policy != QueuePolicy::kBlocking)) {
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
      }
      const std::uint32_t effective_queue_depth =
          selected_queue_depth->value_or(subscriber_plan.requested_queue_depth);
      const std::uint32_t effective_borrowed_samples =
          selected_queue_depth->value_or(subscriber_plan.max_borrowed_samples);
      const std::uint32_t effective_typed_capacity =
          std::max(effective_queue_depth, subscriber_plan.cache_capacity);
      const std::uint32_t effective_readiness_capacity =
          *selected_queue_depth
              ? std::max(subscriber_plan.readiness_capacity, effective_typed_capacity)
              : subscriber_plan.readiness_capacity;
      ChannelOptions options;
      if (!direct_retained) {
        auto resolved_options = process_local_service_options(*service);
        if (!resolved_options) {
          return make_unexpected(std::move(resolved_options).error());
        }
        options = *resolved_options;
        options.subscriber_queue_depth = effective_queue_depth;
        options.max_borrowed_samples = effective_borrowed_samples;
      }

      auto inline_descriptor_contract = find_inline_descriptor_tensor_contract(
          plan, subscriber_plan.source, subscriber_plan.target, *service);
      if (!inline_descriptor_contract) {
        return make_unexpected(std::move(inline_descriptor_contract).error());
      }
      const std::uint64_t effective_tensor_max_byte_span =
          *inline_descriptor_contract ? (*inline_descriptor_contract)->max_byte_span
                                      : service->tensor_max_byte_span;
      const MemoryKind effective_tensor_memory_kind =
          *inline_descriptor_contract ? (*inline_descriptor_contract)->memory_kind
                                      : service->tensor_memory_kind;
      const DeviceId effective_tensor_device = *inline_descriptor_contract
                                                   ? (*inline_descriptor_contract)->device
                                                   : service->tensor_device;
      auto subscriber = std::make_unique<Impl::SubscriberState>();
      subscriber->edge = subscriber_plan.edge;
      subscriber->source = subscriber_plan.source;
      subscriber->target = subscriber_plan.target;
      subscriber->service = service->service_id;
      const PortPlanRecord* const source_port =
          find_port_by_id_and_direction(plan, subscriber->source, PortDirection::kOutput);
      const PortPlanRecord* const target_port =
          find_port_by_id_and_direction(plan, subscriber->target, PortDirection::kInput);
      if (source_port == nullptr || target_port == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (const auto* source_replay =
              find_replay(replay_ports, subscriber_plan.source, PortDirection::kOutput)) {
        subscriber->source_owner = source_replay->owner;
      } else {
        subscriber->source_owner = source_port->owner;
      }
      subscriber->owner = replay->owner;
      auto subscriber_nvtx_initialized =
          initialize_subscriber_nvtx(*subscriber, plan, *source_port, *target_port);
      if (!subscriber_nvtx_initialized) {
        return make_unexpected(std::move(subscriber_nvtx_initialized).error());
      }
      subscriber->queue_policy = service->queue_policy;
      subscriber->readiness_capacity = effective_readiness_capacity;
      subscriber->drain_capacity = effective_queue_depth;
      subscriber->read_claim_capacity = effective_borrowed_samples;
      subscriber->pre_witness_replacement_allowed =
          publisher_plan->replacement_attribution_required;
      subscriber->direct_process_local = direct_retained;
      subscriber->cross_process_transport = service->cross_process_transport;
      subscriber->process_local_transport = !direct_retained && !service->cross_process_transport;
      subscriber->input = input;
      input->batch_receive_nvtx_enabled |= subscriber->receive_nvtx_message != nullptr &&
                                           subscriber->source_identity_nvtx_string != nullptr;
      if (tensor_service) {
        subscriber->tensor_max_byte_span = effective_tensor_max_byte_span;
        subscriber->tensor_memory_kind = effective_tensor_memory_kind;
        subscriber->tensor_device = effective_tensor_device;
        subscriber->tensor_fields = service->tensor_fields;
        if (replay->type_ops->codec_family != RuntimePayloadCodecFamily::kDescriptorTensorSchema ||
            replay->type_ops->descriptor_tensor_static_count <= 1U) {
          subscriber->pass_through_tensor_contract = port_plan->tensor_contract;
        }
        if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
          const auto tensor_count = replay->type_ops->descriptor_tensor_static_count;
          subscriber->descriptor_tensor_scratch.resize(tensor_count);
          subscriber->tensor_schema_scratch.resize(tensor_count);
        }
        if (!*inline_descriptor_contract) {
          auto field_device =
              uniform_tensor_field_device(subscriber->tensor_device, subscriber->tensor_fields);
          if (!field_device) {
            return make_unexpected(std::move(field_device).error());
          }
          subscriber->tensor_device = *field_device;
        }
        if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema &&
            singular_descriptor_storage_contract(*service)) {
          subscriber->pass_through_tensor_layout = port_plan->tensor_layout;
        }
      }
      if (direct_retained) {
        if (service->cross_process_transport || service->event_wake || subscriber_plan.event_wake ||
            (tensor_service && replay->type_ops != runtime_port_type_ops<Tensor>() &&
             replay->type_ops->codec_family !=
                 RuntimePayloadCodecFamily::kDescriptorTensorSchema)) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        if (tensor_service) {
          if (replay->type_ops->codec_family ==
              RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
            if (replay->type_ops->wire_capacity != service->payload_size_bytes ||
                replay->type_ops->descriptor_tensor_sample_metadata == nullptr ||
                replay->type_ops->assign_descriptor_tensors_sample == nullptr) {
              return make_unexpected(Error{ErrorCode::kProtocolError});
            }
            if (*inline_descriptor_contract) {
              subscriber->pass_through_tensor_layout = (*inline_descriptor_contract)->input_layout;
            }
          } else if (service->tensor_memory_kind == MemoryKind::kUnknown) {
            subscriber->pass_through_tensor_layout = port_plan->tensor_layout;
          }
        } else {
          auto scratch =
              AlignedBlock::create(service->payload_size_bytes, alignof(std::max_align_t));
          if (!scratch) {
            return make_unexpected(std::move(scratch).error());
          }
          subscriber->scratch = std::move(*scratch);
          subscriber->scratch_size = service->payload_size_bytes;
        }
      } else if (tensor_service && replay->type_ops->codec_family ==
                                       RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
        auto endpoint = RuntimeChannelBindingAccess::create_tensor_schema_subscriber(
            *impl->runtime_,
            service->service_id,
            replay->type_ops->wire_capacity,
            replay->type_ops->descriptor_tensor_static_count,
            options);
        if (!endpoint) {
          return make_unexpected(std::move(endpoint).error());
        }
        subscriber->tensor_schema_endpoint = std::move(*endpoint);
        auto scratch =
            AlignedBlock::create(replay->type_ops->wire_capacity, alignof(std::max_align_t));
        if (!scratch) {
          return make_unexpected(std::move(scratch).error());
        }
        subscriber->scratch = std::move(*scratch);
        subscriber->scratch_size = replay->type_ops->wire_capacity;
      } else if (tensor_service) {
        auto endpoint = RuntimeChannelBindingAccess::create_tensor_subscriber(
            *impl->runtime_, service->service_id, options);
        if (!endpoint) {
          return make_unexpected(std::move(endpoint).error());
        }
        subscriber->tensor_endpoint.emplace(std::move(*endpoint));
        if (service->tensor_memory_kind == MemoryKind::kUnknown) {
          subscriber->pass_through_tensor_layout = port_plan->tensor_layout;
        }
      } else {
        auto endpoint = RuntimeChannelBindingAccess::create_subscriber(
            *impl->runtime_, service->service_id, service->payload_size_bytes, options);
        if (!endpoint) {
          return make_unexpected(std::move(endpoint).error());
        }
        subscriber->endpoint = std::move(*endpoint);
        auto scratch = AlignedBlock::create(service->payload_size_bytes, alignof(std::max_align_t));
        if (!scratch) {
          return make_unexpected(std::move(scratch).error());
        }
        subscriber->scratch = std::move(*scratch);
        subscriber->scratch_size = service->payload_size_bytes;
      }
      if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
        if (!port_plan->payload_identity.pod) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        subscriber->pod_identity = port_plan->payload_identity.pod;
        subscriber->native_scratch.resize(replay->type_ops->payload_size);
      } else if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
                 !service->cross_process_transport) {
        if (!port_plan->payload_identity.schema) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        subscriber->process_local_schema_identity = port_plan->payload_identity.schema;
      }
      const std::uint32_t typed_slot_capacity = effective_typed_capacity;
      if (effective_readiness_capacity < typed_slot_capacity) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      subscriber->typed_slot_capacity = typed_slot_capacity;
      // Staging is provisioned for any non-blocking subscriber, transported included. A pumped
      // subscriber needs it for the same reason a direct one does: without somewhere to put a
      // sample that arrives while the typed slot is pinned, the newer sample is left in the
      // transport queue and drop-oldest freshness cannot reach it at all.
      subscriber->direct_ingress_capacity =
          service->queue_policy != QueuePolicy::kBlocking ? effective_queue_depth : 0U;
      subscriber->active_direct_ingress_capacity = subscriber->direct_ingress_capacity;
      if (typed_slot_capacity >
          std::numeric_limits<std::uint32_t>::max() - subscriber->direct_ingress_capacity) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      // One physical slot beyond the authored staging depth, reachable only by a freshness
      // reservation. It is deliberately not part of direct_ingress_capacity: the survivor set a
      // witness can see stays the newest N unread, and this slot holds only the record an
      // eviction has already qualified and is about to promote. Without it, reserving that
      // record at depth one closes the producer's only staging destination.
      const std::uint32_t reservation_headroom =
          subscriber->direct_ingress_capacity != 0U ? 1U : 0U;
      if (typed_slot_capacity > std::numeric_limits<std::uint32_t>::max() - reservation_headroom) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const std::uint32_t sample_slot_capacity =
          typed_slot_capacity + subscriber->direct_ingress_capacity + reservation_headroom;
      const std::size_t tensor_fields_per_slot =
          replay->type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema
              ? replay->type_ops->descriptor_tensor_static_count
              : 1U;
      auto prepared_storage = impl->prepare_subscriber_storage(*subscriber,
                                                               typed_slot_capacity,
                                                               sample_slot_capacity,
                                                               tensor_fields_per_slot,
                                                               subscriber_plan.event_wake);
      if (!prepared_storage) {
        return make_unexpected(std::move(prepared_storage).error());
      }
      if (direct_retained) {
        const auto source_output =
            std::find_if(impl->outputs_.begin(), impl->outputs_.end(), [&](const auto& candidate) {
              return candidate->port == subscriber_plan.source;
            });
        if (source_output == impl->outputs_.end() ||
            (*source_output)->direct_subscriber != nullptr ||
            !(*source_output)->endpoints.empty() || !(*source_output)->tensor_endpoints.empty() ||
            !(*source_output)->catalog_tensor_endpoints.empty() ||
            !(*source_output)->tensor_schema_endpoints.empty()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        (*source_output)->direct_subscriber = subscriber.get();
      }
      input->subscribers.push_back(subscriber.get());
      impl->subscribers_.push_back(std::move(subscriber));
    }
  }
  return {};
}

// Phase 5: materialize imported catalog endpoints through the same input and slot builders.
expected<void, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RuntimeChannelSession::Impl::MaterializationTransaction::materialize_external_imports() {
  const auto& plan = plan_;
  const auto replay_ports = replay_ports_;
  const ProcessGroupId process_group = process_group_;
  auto& impl = impl_;

  for (const auto& topic : plan.graph.external_topics()) {
    if (!topic.imported) {
      continue;
    }
    auto internal_port = external_internal_port(plan, PortId{topic.graph_port.value}, true);
    if (!internal_port) {
      if (internal_port.error().code == ErrorCode::kNotFound) {
        continue;
      }
      return make_unexpected(std::move(internal_port).error());
    }
    const auto* replay = find_replay(replay_ports, *internal_port, PortDirection::kInput);
    const auto* port_plan =
        find_port_by_id_and_direction(plan, *internal_port, PortDirection::kInput);
    const auto* service = find_external_channel_service_by_topic(plan, topic.id);
    const auto* publisher = find_external_channel_publisher_by_topic(plan, topic.id);
    const auto* subscriber_plan = find_external_channel_subscriber_by_topic(plan, topic.id);
    if (process_group && port_plan != nullptr &&
        port_plan->owner_kind == PortOwnerKind::kOperator &&
        !worker_owns_vertex(plan, process_group, port_plan->owner)) {
      continue;
    }
    const bool tensor_service = service != nullptr && service->tensor_payload;
    const bool descriptor_tensor_service =
        tensor_service && service->payload_identity.kind == PayloadIdentityKind::kSchema &&
        service->payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const bool runtime_type_matches =
        tensor_service ? replay != nullptr && port_plan != nullptr &&
                             ((replay->type_ops == runtime_port_type_ops<Tensor>() &&
                               port_plan->payload_identity.kind == PayloadIdentityKind::kTensor) ||
                              (descriptor_tensor_service && replay->type_ops != nullptr &&
                               replay->type_ops->codec_family ==
                                   RuntimePayloadCodecFamily::kDescriptorTensorSchema))
                       : replay != nullptr && port_plan != nullptr &&
                             runtime_codec_matches_plan(
                                 replay->type_ops,
                                 *port_plan,
                                 replay->type_ops == nullptr ? 0U : replay->type_ops->wire_capacity,
                                 false);
    if (replay == nullptr || port_plan == nullptr || service == nullptr || publisher == nullptr ||
        subscriber_plan == nullptr || replay->owner != port_plan->owner ||
        replay->declaration == nullptr || !runtime_type_matches ||
        (!tensor_service && replay->type_ops->construct_sample == nullptr) ||
        (descriptor_tensor_service &&
         (replay->type_ops->descriptor_tensor_sample_metadata == nullptr ||
          replay->type_ops->descriptor_tensor_view_metadata_at == nullptr ||
          replay->type_ops->assign_descriptor_tensors_sample == nullptr ||
          replay->type_ops->prepare_sample_storage == nullptr ||
          replay->type_ops->reset_sample_storage == nullptr)) ||
        replay->type_ops->destroy_sample == nullptr || replay->type_ops->bind_input == nullptr ||
        replay->type_ops->unbind_input == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto options = external_service_options(*service, *publisher, *subscriber_plan);
    if (!options) {
      return make_unexpected(std::move(options).error());
    }
    auto catalog_endpoint =
        find_validated_catalog_endpoint_by_boundary_port(plan, PortId{topic.graph_port.value});
    const auto* const catalog_record =
        catalog_endpoint ? ChannelEndpointCatalogAccess::record(*catalog_endpoint) : nullptr;
    if (catalog_record == nullptr || catalog_record->service != service->service_id ||
        catalog_record->service_static.id != service->service_id ||
        ChannelEndpointCatalogAccess::backend_service_binding(*catalog_endpoint).empty()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (tensor_service && runtime_external_tensor_uses_unsupported_strides(
                              replay->type_ops, *port_plan, *service, *catalog_record)) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (tensor_service && !runtime_external_tensor_matches_plan(
                              replay->type_ops, *port_plan, *service, *catalog_record)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto* const tensor_interface_pool =
        tensor_service && !descriptor_tensor_service
            ? find_tensor_interface_pool_by_topic(plan, topic.id)
            : nullptr;
    if (tensor_service && !descriptor_tensor_service &&
        service->tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable &&
        (tensor_interface_pool == nullptr || tensor_interface_pool->topic != topic.id ||
         tensor_interface_pool->graph_port.value() != topic.graph_port.value ||
         !catalog_record->service_static.tensor->memory_kind ||
         tensor_interface_pool->memory_kind !=
             *catalog_record->service_static.tensor->memory_kind ||
         ((tensor_interface_pool->memory_kind == MemoryKind::kPinnedHost ||
           tensor_interface_pool->memory_kind == MemoryKind::kCudaDevice) &&
          !tensor_interface_pool->device.valid()))) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto generation_capacity =
        std::find_if(plan.imported_generation_capacities.begin(),
                     plan.imported_generation_capacities.end(),
                     [&](const auto& record) { return record.topic == topic.id; });
    std::optional<std::uint64_t> attach_max_wait_ns;
    if (catalog_record->attach.max_wait) {
      attach_max_wait_ns = static_cast<std::uint64_t>(catalog_record->attach.max_wait->count());
    }
    if (generation_capacity == plan.imported_generation_capacities.end() ||
        generation_capacity->endpoint_or_service_id != service->service_id ||
        generation_capacity->peer_requirement != catalog_record->attach.peer ||
        generation_capacity->attach_max_wait_ns != attach_max_wait_ns) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const PortId boundary_port{topic.graph_port.value};
    const EndpointClockHealthPlanRecord* const health_plan =
        find_endpoint_clock_health(plan, boundary_port);
    if (health_plan == nullptr || catalog_record == nullptr ||
        health_plan->endpoint != boundary_port ||
        health_plan->plan_generation != plan.plan_generation ||
        catalog_record->direction != TopicDirection::kImport ||
        catalog_record->boundary_port != boundary_port ||
        catalog_record->clock_health.capability != health_plan->capability ||
        catalog_record->clock_health.requirement != health_plan->requirement ||
        catalog_record->clock_health.max_report_age != health_plan->max_report_age) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const bool monitored = health_plan->capability == ClockHealthCapability::kStatusReports;
    if (monitored &&
        (health_plan->requirement != ClockHealthRequirement::kFreshHealthyReport ||
         !health_plan->max_report_age.has_value() || health_plan->max_report_age->count() <= 0 ||
         health_plan->fixed_report_ingress_capacity != 1U ||
         health_plan->fixed_update_capacity != 4U ||
         health_plan->per_dispatch_report_budget != 1U)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!monitored && (health_plan->capability != ClockHealthCapability::kUnavailable ||
                       health_plan->requirement != ClockHealthRequirement::kNotRequired ||
                       health_plan->max_report_age.has_value() ||
                       health_plan->fixed_report_ingress_capacity != 0U ||
                       health_plan->fixed_update_capacity != 0U ||
                       health_plan->per_dispatch_report_budget != 0U)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto capture_clock_binding =
        bind_catalog_capture_clock(plan, *catalog_endpoint, health_plan->capture_clock);
    if (!capture_clock_binding) {
      return make_unexpected(std::move(capture_clock_binding).error());
    }
    const auto payload_edge =
        std::find_if(plan.payload_edges.begin(), plan.payload_edges.end(), [&](const auto& edge) {
          return edge.source == PortId(topic.graph_port.value) && edge.target == *internal_port;
        });
    const std::uint32_t cache_capacity =
        payload_edge == plan.payload_edges.end() ? 0U : payload_edge->cache_capacity;
    const std::uint32_t typed_slot_capacity =
        std::max(subscriber_plan->requested_queue_depth, cache_capacity);
    if (typed_slot_capacity == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (typed_slot_capacity == std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    // Every catalog import can make all retained data slots ready and then accept one ordered
    // EOS control record before the controller freezes its candidate cut. Reserve that terminal
    // witness explicitly; EOS owns no typed payload slot.
    const std::uint32_t readiness_capacity = typed_slot_capacity + 1U;
    // A raw catalog Tensor import can retain every typed slot while one replacement is
    // materialized before the evicted backing is released. Reserve that transient receive-
    // control claim independently; it does not represent the ordered EOS readiness witness.
    const std::uint32_t tensor_receive_control_capacity = typed_slot_capacity + 1U;
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> endpoint;
    std::optional<ChannelSubscriber<Tensor>> tensor_endpoint;
    std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> tensor_schema_endpoint;
    if (descriptor_tensor_service) {
      if (!catalog_record->service_static.schema_identity) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto created = RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
          *impl->runtime_,
          *catalog_endpoint,
          *catalog_record->service_static.schema_identity,
          replay->type_ops->wire_capacity,
          replay->type_ops->descriptor_tensor_static_count);
      if (!created) {
        return make_unexpected(std::move(created).error());
      }
      tensor_schema_endpoint = std::move(*created);
    } else if (tensor_service) {
      auto created = RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
          *impl->runtime_, *catalog_endpoint, tensor_receive_control_capacity);
      if (!created) {
        return make_unexpected(std::move(created).error());
      }
      tensor_endpoint.emplace(std::move(*created));
    } else {
      auto created = channel_internal::CatalogChannelBindingAccess::create_payload_subscriber(
          *impl->runtime_, *catalog_endpoint, replay->type_ops->wire_capacity);
      if (!created) {
        return make_unexpected(std::move(created).error());
      }
      endpoint = std::move(*created);
    }

    auto ensured_input = impl->ensure_input_state(plan, *replay);
    if (!ensured_input) {
      return make_unexpected(std::move(ensured_input).error());
    }
    Impl::InputPortState* const input = *ensured_input;
    if ((input->capture_time_required || input->capture_clock.valid()) &&
        input->capture_clock != capture_clock_binding->plan) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    auto subscriber = std::make_unique<Impl::SubscriberState>();
    subscriber->edge = payload_edge == plan.payload_edges.end() ? EdgeId{} : payload_edge->edge;
    subscriber->source = PortId{topic.graph_port.value};
    subscriber->target = *internal_port;
    const PortPlanRecord* const source_port =
        find_port_by_id_and_direction(plan, subscriber->source, PortDirection::kOutput);
    if (source_port == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    subscriber->source_owner = source_port->owner;
    subscriber->owner = replay->owner;
    auto subscriber_nvtx_initialized =
        initialize_subscriber_nvtx(*subscriber, plan, *source_port, *port_plan);
    if (!subscriber_nvtx_initialized) {
      return make_unexpected(std::move(subscriber_nvtx_initialized).error());
    }
    subscriber->queue_policy = service->queue_policy;
    subscriber->readiness_capacity = readiness_capacity;
    subscriber->drain_capacity = subscriber_plan->requested_queue_depth;
    subscriber->read_claim_capacity = subscriber_plan->max_borrowed_samples;
    subscriber->pre_witness_replacement_allowed = publisher->replacement_attribution_required;
    subscriber->externally_driven_import = true;
    subscriber->catalog_tensor_import = tensor_service;
    subscriber->input = input;
    input->batch_receive_nvtx_enabled |= subscriber->receive_nvtx_message != nullptr &&
                                         subscriber->source_identity_nvtx_string != nullptr;
    subscriber->endpoint = std::move(endpoint);
    subscriber->tensor_schema_endpoint = std::move(tensor_schema_endpoint);
    if (tensor_endpoint) {
      subscriber->tensor_endpoint.emplace(std::move(*tensor_endpoint));
    }
    if (descriptor_tensor_service) {
      subscriber->catalog_schema_identity = catalog_record->service_static.schema_identity;
      auto fields = external_descriptor_tensor_storage_contracts(
          plan, topic.id, boundary_port, service->tensor_fields);
      if (!fields || replay->type_ops->descriptor_tensor_static_count != fields->size()) {
        return make_unexpected(fields ? Error{ErrorCode::kProtocolError}
                                      : std::move(fields).error());
      }
      subscriber->tensor_fields = std::move(*fields);
      subscriber->tensor_logical_fields = service->tensor_fields;
      subscriber->descriptor_tensor_scratch.resize(
          replay->type_ops->descriptor_tensor_static_count);
      subscriber->tensor_schema_scratch.resize(replay->type_ops->descriptor_tensor_static_count);
    } else if (tensor_service) {
      subscriber->tensor_max_byte_span = *catalog_record->service_static.tensor->max_byte_span;
      subscriber->tensor_memory_kind = *catalog_record->service_static.tensor->memory_kind;
      subscriber->tensor_device =
          tensor_interface_pool != nullptr ? tensor_interface_pool->device : DeviceId{};
      subscriber->tensor_logical_requirements = *catalog_record->service_static.tensor;
    }
    subscriber->catalog_capture_clock = capture_clock_binding->catalog;
    subscriber->plan_capture_clock = capture_clock_binding->plan;
    // Admit the plan-frozen retained-generation capacity for this import. A missing or
    // non-restart-capable record keeps every rebind attempt failing closed with kNotSupported;
    // the catalog copy shares immutable state so a later rebind can re-resolve the same record.
    subscriber->rebind_catalog = plan.endpoint_catalog;
    subscriber->rebind_endpoint_name = std::string{catalog_endpoint->name()};
    subscriber->import_generation_capacity = generation_capacity->endpoint_max_generations;
    subscriber->import_restart_capable = generation_capacity->restart_capable_minimum_two;
    if (monitored) {
      subscriber->clock_health_endpoint = boundary_port;
      subscriber->health_dispatch_budget = health_plan->per_dispatch_report_budget;
    }
    if (!tensor_service || descriptor_tensor_service) {
      auto scratch =
          AlignedBlock::create(replay->type_ops->wire_capacity, alignof(std::max_align_t));
      if (!scratch) {
        return make_unexpected(std::move(scratch).error());
      }
      subscriber->scratch = std::move(*scratch);
      subscriber->scratch_size = replay->type_ops->wire_capacity;
    }
    if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
      if (!port_plan->payload_identity.pod) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      subscriber->pod_identity = port_plan->payload_identity.pod;
    }
    const std::size_t tensor_fields_per_slot =
        descriptor_tensor_service ? replay->type_ops->descriptor_tensor_static_count : 1U;
    auto prepared_storage = impl->prepare_subscriber_storage(
        *subscriber, typed_slot_capacity, typed_slot_capacity, tensor_fields_per_slot, true);
    if (!prepared_storage) {
      return make_unexpected(std::move(prepared_storage).error());
    }
    input->subscribers.push_back(subscriber.get());
    impl->subscribers_.push_back(std::move(subscriber));
  }
  return {};
}

// Phase 6: publish input bindings and indexes only after every subscriber and retained slot is
// owned. Session publication remains the final operation after this phase succeeds.
expected<void, Error>
RuntimeChannelSession::Impl::MaterializationTransaction::publish_input_bindings_and_indexes() {
  const auto& plan = plan_;
  auto& impl = impl_;

  for (const auto& input : impl->inputs_) {
    if (input->type_ops->predicate_view_size == 0U ||
        input->type_ops->predicate_view_alignment == 0U ||
        input->type_ops->construct_predicate_view == nullptr ||
        input->type_ops->destroy_predicate_view == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto predicate_storage = AlignedBlock::create(input->type_ops->predicate_view_size,
                                                  input->type_ops->predicate_view_alignment);
    if (!predicate_storage) {
      return make_unexpected(std::move(predicate_storage).error());
    }
    input->predicate_view_storage = std::move(*predicate_storage);
    std::size_t claim_capacity = 0U;
    for (const auto* subscriber : input->subscribers) {
      if (claim_capacity >
          std::numeric_limits<std::size_t>::max() - subscriber->read_claim_capacity) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      claim_capacity += subscriber->read_claim_capacity;
    }
    auto claim_pool = ReadClaimPool::create(claim_capacity);
    if (!claim_pool) {
      return make_unexpected(std::move(claim_pool).error());
    }
    input->claim_pool = *claim_pool;
    input->type_ops->bind_input(input->declaration, &input->binding);
  }
  auto pump_indices = impl->initialize_pump_indices();
  if (!pump_indices) {
    return make_unexpected(std::move(pump_indices).error());
  }
  auto activation_cut_index = impl->initialize_activation_cut_subscriber_index();
  if (!activation_cut_index) {
    return make_unexpected(std::move(activation_cut_index).error());
  }
  auto activation_owners = impl->bind_activation_owners(plan.ports.size());
  if (!activation_owners) {
    return make_unexpected(std::move(activation_owners).error());
  }
  impl->freeze_descriptor_import_inventory();
  impl->ready_handles_.resize(impl->wait_attachment_count_);
  return {};
}

}  // namespace holoscan::detail
