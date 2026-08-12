/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "physical_plan_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan::detail {
namespace {

using namespace physical_plan_test;

// The Graph snapshot must deep-copy exact construction evidence rather than infer it from a POD
// layout or generated schema identity. Constructor-only POD publication remains admitted through
// emit(T), but does not fabricate an allocate() capability or a generic-loan pool.
// REQ: COMPILE-009
// REQ: COMPILE-047
// REQ: CHAN-074
// REQ: CHAN-075
// REQ: CHAN-120
TEST(CompilePhysicalPlanArtifact, FreezesTypedGenericLoanEvidenceWithoutFunctionPointers) {
  Graph graph{"generic-loan-evidence"};
  (void)graph.op<GenericLoanEvidenceOperator>("source");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto find_port_by_name = [&](std::string_view name) {
    return std::find_if(plan.ports().begin(), plan.ports().end(), [&](const auto& port) {
      return port.local_name == name;
    });
  };
  const auto pod = find_port_by_name("pod");
  const auto schema_port = find_port_by_name("schema");
  const auto constructor_only = find_port_by_name("constructor-only");
  ASSERT_NE(pod, plan.ports().end());
  ASSERT_NE(schema_port, plan.ports().end());
  ASSERT_NE(constructor_only, plan.ports().end());
  EXPECT_TRUE(pod->generic_loan_supported);
  EXPECT_EQ(pod->generic_loan_slot_size_bytes, sizeof(CompileLoanPod));
  EXPECT_TRUE(schema_port->generic_loan_supported);
  EXPECT_EQ(schema_port->generic_loan_slot_size_bytes, sizeof(schema::StringT));
  EXPECT_FALSE(constructor_only->generic_loan_supported);
  EXPECT_EQ(constructor_only->generic_loan_slot_size_bytes, 0U);

  const auto find_loan = [&](PortId output) {
    return std::find_if(plan.generic_output_loans().begin(),
                        plan.generic_output_loans().end(),
                        [&](const auto& loan) { return loan.output == output; });
  };
  const auto pod_loan = find_loan(pod->id);
  const auto schema_loan = find_loan(schema_port->id);
  ASSERT_NE(pod_loan, plan.generic_output_loans().end());
  ASSERT_NE(schema_loan, plan.generic_output_loans().end());
  EXPECT_EQ(pod_loan->trait_evidence, GenericLoanTraitEvidence::kNothrowDefaultConstructiblePOD);
  EXPECT_EQ(pod_loan->slot_size_bytes, sizeof(CompileLoanPod));
  EXPECT_EQ(schema_loan->trait_evidence,
            GenericLoanTraitEvidence::kNothrowDefaultConstructibleSchema);
  EXPECT_EQ(schema_loan->slot_size_bytes, sizeof(schema::StringT));
  EXPECT_EQ(find_loan(constructor_only->id), plan.generic_output_loans().end());
  EXPECT_NE(
      std::find_if(plan.publication_outcomes().begin(),
                   plan.publication_outcomes().end(),
                   [&](const auto& outcome) { return outcome.output == constructor_only->id; }),
      plan.publication_outcomes().end());
  EXPECT_NE(plan.json().find("\"generic_loan_supported\":true"), std::string::npos);
  EXPECT_NE(plan.json().find("\"trait_evidence\":1"), std::string::npos);
  EXPECT_NE(plan.json().find("\"trait_evidence\":2"), std::string::npos);
}

// Generated package evidence is deep-owned by the immutable plan and serialized in one canonical
// provenance block. Repeated schema ports coalesce exact evidence without retaining package views
// or callable addresses.
//
//   source(GenericLoanEvidenceOperator)   (isolated schema output; no edges)
//
// REQ: CHAN-121
TEST(CompilePhysicalPlanArtifact, PublishesCanonicalSchemaPackageProvenance) {
  Graph graph{"schema-package-provenance"};
  (void)graph.op<GenericLoanEvidenceOperator>("source");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.schema_packages().size(), 1U);

  const SchemaPackagePlanRecord expected =
      schema_package_plan_record(schema_package<schema::StringT>());
  EXPECT_EQ(plan.schema_packages().front(), expected);

  // When the pinned FlatBuffers tool or schema inputs change, rebuild
  // holoscan_flatbuffer_schemas_generated and update this exact golden from the failing plan JSON.
  constexpr std::string_view kCanonicalGolden{
      "\"schema_packages\":[{\"binary_schema_size\":304,\"flatc_build\":\"flatc "
      "25.12.19\",\"generator_format_version\":1,\"include_closure_sha256\":\"e414ba3347e21d"
      "c8490492ab2eba81b29b46f6384a0f8f8b7e3ba36066aedc6b\",\"ordered_flags\":[\"--bfbs-"
      "gen-embed\",\"--cpp\",\"--cpp-ptr-type\",\"std::shared_ptr\",\"--gen-object-api\","
      "\"--reflect-names\",\"--reflect-types\",\"--schema\",\"-o\",\"<generated-schema-dir>\","
      "\"-I\",\"<schema-root-0>\",\"<ordered-root-schema-files>\"],\"package_format_version\":1,"
      "\"runtime_abi\":{\"major\":25,\"minor\":12,\"revision\":19},\"schema_id\":\"d0a5cc971"
      "394c0e0cf0c846cf1839203\"}]"};
  EXPECT_NE(plan.json().find(kCanonicalGolden), std::string::npos) << plan.json();
}
// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-040
// REQ: COMPILE-048
TEST(CompilePhysicalPlanArtifact, SerializesEveryPublicPhysicalFamilyAndSemanticFieldCanonically) {
  auto first = physical_artifact_state();
  auto second = physical_artifact_state();
  ASSERT_FALSE(first.ports.empty());
  ASSERT_FALSE(first.interface_port_bindings.empty());
  ASSERT_FALSE(first.payload_edges.empty());
  ASSERT_FALSE(first.channel_locality.empty());
  ASSERT_FALSE(first.channel_services.empty());
  ASSERT_FALSE(first.channel_publishers.empty());
  ASSERT_FALSE(first.channel_subscribers.empty());
  ASSERT_FALSE(first.process_local_channel_services.empty());
  ASSERT_FALSE(first.process_local_channel_publishers.empty());
  ASSERT_FALSE(first.process_local_channel_subscribers.empty());
  ASSERT_EQ(first.process_local_channel_services.size(),
            second.process_local_channel_services.size());
  ASSERT_EQ(first.process_local_channel_publishers.size(),
            second.process_local_channel_publishers.size());
  ASSERT_EQ(first.process_local_channel_subscribers.size(),
            second.process_local_channel_subscribers.size());
  for (std::size_t index = 0U; index < first.process_local_channel_services.size(); ++index) {
    const auto& left = first.process_local_channel_services[index];
    const auto& right = second.process_local_channel_services[index];
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.service_id, right.service_id);
    EXPECT_EQ(left.payload_identity.kind, right.payload_identity.kind);
    EXPECT_EQ(left.payload_identity.payload_kind, right.payload_identity.payload_kind);
    ASSERT_EQ(left.payload_identity.scalar.has_value(), right.payload_identity.scalar.has_value());
    if (left.payload_identity.scalar) {
      EXPECT_EQ(left.payload_identity.scalar->kind, right.payload_identity.scalar->kind);
    }
    EXPECT_EQ(left.payload_size_bytes, right.payload_size_bytes);
    EXPECT_EQ(left.tensor_payload, right.tensor_payload);
    EXPECT_EQ(left.tensor_max_byte_span, right.tensor_max_byte_span);
    EXPECT_EQ(left.tensor_pool_id, right.tensor_pool_id);
    EXPECT_EQ(left.tensor_memory_kind, right.tensor_memory_kind);
    EXPECT_EQ(left.tensor_device, right.tensor_device);
    EXPECT_EQ(left.delivery_kind, right.delivery_kind);
    EXPECT_EQ(left.queue_policy, right.queue_policy);
    EXPECT_EQ(left.max_subscribers, right.max_subscribers);
    EXPECT_EQ(left.max_subscriber_queue_depth, right.max_subscriber_queue_depth);
    EXPECT_EQ(left.max_subscriber_borrowed_samples, right.max_subscriber_borrowed_samples);
    EXPECT_EQ(left.publisher_history_depth, right.publisher_history_depth);
    EXPECT_EQ(left.event_wake, right.event_wake);
  }
  for (std::size_t index = 0U; index < first.process_local_channel_publishers.size(); ++index) {
    const auto& left = first.process_local_channel_publishers[index];
    const auto& right = second.process_local_channel_publishers[index];
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.service_id, right.service_id);
    EXPECT_EQ(left.max_loaned_samples, right.max_loaned_samples);
    EXPECT_EQ(left.replacement_attribution_required, right.replacement_attribution_required);
  }
  for (std::size_t index = 0U; index < first.process_local_channel_subscribers.size(); ++index) {
    const auto& left = first.process_local_channel_subscribers[index];
    const auto& right = second.process_local_channel_subscribers[index];
    EXPECT_EQ(left.edge, right.edge);
    EXPECT_EQ(left.source, right.source);
    EXPECT_EQ(left.target, right.target);
    EXPECT_EQ(left.service_id, right.service_id);
    EXPECT_EQ(left.requested_queue_depth, right.requested_queue_depth);
    EXPECT_EQ(left.max_borrowed_samples, right.max_borrowed_samples);
    EXPECT_EQ(left.cache_capacity, right.cache_capacity);
    EXPECT_EQ(left.readiness_capacity, right.readiness_capacity);
    EXPECT_EQ(left.event_wake, right.event_wake);
  }
  ASSERT_FALSE(first.generic_output_loans.empty());
  ASSERT_FALSE(first.tensor_allocations.empty());
  ASSERT_FALSE(first.tensor_interface_pools.empty());
  ASSERT_FALSE(first.tensor_device_resources.empty());
  ASSERT_FALSE(first.tensor_inline_records.empty());
  ASSERT_FALSE(first.imported_generation_capacities.empty());
  ASSERT_FALSE(first.tensor_transfer_plans.empty());
  ASSERT_FALSE(first.pool_capacity_contributors.empty());
  ASSERT_FALSE(first.publication_outcomes.empty());
  ASSERT_FALSE(first.aligned_triggers.empty());
  ASSERT_FALSE(first.clock_triggers.empty());
  ASSERT_FALSE(first.deadlines.empty());
  ASSERT_FALSE(first.readable_triggers.empty());
  ASSERT_FALSE(first.missing_aggregates.empty());
  EXPECT_EQ(first.generic_output_loans.front().plan_generation, 71U);
  EXPECT_EQ(first.generic_output_loans.front().run_generation, 0U);
  EXPECT_EQ(first.tensor_allocations.front().plan_generation, 71U);
  EXPECT_EQ(first.tensor_allocations.front().run_generation, 0U);
  EXPECT_EQ(first.tensor_interface_pools.front().plan_generation, 71U);
  EXPECT_EQ(first.tensor_interface_pools.front().run_generation, 0U);
  EXPECT_EQ(first.tensor_device_resources.front().plan_generation, 71U);
  EXPECT_EQ(first.tensor_device_resources.front().run_generation, 0U);

  const auto first_finalized = finalize_plan_artifact(first);
  const auto second_finalized = finalize_plan_artifact(second);
  ASSERT_TRUE(first_finalized) << first_finalized.error().message;
  ASSERT_TRUE(second_finalized) << second_finalized.error().message;
  EXPECT_EQ(first.digest, second.digest);
  EXPECT_EQ(first.json, second.json);

  constexpr std::array required_tokens{
      "\"ports\"",
      "\"interface_port_bindings\"",
      "\"payload_edges\"",
      "\"channel_locality\"",
      "\"channel_services\"",
      "\"channel_publishers\"",
      "\"channel_subscribers\"",
      "\"generic_output_loans\"",
      "\"tensor_allocations\"",
      "\"tensor_device_resources\"",
      "\"tensor_inline_records\"",
      "\"tensor_interface_pools\"",
      "\"imported_generation_capacities\"",
      "\"tensor_transfer_plans\"",
      "\"pool_capacity_contributors\"",
      "\"publication_outcomes\"",
      "\"aligned_triggers\"",
      "\"clock_triggers\"",
      "\"deadlines\"",
      "\"readable_triggers\"",
      "\"missing_aggregates\"",
      "\"local_name\"",
      "\"owner_kind\"",
      "\"late_joiner_history_depth\"",
      "\"tensor_layout\"",
      "\"internal_port\"",
      "\"public_port\"",
      "\"max_inflight_samples\"",
      "\"producer_max_emits_per_compute\"",
      "\"fanout\"",
      "\"cache_capacity\"",
      "\"requires_runtime_attach\"",
      "\"max_subscriber_queue_depth\"",
      "\"max_subscriber_borrowed_samples\"",
      "\"publisher_history_depth\"",
      "\"tensor_emit_policy\"",
      "\"max_loaned_samples\"",
      "\"replacement_attribution_required\"",
      "\"requested_queue_depth\"",
      "\"max_borrowed_samples\"",
      "\"trait_evidence\"",
      "\"max_live_loans\"",
      "\"materialize_before\"",
      "\"pre_submit_preserves_same_activation_loan\"",
      "\"plan_generation\"",
      "\"run_generation\"",
      "\"allocator_root_id\"",
      "\"allocator_process_group\"",
      "\"aggregate_byte_capacity\"",
      "\"max_outstanding_loans\"",
      "\"control_capacity\"",
      "\"readiness_capacity\"",
      "\"release_capacity\"",
      "\"retirement_capacity\"",
      "\"quarantine_capacity\"",
      "\"exporter_subpool_id\"",
      "\"logical\"",
      "\"publisher_loan_capacity\"",
      "\"stream_id\"",
      "\"fused_linear_edges\"",
      "\"pool_count\"",
      "\"per_partition_stream\"",
      "\"per_partition_event\"",
      "\"per_partition_green_context\"",
      "\"gpu_resident_backend_selected\"",
      "\"fixed_record_bucket_id\"",
      "\"fixed_record_bucket_bytes\"",
      "\"fixed_metadata_overhead_bytes\"",
      "\"provider_max_inline_logical_bytes\"",
      "\"profile_max_inline_logical_bytes\"",
      "\"effective_max_inline_logical_bytes\"",
      "\"accepted_max_inline_logical_bytes\"",
      "\"record_format_id\"",
      "\"record_format_version\"",
      "\"inline_discriminator\"",
      "\"opaque_descriptor_discriminator\"",
      "\"inline_arm_schema_id\"",
      "\"opaque_descriptor_arm_schema_id\"",
      "\"endpoint_max_generations\"",
      "\"service_max_generations\"",
      "\"runtime_max_generations\"",
      "\"no_wrap_aba_generation\"",
      "\"attach_max_wait_ns\"",
      "\"peer_requirement\"",
      "\"candidate_reservation_capacity\"",
      "\"release_record_capacity\"",
      "\"quarantine_record_capacity\"",
      "\"source_class\"",
      "\"realization\"",
      "\"min_logical_bytes\"",
      "\"max_logical_bytes\"",
      "\"delivered_read_only\"",
      "\"contributor_id\"",
      "\"maximum_distinct_slots\"",
      "\"pre_submit_error\"",
      "\"post_submit_error\"",
      "\"post_submit_partial_error\"",
      "\"post_submit_consumes_authority\"",
      "\"producer_loan\"",
      "\"retirement_service\"",
  };
  for (const std::string_view token : required_tokens) {
    EXPECT_NE(first.json.find(token), std::string::npos) << token;
  }
  EXPECT_NE(first.json.find("\"post_submit_error\":22"), std::string::npos);
  EXPECT_NE(first.json.find("\"post_submit_partial_error\":36"), std::string::npos);
  for (const std::string_view private_token : {"process_local_channel_services",
                                               "process_local_channel_publishers",
                                               "process_local_channel_subscribers"}) {
    EXPECT_EQ(first.json.find(private_token), std::string::npos) << private_token;
  }
}

// Package provenance may honestly differ between processes that use the same pinned schema
// identity. The public diagnostic artifact must show that difference while the semantic plan
// digest remains stable.
// REQ: CHAN-121
TEST(CompilePhysicalPlanArtifact, SchemaPackageEvidenceDoesNotEnterSemanticDigest) {
  auto first = physical_artifact_state();
  auto second = physical_artifact_state();
  SchemaPackagePlanRecord first_record =
      schema_package_plan_record(schema_package<schema::StringT>());
  SchemaPackagePlanRecord second_record = first_record;
  second_record.flatc_build = "flatc independently-reconstructed";
  second_record.ordered_flags.push_back("--diagnostic-provenance-only");
  second_record.generator_format_version = 7U;
  first.schema_packages.push_back(std::move(first_record));
  second.schema_packages.push_back(std::move(second_record));

  const auto first_finalized = finalize_plan_artifact(first);
  const auto second_finalized = finalize_plan_artifact(second);

  ASSERT_TRUE(first_finalized) << first_finalized.error().message;
  ASSERT_TRUE(second_finalized) << second_finalized.error().message;
  EXPECT_EQ(first.digest, second.digest);
  EXPECT_NE(first.json, second.json);
  EXPECT_NE(first.json.find("\"flatc_build\":\"flatc 25.12.19\""), std::string::npos);
  EXPECT_NE(second.json.find("\"flatc_build\":\"flatc independently-reconstructed\""),
            std::string::npos);
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-040
// REQ: COMPILE-048
TEST(CompilePhysicalPlanArtifact, EveryPublicPhysicalFamilyPerturbsSemanticIdentity) {
  auto baseline = physical_artifact_state();
  const auto baseline_finalized = finalize_plan_artifact(baseline);
  ASSERT_TRUE(baseline_finalized) << baseline_finalized.error().message;
  const auto expect_change = [&](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_TRUE(finalized) << finalized.error().message;
    EXPECT_NE(candidate.digest, baseline.digest);
    EXPECT_NE(candidate.json, baseline.json);
  };

  expect_change([](auto& state) { state.ports.front().local_name = "changed"; });
  expect_change([](auto& state) {
    state.interface_port_bindings.front().direction = PortDirection::kOutput;
  });
  expect_change([](auto& state) { ++state.payload_edges.front().max_inflight_samples; });
  expect_change([](auto& state) {
    // Locality acceptance and the payload-edge locality are one closed
    // compiler relation. Perturb the external edge coherently so this test
    // continues to exercise artifact identity rather than corruption rejection.
    state.payload_edges.back().locality = ChannelLocalityKind::kExternalExport;
    state.channel_locality.back().accepted = ChannelLocalityKind::kExternalExport;
  });
  expect_change([](auto& state) { ++state.channel_services.front().max_subscriber_queue_depth; });
  expect_change([](auto& state) { ++state.channel_publishers.front().max_loaned_samples; });
  expect_change([](auto& state) { ++state.channel_subscribers.front().max_borrowed_samples; });
  expect_change([](auto& state) { ++state.generic_output_loans.front().slot_count; });
  expect_change([](auto& state) { ++state.tensor_allocations.front().bounds.max_rank; });
  expect_change([](auto& state) { ++state.tensor_device_resources.front().slot_capacity; });
  expect_change([](auto& state) {
    state.tensor_device_resources.front().fused_linear_edges.push_back(EdgeId{9U});
  });
  expect_change([](auto& state) { ++*state.tensor_interface_pools.front().logical.max_byte_span; });
  expect_change(
      [](auto& state) { ++state.tensor_inline_records.front().fixed_record_bucket_bytes; });
  expect_change(
      [](auto& state) { ++state.tensor_inline_records.front().provider_max_inline_logical_bytes; });
  expect_change(
      [](auto& state) { state.tensor_inline_records.front().record_format_version += 1U; });
  expect_change(
      [](auto& state) { ++state.imported_generation_capacities.front().runtime_max_generations; });
  expect_change(
      [](auto& state) { state.imported_generation_capacities.front().attach_max_wait_ns = 1U; });
  expect_change(
      [](auto& state) { state.tensor_transfer_plans.front().delivered_read_only = false; });
  expect_change([](auto& state) { ++state.tensor_transfer_plans.front().min_logical_bytes; });
  expect_change(
      [](auto& state) { ++state.pool_capacity_contributors.front().maximum_distinct_slots; });
  expect_change(
      [](auto& state) { state.publication_outcomes.front().runtime_retry_enabled = true; });
  expect_change([](auto& state) {
    state.publication_outcomes.front().post_submit_error = ErrorCode::kFailure;
  });
  expect_change([](auto& state) {
    state.publication_outcomes.front().post_submit_partial_error = ErrorCode::kFailure;
  });
  expect_change([](auto& state) { ++state.aligned_triggers.front().inclusive_window_ns; });
  expect_change([](auto& state) { ++state.clock_triggers.front().period_ns; });
  expect_change([](auto& state) { ++state.deadlines.front().budget_ns; });
  expect_change([](auto& state) { ++state.readable_triggers.front().effective_capacity; });
  expect_change([](auto& state) { ++state.missing_aggregates.front().degraded_callback_count; });
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanArtifact, RejectsProcessLocalConstructionFactDrift) {
  const auto expect_rejection = [&](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  };

  expect_rejection(
      [](auto& state) { ++state.process_local_channel_services.front().payload_size_bytes; });
  expect_rejection(
      [](auto& state) { ++state.process_local_channel_publishers.front().max_loaned_samples; });
  expect_rejection(
      [](auto& state) { ++state.process_local_channel_subscribers.front().cache_capacity; });

  auto ineligible_direct = physical_artifact_state();
  ASSERT_EQ(ineligible_direct.process_local_channel_services.front().delivery_kind,
            ProcessLocalChannelDeliveryKind::kDirectRetained);
  ineligible_direct.payload_edges.front().fanout = 2U;
  const auto finalized = finalize_plan_artifact(ineligible_direct);
  ASSERT_FALSE(finalized);
  EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanArtifact, RejectsBrokenProcessLocalChannelJoins) {
  {
    auto missing_subscriber = physical_artifact_state();
    ASSERT_FALSE(missing_subscriber.process_local_channel_subscribers.empty());
    missing_subscriber.process_local_channel_subscribers.clear();
    const auto finalized = finalize_plan_artifact(missing_subscriber);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto missing_service_identity = physical_artifact_state();
    ASSERT_FALSE(missing_service_identity.process_local_channel_services.empty());
    missing_service_identity.process_local_channel_services.front().service_id.clear();
    const auto finalized = finalize_plan_artifact(missing_service_identity);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto duplicate_service = physical_artifact_state();
    ASSERT_FALSE(duplicate_service.process_local_channel_services.empty());
    duplicate_service.process_local_channel_services.push_back(
        duplicate_service.process_local_channel_services.front());
    const auto finalized = finalize_plan_artifact(duplicate_service);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto mismatched_payload = physical_artifact_state();
    ASSERT_FALSE(mismatched_payload.process_local_channel_services.empty());
    ASSERT_TRUE(mismatched_payload.process_local_channel_services.front().payload_identity.scalar);
    // Int32 has the same physical width as UInt32. The exact payload identity,
    // not only its byte count, is part of the private construction join.
    mismatched_payload.process_local_channel_services.front().payload_identity.scalar->kind =
        ScalarIdentityKind::kInt32;
    const auto finalized = finalize_plan_artifact(mismatched_payload);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto mismatched_source = physical_artifact_state();
    ASSERT_FALSE(mismatched_source.process_local_channel_publishers.empty());
    mismatched_source.process_local_channel_publishers.front().source = PortId{2U};
    const auto finalized = finalize_plan_artifact(mismatched_source);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto duplicate_publisher = physical_artifact_state();
    ASSERT_FALSE(duplicate_publisher.process_local_channel_publishers.empty());
    duplicate_publisher.process_local_channel_publishers.push_back(
        duplicate_publisher.process_local_channel_publishers.front());
    const auto finalized = finalize_plan_artifact(duplicate_publisher);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    auto duplicate_subscriber = physical_artifact_state();
    ASSERT_FALSE(duplicate_subscriber.process_local_channel_subscribers.empty());
    duplicate_subscriber.process_local_channel_subscribers.push_back(
        duplicate_subscriber.process_local_channel_subscribers.front());
    const auto finalized = finalize_plan_artifact(duplicate_subscriber);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
  {
    GraphSnapshotData snapshot;
    snapshot.generation = GraphGeneration{1U};
    snapshot.vertices = {vertex(1U, "tensor-source"), vertex(2U, "tensor-sink")};
    snapshot.ports = {
        output_port(1U, 1U, "out", tensor_payload()),
        input_port(2U, 2U, "in", tensor_payload(), 2U),
    };
    snapshot.ports.front().setup.tensor_layout = TensorPortLayout{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    snapshot.ports.front().setup.tensor_allocation =
        TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 64U};
    snapshot.edges = {edge(1U, 1U, 2U, 2U)};
    const auto profiles = standard_profiles(snapshot.vertices.size());
    auto physical = lower_for_test(snapshot, 37U, profiles);
    ASSERT_TRUE(physical.errors.empty());
    ASSERT_EQ(physical.tensor_allocations.size(), 1U);
    ASSERT_EQ(physical.process_local_channel_services.size(), 1U);

    ExecutionPlanState ambiguous_tensor_pool;
    ambiguous_tensor_pool.plan_generation = 37U;
    detail::publish_physical_plan_records(std::move(physical), ambiguous_tensor_pool);
    ASSERT_TRUE(valid_process_local_channel_plan(ambiguous_tensor_pool));
    const auto& direct = ambiguous_tensor_pool.tensor_allocations.front();
    TensorInterfacePoolPlanRecord duplicate_pool;
    duplicate_pool.graph_port = direct.output;
    duplicate_pool.pool_id = direct.pool_id + ".ambiguous";
    duplicate_pool.logical.max_byte_span = direct.bounds.max_byte_span;
    duplicate_pool.memory_kind = direct.memory_kind;
    duplicate_pool.device = direct.device;
    ambiguous_tensor_pool.tensor_interface_pools.push_back(std::move(duplicate_pool));
    const auto finalized = finalize_plan_artifact(ambiguous_tensor_pool);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  }
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: CHAN-065
TEST(CompilePhysicalPlanArtifact, RejectsProcessLocalAndExternalLocalityJoinDrift) {
  const auto expect_rejection = [](auto mutate) {
    auto candidate = physical_artifact_state();
    mutate(candidate);
    const auto finalized = finalize_plan_artifact(candidate);
    ASSERT_FALSE(finalized);
    EXPECT_EQ(finalized.error().message, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID");
  };

  expect_rejection([](auto& state) {
    const auto found = std::find_if(
        state.channel_locality.begin(), state.channel_locality.end(), [](const auto& row) {
          return row.accepted == ChannelLocalityKind::kProcessLocal;
        });
    ASSERT_NE(found, state.channel_locality.end());
    found->accepted = ChannelLocalityKind::kExternalExport;
    found->requires_runtime_attach = true;
  });
  expect_rejection([](auto& state) {
    const auto found = std::find_if(
        state.channel_locality.begin(), state.channel_locality.end(), [](const auto& row) {
          return row.accepted != ChannelLocalityKind::kProcessLocal;
        });
    ASSERT_NE(found, state.channel_locality.end());
    found->accepted = ChannelLocalityKind::kProcessLocal;
    found->requires_runtime_attach = false;
  });
}

// REQ: COMPILE-002
// REQ: COMPILE-040
// REQ: COMPILE-048
TEST(CompilePhysicalPlanArtifact, RebindsProcessLocalServiceIdentityAsOneJoinFamily) {
  auto state = physical_artifact_state();
  ASSERT_FALSE(state.process_local_channel_services.empty());
  ASSERT_FALSE(state.tensor_device_resources.empty());
  const std::string original_stream_id = state.tensor_device_resources.front().stream_id;
  state.profile_manifest.derived_capacities.push_back(ProfileDerivedCapacityRecord{
      .object_id = original_stream_id,
      .capacity_id = "slot_count",
      .limit = state.tensor_device_resources.front().slot_capacity,
      .unit = "count",
  });
  state.profile_manifest.materialization_obligations.push_back(
      ProfileMaterializationObligationRecord{
          .object_id = original_stream_id,
          .strategy_id = "holoscan.runtime.tensor_device_stream/v1",
          .complete_before_stage = LifecycleStage::kArm,
          .cleanup_stage_mask = 0xC0U,
          .max_cleanup_actions = 1U,
          .evidence_id = "holoscan.compile.tensor_device_resource_plan/v1",
      });

  rebind_plan_generation(state, 99U);

  const auto& service = state.process_local_channel_services.front();
  EXPECT_EQ(service.service_id, "process-local-channel.99.source.1.policy.0");
  ASSERT_FALSE(state.process_local_channel_publishers.empty());
  ASSERT_FALSE(state.process_local_channel_subscribers.empty());
  EXPECT_EQ(state.process_local_channel_publishers.front().service_id, service.service_id);
  EXPECT_EQ(state.process_local_channel_subscribers.front().service_id, service.service_id);
  EXPECT_NE(state.tensor_device_resources.front().stream_id, original_stream_id);
  ASSERT_EQ(state.profile_manifest.derived_capacities.size(), 1U);
  EXPECT_EQ(state.profile_manifest.derived_capacities.front().object_id,
            state.tensor_device_resources.front().stream_id);
  ASSERT_EQ(state.profile_manifest.materialization_obligations.size(), 1U);
  EXPECT_EQ(state.profile_manifest.materialization_obligations.front().object_id,
            state.tensor_device_resources.front().stream_id);
  const auto finalized = finalize_plan_artifact(state);
  ASSERT_TRUE(finalized) << finalized.error().message;
}

// REQ: COMPILE-009
// REQ: COMPILE-040
TEST(CompilePhysicalPlanManifest, PublishesFiniteUniqueCanonicalPhysicalRequirements) {
  Graph graph{"physical-manifest"};
  const auto source = graph.op<ManifestTensorOperator>("source");
  constexpr std::array axes{TensorAxisBounds{.max_extent = 16U},
                            TensorAxisBounds{.max_extent = 16U}};
  TopicOptions options;
  options.tensor = LogicalTensorRequirements{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 2U,
      .logical_bounds = LogicalTensorBounds{.max_rank = 2U, .max_byte_span = 4096U, .axes = axes},
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
      .max_inline_bytes = 1024U,
  };
  auto exported = graph.export_topic<Tensor>("holomq://manifest-tensor", options);
  graph.add_flow(source->tensor, exported.input());

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.generic_output_loans().size(), 1U);
  ASSERT_EQ(plan.tensor_allocations().size(), 1U);
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  EXPECT_TRUE(plan.tensor_device_resources().empty());
  ASSERT_EQ(plan.tensor_inline_records().size(), 1U);
  const auto& manifest = plan.profile_manifest();

  EXPECT_TRUE(std::is_sorted(manifest.required_capabilities.begin(),
                             manifest.required_capabilities.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.capability_id, lhs.version} <
                                      std::pair{rhs.capability_id, rhs.version};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.derived_capacities.begin(),
                             manifest.derived_capacities.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.object_id, lhs.capacity_id} <
                                      std::pair{rhs.object_id, rhs.capacity_id};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.materialization_obligations.begin(),
                             manifest.materialization_obligations.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.object_id, lhs.strategy_id} <
                                      std::pair{rhs.object_id, rhs.strategy_id};
                             }));
  EXPECT_EQ(std::adjacent_find(manifest.required_capabilities.begin(),
                               manifest.required_capabilities.end(),
                               [](const auto& lhs, const auto& rhs) {
                                 return lhs.capability_id == rhs.capability_id &&
                                        lhs.version == rhs.version;
                               }),
            manifest.required_capabilities.end());
  EXPECT_EQ(std::adjacent_find(manifest.derived_capacities.begin(),
                               manifest.derived_capacities.end(),
                               [](const auto& lhs, const auto& rhs) {
                                 return lhs.object_id == rhs.object_id &&
                                        lhs.capacity_id == rhs.capacity_id;
                               }),
            manifest.derived_capacities.end());
  EXPECT_TRUE(std::all_of(manifest.derived_capacities.begin(),
                          manifest.derived_capacities.end(),
                          [](const auto& capacity) {
                            return capacity.limit < std::numeric_limits<std::uint64_t>::max();
                          }));
  const auto has_capability = [&](std::string_view id) {
    return std::any_of(manifest.required_capabilities.begin(),
                       manifest.required_capabilities.end(),
                       [&](const auto& capability) { return capability.capability_id == id; });
  };
  EXPECT_TRUE(has_capability("holoscan.runtime.payload_edges"));
  EXPECT_TRUE(has_capability("holoscan.runtime.channel_service_attach"));
  EXPECT_TRUE(has_capability("holoscan.runtime.generic_output_loans"));
  EXPECT_TRUE(has_capability("holoscan.runtime.tensor_pool_materialization"));
  EXPECT_FALSE(has_capability("holoscan.runtime.tensor_device_stream_materialization"));
  EXPECT_TRUE(has_capability("holoscan.runtime.tensor_transfer_matrix"));
  EXPECT_TRUE(has_capability(kTensorInlineRecordCapabilityId));
  const auto& inline_record = plan.tensor_inline_records().front();
  EXPECT_TRUE(std::any_of(manifest.derived_capacities.begin(),
                          manifest.derived_capacities.end(),
                          [&](const auto& capacity) {
                            return capacity.object_id == inline_record.fixed_record_bucket_id &&
                                   capacity.capacity_id == "fixed_record_bytes" &&
                                   capacity.limit == inline_record.fixed_record_bucket_bytes;
                          }));
  EXPECT_TRUE(std::any_of(manifest.materialization_obligations.begin(),
                          manifest.materialization_obligations.end(),
                          [&](const auto& obligation) {
                            return obligation.object_id == inline_record.fixed_record_bucket_id &&
                                   obligation.strategy_id ==
                                       "holoscan.runtime.tensor_inline_record_bucket/v1" &&
                                   obligation.evidence_id == inline_record.capability_evidence_id;
                          }));
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlanManifest, AggregatesMultipleImportedTensorGenerationScopes) {
  Graph graph{"multi-import-generation-manifest"};
  test::add_compile_test_default_clock(graph);
  const auto tensor_options = [] {
    TopicOptions options;
    options.tensor = LogicalTensorRequirements{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
        .logical_bounds =
            LogicalTensorBounds{
                .max_rank = 1U,
                .max_byte_span = 16U,
            },
        .foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable,
    };
    return options;
  };

  auto first_options = tensor_options();
  first_options.endpoint_name = "shared-first";
  auto second_options = tensor_options();
  second_options.endpoint_name = "shared-second";
  auto third_options = tensor_options();
  third_options.endpoint_name = "distinct-third";
  auto first = graph.import_topic<Tensor>("holomq://shared-import", first_options);
  auto second = graph.import_topic<Tensor>("holomq://shared-import", second_options);
  auto third = graph.import_topic<Tensor>("holomq://distinct-import", third_options);
  const auto first_sink = graph.op<ManifestTensorSinkOperator>("first-sink");
  const auto second_sink = graph.op<ManifestTensorSinkOperator>("second-sink");
  const auto third_sink = graph.op<ManifestTensorSinkOperator>("third-sink");
  graph.add_flow(first.output(), first_sink->input);
  graph.add_flow(second.output(), second_sink->input);
  graph.add_flow(third.output(), third_sink->input);
  graph.partition("first-import").add(first_sink);
  graph.partition("second-import").add(second_sink);
  graph.partition("third-import").add(third_sink);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.imported_generation_capacities().size(), 3U);

  std::vector<std::string> services;
  std::vector<std::string> runtime_scopes;
  for (const auto& generation : plan.imported_generation_capacities()) {
    services.push_back(generation.endpoint_or_service_id);
    runtime_scopes.push_back(generation.runtime_scope_id);
    EXPECT_EQ(generation.runtime_max_generations, 3U);
  }
  std::sort(services.begin(), services.end());
  services.erase(std::unique(services.begin(), services.end()), services.end());
  std::sort(runtime_scopes.begin(), runtime_scopes.end());
  runtime_scopes.erase(std::unique(runtime_scopes.begin(), runtime_scopes.end()),
                       runtime_scopes.end());
  EXPECT_EQ(services.size(), 3U);
  ASSERT_EQ(runtime_scopes.size(), 1U);

  const auto& capacities = plan.profile_manifest().derived_capacities;
  const auto capacity_count = [&](std::string_view id) {
    return std::count_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
      return capacity.capacity_id == id;
    });
  };
  EXPECT_EQ(capacity_count("endpoint_generations"), 3);
  EXPECT_EQ(capacity_count("candidate_generation_reservations"), 3);
  EXPECT_EQ(capacity_count("generation_release_records"), 3);
  EXPECT_EQ(capacity_count("generation_quarantine_records"), 3);
  EXPECT_EQ(capacity_count("service_generations"), 3);
  EXPECT_EQ(capacity_count("runtime_generations"), 1);
  const auto runtime_capacity =
      std::find_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
        return capacity.object_id == runtime_scopes.front() &&
               capacity.capacity_id == "runtime_generations";
      });
  ASSERT_NE(runtime_capacity, capacities.end());
  EXPECT_EQ(runtime_capacity->limit, 3U);

  const auto& materializations = plan.profile_manifest().materialization_obligations;
  EXPECT_EQ(std::count_if(materializations.begin(),
                          materializations.end(),
                          [](const auto& item) {
                            return item.strategy_id == "holoscan.runtime.channel_service_attach/v1";
                          }),
            3);
}

// REQ: COMPILE-040
TEST(CompilePhysicalPlanManifest, DeduplicatesSharedServiceCapacityProjection) {
  ExecutionPlanState state;
  state.channel_services = {
      ChannelServiceStaticPlanRecord{
          .topic = ExternalTopicId{1U},
          .service_id = "shared-service",
          .max_subscriber_queue_depth = 4U,
          .max_subscriber_borrowed_samples = 3U,
      },
      ChannelServiceStaticPlanRecord{
          .topic = ExternalTopicId{2U},
          .service_id = "shared-service",
          .max_subscriber_queue_depth = 4U,
          .max_subscriber_borrowed_samples = 3U,
      },
  };
  state.imported_generation_capacities = {
      ImportedGenerationCapacityPlanRecord{
          .topic = ExternalTopicId{1U},
          .runtime_scope_id = "runtime-scope",
          .endpoint_or_service_id = "shared-service",
          .endpoint_max_generations = 1U,
          .service_max_generations = 2U,
          .runtime_max_generations = 2U,
          .release_record_capacity = 1U,
          .quarantine_record_capacity = 1U,
      },
      ImportedGenerationCapacityPlanRecord{
          .topic = ExternalTopicId{2U},
          .runtime_scope_id = "runtime-scope",
          .endpoint_or_service_id = "shared-service",
          .endpoint_max_generations = 2U,
          .service_max_generations = 2U,
          .runtime_max_generations = 2U,
          .candidate_reservation_capacity = 1U,
          .release_record_capacity = 2U,
          .quarantine_record_capacity = 2U,
      },
  };
  auto& manifest = state.profile_manifest;
  manifest.disposition = ProfileAdmissionDisposition::kAccepted;
  manifest.requested_profile = Profile::kStandard;
  manifest.effective_graph_profile = Profile::kStandard;
  manifest.profile_policy_id = profile_policy_id_v1();
  manifest.plan_schema_version = "holoscan.execution_plan.private.v1";
  manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  manifest.target_triple = std::string(compile_target_triple());

  const auto finalized = finalize_profile_manifest(CompileOptions{}, state);
  ASSERT_TRUE(finalized) << finalized.error().message;
  const auto& capacities = state.profile_manifest.derived_capacities;
  const auto matching_capacities = [&](std::string_view object_id, std::string_view capacity_id) {
    return std::count_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
      return capacity.object_id == object_id && capacity.capacity_id == capacity_id;
    });
  };
  const auto capacity_limit = [&](std::string_view object_id, std::string_view capacity_id) {
    const auto found =
        std::find_if(capacities.begin(), capacities.end(), [&](const auto& capacity) {
          return capacity.object_id == object_id && capacity.capacity_id == capacity_id;
        });
    EXPECT_NE(found, capacities.end());
    return found == capacities.end() ? 0U : found->limit;
  };

  EXPECT_EQ(matching_capacities("shared-service", "max_subscriber_queue_depth"), 1);
  EXPECT_EQ(capacity_limit("shared-service", "max_subscriber_queue_depth"), 4U);
  EXPECT_EQ(capacity_limit("shared-service", "max_subscriber_borrowed_samples"), 3U);
  EXPECT_EQ(capacity_limit("shared-service", "publisher_history_depth"), 0U);
  EXPECT_EQ(matching_capacities("shared-service", "service_generations"), 1);
  EXPECT_EQ(capacity_limit("shared-service", "service_generations"), 2U);
  EXPECT_EQ(matching_capacities("runtime-scope", "runtime_generations"), 1);
  EXPECT_EQ(capacity_limit("runtime-scope", "runtime_generations"), 2U);
  EXPECT_EQ(matching_capacities("imported-generation-endpoint:1", "endpoint_generations"), 1);
  EXPECT_EQ(matching_capacities("imported-generation-endpoint:2", "endpoint_generations"), 1);

  const auto& materializations = state.profile_manifest.materialization_obligations;
  EXPECT_EQ(std::count_if(materializations.begin(),
                          materializations.end(),
                          [](const auto& item) {
                            return item.object_id == "shared-service" &&
                                   item.strategy_id == "holoscan.runtime.channel_service_attach/v1";
                          }),
            1);
}

}  // namespace
}  // namespace holoscan::detail
// NOLINTEND(bugprone-unchecked-optional-access)
