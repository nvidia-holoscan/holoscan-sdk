/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/execution_context.hpp>

#include "parameter_dso_test_types.hpp"

namespace holoscan::test {

expected<void, Error> DSOParameterOperator::compute(ExecutionContext& context) {
  observed_params_ = &context.params<DSOParams>();
  return {};
}

expected<void, Error> DSOOpaqueParameterOperator::compute(ExecutionContext&) {
  return {};
}

expected<void, Error> DSOContractParameterOperator::compute(ExecutionContext&) {
  return {};
}

expected<std::uint64_t, Error> dso_shared_byte_span_projector(
    const DSOProjectorParams& params) noexcept {
  return params.bytes;
}

void DSOProjectedOperator::setup(OperatorSpec& spec) {
  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          parameter_projected_byte_span<&dso_shared_byte_span_projector>(
              "holoscan.test.shared-dso-projector.v1", param_ref<&DSOProjectorParams::bytes>()));
}

expected<void, Error> DSOProjectedOperator::compute(ExecutionContext&) {
  return {};
}

DSOParameterOperator* create_dso_parameter_operator() {
  return new DSOParameterOperator{};
}

void destroy_dso_parameter_operator(DSOParameterOperator* op) noexcept {
  delete op;
}

const void* dso_aggregate_type_token() noexcept {
  return param_type_descriptor<DSOParams>().local_type_token;
}

const void* dso_field_type_token() noexcept {
  return param_schema_descriptor<DSOParams>()
      ->fields.front()
      .local_operations.local_field_type_token;
}

const void* dso_opaque_aggregate_type_token() noexcept {
  return param_type_descriptor<DSOOpaqueParams>().local_type_token;
}

DSOParameterGraph::DSOParameterGraph() : Graph("dso-parameter-graph") {}

void DSOParameterGraph::compose() {
  parameter_operator_ = op<DSOParameterOperator>("op");
  opaque_parameter_operator_ = op<DSOOpaqueParameterOperator>("opaque");
  contract_parameter_operator_ = op<DSOContractParameterOperator>("contract");
}

OperatorHandle<DSOParameterOperator> DSOParameterGraph::parameter_operator() const {
  return parameter_operator_.value();
}

OperatorHandle<DSOOpaqueParameterOperator> DSOParameterGraph::opaque_parameter_operator() const {
  return opaque_parameter_operator_.value();
}

OperatorHandle<DSOContractParameterOperator> DSOParameterGraph::contract_parameter_operator()
    const {
  return contract_parameter_operator_.value();
}

DSOParameterGraph* create_dso_parameter_graph() {
  return new DSOParameterGraph{};
}

void destroy_dso_parameter_graph(DSOParameterGraph* graph) noexcept {
  delete graph;
}

OperatorHandle<DSOParameterOperator> dso_parameter_operator_handle(DSOParameterGraph* graph) {
  (void)graph->view();
  return graph->parameter_operator();
}

OperatorHandle<DSOOpaqueParameterOperator> dso_opaque_parameter_operator_handle(
    DSOParameterGraph* graph) {
  (void)graph->view();
  return graph->opaque_parameter_operator();
}

OperatorHandle<DSOContractParameterOperator> dso_contract_parameter_operator_handle(
    DSOParameterGraph* graph) {
  (void)graph->view();
  return graph->contract_parameter_operator();
}

}  // namespace holoscan::test
