/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_memory_kind_derivation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/operator_spec.hpp>

#include "../core/memory_kind_validation.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/tensor_contract_resolution_internal.hpp"
#include "derivation_replay.hpp"
#include "diagnostic_detail.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kDerivationInvalid = "PHYSICAL_TENSOR_MEMORY_KIND_DERIVATION_INVALID";
constexpr std::string_view kDependencyInvalid = "PHYSICAL_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID";
constexpr std::string_view kParameterMutabilityInvalid =
    "PHYSICAL_TENSOR_MEMORY_KIND_PARAMETER_MUTABILITY_INVALID";
constexpr std::string_view kParameterTypeInvalid =
    "PHYSICAL_TENSOR_MEMORY_KIND_PARAMETER_TYPE_INVALID";
constexpr std::string_view kMappingInvalid = "PHYSICAL_TENSOR_MEMORY_KIND_MAPPING_INVALID";
constexpr std::string_view kValueInvalid = "PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID";
constexpr std::string_view kContractConflict = "PHYSICAL_TENSOR_MEMORY_KIND_CONTRACT_CONFLICT";

struct Candidate {
  VertexId owner{};
  PortId output{};
  std::optional<std::uint16_t> field_id;
  std::optional<std::uint16_t> tensor_index;
  OperatorInstanceId operator_id{};
  std::optional<TensorContractRecord> contract;
  const TensorMemoryKindExpressionRecord* expression{};
  const TensorMemoryKindExpression* authored_expression{};
  std::optional<TensorMemoryKindExpression> owned_authored_expression;
  const void* initial_parameters{};
  const void* parameter_type_token{};
};

struct CandidateKey {
  PortId output{};
  std::optional<std::uint16_t> field_id;
  std::optional<std::uint16_t> tensor_index;

  friend bool operator==(const CandidateKey&, const CandidateKey&) = default;
};

enum class ReplayDisposition : std::uint8_t { kSkip, kReady, kFatal };

[[nodiscard]] auto candidate_key_tuple(const CandidateKey& key) noexcept {
  return std::tuple{key.output.value(), key.field_id, key.tensor_index};
}

[[nodiscard]] bool candidate_key_less(const CandidateKey& left,
                                      const CandidateKey& right) noexcept {
  return candidate_key_tuple(left) < candidate_key_tuple(right);
}

[[nodiscard]] DiagnosticSource source_for(const Candidate& candidate) noexcept {
  return DiagnosticSource{.vertex = candidate.owner, .port = candidate.output};
}

[[nodiscard]] std::string expression_pointer(const Candidate& candidate) {
  std::string pointer = "/ports/" + std::to_string(candidate.output.value());
  if (candidate.field_id) {
    pointer += "/tensor_fields/" + std::to_string(*candidate.field_id);
  }
  pointer += "/tensor_memory_kind_expression";
  return pointer;
}

void reject(TensorMemoryKindDerivationLowering& result, std::string_view token,
            const Candidate& candidate, std::string required = {}, std::string provided = {}) {
  result.diagnostics.push_back(PlanLoweringDiagnostic{
      .token = std::string{token},
      .source = source_for(candidate),
      .json_pointer = expression_pointer(candidate),
      .required = std::move(required),
      .provided = std::move(provided),
  });
  auto& detail = result.diagnostics.back().detail.emplace(DiagnosticDetail{
      .operation = DiagnosticOperation::kCompile, .error = Error{ErrorCode::kSuccess}});
  const auto* expression = candidate.owned_authored_expression
                               ? &*candidate.owned_authored_expression
                               : candidate.authored_expression;
  if (expression == nullptr || candidate.expression == nullptr) {
    return;
  }
  const auto authored = OperatorSpecCompilerAccess::tensor_memory_kind_expression(*expression);
  if (!authored.parameter || authored.parameter.has_sensitive_member()) {
    return;
  }
  append_derivation_context(detail, "parameter_schema", candidate.expression->parameter_schema_id);
  append_derivation_context(detail, "parameter_path", candidate.expression->parameter_path);
  try {
    append_derivation_context(
        detail, "mapping_entries", std::to_string(candidate.expression->mapping.size()));
  } catch (...) {
    detail.truncated = true;
  }
}

[[nodiscard]] const PortRecord* find_owned_port(const GraphSnapshotData& snapshot, VertexId owner,
                                                PortDirection direction,
                                                std::string_view name) noexcept {
  const auto found =
      std::find_if(snapshot.ports.begin(), snapshot.ports.end(), [&](const auto& port) {
        return port.owner == owner && port.direction == direction && port.name == name;
      });
  return found == snapshot.ports.end() ? nullptr : &*found;
}

[[nodiscard]] PortRecord* find_mutable_port(GraphSnapshotData& snapshot, PortId output) noexcept {
  const auto found =
      std::find_if(snapshot.ports.begin(), snapshot.ports.end(), [&](const auto& port) {
        return port.id.value == output.value();
      });
  return found == snapshot.ports.end() ? nullptr : &*found;
}

[[nodiscard]] const ParameterFieldPlanRecord* find_parameter_owner_field(
    std::span<const ParameterFieldPlanRecord> fields, OperatorInstanceId owner,
    const ParamFieldDescriptor& root) noexcept {
  const auto found = std::find_if(fields.begin(), fields.end(), [&](const auto& field) {
    return field.owner.id == owner && field.field == root.id && field.full_id == root.full_id &&
           field.canonical_name == root.canonical_name && field.logical_type == root.logical_type;
  });
  return found == fields.end() ? nullptr : &*found;
}

template <typename T>
[[nodiscard]] const T* optional_value_pointer(const std::optional<T>& value) noexcept {
  return value ? &*value : nullptr;
}

[[nodiscard]] bool port_has_memory_kind_derivation(const PortRecord& port) noexcept {
  return port.setup.tensor_memory_kind_expression.has_value() ||
         std::any_of(
             port.setup.tensor_fields.begin(),
             port.setup.tensor_fields.end(),
             [](const auto& field) { return field.tensor_memory_kind_expression.has_value(); });
}

[[nodiscard]] std::vector<VertexId> memory_kind_derivation_owners(
    const GraphSnapshotData& snapshot) {
  std::vector<VertexId> result;
  for (const auto& port : snapshot.ports) {
    if (port_has_memory_kind_derivation(port)) {
      result.push_back(port.owner);
    }
  }
  std::sort(result.begin(), result.end(), [](VertexId left, VertexId right) {
    return left.value < right.value;
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

[[nodiscard]] ReplayDisposition replay_derivation_operator(
    const ExecutableOperatorFactoryRecord& entry, DerivationReplay& replay,
    TensorMemoryKindDerivationLowering& result) {
  if (!entry.factory || entry.operator_id == 0U || !entry.vertex) {
    result.diagnostics.push_back(PlanLoweringDiagnostic{
        .token = std::string{kDerivationInvalid},
        .source = DiagnosticSource{.vertex = entry.vertex},
    });
    return ReplayDisposition::kSkip;
  }
  DiagnosticCapture capture;
  const auto outcome = replay_derivation(*entry.factory, replay, capture);
  if (outcome == DerivationReplayOutcome::kReady) {
    return ReplayDisposition::kReady;
  }
  if (outcome == DerivationReplayOutcome::kConstructionFailed) {
    result.diagnostics.push_back(PlanLoweringDiagnostic{
        .token = std::string{kDerivationInvalid},
        .source = DiagnosticSource{.vertex = entry.vertex},
    });
    attach_diagnostic_capture(
        result.diagnostics.back().detail, capture, DiagnosticOperation::kConstruct);
    return ReplayDisposition::kSkip;
  }
  const Error& declaration_error = OperatorSpecCompilerAccess::declaration_error(*replay.spec);
  result.diagnostics.push_back(PlanLoweringDiagnostic{
      .token = std::string{kDerivationInvalid},
      .source = DiagnosticSource{.vertex = entry.vertex},
      .required = "successful Operator setup replay",
      .provided = outcome == DerivationReplayOutcome::kSetupThrew ? "setup threw"
                  : declaration_error.message.empty()
                      ? std::to_string(static_cast<std::uint32_t>(declaration_error.code))
                      : declaration_error.message,
  });
  attach_diagnostic_capture(result.diagnostics.back().detail, capture, DiagnosticOperation::kSetup);
  return ReplayDisposition::kFatal;
}

void append_port_candidate(const ExecutableOperatorFactoryRecord& entry, const PortRecord& output,
                           const OperatorSpecCompilerAccess::PortMetadata& metadata,
                           std::vector<Candidate>& candidates,
                           TensorMemoryKindDerivationLowering& result) {
  if (!metadata.tensor_memory_kind_expression) {
    return;
  }
  if (output.setup.tensor_memory_kind_expression != metadata.tensor_memory_kind_expression) {
    result.diagnostics.push_back(PlanLoweringDiagnostic{
        .token = std::string{kDependencyInvalid},
        .source = DiagnosticSource{.vertex = entry.vertex, .port = PortId{output.id.value}},
    });
    return;
  }
  candidates.push_back(Candidate{
      .owner = entry.vertex,
      .output = PortId{output.id.value},
      .operator_id = entry.operator_id,
      .contract = resolve_output_tensor_contract(output.setup.tensor_contract,
                                                 output.graph_override.tensor_contract),
      .expression = optional_value_pointer(output.setup.tensor_memory_kind_expression),
      .authored_expression = metadata.authored_tensor_memory_kind_expression,
      .initial_parameters = entry.factory->initial_parameter_value(),
      .parameter_type_token = entry.factory->parameter_type().local_type_token,
  });
}

void append_field_candidates(const ExecutableOperatorFactoryRecord& entry, const PortRecord& output,
                             const OperatorSpecCompilerAccess::PortMetadata& metadata,
                             std::vector<Candidate>& candidates,
                             TensorMemoryKindDerivationLowering& result) {
  for (const auto& authored_field : metadata.tensor_fields) {
    if (!authored_field.tensor_memory_kind_expression ||
        !authored_field.configuration.tensor_memory_kind_expression) {
      continue;
    }
    const auto& replay_field = authored_field.configuration;
    const TensorFieldConfigurationRecord* graph_field{};
    if (metadata.descriptor_tensor_field_count != 1U || !output.setup.tensor_fields.empty()) {
      const auto found = std::find_if(output.setup.tensor_fields.begin(),
                                      output.setup.tensor_fields.end(),
                                      [&](const auto& field) {
                                        return field.field_id == replay_field.field_id &&
                                               field.tensor_index == replay_field.tensor_index;
                                      });
      if (found != output.setup.tensor_fields.end()) {
        graph_field = &*found;
      }
    }
    const TensorContractRecord* contract =
        graph_field != nullptr ? optional_value_pointer(graph_field->tensor_contract)
                               : optional_value_pointer(output.setup.tensor_contract);
    const std::optional<TensorContractRecord> effective_contract =
        graph_field == nullptr
            ? resolve_output_tensor_contract(output.setup.tensor_contract,
                                             output.graph_override.tensor_contract)
            : (contract != nullptr ? std::optional<TensorContractRecord>{*contract} : std::nullopt);
    const TensorMemoryKindExpressionRecord* expression =
        graph_field != nullptr ? optional_value_pointer(graph_field->tensor_memory_kind_expression)
                               : optional_value_pointer(output.setup.tensor_memory_kind_expression);
    if (expression == nullptr ||
        *expression != replay_field.tensor_memory_kind_expression.value()) {
      result.diagnostics.push_back(PlanLoweringDiagnostic{
          .token = std::string{kDependencyInvalid},
          .source = DiagnosticSource{.vertex = entry.vertex, .port = PortId{output.id.value}},
      });
      continue;
    }
    candidates.push_back(Candidate{
        .owner = entry.vertex,
        .output = PortId{output.id.value},
        .field_id = replay_field.field_id,
        .tensor_index = replay_field.tensor_index,
        .operator_id = entry.operator_id,
        .contract = effective_contract,
        .expression = expression,
        .owned_authored_expression = authored_field.tensor_memory_kind_expression,
        .initial_parameters = entry.factory->initial_parameter_value(),
        .parameter_type_token = entry.factory->parameter_type().local_type_token,
    });
  }
}

void collect_candidates(const ExecutableOperatorFactoryRecord& entry,
                        const GraphSnapshotData& snapshot, const OperatorSpec& spec,
                        std::vector<Candidate>& candidates,
                        TensorMemoryKindDerivationLowering& result) {
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    if (metadata.direction != PortDirection::kOutput) {
      continue;
    }
    const bool has_derivation =
        metadata.tensor_memory_kind_expression.has_value() ||
        std::any_of(
            metadata.tensor_fields.begin(), metadata.tensor_fields.end(), [](const auto& field) {
              return field.tensor_memory_kind_expression.has_value();
            });
    if (!has_derivation) {
      continue;
    }
    const PortRecord* output =
        find_owned_port(snapshot, entry.vertex, metadata.direction, metadata.name);
    if (output == nullptr) {
      result.diagnostics.push_back(PlanLoweringDiagnostic{
          .token = std::string{kDependencyInvalid},
          .source = DiagnosticSource{.vertex = entry.vertex},
      });
      continue;
    }
    append_port_candidate(entry, *output, metadata, candidates, result);
    append_field_candidates(entry, *output, metadata, candidates, result);
  }
}

// Expected, replayed, missing, and duplicate candidate identities are compared in one pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void validate_candidate_coverage(const GraphSnapshotData& snapshot,
                                 std::span<const Candidate> candidates,
                                 TensorMemoryKindDerivationLowering& result) {
  std::vector<CandidateKey> expected;
  std::vector<CandidateKey> replayed;
  for (const auto& port : snapshot.ports) {
    if (port.setup.tensor_memory_kind_expression) {
      expected.push_back(CandidateKey{.output = PortId{port.id.value}});
    }
    for (const auto& field : port.setup.tensor_fields) {
      if (field.tensor_memory_kind_expression) {
        expected.push_back(CandidateKey{
            .output = PortId{port.id.value},
            .field_id = field.field_id,
            .tensor_index = field.tensor_index,
        });
      }
    }
  }
  replayed.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const PortRecord* const port = find_port(snapshot, GraphPortId{candidate.output.value()});
    const bool collapsed_singular = port != nullptr &&
                                    port->setup.tensor_memory_kind_expression.has_value() &&
                                    port->setup.tensor_fields.empty();
    replayed.push_back(CandidateKey{
        .output = candidate.output,
        .field_id = collapsed_singular ? std::nullopt : candidate.field_id,
        .tensor_index = collapsed_singular ? std::nullopt : candidate.tensor_index,
    });
  }
  std::sort(expected.begin(), expected.end(), candidate_key_less);
  std::sort(replayed.begin(), replayed.end(), candidate_key_less);
  if (expected == replayed) {
    return;
  }
  for (const auto& key : expected) {
    const auto matches =
        std::equal_range(replayed.begin(), replayed.end(), key, candidate_key_less);
    if (std::distance(matches.first, matches.second) == 1) {
      continue;
    }
    const PortRecord* const port = find_port(snapshot, GraphPortId{key.output.value()});
    const Candidate missing{
        .owner = port != nullptr ? port->owner : VertexId{},
        .output = key.output,
        .field_id = key.field_id,
        .tensor_index = key.tensor_index,
    };
    reject(result, kDerivationInvalid, missing, "exactly one replay candidate");
  }
  for (std::size_t index = 0U; index < replayed.size(); ++index) {
    const auto& key = replayed[index];
    if ((index > 0U && key == replayed[index - 1U]) ||
        std::binary_search(expected.begin(), expected.end(), key, candidate_key_less)) {
      continue;
    }
    const PortRecord* const port = find_port(snapshot, GraphPortId{key.output.value()});
    const Candidate unexpected{
        .owner = port != nullptr ? port->owner : VertexId{},
        .output = key.output,
        .field_id = key.field_id,
        .tensor_index = key.tensor_index,
    };
    reject(result, kDerivationInvalid, unexpected, "no undeclared replay candidates");
  }
}

[[nodiscard]] bool validate_parameter_field(const Candidate& candidate,
                                            const ParamFieldDescriptor& field,
                                            TensorMemoryKindDerivationPlanRecord& record,
                                            TensorMemoryKindDerivationLowering& result) {
  if (field.mutability != ParamMutability::kCompileTime &&
      field.mutability != ParamMutability::kRecompile) {
    reject(result,
           kParameterMutabilityInvalid,
           candidate,
           "compile-time or recompile-required",
           candidate.expression->parameter_path);
    return false;
  }
  if (field.sensitivity == ParamSensitivity::kSensitive) {
    reject(result, kDependencyInvalid, candidate, "non-sensitive memory-kind Parameter");
    return false;
  }
  if (field.mutability == ParamMutability::kRecompile) {
    record.update_disposition = TensorMemoryKindDerivationUpdateDisposition::kRecompileRequired;
  }
  return true;
}

[[nodiscard]] bool validate_mapping_against_field(
    const Candidate& candidate, const ParamFieldDescriptor& field,
    const OperatorSpecCompilerAccess::TensorMemoryKindAuthoringView& authored,
    TensorMemoryKindDerivationLowering& result) {
  if (authored.form == TensorMemoryKindExpressionForm::kDirect) {
    if (authored.parameter.source != TensorMemoryKindParameterSource::kMemoryKind ||
        !candidate.expression->mapping.empty() || !valid_direct_memory_kind_domain(field)) {
      reject(result,
             kParameterTypeInvalid,
             candidate,
             "exact MemoryKind Parameter with a nonempty concrete enum domain");
      return false;
    }
    return true;
  }
  if (authored.form != TensorMemoryKindExpressionForm::kMapped || authored.mapping.empty() ||
      authored.mapping.size() != candidate.expression->mapping.size()) {
    reject(result, kMappingInvalid, candidate, "complete canonical Boolean or enum mapping");
    return false;
  }
  if (authored.parameter.source == TensorMemoryKindParameterSource::kBoolean) {
    const auto covers_boolean_value = [&](bool value) {
      const ParamScalarValue source = ParamScalarValue::from(value);
      return std::any_of(candidate.expression->mapping.begin(),
                         candidate.expression->mapping.end(),
                         [&](const auto& row) { return row.source == source; });
    };
    if (field.logical_type != ParamLogicalType::kBool ||
        candidate.expression->mapping.size() != 2U || !covers_boolean_value(false) ||
        !covers_boolean_value(true)) {
      reject(result, kParameterTypeInvalid, candidate, "Boolean mapping source");
      return false;
    }
    return true;
  }
  if (authored.parameter.source != TensorMemoryKindParameterSource::kEnum ||
      field.enum_choices.count == 0U ||
      field.enum_choices.count != candidate.expression->mapping.size()) {
    reject(result, kParameterTypeInvalid, candidate, "schema-declared enum mapping source");
    return false;
  }
  for (std::size_t index = 0U; index < field.enum_choices.count; ++index) {
    const auto& choice = field.enum_choices.values[index];
    if (std::none_of(candidate.expression->mapping.begin(),
                     candidate.expression->mapping.end(),
                     [&](const auto& row) { return row.source == choice.value; })) {
      reject(result, kMappingInvalid, candidate, "one row per admitted enum value");
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<MemoryKind> evaluate_candidate(
    const Candidate& candidate, std::span<const ParameterFieldPlanRecord> fields,
    PlanGenerationId plan_generation, TensorMemoryKindDerivationPlanRecord& record,
    TensorMemoryKindDerivationLowering& result) {
  const TensorMemoryKindExpression* authored_expression =
      candidate.owned_authored_expression ? &*candidate.owned_authored_expression
                                          : candidate.authored_expression;
  if (!candidate.contract || candidate.expression == nullptr || authored_expression == nullptr ||
      candidate.contract->storage != TensorOutputStorage::kRuntimePool ||
      candidate.contract->representation.memory_kind || plan_generation == 0U) {
    reject(result, kContractConflict, candidate, "unresolved runtime-pool memory kind");
    return std::nullopt;
  }
  const auto authored =
      OperatorSpecCompilerAccess::tensor_memory_kind_expression(*authored_expression);
  if (!authored.parameter || authored.parameter.root_type_token != candidate.parameter_type_token) {
    reject(result, kDependencyInvalid, candidate, "one exact owning Parameter field");
    return std::nullopt;
  }
  const ParamSchemaDescriptor* schema = authored.parameter.schema();
  const ParamFieldDescriptor* root_field = authored.parameter.root();
  const ParamFieldDescriptor* field = authored.parameter.field();
  if (schema == nullptr || root_field == nullptr || field == nullptr ||
      authored.parameter.path() != candidate.expression->parameter_path ||
      schema->stable_id != candidate.expression->parameter_schema_id ||
      schema->version != candidate.expression->parameter_schema_version ||
      field->id != candidate.expression->parameter_field ||
      field->full_id != candidate.expression->parameter_full_id ||
      find_parameter_owner_field(fields, candidate.operator_id, *root_field) == nullptr) {
    reject(result, kDependencyInvalid, candidate, "one exact owning Parameter field");
    return std::nullopt;
  }
  if (authored.parameter.sensitive()) {
    reject(result, kDependencyInvalid, candidate, "non-sensitive memory-kind Parameter path");
    return std::nullopt;
  }
  if (!validate_parameter_field(candidate, *field, record, result) ||
      !validate_mapping_against_field(candidate, *field, authored, result)) {
    return std::nullopt;
  }

  ParamScalarValue source;
  if (!authored.parameter.read(candidate.initial_parameters, source) ||
      source.kind == ParamScalarKind::kNone) {
    reject(result, kValueInvalid, candidate, "one valid generation-zero scalar");
    return std::nullopt;
  }
  if (authored.form == TensorMemoryKindExpressionForm::kDirect) {
    MemoryKind memory_kind{MemoryKind::kUnknown};
    if (!source.try_as(memory_kind)) {
      reject(result, kParameterTypeInvalid, candidate, "exact MemoryKind Parameter");
      return std::nullopt;
    }
    if (!valid_concrete_memory_kind(memory_kind)) {
      reject(result, kValueInvalid, candidate, "concrete non-unknown memory kind");
      return std::nullopt;
    }
    return memory_kind;
  }
  const auto mapped = std::find_if(candidate.expression->mapping.begin(),
                                   candidate.expression->mapping.end(),
                                   [&](const auto& row) { return row.source == source; });
  if (mapped == candidate.expression->mapping.end() ||
      !valid_concrete_memory_kind(mapped->memory_kind)) {
    reject(result, kValueInvalid, candidate, "one admitted mapped generation-zero value");
    return std::nullopt;
  }
  return mapped->memory_kind;
}

[[nodiscard]] bool apply_resolved_memory_kind(GraphSnapshotData& snapshot,
                                              const Candidate& candidate, MemoryKind memory_kind) {
  PortRecord* output = find_mutable_port(snapshot, candidate.output);
  if (output == nullptr || output->graph_override.tensor_memory_kind_expression ||
      (output->graph_override.tensor_contract &&
       output->graph_override.tensor_contract->representation.memory_kind)) {
    return false;
  }
  TensorContractRecord* contract{};
  std::optional<TensorRepresentation>* layout{};
  if (candidate.field_id && !output->setup.tensor_fields.empty()) {
    const auto field = std::find_if(output->setup.tensor_fields.begin(),
                                    output->setup.tensor_fields.end(),
                                    [&](const auto& item) {
                                      return item.field_id == candidate.field_id &&
                                             item.tensor_index == candidate.tensor_index;
                                    });
    if (field == output->setup.tensor_fields.end() || !field->tensor_contract) {
      return false;
    }
    contract = &*field->tensor_contract;  // NOLINT(bugprone-unchecked-optional-access)
    layout = &field->tensor_layout;
  } else {
    if (!output->setup.tensor_contract) {
      return false;
    }
    contract = &*output->setup.tensor_contract;
    layout = &output->setup.tensor_layout;
  }
  if (contract->representation.memory_kind ||
      contract->storage != TensorOutputStorage::kRuntimePool) {
    return false;
  }
  contract->representation.memory_kind = memory_kind;
  contract->origins.memory_kind = TensorContractValueOrigin::kCompilerDerivedMemoryKind;
  if (!*layout) {
    *layout = TensorRepresentation{};
  }
  (*layout)->memory_kind = memory_kind;
  return true;
}

void evaluate_candidates(GraphSnapshotData& snapshot,
                         std::span<const ParameterFieldPlanRecord> parameter_fields,
                         PlanGenerationId plan_generation, const std::vector<Candidate>& candidates,
                         TensorMemoryKindDerivationLowering& result) {
  for (const auto& candidate : candidates) {
    TensorMemoryKindDerivationPlanRecord record{
        .output = candidate.output,
        .field_id = candidate.field_id,
        .tensor_index = candidate.tensor_index,
        .plan_generation = plan_generation,
        .run_generation = 0U,
    };
    const auto memory_kind =
        evaluate_candidate(candidate, parameter_fields, plan_generation, record, result);
    if (!memory_kind || !apply_resolved_memory_kind(snapshot, candidate, *memory_kind)) {
      if (memory_kind) {
        reject(result, kContractConflict, candidate, "one unresolved physical contract target");
      }
      continue;
    }
    record.parameter_schema_id = candidate.expression->parameter_schema_id;
    record.parameter_schema_version = candidate.expression->parameter_schema_version;
    record.parameter_field = candidate.expression->parameter_field;
    record.parameter_full_id = candidate.expression->parameter_full_id;
    record.parameter_member_full_ids = candidate.expression->parameter_member_full_ids;
    record.parameter_path = candidate.expression->parameter_path;
    record.form = candidate.expression->form;
    record.mapping = candidate.expression->mapping;
    record.resolved_memory_kind = *memory_kind;
    result.records.push_back(std::move(record));
  }
  std::sort(result.records.begin(), result.records.end(), [](const auto& left, const auto& right) {
    return std::tuple{left.output.value(), left.field_id, left.tensor_index} <
           std::tuple{right.output.value(), right.field_id, right.tensor_index};
  });
}

}  // namespace

bool has_tensor_memory_kind_derivations(const GraphSnapshotData& snapshot) noexcept {
  return std::any_of(snapshot.ports.begin(), snapshot.ports.end(), port_has_memory_kind_derivation);
}

TensorMemoryKindDerivationLowering lower_tensor_memory_kind_derivations(
    const ExecutableFactoryInventory& factories, GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const ParameterFieldPlanRecord> parameter_fields) {
  TensorMemoryKindDerivationLowering result;
  std::vector<DerivationReplay> replay_storage;
  std::vector<Candidate> candidates;
  const std::vector<VertexId> owners = memory_kind_derivation_owners(snapshot);
  replay_storage.reserve(factories.operators.size());
  for (const auto& entry : factories.operators) {
    if (!std::binary_search(
            owners.begin(), owners.end(), entry.vertex, [](VertexId left, VertexId right) {
              return left.value < right.value;
            })) {
      continue;
    }
    DerivationReplay replay;
    const ReplayDisposition disposition = replay_derivation_operator(entry, replay, result);
    if (disposition == ReplayDisposition::kSkip) {
      continue;
    }
    if (disposition == ReplayDisposition::kFatal) {
      return result;
    }
    collect_candidates(entry, snapshot, *replay.spec, candidates, result);
    replay_storage.push_back(std::move(replay));
  }
  validate_candidate_coverage(snapshot, candidates, result);
  evaluate_candidates(snapshot, parameter_fields, plan_generation, std::move(candidates), result);
  return result;
}

}  // namespace holoscan::detail
