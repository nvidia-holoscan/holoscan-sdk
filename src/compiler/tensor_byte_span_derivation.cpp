/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensor_byte_span_derivation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../core/operator_spec_compiler_access.hpp"
#include "../core/tensor_contract_resolution_internal.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kDerivationInvalid = "PHYSICAL_TENSOR_BYTE_SPAN_DERIVATION_INVALID";
constexpr std::string_view kParameterTypeInvalid =
    "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_TYPE_INVALID";
constexpr std::string_view kParameterMutabilityInvalid =
    "PHYSICAL_TENSOR_BYTE_SPAN_PARAMETER_MUTABILITY_INVALID";
constexpr std::string_view kValueInvalid = "PHYSICAL_TENSOR_BYTE_SPAN_VALUE_INVALID";
constexpr std::string_view kInputUnbounded = "PHYSICAL_TENSOR_BYTE_SPAN_INPUT_UNBOUNDED";
constexpr std::string_view kDependencyInvalid = "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID";
constexpr std::string_view kOverflow = "PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW";
constexpr std::string_view kContractConflict = "PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT";

struct Candidate {
  VertexId owner{};
  PortId output{};
  std::optional<std::uint16_t> field_id;
  std::optional<std::uint16_t> tensor_index;
  OperatorInstanceId operator_id{};
  std::optional<TensorContractRecord> contract;
  const TensorByteSpanExpressionRecord* expression{};
  const TensorByteSpanExpression* authored_expression{};
  std::optional<TensorByteSpanExpression> owned_authored_expression;
  const void* initial_parameters{};
  const void* parameter_type_token{};
  bool input_source_eligible{};
};

struct CandidateKey {
  PortId output{};
  std::optional<std::uint16_t> field_id;
  std::optional<std::uint16_t> tensor_index;

  friend bool operator==(const CandidateKey&, const CandidateKey&) = default;
};

[[nodiscard]] auto candidate_key_tuple(const CandidateKey& key) noexcept {
  return std::tuple{key.output.value(), key.field_id, key.tensor_index};
}

[[nodiscard]] bool candidate_key_less(const CandidateKey& left,
                                      const CandidateKey& right) noexcept {
  return candidate_key_tuple(left) < candidate_key_tuple(right);
}

struct ReplayStorage {
  std::unique_ptr<OperatorBase> instance;
  std::unique_ptr<OperatorSpec> spec;
};

enum class VisitState : std::uint8_t { kUnvisited, kVisiting, kComplete, kFailed };

[[nodiscard]] DiagnosticSource source_for(const Candidate& candidate) noexcept {
  return DiagnosticSource{.vertex = candidate.owner, .port = candidate.output};
}

[[nodiscard]] std::string expression_pointer(const Candidate& candidate,
                                             std::optional<std::size_t> node_index = std::nullopt) {
  std::string pointer = "/ports/" + std::to_string(candidate.output.value());
  if (candidate.field_id) {
    pointer += "/tensor_fields/" + std::to_string(*candidate.field_id);
  }
  pointer += "/tensor_byte_span_expression";
  if (node_index) {
    pointer += "/nodes/" + std::to_string(*node_index);
  }
  return pointer;
}

void reject(TensorByteSpanDerivationLowering& result, std::string_view token,
            const Candidate& candidate, std::optional<std::size_t> node_index = std::nullopt,
            std::string required = {}, std::string provided = {}) {
  result.diagnostics.push_back(PlanLoweringDiagnostic{
      .token = std::string{token},
      .source = source_for(candidate),
      .json_pointer = expression_pointer(candidate, node_index),
      .required = std::move(required),
      .provided = std::move(provided),
  });
}

[[nodiscard]] std::string node_evidence(const TensorByteSpanExpressionNodeRecord& node) {
  std::string evidence = "operation=" + std::to_string(static_cast<std::uint32_t>(node.operation));
  if (!node.parameter_path.empty()) {
    evidence += ",parameter=" + node.parameter_path;
  }
  if (!node.input_name.empty()) {
    evidence += ",input=" + node.input_name;
  }
  return evidence;
}

[[nodiscard]] bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                    std::uint64_t& result) noexcept {
  if (left == 0U || right == 0U || left > std::numeric_limits<std::uint64_t>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool checked_ceiling_divide(std::uint64_t numerator, std::uint64_t denominator,
                                          std::uint64_t& result) noexcept {
  if (numerator == 0U || denominator == 0U) {
    return false;
  }
  result = numerator / denominator + (numerator % denominator != 0U ? 1U : 0U);
  return result != 0U;
}

[[nodiscard]] bool checked_align_up(std::uint64_t value, std::uint64_t alignment,
                                    std::uint64_t& result) noexcept {
  if (value == 0U || alignment == 0U || (alignment & (alignment - 1U)) != 0U ||
      value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U)) {
    return false;
  }
  result = (value + alignment - 1U) & ~(alignment - 1U);
  return result != 0U;
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

[[nodiscard]] const ParameterFieldPlanRecord* find_parameter_field(
    std::span<const ParameterFieldPlanRecord> fields, OperatorInstanceId owner,
    const TensorByteSpanExpressionNodeRecord& node) noexcept {
  const auto found = std::find_if(fields.begin(), fields.end(), [&](const auto& field) {
    return field.owner.id == owner && field.field == node.parameter_field &&
           field.full_id == node.parameter_full_id && field.canonical_name == node.parameter_path;
  });
  return found == fields.end() ? nullptr : &*found;
}

template <typename T>
[[nodiscard]] const T* optional_value_pointer(const std::optional<T>& value) noexcept {
  return value.has_value() ? &value.value() : nullptr;
}

void validate_candidate_coverage(const GraphSnapshotData& snapshot,
                                 std::span<const Candidate> candidates,
                                 TensorByteSpanDerivationLowering& result) {
  std::vector<CandidateKey> expected;
  std::vector<CandidateKey> replayed;
  for (const auto& port : snapshot.ports) {
    if (port.setup.tensor_byte_span_expression) {
      expected.push_back(CandidateKey{.output = PortId{port.id.value}});
    }
    for (const auto& field : port.setup.tensor_fields) {
      if (field.tensor_byte_span_expression) {
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
                                    port->setup.tensor_byte_span_expression.has_value() &&
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
    reject(result, kDerivationInvalid, missing, std::nullopt, "exactly one replay candidate");
  }
}

class Evaluator {
 public:
  Evaluator(const GraphSnapshotData& snapshot, std::span<const ParameterFieldPlanRecord> fields,
            std::vector<Candidate> candidates, TensorByteSpanDerivationLowering& result)
      : snapshot_(snapshot), fields_(fields), candidates_(std::move(candidates)), result_(result) {
    states_.resize(candidates_.size());
    resolved_.resize(candidates_.size());
    dispositions_.resize(candidates_.size());
    for (std::size_t index = 0U; index < candidates_.size(); ++index) {
      if (candidates_[index].input_source_eligible) {
        candidate_by_output_.emplace(candidates_[index].output.value(), index);
      }
    }
  }

  void evaluate_all(PlanGenerationId plan_generation) {
    for (std::size_t index = 0U; index < candidates_.size(); ++index) {
      (void)evaluate(index, plan_generation);
    }
    std::sort(
        result_.records.begin(), result_.records.end(), [](const auto& left, const auto& right) {
          return std::tuple{left.output.value(), left.field_id, left.tensor_index} <
                 std::tuple{right.output.value(), right.field_id, right.tensor_index};
        });
  }

 private:
  struct NodeEvaluation {
    std::uint64_t value{};
    PortId input{};
  };

  void reject_node(std::string_view token, const Candidate& candidate, std::size_t node_index,
                   std::string required, const TensorByteSpanExpressionNodeRecord& normalized) {
    reject(result_, token, candidate, node_index, std::move(required), node_evidence(normalized));
  }

  [[nodiscard]] const PortRecord* resolve_input_producer(
      const Candidate& candidate, const TensorByteSpanExpressionNodeRecord& node,
      std::size_t node_index, PortId& input_id) {
    const auto reject_input = [&](std::string required) {
      reject_node(kDependencyInvalid, candidate, node_index, std::move(required), node);
    };
    const PortRecord* input =
        find_owned_port(snapshot_, candidate.owner, PortDirection::kInput, node.input_name);
    if (input == nullptr) {
      reject_input("one owning input declaration");
      return nullptr;
    }
    input_id = PortId{input->id.value};
    const EdgeRecord* edge{};
    for (const auto& candidate_edge : snapshot_.edges) {
      if (resolve_interface_port(snapshot_, candidate_edge.target) != input->id) {
        continue;
      }
      if (edge != nullptr) {
        reject_input("one incident producer edge");
        return nullptr;
      }
      edge = &candidate_edge;
    }
    if (edge == nullptr) {
      reject_input("one incident producer edge");
      return nullptr;
    }
    const PortRecord* producer =
        find_port(snapshot_, resolve_interface_port(snapshot_, edge->source));
    if (producer == nullptr) {
      reject_input("one resolved producer output");
      return nullptr;
    }
    if (find_vertex(snapshot_, candidate.owner) == nullptr ||
        find_vertex(snapshot_, producer->owner) == nullptr) {
      reject_input("resolved producer and consumer ownership");
      return nullptr;
    }
    return producer;
  }

  [[nodiscard]] std::optional<std::uint64_t> input_capacity(
      const Candidate& candidate, const TensorByteSpanExpressionNodeRecord& node,
      std::size_t node_index, PlanGenerationId plan_generation, PortId& input_id,
      TensorByteSpanDerivationUpdateDisposition& update_disposition) {
    const auto reject_input = [&](std::string_view token, std::string required) {
      reject(result_, token, candidate, node_index, std::move(required), node_evidence(node));
    };
    const PortRecord* producer = resolve_input_producer(candidate, node, node_index, input_id);
    if (producer == nullptr) {
      return std::nullopt;
    }
    std::optional<std::uint64_t> producer_bound;
    const auto derived = candidate_by_output_.find(producer->id.value);
    if (derived != candidate_by_output_.end()) {
      producer_bound = evaluate(derived->second, plan_generation);
      if (!producer_bound) {
        return std::nullopt;
      }
      if (dispositions_[derived->second] ==
          TensorByteSpanDerivationUpdateDisposition::kRecompileRequired) {
        update_disposition = TensorByteSpanDerivationUpdateDisposition::kRecompileRequired;
      }
    } else {
      const auto& allocation = producer->graph_override.tensor_allocation
                                   ? producer->graph_override.tensor_allocation
                                   : producer->setup.tensor_allocation;
      const auto contract = resolve_output_tensor_contract(
          producer->setup.tensor_contract, producer->graph_override.tensor_contract);
      if (allocation) {
        producer_bound = allocation->max_byte_span;
      } else if (contract && contract->max_byte_span) {
        producer_bound = contract->max_byte_span;
      }
      const auto imported =
          std::find_if(snapshot_.external_topics.begin(),
                       snapshot_.external_topics.end(),
                       [&](const auto& topic) {
                         return topic.imported &&
                                resolve_interface_port(snapshot_, topic.graph_port) == producer->id;
                       });
      // Both optionals are established by the catalog predicate before their values are copied.
      if (!producer_bound && imported != snapshot_.external_topics.end() && imported->tensor &&
          imported->tensor->max_byte_span) {  // NOLINT(bugprone-unchecked-optional-access)
        producer_bound =
            imported->tensor->max_byte_span;  // NOLINT(bugprone-unchecked-optional-access)
      }
    }
    if (!producer_bound) {
      reject_input(kInputUnbounded, "finite effective producer byte span");
      return std::nullopt;
    }
    return producer_bound;
  }

  [[nodiscard]] std::optional<std::uint64_t> parameter_capacity(
      const Candidate& candidate, const TensorContractRecord& contract,
      const TensorByteSpanAuthoringNode& authored,
      const TensorByteSpanExpressionNodeRecord& normalized, std::size_t node_index,
      TensorByteSpanDerivationPlanRecord& record) {
    const ParameterFieldPlanRecord* field =
        find_parameter_field(fields_, candidate.operator_id, normalized);
    if (field == nullptr || !authored.parameter ||
        authored.parameter.root_type_token != candidate.parameter_type_token) {
      reject_node(kDependencyInvalid,
                  candidate,
                  node_index,
                  "one exact owning Parameter field",
                  normalized);
      return std::nullopt;
    }
    if (field->mutability != ParamMutability::kCompileTime &&
        field->mutability != ParamMutability::kRecompile) {
      reject_node(kParameterMutabilityInvalid,
                  candidate,
                  node_index,
                  "compile-time or recompile-required",
                  normalized);
      return std::nullopt;
    }
    if (field->sensitivity == ParamSensitivity::kSensitive) {
      reject_node(kDependencyInvalid,
                  candidate,
                  node_index,
                  "non-sensitive capacity Parameter",
                  normalized);
      return std::nullopt;
    }
    if (field->mutability == ParamMutability::kRecompile) {
      record.update_disposition = TensorByteSpanDerivationUpdateDisposition::kRecompileRequired;
    }
    const bool shape = authored.operation == TensorByteSpanOperation::kParameterContiguousShape;
    if (shape && (field->constraints.max_elements == 0U ||
                  field->constraints.max_elements > kMaximumTensorByteSpanParameterExtents)) {
      reject_node(kParameterTypeInvalid,
                  candidate,
                  node_index,
                  "bounded integral shape with supported max_elements",
                  normalized);
      return std::nullopt;
    }

    TensorByteSpanParameterValue parameter_value;
    const auto status =
        authored.parameter.read_initial(candidate.initial_parameters, parameter_value);
    if (status == TensorByteSpanParameterReadStatus::kTypeInvalid) {
      reject_node(kParameterTypeInvalid,
                  candidate,
                  node_index,
                  "positive integral scalar or bounded shape",
                  normalized);
      return std::nullopt;
    }
    if (status != TensorByteSpanParameterReadStatus::kSuccess || parameter_value.count == 0U) {
      reject_node(
          kValueInvalid, candidate, node_index, "positive generation-zero value", normalized);
      return std::nullopt;
    }
    if (!shape) {
      if (parameter_value.count != 1U) {
        reject_node(kParameterTypeInvalid,
                    candidate,
                    node_index,
                    "one positive integral scalar",
                    normalized);
        return std::nullopt;
      }
      return parameter_value.values[0];
    }
    if (!contract.representation.rank || !contract.representation.dtype) {
      reject_node(kContractConflict,
                  candidate,
                  node_index,
                  "output contract with exact rank and dtype",
                  normalized);
      return std::nullopt;
    }
    if (parameter_value.count != contract.representation.rank.value()) {
      reject_node(kParameterTypeInvalid,
                  candidate,
                  node_index,
                  "shape length equal to output rank",
                  normalized);
      return std::nullopt;
    }
    std::uint64_t value = dlpack_dtype_item_size_bytes(contract.representation.dtype.value());
    for (std::size_t extent = 0U; extent < parameter_value.count; ++extent) {
      std::uint64_t next{};
      if (!checked_multiply(value, parameter_value.values[extent], next)) {
        reject_node(kOverflow,
                    candidate,
                    node_index,
                    "shape product representable as uint64 bytes",
                    normalized);
        return std::nullopt;
      }
      value = next;
    }
    return value;
  }

  [[nodiscard]] std::optional<NodeEvaluation> evaluate_node(
      const Candidate& candidate, const TensorContractRecord& contract,
      const TensorByteSpanAuthoringNode& authored,
      const TensorByteSpanExpressionNodeRecord& normalized, std::size_t node_index,
      PlanGenerationId plan_generation, std::span<const std::uint64_t> values,
      TensorByteSpanDerivationPlanRecord& record) {
    NodeEvaluation evaluated;
    switch (authored.operation) {
      case TensorByteSpanOperation::kPositiveConstant:
        evaluated.value = authored.value;
        break;
      case TensorByteSpanOperation::kParameterByteSpan:
      case TensorByteSpanOperation::kParameterContiguousShape: {
        const auto value =
            parameter_capacity(candidate, contract, authored, normalized, node_index, record);
        if (!value) {
          return std::nullopt;
        }
        evaluated.value = *value;
        break;
      }
      case TensorByteSpanOperation::kInputMaxByteSpan: {
        const auto value = input_capacity(candidate,
                                          normalized,
                                          node_index,
                                          plan_generation,
                                          evaluated.input,
                                          record.update_disposition);
        if (!value) {
          return std::nullopt;
        }
        evaluated.value = *value;
        break;
      }
      case TensorByteSpanOperation::kMultiply:
        if (authored.left >= values.size() || authored.right >= values.size() ||
            !checked_multiply(values[authored.left], values[authored.right], evaluated.value)) {
          reject_node(
              kOverflow, candidate, node_index, "checked positive multiplication", normalized);
          return std::nullopt;
        }
        break;
      case TensorByteSpanOperation::kCeilingDivide:
        if (authored.left >= values.size() || authored.right >= values.size() ||
            !checked_ceiling_divide(
                values[authored.left], values[authored.right], evaluated.value)) {
          reject_node(kValueInvalid,
                      candidate,
                      node_index,
                      "checked positive ceiling division",
                      normalized);
          return std::nullopt;
        }
        break;
      case TensorByteSpanOperation::kMaximum:
        if (authored.left >= values.size() || authored.right >= values.size()) {
          reject_node(
              kDerivationInvalid, candidate, node_index, "postorder child indices", normalized);
          return std::nullopt;
        }
        evaluated.value = std::max(values[authored.left], values[authored.right]);
        break;
      case TensorByteSpanOperation::kAlignUp:
        if (authored.left >= values.size() ||
            !checked_align_up(values[authored.left], authored.value, evaluated.value)) {
          reject_node(
              kOverflow, candidate, node_index, "checked power-of-two alignment", normalized);
          return std::nullopt;
        }
        break;
      default:
        reject_node(kDerivationInvalid,
                    candidate,
                    node_index,
                    "supported Tensor byte-span operation",
                    normalized);
        return std::nullopt;
    }
    if (evaluated.value == 0U || evaluated.value > std::numeric_limits<std::size_t>::max()) {
      reject_node(kValueInvalid,
                  candidate,
                  node_index,
                  "positive value representable as size_t",
                  normalized);
      return std::nullopt;
    }
    return evaluated;
  }

  [[nodiscard]] std::optional<std::uint64_t> evaluate(std::size_t index,
                                                      PlanGenerationId plan_generation) {
    if (states_[index] == VisitState::kComplete) {
      return resolved_[index];
    }
    Candidate& candidate = candidates_[index];
    if (states_[index] == VisitState::kVisiting) {
      reject(result_, kDependencyInvalid, candidate);
      states_[index] = VisitState::kFailed;
      return std::nullopt;
    }
    if (states_[index] == VisitState::kFailed) {
      return std::nullopt;
    }
    states_[index] = VisitState::kVisiting;
    const TensorByteSpanExpression* const authored_expression =
        candidate.owned_authored_expression ? &*candidate.owned_authored_expression
                                            : candidate.authored_expression;
    if (!candidate.contract || candidate.expression == nullptr || authored_expression == nullptr ||
        !candidate.contract->max_rank || candidate.contract->max_byte_span ||
        candidate.contract->storage != TensorOutputStorage::kRuntimePool) {
      reject(result_, kContractConflict, candidate);
      states_[index] = VisitState::kFailed;
      return std::nullopt;
    }
    const auto authored_nodes =
        OperatorSpecCompilerAccess::tensor_byte_span_expression_nodes(*authored_expression);
    if (authored_nodes.size() != candidate.expression->nodes.size() || authored_nodes.empty()) {
      reject(result_, kDerivationInvalid, candidate);
      states_[index] = VisitState::kFailed;
      return std::nullopt;
    }
    TensorByteSpanDerivationPlanRecord record{
        .output = candidate.output,
        .field_id = candidate.field_id,
        .tensor_index = candidate.tensor_index,
        .plan_generation = plan_generation,
        .run_generation = 0U,
    };
    record.nodes.reserve(authored_nodes.size());
    std::vector<std::uint64_t> values;
    values.reserve(authored_nodes.size());
    for (std::size_t node_index = 0U; node_index < authored_nodes.size(); ++node_index) {
      const auto& authored = authored_nodes[node_index];
      const auto& normalized = candidate.expression->nodes[node_index];
      const auto evaluated = evaluate_node(candidate,
                                           candidate.contract.value(),
                                           authored,
                                           normalized,
                                           node_index,
                                           plan_generation,
                                           values,
                                           record);
      if (!evaluated) {
        states_[index] = VisitState::kFailed;
        return std::nullopt;
      }
      values.push_back(evaluated->value);
      record.nodes.push_back(TensorByteSpanDerivationNodePlanRecord{
          .operation = authored.operation,
          .left = authored.left,
          .right = authored.right,
          .operand = authored.value,
          .parameter_field = normalized.parameter_field,
          .input = evaluated->input,
          .resolved_value = evaluated->value,
      });
    }
    record.resolved_max_byte_span = values.back();
    if (candidate.contract->transfer && candidate.contract->transfer->foreign_emit_policy !=
                                            InterfaceTensorEmitPolicy::kRequireExportable) {
      const std::uint64_t staging_byte_span =
          candidate.contract->transfer->max_staging_byte_span.value_or(values.back());
      record.resolved_max_staging_byte_span = staging_byte_span;
    }
    resolved_[index] = values.back();
    dispositions_[index] = record.update_disposition;
    result_.records.push_back(std::move(record));
    states_[index] = VisitState::kComplete;
    return resolved_[index];
  }

  const GraphSnapshotData& snapshot_;
  std::span<const ParameterFieldPlanRecord> fields_;
  std::vector<Candidate> candidates_;
  TensorByteSpanDerivationLowering& result_;
  std::unordered_map<std::uint64_t, std::size_t> candidate_by_output_;
  std::vector<VisitState> states_;
  std::vector<std::optional<std::uint64_t>> resolved_;
  std::vector<TensorByteSpanDerivationUpdateDisposition> dispositions_;
};

[[nodiscard]] bool owner_has_byte_span_derivation(const GraphSnapshotData& snapshot,
                                                  VertexId owner) {
  return std::any_of(snapshot.ports.begin(), snapshot.ports.end(), [&](const auto& port) {
    return port.owner == owner &&
           (port.setup.tensor_byte_span_expression.has_value() ||
            std::any_of(
                port.setup.tensor_fields.begin(),
                port.setup.tensor_fields.end(),
                [](const auto& field) { return field.tensor_byte_span_expression.has_value(); }));
  });
}

enum class ReplayDisposition : std::uint8_t { kSkip, kReady, kFatal };

[[nodiscard]] ReplayDisposition replay_derivation_operator(
    const ExecutableOperatorFactoryRecord& entry, ReplayStorage& replay,
    TensorByteSpanDerivationLowering& result) {
  if (!entry.factory || entry.operator_id == 0U || !entry.vertex) {
    result.diagnostics.push_back(PlanLoweringDiagnostic{
        .token = std::string{kDerivationInvalid},
        .source = DiagnosticSource{.vertex = entry.vertex},
    });
    return ReplayDisposition::kSkip;
  }
  auto created = entry.factory->create();
  if (!created || !created.value()) {
    result.diagnostics.push_back(PlanLoweringDiagnostic{
        .token = std::string{kDerivationInvalid},
        .source = DiagnosticSource{.vertex = entry.vertex},
    });
    return ReplayDisposition::kSkip;
  }
  replay = ReplayStorage{
      .instance = std::move(created.value()),
      .spec = std::make_unique<OperatorSpec>(),
  };
  try {
    replay.instance->setup(*replay.spec);
  } catch (...) {
    result.diagnostics.push_back(PlanLoweringDiagnostic{
        .token = std::string{kDerivationInvalid},
        .source = DiagnosticSource{.vertex = entry.vertex},
        .required = "successful Operator setup replay",
        .provided = "setup threw",
    });
    return ReplayDisposition::kFatal;
  }
  const Error& declaration_error = OperatorSpecCompilerAccess::declaration_error(*replay.spec);
  if (declaration_error.code == ErrorCode::kSuccess) {
    return ReplayDisposition::kReady;
  }
  result.diagnostics.push_back(PlanLoweringDiagnostic{
      .token = std::string{kDerivationInvalid},
      .source = DiagnosticSource{.vertex = entry.vertex},
      .required = "successful Operator setup replay",
      .provided = declaration_error.message.empty()
                      ? std::to_string(static_cast<std::uint32_t>(declaration_error.code))
                      : declaration_error.message,
  });
  return ReplayDisposition::kFatal;
}

void append_port_derivation_candidate(const ExecutableOperatorFactoryRecord& entry,
                                      const PortRecord& output,
                                      const OperatorSpecCompilerAccess::PortMetadata& metadata,
                                      std::vector<Candidate>& candidates,
                                      TensorByteSpanDerivationLowering& result) {
  if (!metadata.tensor_byte_span_expression) {
    return;
  }
  if (output.setup.tensor_byte_span_expression != metadata.tensor_byte_span_expression) {
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
      .expression = output.setup.tensor_byte_span_expression
                        ? &*output.setup.tensor_byte_span_expression
                        : nullptr,
      .authored_expression = metadata.authored_tensor_byte_span_expression,
      .initial_parameters = entry.factory->initial_parameter_value(),
      .parameter_type_token = entry.factory->parameter_type().local_type_token,
      .input_source_eligible = true,
  });
}

void append_field_derivation_candidates(const ExecutableOperatorFactoryRecord& entry,
                                        const PortRecord& output,
                                        const OperatorSpecCompilerAccess::PortMetadata& metadata,
                                        std::vector<Candidate>& candidates,
                                        TensorByteSpanDerivationLowering& result) {
  for (const auto& authored_field : metadata.tensor_fields) {
    if (!authored_field.tensor_byte_span_expression ||
        !authored_field.configuration.tensor_byte_span_expression) {
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
    const TensorByteSpanExpressionRecord* expression =
        graph_field != nullptr ? optional_value_pointer(graph_field->tensor_byte_span_expression)
                               : optional_value_pointer(output.setup.tensor_byte_span_expression);
    if (expression == nullptr || *expression != replay_field.tensor_byte_span_expression.value()) {
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
        .owned_authored_expression = authored_field.tensor_byte_span_expression,
        .initial_parameters = entry.factory->initial_parameter_value(),
        .parameter_type_token = entry.factory->parameter_type().local_type_token,
        .input_source_eligible = metadata.descriptor_tensor_field_count == 1U,
    });
  }
}

void collect_derivation_candidates(const ExecutableOperatorFactoryRecord& entry,
                                   const GraphSnapshotData& snapshot, const OperatorSpec& spec,
                                   std::vector<Candidate>& candidates,
                                   TensorByteSpanDerivationLowering& result) {
  for (std::size_t index = 0U; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    if (metadata.direction != PortDirection::kOutput) {
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
    append_port_derivation_candidate(entry, *output, metadata, candidates, result);
    append_field_derivation_candidates(entry, *output, metadata, candidates, result);
  }
}

}  // namespace

TensorByteSpanDerivationLowering lower_tensor_byte_span_derivations(
    const ExecutableFactoryInventory& factories, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const ParameterFieldPlanRecord> parameter_fields) {
  TensorByteSpanDerivationLowering result;
  std::vector<ReplayStorage> replay_storage;
  std::vector<Candidate> candidates;
  replay_storage.reserve(factories.operators.size());
  for (const auto& entry : factories.operators) {
    if (!owner_has_byte_span_derivation(snapshot, entry.vertex)) {
      continue;
    }
    ReplayStorage replay;
    const ReplayDisposition disposition = replay_derivation_operator(entry, replay, result);
    if (disposition == ReplayDisposition::kSkip) {
      continue;
    }
    if (disposition == ReplayDisposition::kFatal) {
      return result;
    }
    collect_derivation_candidates(entry, snapshot, *replay.spec, candidates, result);
    replay_storage.push_back(std::move(replay));
  }
  validate_candidate_coverage(snapshot, candidates, result);
  if (plan_generation == 0U) {
    for (const auto& candidate : candidates) {
      reject(result, kDerivationInvalid, candidate);
    }
    return result;
  }
  Evaluator evaluator{snapshot, parameter_fields, std::move(candidates), result};
  evaluator.evaluate_all(plan_generation);
  return result;
}

}  // namespace holoscan::detail
