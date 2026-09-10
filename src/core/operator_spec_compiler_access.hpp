/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>

namespace holoscan::detail {

/// @brief Expose immutable setup metadata to the cold Graph/compiler path.
///
/// This header is not installed. Returned string views remain valid only while
/// the source `OperatorSpec` lives; Graph immediately copies every value into
/// owned authoring storage.
class OperatorSpecCompilerAccess {
 public:
  struct PortMetadata {
    const void* declaration{};
    const void* local_type_token{};
    const RuntimePortTypeOps* runtime_type_ops{};
    std::uint64_t serialized_payload_capacity_bytes{};
    const TensorByteSpanExpression* authored_tensor_byte_span_expression{};
    const InputBase* follows_input_declaration{};
    std::string_view name;
    std::string_view diagnostic_alias;
    std::optional<TensorPoolBounds> tensor_allocation;
    std::vector<AuthoredTensorFieldConfiguration> tensor_fields;
    std::optional<Rate> nominal_rate;
    std::optional<Rate> expected_rate;
    std::optional<Rate> max_nominal_rate;
    std::optional<TensorByteSpanExpressionRecord> tensor_byte_span_expression;
    Error tensor_contract_error{ErrorCode::kSuccess};
    Error rate_error{ErrorCode::kSuccess};
    std::optional<PayloadOptionsRecord> payload_options;
    std::optional<SchemaPackagePlanRecord> schema_package;
    std::optional<TensorContractRecord> tensor_contract;
    PayloadIdentityRecord payload_identity;
    std::uint32_t max_emits_per_compute{1};
    std::uint32_t late_joiner_history_depth{};
    std::optional<std::uint32_t> queue_depth;
    std::optional<RateRatio> follows_ratio;
    std::optional<TensorRepresentation> tensor_layout;
    std::uint16_t descriptor_tensor_field_count{};
    PortDirection direction{PortDirection::kInput};
    bool tensor{};
    PortRateKind rate_kind{PortRateKind::kUnspecified};
    std::optional<MemoryConversionPolicy> memory_conversion;
    std::optional<LifecycleStage> lifecycle_binding_stage;
  };

  struct SourceMetadata {
    const void* declaration{};
    std::string_view name;
    ReadinessSourceKind kind{ReadinessSourceKind::kReadable};
    std::uint32_t capacity{1};
    std::uint32_t sender_reference_capacity{};
    ClockKey clock;
  };

  struct LifecycleCallbackMetadata {
    LifecycleStage stage{LifecycleStage::kConfigure};
    bool required{};
    LifecycleMethodId method{};
    const void* declaring_type_token{};
  };

  enum class LifecycleFactKind : std::uint8_t {
    kProduces,
    kRequires,
  };

  struct LifecycleFactMetadata {
    LifecycleStage stage{LifecycleStage::kConfigure};
    LifecycleFactKind kind{LifecycleFactKind::kProduces};
    std::string_view stable_name;
  };

  [[nodiscard]] static const Error& declaration_error(const OperatorSpec& spec) noexcept {
    return spec.declaration_error_;
  }

  [[nodiscard]] static bool compute_capture_safe(const OperatorSpec& spec) noexcept {
    return spec.compute_capture_safe_;
  }

  [[nodiscard]] static std::size_t port_count(const OperatorSpec& spec) noexcept {
    return spec.ports_.size();
  }

 private:
  static void normalize_port_byte_span_expression(
      const OperatorSpec& spec, const std::optional<TensorByteSpanExpression>& expression,
      PortMetadata& result) {
    if (!expression) {
      return;
    }
    result.authored_tensor_byte_span_expression = &*expression;
    auto normalized = normalize_tensor_byte_span_expression(spec, *expression);
    if (normalized) {
      result.tensor_byte_span_expression = std::move(*normalized);
    } else if (result.tensor_contract_error.code == ErrorCode::kSuccess) {
      result.tensor_contract_error = std::move(normalized.error());
    }
  }

  static void normalize_field_byte_span_expressions(const OperatorSpec& spec,
                                                    PortMetadata& result) {
    if (result.direction != PortDirection::kOutput) {
      return;
    }
    for (auto& field : result.tensor_fields) {
      if (!field.tensor_byte_span_expression) {
        continue;
      }
      auto normalized =
          normalize_tensor_byte_span_expression(spec, *field.tensor_byte_span_expression);
      if (normalized) {
        field.configuration.tensor_byte_span_expression = std::move(*normalized);
      } else if (result.tensor_contract_error.code == ErrorCode::kSuccess) {
        result.tensor_contract_error =
            tensor_field_contract_error(std::move(normalized.error()), field.configuration);
      }
    }
  }

 public:
  [[nodiscard]] static PortMetadata port(const OperatorSpec& spec, std::size_t index) {
    const auto& record = spec.ports_.at(index);
    PortMetadata result{
        .declaration = record.declaration,
        .local_type_token = record.local_type_token,
        .runtime_type_ops = record.runtime_type_ops,
        .serialized_payload_capacity_bytes =
            record.payload_identity.kind == PayloadIdentityKind::kSchema &&
                    record.runtime_type_ops != nullptr
                ? static_cast<std::uint64_t>(record.runtime_type_ops->wire_capacity)
                : 0U,
        .name = record.name,
        .schema_package = record.schema_package,
        .payload_identity = record.payload_identity,
        .direction = record.direction,
        .tensor = record.tensor,
    };
    if (record.direction == PortDirection::kInput) {
      const auto& builder = spec.input_builders_.at(record.builder_index);
      result.queue_depth = builder.queue_depth_;
      result.rate_kind = builder.rate_.kind;
      result.expected_rate = builder.rate_.expected_rate;
      result.rate_error = builder.rate_.error;
      if (result.rate_error.code != ErrorCode::kSuccess) {
        result.rate_error.message.assign(builder.rate_.error_token);
      }
      result.diagnostic_alias = builder.diagnostic_alias_;
      result.tensor_layout = builder.tensor_layout_;
      result.memory_conversion = builder.memory_conversion_;
      result.payload_options = builder.payload_options_;
      result.tensor_contract = builder.tensor_contract_;
      result.tensor_contract_error = builder.tensor_contract_error_;
      result.descriptor_tensor_field_count = builder.descriptor_tensor_field_count_;
      result.tensor_fields = builder.tensor_fields_;
    } else if (record.tensor) {
      const auto& builder = spec.tensor_output_builders_.at(record.builder_index);
      result.rate_kind = builder.rate_.kind;
      result.nominal_rate = builder.rate_.nominal_rate;
      result.follows_input_declaration = builder.rate_.follows_input;
      result.follows_ratio = builder.rate_.follows_ratio;
      result.lifecycle_binding_stage = builder.rate_.lifecycle_binding_stage;
      result.max_nominal_rate = builder.rate_.max_nominal_rate;
      result.rate_error = builder.rate_.error;
      if (result.rate_error.code != ErrorCode::kSuccess) {
        result.rate_error.message.assign(builder.rate_.error_token);
      }
      result.max_emits_per_compute = builder.max_emits_per_compute_;
      result.late_joiner_history_depth = builder.late_joiner_history_depth_;
      result.diagnostic_alias = builder.diagnostic_alias_;
      result.tensor_layout = builder.tensor_layout_;
      result.payload_options = builder.payload_options_;
      result.tensor_contract = builder.tensor_contract_;
      result.tensor_contract_error = builder.tensor_contract_error_;
      normalize_port_byte_span_expression(spec, builder.tensor_byte_span_expression_, result);
      if (builder.has_tensor_allocation_)
        result.tensor_allocation = builder.tensor_allocation_;
      result.descriptor_tensor_field_count = builder.descriptor_tensor_field_count_;
      result.tensor_fields = builder.tensor_fields_;
    } else {
      const auto& builder = spec.output_builders_.at(record.builder_index);
      result.rate_kind = builder.rate_.kind;
      result.nominal_rate = builder.rate_.nominal_rate;
      result.follows_input_declaration = builder.rate_.follows_input;
      result.follows_ratio = builder.rate_.follows_ratio;
      result.lifecycle_binding_stage = builder.rate_.lifecycle_binding_stage;
      result.max_nominal_rate = builder.rate_.max_nominal_rate;
      result.rate_error = builder.rate_.error;
      if (result.rate_error.code != ErrorCode::kSuccess) {
        result.rate_error.message.assign(builder.rate_.error_token);
      }
      result.max_emits_per_compute = builder.max_emits_per_compute_;
      result.late_joiner_history_depth = builder.late_joiner_history_depth_;
      result.diagnostic_alias = builder.diagnostic_alias_;
      result.tensor_layout = builder.tensor_layout_;
      result.payload_options = builder.payload_options_;
      result.tensor_contract = builder.tensor_contract_;
      result.tensor_contract_error = builder.tensor_contract_error_;
    }
    normalize_field_byte_span_expressions(spec, result);
    const bool shared_tensor_contract =
        result.tensor_contract ||
        std::any_of(
            result.tensor_fields.begin(), result.tensor_fields.end(), [](const auto& field) {
              return field.configuration.tensor_contract.has_value();
            });
    if (result.direction == PortDirection::kOutput && shared_tensor_contract &&
        result.payload_options && !result.payload_options->schema &&
        result.payload_identity.schema) {
      result.payload_options->schema = result.payload_identity.schema;
    }
    return result;
  }

 private:
  [[nodiscard]] static expected<TensorByteSpanExpressionRecord, Error>
  normalize_tensor_byte_span_expression(const OperatorSpec& spec,
                                        const TensorByteSpanExpression& expression) {
    if (expression.invalid_reason_ != TensorByteSpanExpressionInvalidReason::kNone ||
        expression.nodes_.empty() ||
        expression.nodes_.size() > kMaximumTensorByteSpanExpressionNodes) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID"});
    }
    TensorByteSpanExpressionRecord result;
    result.nodes.reserve(expression.nodes_.size());
    std::vector<std::pair<ParamFieldId, ParamSchemaHash>> parameters;
    std::vector<std::string> inputs;
    for (std::size_t index = 0U; index < expression.nodes_.size(); ++index) {
      auto normalized = normalize_tensor_byte_span_node(
          spec, expression.nodes_[index], index, parameters, inputs);
      if (!normalized) {
        return make_unexpected(std::move(normalized.error()));
      }
      result.nodes.push_back(std::move(*normalized));
    }
    if (parameters.size() + inputs.size() > kMaximumTensorByteSpanExpressionDependencies) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID"});
    }
    return result;
  }

  using ParameterDependencies = std::vector<std::pair<ParamFieldId, ParamSchemaHash>>;
  using InputDependencies = std::vector<std::string>;

  [[nodiscard]] static Error invalid_byte_span_derivation() {
    return Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID"};
  }

  template <typename T>
  static void append_unique(std::vector<T>& values, T value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
      values.push_back(std::move(value));
    }
  }

  [[nodiscard]] static expected<void, Error> normalize_parameter_node(
      const TensorByteSpanAuthoringNode& node, TensorByteSpanExpressionNodeRecord& record,
      ParameterDependencies& parameters) {
    if (!node.parameter) {
      return make_unexpected(invalid_byte_span_derivation());
    }
    const ParamSchemaDescriptor* schema = node.parameter.root_schema();
    const ParamFieldDescriptor* field = node.parameter.final_field();
    const std::string path = node.parameter.canonical_path();
    if (schema == nullptr || field == nullptr || field->id == 0U || path.empty()) {
      return make_unexpected(invalid_byte_span_derivation());
    }
    record.parameter_field = field->id;
    record.parameter_full_id = field->full_id;
    record.parameter_path = path;
    record.parameter_schema_id = std::string(schema->stable_id);
    record.parameter_schema_version = schema->version;
    append_unique(parameters, std::pair{field->id, field->full_id});
    return {};
  }

  [[nodiscard]] static expected<void, Error> normalize_input_node(
      const OperatorSpec& spec, const TensorByteSpanAuthoringNode& node,
      TensorByteSpanExpressionNodeRecord& record, InputDependencies& inputs) {
    const auto found = std::find_if(
        spec.ports_.begin(), spec.ports_.end(), [&](const OperatorSpec::PortRecord& port) {
          return port.declaration == node.input_declaration &&
                 port.direction == PortDirection::kInput;
        });
    if (found == spec.ports_.end()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"});
    }
    record.input_name = found->name;
    append_unique(inputs, record.input_name);
    return {};
  }

  [[nodiscard]] static expected<void, Error> normalize_tensor_byte_span_node_payload(
      const OperatorSpec& spec, const TensorByteSpanAuthoringNode& node, const std::size_t index,
      TensorByteSpanExpressionNodeRecord& record, ParameterDependencies& parameters,
      InputDependencies& inputs) {
    // Multiple enum labels intentionally share the same normalized operand rule.
    // NOLINTBEGIN(bugprone-branch-clone)
    switch (node.operation) {
      case TensorByteSpanOperation::kPositiveConstant:
        return node.value == 0U ? make_unexpected(invalid_byte_span_derivation())
                                : expected<void, Error>{};
      case TensorByteSpanOperation::kParameterByteSpan:
      case TensorByteSpanOperation::kParameterContiguousShape:
        return normalize_parameter_node(node, record, parameters);
      case TensorByteSpanOperation::kInputMaxByteSpan:
        return normalize_input_node(spec, node, record, inputs);
      case TensorByteSpanOperation::kMultiply:
      case TensorByteSpanOperation::kCeilingDivide:
      case TensorByteSpanOperation::kMaximum:
        return node.left >= index || node.right >= index
                   ? make_unexpected(invalid_byte_span_derivation())
                   : expected<void, Error>{};
      case TensorByteSpanOperation::kAlignUp:
        return node.left >= index || node.value == 0U || node.value > (1ULL << 30U) ||
                       (node.value & (node.value - 1U)) != 0U
                   ? make_unexpected(invalid_byte_span_derivation())
                   : expected<void, Error>{};
      default:
        return make_unexpected(invalid_byte_span_derivation());
    }
    // NOLINTEND(bugprone-branch-clone)
  }

  [[nodiscard]] static expected<TensorByteSpanExpressionNodeRecord, Error>
  normalize_tensor_byte_span_node(const OperatorSpec& spec, const TensorByteSpanAuthoringNode& node,
                                  const std::size_t index, ParameterDependencies& parameters,
                                  InputDependencies& inputs) {
    if (node.depth == 0U || node.depth > kMaximumTensorByteSpanExpressionDepth) {
      return make_unexpected(invalid_byte_span_derivation());
    }
    TensorByteSpanExpressionNodeRecord record{
        .operation = node.operation,
        .left = node.left,
        .right = node.right,
        .value = node.value,
        .depth = node.depth,
    };
    auto normalized =
        normalize_tensor_byte_span_node_payload(spec, node, index, record, parameters, inputs);
    if (!normalized) {
      return make_unexpected(std::move(normalized.error()));
    }
    return record;
  }

 public:
  [[nodiscard]] static std::span<const TensorByteSpanAuthoringNode>
  tensor_byte_span_expression_nodes(const TensorByteSpanExpression& expression) noexcept {
    return expression.nodes_;
  }

  [[nodiscard]] static std::size_t source_count(const OperatorSpec& spec) noexcept {
    return spec.sources_.size();
  }

  [[nodiscard]] static SourceMetadata source(const OperatorSpec& spec, std::size_t index) {
    const auto& record = spec.sources_.at(index);
    SourceMetadata result{
        .declaration = record.declaration,
        .name = record.name,
        .kind = record.kind,
    };
    switch (record.builder_family) {
      case OperatorSpec::SourceBuilderFamily::kSingleWitness: {
        const auto& builder = spec.single_source_builders_.at(record.builder_index);
        result.capacity = builder.capacity_;
        result.clock = builder.clock_;
        break;
      }
      case OperatorSpec::SourceBuilderFamily::kNotification: {
        const auto& builder = spec.notification_source_builders_.at(record.builder_index);
        result.capacity = builder.capacity_;
        result.sender_reference_capacity = builder.sender_reference_capacity_;
        result.clock = builder.clock_;
        break;
      }
      case OperatorSpec::SourceBuilderFamily::kDoorbell: {
        const auto& builder = spec.doorbell_source_builders_.at(record.builder_index);
        result.capacity = builder.capacity_;
        result.clock = builder.clock_;
        break;
      }
    }
    return result;
  }

  [[nodiscard]] static std::size_t lifecycle_callback_count(const OperatorSpec& spec) noexcept {
    return spec.lifecycle_callbacks_.size();
  }

  [[nodiscard]] static LifecycleCallbackMetadata lifecycle_callback(const OperatorSpec& spec,
                                                                    std::size_t index) {
    const auto& record = spec.lifecycle_callbacks_.at(index);
    return LifecycleCallbackMetadata{
        .stage = record.stage,
        .required = record.required,
        .method = record.method,
        .declaring_type_token = record.binding->declaring_type_token(),
    };
  }

  [[nodiscard]] static std::size_t lifecycle_fact_count(const OperatorSpec& spec) noexcept {
    return spec.lifecycle_facts_.size();
  }

  [[nodiscard]] static LifecycleFactMetadata lifecycle_fact(const OperatorSpec& spec,
                                                            std::size_t index) {
    const auto& record = spec.lifecycle_facts_.at(index);
    static_assert(static_cast<std::uint8_t>(OperatorSpec::LifecycleFactKind::kProduces) ==
                  static_cast<std::uint8_t>(LifecycleFactKind::kProduces));
    static_assert(static_cast<std::uint8_t>(OperatorSpec::LifecycleFactKind::kRequires) ==
                  static_cast<std::uint8_t>(LifecycleFactKind::kRequires));
    return LifecycleFactMetadata{
        .stage = record.stage,
        .kind = static_cast<LifecycleFactKind>(static_cast<std::uint8_t>(record.kind)),
        .stable_name = record.stable_name,
    };
  }
};

}  // namespace holoscan::detail
