/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>

#include <string>
#include <utility>

#include "rate_math_internal.hpp"

namespace holoscan {

namespace {

void record_rate_error(detail::AuthoredPortRate& authored, std::string_view token) {
  if (authored.error.code == ErrorCode::kSuccess) {
    // Preserve the sticky fail-closed state before any diagnostic enrichment.  Builder authoring
    // itself stays allocation-free, including in exception-disabled builds; the cold compiler
    // access path may copy the static token into an owning Error later.
    authored.error.code = ErrorCode::kInvalidArgument;
    authored.error_token = token;
  }
}

[[nodiscard]] bool rate_kind_conflicts(PortRateKind existing, PortRateKind requested) noexcept {
  return existing != PortRateKind::kUnspecified && existing != requested;
}

void declare_expected_rate(detail::AuthoredPortRate& authored, Rate rate) {
  if (authored.error.code != ErrorCode::kSuccess) {
    return;
  }
  const auto normalized = detail::rate_math::normalize(rate);
  if (!normalized) {
    record_rate_error(authored, "PORT_RATE_INVALID");
    return;
  }
  if (rate_kind_conflicts(authored.kind, PortRateKind::kExpectedInput) ||
      (authored.expected_rate && *authored.expected_rate != *normalized)) {
    record_rate_error(authored, "PORT_RATE_CONFLICT");
    return;
  }
  authored.kind = PortRateKind::kExpectedInput;
  authored.expected_rate = *normalized;
}

void declare_nominal_rate(detail::AuthoredPortRate& authored, Rate rate) {
  if (authored.error.code != ErrorCode::kSuccess) {
    return;
  }
  const auto normalized = detail::rate_math::normalize(rate);
  if (!normalized) {
    record_rate_error(authored, "PORT_RATE_INVALID");
    return;
  }
  if (authored.max_nominal_rate ||
      rate_kind_conflicts(authored.kind, PortRateKind::kFixedNominal) ||
      (authored.nominal_rate && *authored.nominal_rate != *normalized)) {
    record_rate_error(authored, "PORT_RATE_CONFLICT");
    return;
  }
  authored.kind = PortRateKind::kFixedNominal;
  authored.nominal_rate = *normalized;
}

void declare_lifecycle_rate(detail::AuthoredPortRate& authored, LifecycleStage stage) {
  if (authored.error.code != ErrorCode::kSuccess) {
    return;
  }
  if (stage != LifecycleStage::kConfigure && stage != LifecycleStage::kDiscover &&
      stage != LifecycleStage::kWarmUp && stage != LifecycleStage::kArm) {
    record_rate_error(authored, "PORT_RATE_BIND_STAGE_INVALID");
    return;
  }
  if (rate_kind_conflicts(authored.kind, PortRateKind::kLifecycleNominal) ||
      (authored.lifecycle_binding_stage && *authored.lifecycle_binding_stage != stage)) {
    record_rate_error(authored, "PORT_RATE_CONFLICT");
    return;
  }
  authored.kind = PortRateKind::kLifecycleNominal;
  authored.lifecycle_binding_stage = stage;
}

void declare_maximum_rate(detail::AuthoredPortRate& authored, Rate rate) {
  if (authored.error.code != ErrorCode::kSuccess) {
    return;
  }
  const auto normalized = detail::rate_math::normalize(rate);
  if (!normalized) {
    record_rate_error(authored, "PORT_RATE_INVALID");
    return;
  }
  if ((authored.kind != PortRateKind::kUnspecified &&
       authored.kind != PortRateKind::kLifecycleNominal) ||
      (authored.max_nominal_rate && *authored.max_nominal_rate != *normalized)) {
    record_rate_error(authored, "PORT_RATE_CONFLICT");
    return;
  }
  authored.max_nominal_rate = *normalized;
}

void declare_follows_rate(detail::AuthoredPortRate& authored, const InputBase& input,
                          RateRatio ratio) {
  if (authored.error.code != ErrorCode::kSuccess) {
    return;
  }
  const auto normalized = detail::rate_math::normalize(ratio);
  if (!normalized) {
    record_rate_error(authored, "PORT_RATE_INVALID");
    return;
  }
  if (authored.max_nominal_rate ||
      rate_kind_conflicts(authored.kind, PortRateKind::kFollowsInput) ||
      (authored.follows_input != nullptr && authored.follows_input != &input) ||
      (authored.follows_ratio && *authored.follows_ratio != *normalized)) {
    record_rate_error(authored, "PORT_RATE_CONFLICT");
    return;
  }
  authored.kind = PortRateKind::kFollowsInput;
  authored.follows_input = &input;
  authored.follows_ratio = *normalized;
}

}  // namespace

InputSpecBuilder& InputSpecBuilder::expected_rate(Rate rate) {
  declare_expected_rate(rate_, rate);
  return *this;
}

OutputSpecBuilder& OutputSpecBuilder::nominal_rate(Rate rate) {
  declare_nominal_rate(rate_, rate);
  return *this;
}

OutputSpecBuilder& OutputSpecBuilder::lifecycle_nominal_rate(LifecycleStage bind_stage) {
  declare_lifecycle_rate(rate_, bind_stage);
  return *this;
}

OutputSpecBuilder& OutputSpecBuilder::max_nominal_rate(Rate rate) {
  declare_maximum_rate(rate_, rate);
  return *this;
}

OutputSpecBuilder& OutputSpecBuilder::follows_input(const InputBase& input, RateRatio ratio) {
  declare_follows_rate(rate_, input, ratio);
  return *this;
}

TensorOutputSpecBuilder& TensorOutputSpecBuilder::nominal_rate(Rate rate) {
  declare_nominal_rate(rate_, rate);
  return *this;
}

TensorOutputSpecBuilder& TensorOutputSpecBuilder::lifecycle_nominal_rate(
    LifecycleStage bind_stage) {
  declare_lifecycle_rate(rate_, bind_stage);
  return *this;
}

TensorOutputSpecBuilder& TensorOutputSpecBuilder::max_nominal_rate(Rate rate) {
  declare_maximum_rate(rate_, rate);
  return *this;
}

TensorOutputSpecBuilder& TensorOutputSpecBuilder::follows_input(const InputBase& input,
                                                                RateRatio ratio) {
  declare_follows_rate(rate_, input, ratio);
  return *this;
}

OperatorSpec::~OperatorSpec() noexcept = default;

LifecycleBuilder& LifecycleBuilder::produces(LifecycleStage stage, LifecycleDependencyKey key) {
  spec_->register_lifecycle_fact(stage, key, OperatorSpec::LifecycleFactKind::kProduces);
  return *this;
}

LifecycleBuilder& LifecycleBuilder::requires_dependency(LifecycleStage stage,
                                                        LifecycleDependencyKey key) {
  spec_->register_lifecycle_fact(stage, key, OperatorSpec::LifecycleFactKind::kRequires);
  return *this;
}

void OperatorSpec::register_lifecycle_fact(LifecycleStage stage, LifecycleDependencyKey key,
                                           LifecycleFactKind kind) {
  if (!is_valid_lifecycle_stage(stage) || key.stable_name.empty()) {
    if (declaration_error_.code == ErrorCode::kSuccess) {
      declaration_error_ =
          Error{ErrorCode::kInvalidArgument, "operator lifecycle dependency is invalid"};
    }
    return;
  }
  const bool duplicate =
      std::any_of(lifecycle_facts_.begin(), lifecycle_facts_.end(), [&](const auto& existing) {
        return existing.stage == stage && existing.kind == kind &&
               existing.stable_name == key.stable_name;
      });
  if (duplicate) {
    if (declaration_error_.code == ErrorCode::kSuccess) {
      declaration_error_ =
          Error{ErrorCode::kDuplicateName, "operator lifecycle dependency is duplicated"};
    }
    return;
  }
  lifecycle_facts_.push_back(LifecycleFactRecord{
      .stage = stage,
      .kind = kind,
      .stable_name = std::string(key.stable_name),
  });
}

void OperatorSpec::validate_declaration_name(std::string_view member_name,
                                             std::string_view setup_name) {
  if (declaration_error_.code != ErrorCode::kSuccess) {
    return;
  }
  if (setup_name.empty()) {
    declaration_error_ = Error{ErrorCode::kInvalidArgument, "operator declaration name is empty"};
    return;
  }
  if (!member_name.empty() && member_name != setup_name) {
    declaration_error_ =
        Error{ErrorCode::kInvalidArgument,
              "setup name conflicts with the port member name; use a diagnostic alias instead"};
  }
}

void OperatorSpec::record_name(std::string_view name) {
  if (name.empty()) {
    return;
  }
  const auto [unused, inserted] = names_.emplace(name);
  (void)unused;
  if (!inserted && declaration_error_.code == ErrorCode::kSuccess) {
    declaration_error_ = Error{ErrorCode::kDuplicateName,
                               "operator declaration name is duplicated across inputs, outputs, "
                               "or readiness sources"};
  }
}

void OperatorSpec::record_declaration(const void* declaration) {
  const auto [unused, inserted] = declarations_.emplace(declaration);
  (void)unused;
  if (!inserted && declaration_error_.code == ErrorCode::kSuccess) {
    declaration_error_ =
        Error{ErrorCode::kInvalidArgument,
              "operator declaration member is bound more than once in one setup pass"};
  }
}

void OperatorSpec::validate_source_declaration(const void* declaration, std::string_view name) {
  validate_declaration_name({}, name);
  record_name(name);
  record_declaration(declaration);
}

InputSpecBuilder& OperatorSpec::bind_input(InputBase& input, std::string_view name,
                                           const void* local_type_token,
                                           PayloadIdentityRecord payload_identity,
                                           std::optional<SchemaPackagePlanRecord> schema_package,
                                           const detail::RuntimePortTypeOps* runtime_type_ops) {
  validate_declaration_name(input.name(), name);
  if (input.name_.empty()) {
    input.name_.assign(name);
  }
  record_name(name);
  record_declaration(&input);
  input_builders_.emplace_back();
  ports_.push_back(PortRecord{
      .declaration = &input,
      .direction = PortDirection::kInput,
      .name = std::string{name},
      .tensor = false,
      .builder_index = input_builders_.size() - 1U,
      .local_type_token = local_type_token,
      .payload_identity = std::move(payload_identity),
      .schema_package = std::move(schema_package),
      .runtime_type_ops = runtime_type_ops,
  });
  return input_builders_.back();
}

OutputSpecBuilder& OperatorSpec::bind_output(OutputBase& output, std::string_view name,
                                             const void* local_type_token,
                                             PayloadIdentityRecord payload_identity,
                                             std::optional<SchemaPackagePlanRecord> schema_package,
                                             const detail::RuntimePortTypeOps* runtime_type_ops) {
  validate_declaration_name(output.name(), name);
  if (output.name_.empty()) {
    output.name_.assign(name);
  }
  record_name(name);
  record_declaration(&output);
  output_builders_.emplace_back();
  ports_.push_back(PortRecord{
      .declaration = &output,
      .direction = PortDirection::kOutput,
      .name = std::string{name},
      .tensor = false,
      .builder_index = output_builders_.size() - 1U,
      .local_type_token = local_type_token,
      .payload_identity = std::move(payload_identity),
      .schema_package = std::move(schema_package),
      .runtime_type_ops = runtime_type_ops,
  });
  return output_builders_.back();
}

TensorOutputSpecBuilder& OperatorSpec::bind_tensor_output(
    OutputBase& output, std::string_view name, const void* local_type_token,
    PayloadIdentityRecord payload_identity, std::optional<SchemaPackagePlanRecord> schema_package,
    const detail::RuntimePortTypeOps* runtime_type_ops) {
  validate_declaration_name(output.name(), name);
  if (output.name_.empty()) {
    output.name_.assign(name);
  }
  record_name(name);
  record_declaration(&output);
  tensor_output_builders_.emplace_back();
  ports_.push_back(PortRecord{
      .declaration = &output,
      .direction = PortDirection::kOutput,
      .name = std::string{name},
      .tensor = true,
      .builder_index = tensor_output_builders_.size() - 1U,
      .local_type_token = local_type_token,
      .payload_identity = std::move(payload_identity),
      .schema_package = std::move(schema_package),
      .runtime_type_ops = runtime_type_ops,
  });
  return tensor_output_builders_.back();
}

SingleWitnessSourceSpecBuilder& OperatorSpec::readable_source(ReadableSource& source,
                                                              std::string_view name) {
  validate_source_declaration(&source, name);
  single_source_builders_.emplace_back();
  sources_.push_back(SourceRecord{
      .declaration = &source,
      .name = std::string{name},
      .kind = ReadinessSourceKind::kReadable,
      .builder_family = SourceBuilderFamily::kSingleWitness,
      .builder_index = single_source_builders_.size() - 1U,
  });
  return single_source_builders_.back();
}

NotificationSourceSpecBuilder& OperatorSpec::notification_source(NotificationSource& source,
                                                                 std::string_view name) {
  validate_source_declaration(&source, name);
  notification_source_builders_.emplace_back();
  sources_.push_back(SourceRecord{
      .declaration = &source,
      .name = std::string{name},
      .kind = ReadinessSourceKind::kNotification,
      .builder_family = SourceBuilderFamily::kNotification,
      .builder_index = notification_source_builders_.size() - 1U,
  });
  return notification_source_builders_.back();
}

DoorbellSourceSpecBuilder& OperatorSpec::doorbell_source(DoorbellSource& source,
                                                         std::string_view name) {
  validate_source_declaration(&source, name);
  doorbell_source_builders_.emplace_back();
  sources_.push_back(SourceRecord{
      .declaration = &source,
      .name = std::string{name},
      .kind = ReadinessSourceKind::kDoorbell,
      .builder_family = SourceBuilderFamily::kDoorbell,
      .builder_index = doorbell_source_builders_.size() - 1U,
  });
  return doorbell_source_builders_.back();
}

SingleWitnessSourceSpecBuilder& OperatorSpec::spin_source(SpinSource& source,
                                                          std::string_view name) {
  validate_source_declaration(&source, name);
  single_source_builders_.emplace_back();
  sources_.push_back(SourceRecord{
      .declaration = &source,
      .name = std::string{name},
      .kind = ReadinessSourceKind::kSpin,
      .builder_family = SourceBuilderFamily::kSingleWitness,
      .builder_index = single_source_builders_.size() - 1U,
  });
  return single_source_builders_.back();
}

}  // namespace holoscan
