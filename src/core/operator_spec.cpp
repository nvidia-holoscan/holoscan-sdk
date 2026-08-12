/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>

#include <string>
#include <utility>

namespace holoscan {

namespace {

[[nodiscard]] bool valid_lifecycle_stage(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(stage) <= static_cast<std::uint8_t>(LifecycleStage::kRelease);
}

}  // namespace

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
  if (!valid_lifecycle_stage(stage) || key.stable_name.empty()) {
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
