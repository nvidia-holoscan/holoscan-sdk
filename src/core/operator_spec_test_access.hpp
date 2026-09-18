/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include <holoscan/core/operator_spec.hpp>

namespace holoscan::detail {

/// @brief Expose immutable OperatorSpec records to source-tree conformance tests only.
///
/// This header is neither installed nor included by a public header. The accessor prevents tests
/// from weakening `OperatorSpec` encapsulation with preprocessor tricks while verifying that cold
/// compilation metadata survives the typed builder boundary.
class OperatorSpecTestAccess {
 public:
  /// @brief Return whether `compute()` was declared eligible for CUDA Graph capture.
  [[nodiscard]] static bool cuda_graph_capture_eligible(const OperatorSpec& spec) noexcept {
    return spec.cuda_graph_capture_eligible_;
  }

  /// @brief Identify which stable builder collection owns one source record.
  enum class SourceBuilderFamily : std::uint8_t {
    kSingleWitness,
    kNotification,
    kDoorbell,
  };

  /// @brief Copy the immutable typed metadata needed by a cold port compiler.
  struct PortMetadata {
    const void* declaration{};
    PortDirection direction{PortDirection::kInput};
    std::string_view name;
    bool tensor{};
    std::size_t builder_index{};
    const void* local_type_token{};
    PayloadKind payload_kind{PayloadKind::kInlineTyped};
    std::optional<SchemaIdentity> schema_identity{};
  };

  /// @brief Copy the immutable association needed by a cold readiness compiler.
  struct SourceMetadata {
    const void* declaration{};
    std::string_view name;
    ReadinessSourceKind kind{static_cast<ReadinessSourceKind>(0)};
    SourceBuilderFamily builder_family{SourceBuilderFamily::kSingleWitness};
    std::size_t builder_index{};
  };

  /// @brief Copy one cold lifecycle callback descriptor without exposing member-pointer storage.
  struct LifecycleCallbackMetadata {
    LifecycleStage stage{LifecycleStage::kConfigure};
    bool required{};
    LifecycleMethodId method{};
    const void* declaring_type_token{};
  };

  /// @brief Classify one Operator-local lifecycle ordering fact.
  enum class LifecycleFactKind : std::uint8_t {
    kProduces,
    kRequires,
  };

  /// @brief Copy one deep-owned lifecycle ordering declaration.
  struct LifecycleFactMetadata {
    LifecycleStage stage{LifecycleStage::kConfigure};
    LifecycleFactKind kind{LifecycleFactKind::kProduces};
    std::string_view stable_name;
  };

  /// @brief Return the number of typed port records retained by the declaration pass.
  [[nodiscard]] static std::size_t port_count(const OperatorSpec& spec) noexcept {
    return spec.ports_.size();
  }

  /// @brief Return one immutable port record by declaration order.
  [[nodiscard]] static PortMetadata port(const OperatorSpec& spec, std::size_t index) {
    const OperatorSpec::PortRecord& record = spec.ports_[index];
    std::optional<SchemaIdentity> schema_identity;
    if (record.payload_identity.schema) {
      const SchemaIdentityRecord& owned = *record.payload_identity.schema;
      schema_identity = SchemaIdentity{
          .canonical_name = owned.canonical_name,
          .compatibility_epoch = owned.compatibility_epoch,
          .family_id = owned.family_id,
          .file_identifier = owned.file_identifier,
      };
    }
    return PortMetadata{
        .declaration = record.declaration,
        .direction = record.direction,
        .name = record.name,
        .tensor = record.tensor,
        .builder_index = record.builder_index,
        .local_type_token = record.local_type_token,
        .payload_kind = record.payload_identity.payload_kind,
        .schema_identity = schema_identity,
    };
  }

  /// @brief Return the number of readiness records retained by the declaration pass.
  [[nodiscard]] static std::size_t source_count(const OperatorSpec& spec) noexcept {
    return spec.sources_.size();
  }

  /// @brief Return one immutable readiness record by declaration order.
  [[nodiscard]] static SourceMetadata source(const OperatorSpec& spec, std::size_t index) {
    const OperatorSpec::SourceRecord& record = spec.sources_[index];
    return SourceMetadata{
        .declaration = record.declaration,
        .name = record.name,
        .kind = record.kind,
        .builder_family = builder_family(record.builder_family),
        .builder_index = record.builder_index,
    };
  }

  /// @brief Return the number of sparse callback registrations.
  [[nodiscard]] static std::size_t lifecycle_callback_count(const OperatorSpec& spec) noexcept {
    return spec.lifecycle_callbacks_.size();
  }

  /// @brief Return one pointer-free callback descriptor plus process-local type token.
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

  /// @brief Invoke one session-local callback binding against a fresh receiver.
  [[nodiscard]] static LifecycleStatus invoke_lifecycle_callback(
      const OperatorSpec& spec, std::size_t index, void* receiver,
      LifecycleContext& context) noexcept {
    return spec.lifecycle_callbacks_[index].binding->invoke(receiver, context);
  }

  /// @brief Return the number of deep-owned lifecycle ordering facts.
  [[nodiscard]] static std::size_t lifecycle_fact_count(const OperatorSpec& spec) noexcept {
    return spec.lifecycle_facts_.size();
  }

  /// @brief Return one lifecycle ordering fact.
  [[nodiscard]] static LifecycleFactMetadata lifecycle_fact(const OperatorSpec& spec,
                                                            std::size_t index) {
    const auto& record = spec.lifecycle_facts_.at(index);
    return LifecycleFactMetadata{
        .stage = record.stage,
        .kind = record.kind == OperatorSpec::LifecycleFactKind::kProduces
                    ? LifecycleFactKind::kProduces
                    : LifecycleFactKind::kRequires,
        .stable_name = record.stable_name,
    };
  }

  /// @brief Replace one lifecycle callback stage with a raw value for rejection tests.
  static void set_lifecycle_callback_stage(OperatorSpec& spec, std::size_t index,
                                           std::uint8_t stage) {
    // Rejection tests deliberately inject an out-of-range representation through this private
    // test seam so the production snapshot validator can prove it fails closed.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    spec.lifecycle_callbacks_.at(index).stage = static_cast<LifecycleStage>(stage);
  }

  /// @brief Replace one lifecycle fact stage with a raw value for rejection tests.
  static void set_lifecycle_fact_stage(OperatorSpec& spec, std::size_t index, std::uint8_t stage) {
    // Rejection tests deliberately inject an out-of-range representation through this private
    // test seam so the production snapshot validator can prove it fails closed.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    spec.lifecycle_facts_.at(index).stage = static_cast<LifecycleStage>(stage);
  }

  /// @brief Replace one lifecycle fact kind with a raw value for rejection tests.
  static void set_lifecycle_fact_kind(OperatorSpec& spec, std::size_t index, std::uint8_t kind) {
    // Rejection tests deliberately inject an out-of-range representation through this private
    // test seam so the production snapshot validator can prove it fails closed.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    spec.lifecycle_facts_.at(index).kind = static_cast<OperatorSpec::LifecycleFactKind>(kind);
  }

 private:
  [[nodiscard]] static constexpr SourceBuilderFamily builder_family(
      OperatorSpec::SourceBuilderFamily family) noexcept {
    switch (family) {
      case OperatorSpec::SourceBuilderFamily::kSingleWitness:
        return SourceBuilderFamily::kSingleWitness;
      case OperatorSpec::SourceBuilderFamily::kNotification:
        return SourceBuilderFamily::kNotification;
      case OperatorSpec::SourceBuilderFamily::kDoorbell:
        return SourceBuilderFamily::kDoorbell;
    }
    return SourceBuilderFamily::kSingleWitness;
  }
};

}  // namespace holoscan::detail
