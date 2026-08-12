/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

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
    PortDirection direction{PortDirection::kInput};
    std::string_view name;
    bool tensor{};
    const void* local_type_token{};
    PayloadIdentityRecord payload_identity;
    std::optional<SchemaPackagePlanRecord> schema_package;
    const RuntimePortTypeOps* runtime_type_ops{};
    std::uint64_t serialized_payload_capacity_bytes{};
    std::uint32_t queue_depth{1};
    std::uint32_t max_emits_per_compute{1};
    std::uint32_t late_joiner_history_depth{};
    std::string_view diagnostic_alias;
    std::optional<TensorPortLayout> tensor_layout;
    std::optional<MemoryConversionPolicy> memory_conversion;
    std::optional<PayloadOptionsRecord> payload_options;
    std::optional<TensorAllocationBounds> tensor_allocation;
    std::uint16_t descriptor_tensor_field_count{};
    std::span<const AuthoredTensorFieldConfiguration> tensor_fields;
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

  [[nodiscard]] static PortMetadata port(const OperatorSpec& spec, std::size_t index) {
    const auto& record = spec.ports_.at(index);
    PortMetadata result{
        .declaration = record.declaration,
        .direction = record.direction,
        .name = record.name,
        .tensor = record.tensor,
        .local_type_token = record.local_type_token,
        .payload_identity = record.payload_identity,
        .schema_package = record.schema_package,
        .runtime_type_ops = record.runtime_type_ops,
        .serialized_payload_capacity_bytes =
            record.payload_identity.kind == PayloadIdentityKind::kSchema &&
                    record.runtime_type_ops != nullptr
                ? static_cast<std::uint64_t>(record.runtime_type_ops->wire_capacity)
                : 0U,
    };
    if (record.direction == PortDirection::kInput) {
      const auto& builder = spec.input_builders_.at(record.builder_index);
      result.queue_depth = builder.queue_depth_;
      result.diagnostic_alias = builder.diagnostic_alias_;
      result.tensor_layout = builder.tensor_layout_;
      result.memory_conversion = builder.memory_conversion_;
      result.payload_options = builder.payload_options_;
      result.descriptor_tensor_field_count = builder.descriptor_tensor_field_count_;
      result.tensor_fields = builder.tensor_fields_;
    } else if (record.tensor) {
      const auto& builder = spec.tensor_output_builders_.at(record.builder_index);
      result.max_emits_per_compute = builder.max_emits_per_compute_;
      result.late_joiner_history_depth = builder.late_joiner_history_depth_;
      result.diagnostic_alias = builder.diagnostic_alias_;
      result.tensor_layout = builder.tensor_layout_;
      result.payload_options = builder.payload_options_;
      if (builder.has_tensor_allocation_)
        result.tensor_allocation = builder.tensor_allocation_;
      result.descriptor_tensor_field_count = builder.descriptor_tensor_field_count_;
      result.tensor_fields = builder.tensor_fields_;
    } else {
      const auto& builder = spec.output_builders_.at(record.builder_index);
      result.max_emits_per_compute = builder.max_emits_per_compute_;
      result.late_joiner_history_depth = builder.late_joiner_history_depth_;
      result.diagnostic_alias = builder.diagnostic_alias_;
      result.tensor_layout = builder.tensor_layout_;
      result.payload_options = builder.payload_options_;
    }
    return result;
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
