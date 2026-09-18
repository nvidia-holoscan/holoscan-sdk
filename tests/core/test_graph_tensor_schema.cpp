/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "core/graph_compiler_access.hpp"
#include "core/operator_spec_compiler_access.hpp"

#include "graph_test_support.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

using GraphSchemaPayload = schema::StringT;

// Tag types that alias schema::StringT but carry different max_serialized_size values.
// Used to verify that serialized capacity participates in the canonical graph hash and
// that capacities exceeding the uint32_t sample-metadata limit are rejected.
// Representative test: ExternalSchemaCapacityParticipatesInCanonicalHash.
struct GraphSmallCapacitySchema {};
struct GraphLargeCapacitySchema {};
struct GraphOverflowCapacitySchema {};

// Generated owning type whose deliberately incomplete package exercises admission validation.
// Its generated root remains String so ordinary generated-schema Runtime operations are valid.
struct GraphInvalidSchemaPackage : schema::StringT {
  using schema::StringT::StringT;
  GraphInvalidSchemaPackage() = default;
};

// Three-field descriptor schema used by Tensor field declaration tests.
// Representative tests: MultiTensorFieldsRetainCanonicalFieldContracts and
// MultiTensorFieldsRejectInvalidSelectorsAndUnqualifiedContracts.
struct MultiTensorGraphSchema {
  std::shared_ptr<Tensor> left;
  std::shared_ptr<Tensor> right;
  std::shared_ptr<Tensor> calibration;
};

struct MultiTensorMissingFieldIds {};
struct MultiTensorShortFieldIds {};
struct MultiTensorOutOfRangeFieldIds {};

struct MultiTensorGraphFields {
  static constexpr TensorFieldSelector<MultiTensorGraphSchema> left{1U, 0U};
  static constexpr TensorFieldSelector<MultiTensorGraphSchema> right{3U, 1U};
  static constexpr TensorFieldSelector<MultiTensorGraphSchema> calibration{7U, 2U};
};

struct GraphMemoryKindParams {
  MemoryKind memory_kind{MemoryKind::kHost};
};

HOLOSCAN_PARAM_SCHEMA(GraphMemoryKindParams,
                      holoscan::param_schema("nvidia.holoscan.test.GraphMemoryKindParams", 1),
                      holoscan::field<&GraphMemoryKindParams::memory_kind>("memory_kind")
                          .enum_values({param_enum_value("host", MemoryKind::kHost),
                                        param_enum_value("pinned_host", MemoryKind::kPinnedHost),
                                        param_enum_value("cuda_device", MemoryKind::kCudaDevice),
                                        param_enum_value("cuda_managed", MemoryKind::kCudaManaged)})
                          .recompile_required());

template <>
consteval SchemaIdentity schema_identity<GraphSmallCapacitySchema>() {
  return schema_identity<schema::StringT>();
}

template <>
consteval SchemaIdentity schema_identity<GraphLargeCapacitySchema>() {
  return schema_identity<schema::StringT>();
}

template <>
consteval SchemaIdentity schema_identity<GraphOverflowCapacitySchema>() {
  return schema_identity<schema::StringT>();
}

template <>
consteval SchemaIdentity schema_identity<GraphInvalidSchemaPackage>() {
  return schema_identity<schema::StringT>();
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<GraphSmallCapacitySchema>() {
  return schema_family_identity<schema::StringT>();
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<GraphLargeCapacitySchema>() {
  return schema_family_identity<schema::StringT>();
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<GraphOverflowCapacitySchema>() {
  return schema_family_identity<schema::StringT>();
}

template <>
consteval SchemaFamilyIdentity schema_family_identity<GraphInvalidSchemaPackage>() {
  return schema_family_identity<schema::StringT>();
}

template <>
consteval SchemaIdentity schema_identity<MultiTensorGraphSchema>() {
  return schema_identity<schema::ImageT>();
}

template <>
struct DescriptorTensorSchemaNativeTraits<MultiTensorGraphSchema> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 3U;
  static constexpr std::array<std::uint16_t, 3U> tensor_field_ids{1U, 3U, 7U};

  [[nodiscard]] static const Tensor* tensor_at(const MultiTensorGraphSchema& value,
                                               const std::size_t index) noexcept {
    switch (index) {
      case 0U:
        return value.left.get();
      case 1U:
        return value.right.get();
      case 2U:
        return value.calibration.get();
      default:
        return nullptr;
    }
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<MultiTensorMissingFieldIds> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 64U;
  static constexpr std::size_t tensor_count = 2U;
  [[nodiscard]] static const Tensor* tensor_at(const MultiTensorMissingFieldIds&,
                                               std::size_t) noexcept {
    return nullptr;
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<MultiTensorShortFieldIds> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 64U;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, 1U> tensor_field_ids{1U};
  [[nodiscard]] static const Tensor* tensor_at(const MultiTensorShortFieldIds&,
                                               std::size_t) noexcept {
    return nullptr;
  }
};

template <>
struct DescriptorTensorSchemaNativeTraits<MultiTensorOutOfRangeFieldIds> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 64U;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint32_t, 2U> tensor_field_ids{1U, 65536U};
  [[nodiscard]] static const Tensor* tensor_at(const MultiTensorOutOfRangeFieldIds&,
                                               std::size_t) noexcept {
    return nullptr;
  }
};

static_assert(detail::HasDescriptorTensorSchemaNativeTraits<MultiTensorGraphSchema>);
static_assert(!detail::HasDescriptorTensorSchemaNativeTraits<MultiTensorMissingFieldIds>);
static_assert(!detail::HasDescriptorTensorSchemaNativeTraits<MultiTensorShortFieldIds>);
static_assert(!detail::HasDescriptorTensorSchemaNativeTraits<MultiTensorOutOfRangeFieldIds>);

template <typename ContractT, typename SelectorT>
concept AcceptsDescriptorTopicField =
    requires(ContractT contract, SelectorT selector) { contract.field(selector); };

static_assert(AcceptsDescriptorTopicField<DescriptorTensorTopicContract<MultiTensorGraphSchema>,
                                          TensorFieldSelector<MultiTensorGraphSchema>>);
static_assert(!AcceptsDescriptorTopicField<DescriptorTensorTopicContract<MultiTensorGraphSchema>,
                                           TensorFieldSelector<schema::ImageT>>);

template <>
struct GeneratedSchemaNativeTraits<GraphSmallCapacitySchema> {
  using root_type = schema::String;
  static constexpr std::size_t max_serialized_size = 64U;
};

template <>
struct GeneratedSchemaNativeTraits<GraphLargeCapacitySchema> {
  using root_type = schema::String;
  static constexpr std::size_t max_serialized_size = 128U;
};

template <>
struct GeneratedSchemaNativeTraits<GraphOverflowCapacitySchema> {
  using root_type = schema::String;
  static constexpr std::size_t max_serialized_size =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
};

template <>
struct GeneratedSchemaNativeTraits<GraphInvalidSchemaPackage> {
  using root_type = schema::String;
  static constexpr std::size_t max_serialized_size = 64U;
};

template <>
const SchemaPackageDescriptor<GraphSmallCapacitySchema>&
schema_package<GraphSmallCapacitySchema>() noexcept {
  static const SchemaPackageDescriptor<GraphSmallCapacitySchema> package{
      schema_identity<GraphSmallCapacitySchema>(),
      {},
      SchemaCodecCapacity{.max_serialized_size = 64U},
      {}};
  return package;
}

template <>
const SchemaPackageDescriptor<GraphLargeCapacitySchema>&
schema_package<GraphLargeCapacitySchema>() noexcept {
  static const SchemaPackageDescriptor<GraphLargeCapacitySchema> package{
      schema_identity<GraphLargeCapacitySchema>(),
      {},
      SchemaCodecCapacity{.max_serialized_size = 128U},
      {}};
  return package;
}

template <>
const SchemaPackageDescriptor<GraphOverflowCapacitySchema>&
schema_package<GraphOverflowCapacitySchema>() noexcept {
  static const SchemaPackageDescriptor<GraphOverflowCapacitySchema> package{
      schema_identity<GraphOverflowCapacitySchema>(),
      {},
      SchemaCodecCapacity{
          .max_serialized_size =
              GeneratedSchemaNativeTraits<GraphOverflowCapacitySchema>::max_serialized_size},
      {}};
  return package;
}

template <>
const SchemaPackageDescriptor<GraphInvalidSchemaPackage>&
schema_package<GraphInvalidSchemaPackage>() noexcept {
  static const SchemaPackageDescriptor<GraphInvalidSchemaPackage> package{
      schema_identity<GraphInvalidSchemaPackage>(),
      {},
      SchemaCodecCapacity{.max_serialized_size = 64U},
      SchemaTransportProfile{},
  };
  return package;
}

namespace detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<MultiTensorGraphSchema>() noexcept {
  static const RuntimePortTypeOps ops = schema::make_descriptor_tensor_port_type_ops<
      MultiTensorGraphSchema,
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>(
      [](void*) noexcept -> expected<void, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*) noexcept {},
      [](const MultiTensorGraphSchema&,
         schema::DescriptorTensorReferences,
         std::span<std::byte>) noexcept -> expected<std::size_t, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*, const schema::Image&, std::span<Tensor>, SampleMetadata) noexcept
          -> expected<void, Error> { return make_unexpected(Error{ErrorCode::kNotImplemented}); });
  return &ops;
}

}  // namespace detail

namespace {

using namespace test::graph_support;

// Authors a raw Tensor output whose pool capacity comes from a byte-span expression.
// Used by TensorConfiguratorPreservesDerivedAllocationForPassThroughRefinement.
class DerivedTensorOutputOp final : public Operator<> {
 public:
  explicit DerivedTensorOutputOp(std::uint64_t bytes = 4096U) : bytes_(bytes) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            tensor_byte_span(bytes_));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;

 private:
  std::uint64_t bytes_;
};

class DerivedMemoryKindOutputOp final : public Operator<GraphMemoryKindParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.rank = 1U},
                .bounds = tensor_bounds(4096U),
                .storage = TensorOutputStorage::kRuntimePool,
            },
            TensorOutputDerivations{
                .memory_kind =
                    parameter_memory_kind(param_ref<&GraphMemoryKindParams::memory_kind>()),
            });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Authors field-scoped derived capacities for a two-Tensor descriptor schema.
// Used by TensorConfiguratorRequiresFieldScopeForDerivedMultiTensorOutput.
class DerivedMultiTensorOutputOp final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& builder = spec.output(output, "output");
    builder.tensor_field(MultiTensorGraphFields::left)
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            tensor_byte_span(1024U));
    builder.tensor_field(MultiTensorGraphFields::right)
        .produces_tensor(
            TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kCudaDevice, .rank = 2U},
                .storage = TensorOutputStorage::kRuntimePool,
            },
            tensor_byte_span(2048U));
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<MultiTensorGraphSchema> output;
};

// Passes a descriptor-backed Image schema through an ordinary typed edge.
// Used by DescriptorTensorImageComposesAsTypedGraphFlow.
class ImagePassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;
  Output<schema::ImageT> output;
};

// Raw Tensor pass-through authored through the shared directional contracts.
// Representative test: CommonTensorContractsNormalizeIntoOwnedPortRecords.
class CommonTensorPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    constexpr std::array<std::int64_t, 2U> shape{16, 8};
    const TensorRepresentation representation{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
    };
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{
            .representation = representation,
            .bounds = tensor_bounds(shape),
            .memory_conversion = MemoryConversionPolicy::kDisable,
        });
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = representation,
            .bounds = tensor_bounds(shape),
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
  Output<Tensor> output;
};

// Single-Tensor Image pass-through that exercises every unqualified Tensor port configurator.
// Representative test: SingleTensorSchemaPreservesUnqualifiedFieldAuthoring.
class SingleTensorUnqualifiedPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation layout{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
    };
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{
            .representation = layout,
            .memory_conversion = MemoryConversionPolicy::kDisable,
        });
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = layout,
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;
  Output<schema::ImageT> output;
};

// Single-Tensor Image pass-through that authors the generated field selector form.
// Representative test: SingleTensorQualifiedFieldNormalizesToLegacyPortContract.
class SingleTensorQualifiedPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation layout{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
    };
    constexpr TensorFieldSelector<schema::ImageT> kData{4U, 0U};
    spec.input(input, "input")
        .tensor_field(kData)
        .expects_tensor(TensorInputSpec{
            .representation = layout,
            .memory_conversion = MemoryConversionPolicy::kDisable,
        });
    spec.output(output, "output")
        .tensor_field(kData)
        .produces_tensor(TensorOutputSpec{
            .representation = layout,
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;
  Output<schema::ImageT> output;
};

// Mixes qualified and legacy unqualified declarations for one Tensor field.
// Representative test: SingleTensorQualifiedFieldRejectsMixedScope.
class InvalidSingleTensorMixedScopeOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    constexpr TensorFieldSelector<schema::ImageT> kData{4U, 0U};
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost}})
        .tensor_field(kData)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<schema::ImageT> output;
};

// Three-Tensor pass-through with field-scoped host, device, and pinned-host contracts.
// Representative test: MultiTensorFieldsPreserveCanonicalTypedContracts.
class MultiTensorFieldPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation host_image{
        .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U};
    const TensorRepresentation device_image{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U};
    const TensorRepresentation calibration{
        .memory_kind = MemoryKind::kPinnedHost, .dtype = DLDataType{kDLFloat, 32U, 1U}, .rank = 1U};

    auto& input_builder = spec.input(input, "input");
    input_builder.tensor_field(MultiTensorGraphFields::left)
        .expects_tensor(TensorInputSpec{.representation = host_image});
    input_builder.tensor_field(MultiTensorGraphFields::right)
        .expects_tensor(TensorInputSpec{.representation = device_image});
    input_builder.tensor_field(MultiTensorGraphFields::calibration)
        .expects_tensor(TensorInputSpec{
            .representation = calibration,
            .memory_conversion = MemoryConversionPolicy::kDisable,
        });

    auto& output_builder = spec.output(output, "output");
    output_builder.tensor_field(MultiTensorGraphFields::left)
        .produces_tensor(TensorOutputSpec{
            .representation = host_image,
            .bounds = tensor_bounds(4096U),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
    output_builder.tensor_field(MultiTensorGraphFields::right)
        .produces_tensor(TensorOutputSpec{
            .representation = device_image,
            .bounds = tensor_bounds(8192U),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
    output_builder.tensor_field(MultiTensorGraphFields::calibration)
        .produces_tensor(TensorOutputSpec{
            .representation = calibration,
            .bounds = tensor_bounds(256U),
            .storage = TensorOutputStorage::kRuntimePool,
            .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                 InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<MultiTensorGraphSchema> input;
  Output<MultiTensorGraphSchema> output;
};

// Three-Tensor pass-through that uses shared direction-specific field contracts.
// Representative test: MultiTensorContractsLowerToCanonicalFieldRecords.
class MultiTensorContractPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    constexpr std::array kImageFields{MultiTensorGraphFields::left, MultiTensorGraphFields::right};
    const TensorRepresentation image_representation{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
    };
    const TensorRepresentation calibration_representation{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 32U, 1U},
        .rank = 0U,
    };
    const TensorInputSpec image_input{
        .representation = image_representation,
        .memory_conversion = MemoryConversionPolicy::kDisable,
    };
    const TensorOutputSpec image_output{
        .representation = image_representation,
        .bounds = tensor_bounds(4096U),
        .storage = TensorOutputStorage::kRuntimePool,
        .transfer =
            TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };

    auto& input_builder = spec.input(input, "input");
    input_builder.tensor_fields(kImageFields, image_input);
    input_builder.tensor_field(MultiTensorGraphFields::calibration,
                               TensorInputSpec{.representation = calibration_representation});

    auto& output_builder = spec.output(output, "output");
    output_builder.tensor_fields(kImageFields, image_output);
    output_builder.tensor_field(MultiTensorGraphFields::calibration,
                                TensorOutputSpec{
                                    .representation = calibration_representation,
                                    .bounds = tensor_bounds(256U),
                                    .storage = TensorOutputStorage::kRuntimePool,
                                });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<MultiTensorGraphSchema> input;
  Output<MultiTensorGraphSchema> output;
};

// Three-Tensor pass-through spelling the concise contracts through the original fluent API.
// Representative test: ConciseMultiTensorContractsMatchFluentGraphIdentity.
class MultiTensorFluentContractPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation image_representation{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
    };
    const TensorRepresentation calibration_representation{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 32U, 1U},
        .rank = 0U,
    };

    auto& input_builder = spec.input(input, "input");
    for (const auto field : {MultiTensorGraphFields::left, MultiTensorGraphFields::right}) {
      input_builder.tensor_field(field).expects_tensor(TensorInputSpec{
          .representation = image_representation,
          .memory_conversion = MemoryConversionPolicy::kDisable,
      });
    }
    input_builder.tensor_field(MultiTensorGraphFields::calibration)
        .expects_tensor(TensorInputSpec{.representation = calibration_representation});

    auto& output_builder = spec.output(output, "output");
    for (const auto field : {MultiTensorGraphFields::left, MultiTensorGraphFields::right}) {
      output_builder.tensor_field(field).produces_tensor(TensorOutputSpec{
          .representation = image_representation,
          .bounds = tensor_bounds(4096U),
          .storage = TensorOutputStorage::kRuntimePool,
          .transfer =
              TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
      });
    }
    output_builder.tensor_field(MultiTensorGraphFields::calibration)
        .produces_tensor(TensorOutputSpec{
            .representation = calibration_representation,
            .bounds = tensor_bounds(256U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<MultiTensorGraphSchema> input;
  Output<MultiTensorGraphSchema> output;
};

// Three-Tensor pass-through with a partial input and complete output heterogeneous aggregate.
// Representative test: MultiTensorAggregateContractsCanonicalizeAuthoredOrder.
class MultiTensorAggregateContractPassOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation image_layout{
        .memory_kind = MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 2U,
    };
    const TensorRepresentation calibration_layout{
        .memory_kind = MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 32U, 1U},
        .rank = 1U,
    };
    const TensorInputSpec image_input{.representation = image_layout};
    const TensorInputSpec calibration_input{
        .representation = calibration_layout,
        .memory_conversion = MemoryConversionPolicy::kDisable,
    };
    const TensorOutputSpec image_output{
        .representation = image_layout,
        .bounds = tensor_bounds(4096U),
        .storage = TensorOutputStorage::kRuntimePool,
    };
    const TensorOutputSpec calibration_output{
        .representation = calibration_layout,
        .bounds = tensor_bounds(256U),
        .storage = TensorOutputStorage::kRuntimePool,
    };

    spec.input(input, "input")
        .tensor_fields(std::array{
            holoscan::tensor_field(MultiTensorGraphFields::right, image_input),
            holoscan::tensor_field(MultiTensorGraphFields::calibration, calibration_input),
        });
    spec.output(output, "output")
        .tensor_fields(std::array{
            holoscan::tensor_field(MultiTensorGraphFields::right, image_output),
            holoscan::tensor_field(MultiTensorGraphFields::calibration, calibration_output),
            holoscan::tensor_field(MultiTensorGraphFields::left, image_output),
        });
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<MultiTensorGraphSchema> input;
  Output<MultiTensorGraphSchema> output;
};

enum class InvalidMultiTensorDeclarationKind : std::uint8_t {
  kDuplicateAggregateSelector,
  kUnqualifiedField,
  kForeignSelector,
  kDuplicateSelector,
  kUnknownField,
  kInvalidDtype,
  kInvalidAllocation,
  kUnresolvedRank,
  kZeroInlineBytes,
  kZeroTensorSpan,
  kInconsistentInlineSpan,
};

// Configurable malformed declaration model. Each case isolates one authoring error so the
// parameterized test below retains a separate, named diagnostic identity without duplicating
// eleven Operator classes.
// Used by InvalidMultiTensorDeclarationTest.RejectsBeforeGraphPublication.
// REQ: OPERATOR-036
class InvalidMultiTensorDeclarationOp : public Operator<> {
 public:
  explicit InvalidMultiTensorDeclarationOp(InvalidMultiTensorDeclarationKind kind) : kind_(kind) {}

  void setup(OperatorSpec& spec) override {
    switch (kind_) {
      case InvalidMultiTensorDeclarationKind::kDuplicateAggregateSelector: {
        const TensorOutputSpec contract{
            .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
            .bounds = tensor_bounds(256U),
            .storage = TensorOutputStorage::kRuntimePool,
        };
        spec.output(output, "output")
            .tensor_fields(std::array{
                holoscan::tensor_field(MultiTensorGraphFields::left, contract),
                holoscan::tensor_field(MultiTensorGraphFields::left, contract),
                holoscan::tensor_field(MultiTensorGraphFields::calibration, contract),
            });
        break;
      }
      case InvalidMultiTensorDeclarationKind::kUnqualifiedField:
        spec.output(output, "output")
            .produces_tensor(
                TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost}});
        break;
      case InvalidMultiTensorDeclarationKind::kForeignSelector:
        spec.output(output, "output")
            .tensor_field(TensorFieldSelector<schema::ImageT>{4U, 0U})
            .produces_tensor(
                TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost}});
        break;
      case InvalidMultiTensorDeclarationKind::kDuplicateSelector: {
        auto& builder = spec.output(output, "output");
        builder.tensor_field(MultiTensorGraphFields::left)
            .produces_tensor(
                TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost}});
        builder.tensor_field(MultiTensorGraphFields::left)
            .produces_tensor(
                TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost}});
        break;
      }
      case InvalidMultiTensorDeclarationKind::kUnknownField:
        spec.output(output, "output")
            .tensor_field(TensorFieldSelector<MultiTensorGraphSchema>{9U, 1U})
            .produces_tensor(
                TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kHost}});
        break;
      case InvalidMultiTensorDeclarationKind::kInvalidDtype:
        spec.output(output, "output")
            .tensor_field(MultiTensorGraphFields::left)
            .produces_tensor(
                TensorOutputSpec{.representation = {.dtype = DLDataType{kDLUInt, 0U, 1U}}});
        break;
      case InvalidMultiTensorDeclarationKind::kInvalidAllocation:
        spec.output(output, "output")
            .tensor_field(MultiTensorGraphFields::left)
            .produces_tensor(TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost},
                .bounds = TensorBounds{.max_rank = 0U, .max_byte_span = 4096U},
                .storage = TensorOutputStorage::kRuntimePool,
            });
        break;
      case InvalidMultiTensorDeclarationKind::kUnresolvedRank:
        spec.output(output, "output")
            .tensor_field(MultiTensorGraphFields::left,
                          TensorOutputSpec{
                              .representation = {.memory_kind = MemoryKind::kHost},
                              .bounds = tensor_bounds(4096U),
                              .storage = TensorOutputStorage::kRuntimePool,
                          });
        break;
      case InvalidMultiTensorDeclarationKind::kZeroInlineBytes:
      case InvalidMultiTensorDeclarationKind::kZeroTensorSpan:
      case InvalidMultiTensorDeclarationKind::kInconsistentInlineSpan: {
        TensorTransferPolicy transfer;
        if (kind_ == InvalidMultiTensorDeclarationKind::kZeroInlineBytes) {
          transfer.max_inline_bytes = 0U;
          transfer.foreign_emit_policy =
              InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
        } else if (kind_ == InvalidMultiTensorDeclarationKind::kZeroTensorSpan) {
          transfer.max_staging_byte_span = 0U;
          transfer.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
        } else {
          transfer.max_inline_bytes = 64U;
          transfer.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
        }
        spec.output(output, "output")
            .tensor_field(MultiTensorGraphFields::left)
            .produces_tensor(TensorOutputSpec{
                .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
                .bounds = tensor_bounds(256U),
                .storage = TensorOutputStorage::kRuntimePool,
                .transfer = transfer,
            });
        break;
      }
    }
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<MultiTensorGraphSchema> output;

 private:
  InvalidMultiTensorDeclarationKind kind_;
};

// Carries GraphSchemaPayload without marking any tensor-emit policy.
// Verifies that PayloadOptions are rejected for schema ports not declared as tensor carriers.
// Representative test: PayloadOptionsRejectScalarAndUnmarkedCarrierProvenance.
class UnmarkedCarrierOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<GraphSchemaPayload> input;
  Output<GraphSchemaPayload> output;
};

// Declares a generated-schema output backed by an invalid local package so graph admission
// can prove it rejects package defects before recording the operator or its ports.
// Representative test: RejectsInvalidGeneratedPackageDuringOperatorSetup.
class InvalidSchemaPackageOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<GraphInvalidSchemaPackage> output;
};

// Carries GraphSchemaPayload alongside a Tensor sidecar using an inline-host emit policy.
// Exercises schema-carrier payload-option normalization through a typed schema port.
// Representative test: TensorBearingSchemaCarrierPreservesExactPayloadOptions.
class TensorCarrierOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorRepresentation layout{
        .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLUInt, 8, 1}, .rank = 1};
    PayloadOptions options;
    options.schema = schema_identity<GraphSchemaPayload>();
    options.interface_tensor_emit_policy =
        InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
    options.max_inline_bytes = 256;
    PayloadOptions output_options;
    output_options.schema = schema_identity<GraphSchemaPayload>();
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{.representation = layout})
        .payload_options(options);
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = layout,
            .bounds = tensor_bounds(4096U),
            .transfer =
                TensorTransferPolicy{
                    .foreign_emit_policy =
                        InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
                    .max_inline_bytes = 256U,
                },
        })
        .payload_options(output_options);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<GraphSchemaPayload> input;
  Output<GraphSchemaPayload> output;
};

// Sets PayloadOptions on a plain scalar (uint32_t) output, which is not permitted.
// Representative test: PayloadOptionsRejectScalarAndUnmarkedCarrierProvenance.
class InvalidScalarPayloadOptionsOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").payload_options(PayloadOptions{});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<std::uint32_t> output;
};

// Sets PayloadOptions on a POD output, exercising the same scalar-payload rejection path.
// Representative test: PayloadOptionsRejectScalarAndUnmarkedCarrierProvenance.
class InvalidPodPayloadOptionsOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").payload_options(PayloadOptions{});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<GraphPodPayload> output;
};

// Applies kCopyIfNeeded emit policy to a schema output without a tensor layout annotation.
// Representative test: PayloadOptionsRejectScalarAndUnmarkedCarrierProvenance.
class InvalidUnmarkedCarrierPolicyOp : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    PayloadOptions options;
    options.schema = schema_identity<GraphSchemaPayload>();
    options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
    spec.output(output, "output").payload_options(options);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<GraphSchemaPayload> output;
};

struct EdgePorts {
  const PortRecord* source{};
  const PortRecord* target{};
};

// Resolves the planned source and target declarations for a one-edge Graph snapshot.
// Tests retain the size assertion so malformed topology still reports at the call site.
EdgePorts single_edge_ports(const GraphView& view) {
  if (view.edges().size() != 1U) {
    return {};
  }
  const auto& edge = view.edges().front();
  const auto find_port = [&](const GraphPortId id) -> const PortRecord* {
    const auto port = std::ranges::find_if(
        view.ports(), [&](const PortRecord& record) { return record.id == id; });
    return port == view.ports().end() ? nullptr : &*port;
  };
  return {.source = find_port(edge.source), .target = find_port(edge.target)};
}
template <typename T>
concept HasTensorAllocationConfigurator = requires(Graph& graph, Output<T>& output) {
  graph.configure(output).tensor_allocation(TensorPoolBounds{.max_rank = 1, .max_byte_span = 1});
};
template <typename Port>
concept HasTensorAllocation = requires(Port& port, TensorLoanRequest request) {
  { port.allocate_tensor(request) } -> std::same_as<expected<TensorOutputLoan, Error>>;
};
template <typename Port>
concept HasTensorShapeAllocation = requires(Port& port, TensorShapeView shape) {
  { port.allocate_tensor(shape) } -> std::same_as<expected<TensorOutputLoan, Error>>;
};
template <typename Port, typename Descriptor>
concept HasDescriptorTensorEmit =
    requires(Port& port, TensorOutputLoan loan, const Descriptor& descriptor) {
      { port.emit_tensor(std::move(loan), descriptor) } -> std::same_as<expected<void, Error>>;
    };
template <typename Port, typename Schema>
concept HasTensorBundleAllocation =
    requires(Port& port, std::span<const TensorFieldLoanRequest<Schema>> requests) {
      {
        port.allocate_tensors(requests)
      } -> std::same_as<expected<TensorSchemaOutputLoan<Schema>, Error>>;
    };
template <typename Port, typename Schema>
concept HasTensorShapeBundleAllocation =
    requires(Port& port, std::span<const TensorFieldShapeRequest<Schema>> requests) {
      {
        port.allocate_tensors(requests)
      } -> std::same_as<expected<TensorSchemaOutputLoan<Schema>, Error>>;
    };
template <typename Port, typename Schema>
concept HasTensorBundleEmit =
    requires(Port& port, TensorSchemaOutputLoan<Schema> loan, const Schema& descriptor) {
      { port.emit_tensors(std::move(loan), descriptor) } -> std::same_as<expected<void, Error>>;
    };
static_assert(!HasTensorAllocationConfigurator<Tensor>);
// REQ: OPERATOR-031
// REQ: OPERATOR-035
static_assert(!HasTensorAllocationConfigurator<schema::ImageT>);
static_assert(HasTensorAllocation<Output<schema::ImageT>>);
static_assert(HasTensorShapeAllocation<Output<schema::ImageT>>);
static_assert(HasDescriptorTensorEmit<Output<schema::ImageT>, schema::ImageT>);
static_assert(!HasTensorAllocationConfigurator<schema::TensorPacketT>);
static_assert(HasTensorAllocation<Output<schema::TensorPacketT>>);
static_assert(HasDescriptorTensorEmit<Output<schema::TensorPacketT>, schema::TensorPacketT>);
static_assert(!HasDescriptorTensorEmit<Output<Tensor>, Tensor>);
static_assert(!HasTensorAllocation<Output<std::uint32_t>>);
static_assert(!HasTensorShapeAllocation<Output<std::uint32_t>>);
static_assert(!HasDescriptorTensorEmit<Output<std::uint32_t>, std::uint32_t>);
static_assert(!HasTensorAllocationConfigurator<std::uint32_t>);
// REQ: OPERATOR-037
static_assert(HasTensorBundleAllocation<Output<MultiTensorGraphSchema>, MultiTensorGraphSchema>);
static_assert(
    HasTensorShapeBundleAllocation<Output<MultiTensorGraphSchema>, MultiTensorGraphSchema>);
static_assert(HasTensorBundleEmit<Output<MultiTensorGraphSchema>, MultiTensorGraphSchema>);
static_assert(!HasTensorAllocation<Output<MultiTensorGraphSchema>>);
static_assert(!HasDescriptorTensorEmit<Output<MultiTensorGraphSchema>, MultiTensorGraphSchema>);
static_assert(!HasTensorBundleAllocation<Output<schema::ImageT>, schema::ImageT>);
static_assert(!HasTensorShapeBundleAllocation<Output<schema::ImageT>, schema::ImageT>);
static_assert(!HasTensorBundleEmit<Output<schema::ImageT>, schema::ImageT>);
static_assert(!HasTensorBundleAllocation<Output<Tensor>, Tensor>);
static_assert(HasTensorShapeAllocation<Output<Tensor>>);
static_assert(!HasTensorBundleEmit<Output<std::uint32_t>, std::uint32_t>);
static_assert(std::is_nothrow_move_constructible_v<TensorSchemaOutputLoan<MultiTensorGraphSchema>>);
static_assert(!std::is_copy_constructible_v<TensorSchemaOutputLoan<MultiTensorGraphSchema>>);
static_assert(!std::is_move_assignable_v<TensorSchemaOutputLoan<MultiTensorGraphSchema>>);

struct TensorShapeRequestCapture {
  detail::TensorSchemaLoanRequestList requests;
  std::size_t call_count{};
};

expected<detail::TensorSchemaLoanBundleStorage, Error> capture_tensor_shape_requests(
    void* state, const detail::TensorSchemaLoanRequestList& requests) noexcept {
  auto& capture = *static_cast<TensorShapeRequestCapture*>(state);
  capture.requests = requests;
  ++capture.call_count;
  return make_unexpected(Error{ErrorCode::kResourceExhausted});
}

const detail::TensorOutputBindingOps kTensorShapeCaptureOps{
    .allocate_tensors = &capture_tensor_shape_requests,
};

TEST(RuntimePortBindingAccess, EmptyDescriptorTensorSpanIsCallerErrorBeforeReadiness) {
  Output<MultiTensorGraphSchema> output;
  MultiTensorGraphSchema descriptor;
  // Key setup: call the private Runtime adapter with no active binding and either an empty or
  // one-element authoritative Tensor inventory.
  const auto empty = detail::RuntimePortBindingAccess::emit_descriptor_tensors(
      output, &descriptor, std::span<const Tensor>{}, {});
  const std::array<Tensor, 1U> one_tensor{};
  const auto unbound = detail::RuntimePortBindingAccess::emit_descriptor_tensors(
      output, &descriptor, std::span<const Tensor>{one_tensor}, {});
  // Key assertion: an empty inventory is malformed caller input; a valid inventory still reports
  // the independently missing Runtime binding.
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(unbound);
  EXPECT_EQ(unbound.error().code, ErrorCode::kNotReady);
}

TEST(RuntimePortBindingAccess, TensorShapeRequestsUseExactBoundFieldDtypes) {
  constexpr std::array<std::int64_t, 1U> kShape{4};
  const std::array<std::optional<DLDataType>, 3U> dtypes{
      DLDataType{kDLFloat, 32U, 1U},
      DLDataType{kDLUInt, 8U, 1U},
      DLDataType{kDLInt, 16U, 1U},
  };
  const std::array requests{
      tensor_shape(MultiTensorGraphFields::calibration, kShape),
      tensor_shape(MultiTensorGraphFields::left, kShape),
      tensor_shape(MultiTensorGraphFields::right, kShape),
  };
  TensorShapeRequestCapture capture;
  Output<MultiTensorGraphSchema> output;
  detail::TensorOutputPortBindingAccess::bind(
      output, {.state = &capture, .ops = &kTensorShapeCaptureOps, .tensor_field_dtypes = dtypes});

  // Key setup: allocate a noncanonical shapes-only inventory through an exact-dtype binding.
  const auto allocated = output.allocate_tensors(requests);

  // Key assertion: the erased request passed to Runtime carries each canonical contract dtype.
  ASSERT_FALSE(allocated);
  EXPECT_EQ(allocated.error().code, ErrorCode::kResourceExhausted);
  ASSERT_EQ(capture.call_count, 1U);
  ASSERT_EQ(capture.requests.field_count, 3U);
  EXPECT_EQ(capture.requests.fields[0].tensor.dtype.code, kDLInt);
  EXPECT_EQ(capture.requests.fields[1].tensor.dtype.code, kDLFloat);
  EXPECT_EQ(capture.requests.fields[2].tensor.dtype.code, kDLUInt);
  detail::TensorOutputPortBindingAccess::unbind(output);
}

TEST(RuntimePortBindingAccess, TensorShapeRequestsRejectUnpinnedFieldDtype) {
  constexpr std::array<std::int64_t, 1U> kShape{4};
  const std::array<std::optional<DLDataType>, 3U> dtypes{
      DLDataType{kDLFloat, 32U, 1U},
      std::nullopt,
      DLDataType{kDLInt, 16U, 1U},
  };
  const std::array requests{
      tensor_shape(MultiTensorGraphFields::left, kShape),
      tensor_shape(MultiTensorGraphFields::right, kShape),
      tensor_shape(MultiTensorGraphFields::calibration, kShape),
  };
  TensorShapeRequestCapture capture;
  Output<MultiTensorGraphSchema> output;
  detail::TensorOutputPortBindingAccess::bind(
      output, {.state = &capture, .ops = &kTensorShapeCaptureOps, .tensor_field_dtypes = dtypes});

  // Key setup: request a field whose admitted contract deliberately omits exact dtype.
  const auto allocated = output.allocate_tensors(requests);

  // Key assertion: failure identifies the field before any Runtime allocation callback runs.
  ASSERT_FALSE(allocated);
  EXPECT_EQ(allocated.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(allocated.error().message,
            "TENSOR_SCHEMA_BUNDLE_FIELD_DTYPE_UNRESOLVED (field_id=3, tensor_index=1)");
  EXPECT_EQ(capture.call_count, 0U);
  detail::TensorOutputPortBindingAccess::unbind(output);
}
// REQ: GRAPH-009
// REQ: GRAPH-010
// REQ: GRAPH-011
// Graph topology:
//   holomq://schema --> imported schema port
//   exported POD port --> holomq://pod
// REQ: GRAPH-034
TEST(GraphComposition, SnapshotsDeepOwnNamesSchemaAndPodIdentityAfterGraphDestruction) {
  GraphView saved;
  {
    Graph graph{"root"};
    std::string endpoint = "schema_endpoint";
    TopicOptions schema_options;
    schema_options.endpoint_name = endpoint;
    (void)graph.import_topic<GraphSchemaPayload>("holomq://schema", schema_options);
    endpoint.assign("mutated");
    (void)graph.export_topic<GraphPodPayload>("holomq://pod");
    saved = graph.view();
  }

  ASSERT_EQ(saved.external_topics().size(), 2);
  EXPECT_EQ(saved.external_topics()[0].endpoint_name, "schema_endpoint");
  ASSERT_TRUE(saved.external_topics()[0].payload_identity.schema);
  // GoogleTest assertions establish these invariants, but clang-tidy does not model the macros.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(saved.external_topics()[0].payload_identity.schema.value().canonical_name,
            "holoscan.schema.String");
  ASSERT_TRUE(saved.external_topics()[1].payload_identity.pod);
  ASSERT_EQ(saved.external_topics()[1].payload_identity.pod.value().fields.size(), 1);
  EXPECT_EQ(saved.external_topics()[1].payload_identity.pod.value().fields[0].name, "value");
  EXPECT_EQ(saved.external_topics()[1].payload_identity.pod.value().fields[0].byte_offset, 0);
  // NOLINTEND(bugprone-unchecked-optional-access)
}
// REQ: GRAPH-024
// REQ: CHAN-075
// REQ: CHAN-120
// Graph topologies:
//   holomq://capacity --> small imported schema port
//   holomq://capacity --> large imported schema port
TEST(GraphComposition, ExternalSchemaCapacityParticipatesInCanonicalHash) {
  Graph small{"same"};
  (void)small.import_topic<GraphSmallCapacitySchema>("holomq://capacity");
  Graph large{"same"};
  (void)large.import_topic<GraphLargeCapacitySchema>("holomq://capacity");

  const GraphView small_view = small.view();
  const GraphView large_view = large.view();
  ASSERT_EQ(small_view.external_topics().size(), 1U);
  ASSERT_EQ(large_view.external_topics().size(), 1U);
  ASSERT_TRUE(small_view.external_topics().front().payload_identity.schema);
  ASSERT_TRUE(large_view.external_topics().front().payload_identity.schema);
  // GTest's fatal assertions establish engagement, but clang-tidy does not model that macro edge.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(small_view.external_topics().front().payload_identity.schema->family_id,
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            large_view.external_topics().front().payload_identity.schema->family_id);
  EXPECT_EQ(small_view.external_topics().front().serialized_payload_capacity_bytes, 64U);
  EXPECT_EQ(large_view.external_topics().front().serialized_payload_capacity_bytes, 128U);
  EXPECT_NE(small_view.identity().hash, large_view.identity().hash);
}

// REQ: GRAPH-024
// REQ: CHAN-075
// REQ: CHAN-120
// Graph topology:
//   holomq://capacity-overflow -X-> imported schema port
TEST(GraphComposition, ExternalSchemaCapacityMustFitSampleMetadata) {
  Graph graph{"schema-capacity-bound"};
  const Error error = graph_error_from(
      [&] { (void)graph.import_topic<GraphOverflowCapacitySchema>("holomq://capacity-overflow"); });
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_SCHEMA_CAPACITY_INVALID");
}
// REQ: GRAPH-009
// REQ: GRAPH-012
// REQ: GRAPH-014
// Graph topology:
//   holomq://tensor --> imported raw Tensor port
TEST(GraphComposition, ExternalTensorOptionsAreDeepOwnedAndFailClosed) {
  Graph graph{"root"};
  std::array<std::int64_t, 2U> shape{1920, 1080};
  std::array<std::uint64_t, 2U> strides{4U, 7680U};
  TopicOptions options;
  options.endpoint_name = "tensor";
  options.memory_conversion = MemoryConversionPolicy::kAllowIfSafe;
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = std::chrono::milliseconds{10},
  };
  options.tensor = TensorBoundaryContract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kCudaDevice,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
          },
      .bounds = tensor_bounds(shape,
                              TensorBoundsOptions{
                                  .max_byte_span = 8U * 1024U * 1024U,
                                  .max_abs_byte_strides = strides,
                              }),
  };
  (void)graph.import_topic<Tensor>("holomq://tensor", options);
  shape.fill(1);
  strides.fill(1U);
  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 1);
  ASSERT_TRUE(view.external_topics()[0].tensor);
  // GoogleTest assertions establish this invariant, but clang-tidy does not model the macro.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(view.external_topics()[0].tensor.value().axes.size(), 2);
  EXPECT_EQ(view.external_topics()[0].tensor.value().axes[1].max_extent, 1080);
  EXPECT_EQ(view.external_topics()[0].tensor.value().axes[1].max_abs_stride_bytes, 7680U);
  // NOLINTEND(bugprone-unchecked-optional-access)

  TopicOptions invalid_health;
  invalid_health.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
  };
  EXPECT_THROW((void)graph.import_topic<Tensor>("holomq://invalid_health", invalid_health),
               RuntimeError);

  TopicOptions irrelevant;
  irrelevant.tensor = TensorBoundaryContract{};
  EXPECT_THROW((void)graph.import_topic<std::uint32_t>("holomq://not_tensor", irrelevant),
               RuntimeError);

  TopicOptions invalid_dtype;
  invalid_dtype.tensor = TensorBoundaryContract{
      .representation =
          holoscan::TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{static_cast<std::uint8_t>(0xFF), 8, 1},
              .rank = 1U,
          },
      .bounds = TensorBounds{.max_byte_span = 8U},
  };
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<Tensor>("holomq://invalid-dtype", invalid_dtype);
            }).message,
            "GRAPH_TENSOR_DTYPE_INVALID");
}

// REQ: GRAPH-012
// REQ: EXTERNAL-010
// Graph topology:
//   holomq://multi-tensor --> imported three-field descriptor port
TEST(GraphComposition, ExternalDescriptorTensorFieldsAreCanonicalAndDeepOwned) {
  constexpr DLDataType kByte{.code = kDLUInt, .bits = 8U, .lanes = 1U};
  std::array<std::int64_t, 1U> calibration_shape{64};
  DescriptorTensorTopicContract<MultiTensorGraphSchema> tensors;
  tensors
      .field(MultiTensorGraphFields::calibration,
             TensorBoundaryContract{
                 .representation = {.memory_kind = MemoryKind::kHost, .dtype = kByte, .rank = 1U},
                 .bounds = tensor_bounds(calibration_shape),
             })
      .field(MultiTensorGraphFields::left,
             TensorBoundaryContract{
                 .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                    .dtype = kByte,
                                    .rank = 2U},
                 .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 4096U},
             })
      .field(MultiTensorGraphFields::right,
             TensorBoundaryContract{
                 .representation = {.memory_kind = MemoryKind::kPinnedHost,
                                    .dtype = kByte,
                                    .rank = 1U},
                 .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 1024U},
                 .transfer =
                     TensorTransferPolicy{
                         .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                     },
             });

  // Key setup: move a complete three-field contract into graph authoring, then mutate the
  // caller-owned axis storage to expose any borrowed-view or noncanonical ordering bug.
  Graph graph;
  (void)graph.import_topic<MultiTensorGraphSchema>(
      "holomq://multi-tensor", TopicOptions{}, std::move(tensors));
  calibration_shape[0] = 1;

  // Key assertion: the graph owns a canonical field-id inventory and retains the original axis
  // bound independently of the caller's mutated storage.
  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 1U);
  ASSERT_EQ(view.ports().size(), 1U);
  const auto& topic = view.external_topics().front();
  EXPECT_EQ(topic.serialized_payload_capacity_bytes,
            detail::runtime_port_type_ops<MultiTensorGraphSchema>()->wire_capacity);
  ASSERT_EQ(topic.tensor_fields.size(), 3U);
  EXPECT_EQ(topic.tensor_fields[0].field_id, MultiTensorGraphFields::left.field_id);
  EXPECT_EQ(topic.tensor_fields[0].tensor_index, 0U);
  EXPECT_EQ(topic.tensor_fields[0].tensor.memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(topic.tensor_fields[1].field_id, MultiTensorGraphFields::right.field_id);
  EXPECT_EQ(topic.tensor_fields[1].tensor_index, 1U);
  EXPECT_EQ(topic.tensor_fields[1].tensor.foreign_emit_policy,
            InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_EQ(topic.tensor_fields[2].field_id, MultiTensorGraphFields::calibration.field_id);
  EXPECT_EQ(topic.tensor_fields[2].tensor_index, 2U);
  ASSERT_EQ(topic.tensor_fields[2].tensor.axes.size(), 1U);
  EXPECT_EQ(topic.tensor_fields[2].tensor.axes[0].max_extent, 64U);
  EXPECT_EQ(view.ports().front().descriptor_tensor_field_count, 3U);
}

// REQ: GRAPH-012
// REQ: EXTERNAL-010
// Graph topology:
//   exported Image port --> holomq://image
TEST(GraphComposition, ExternalSingularDescriptorTensorUsesTheSameFieldInventory) {
  constexpr TensorFieldSelector<schema::ImageT> kImageData{4U, 0U};
  DescriptorTensorTopicContract<schema::ImageT> tensors;
  tensors.field(kImageData,
                TensorBoundaryContract{
                    .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                       .dtype = DLDataType{kDLUInt, 8U, 1U},
                                       .rank = 3U},
                    .bounds = TensorBounds{.max_rank = 3U, .max_byte_span = 8U * 1024U * 1024U},
                });

  // Key setup: author the singular Image schema through the same explicit field inventory used
  // by multi-Tensor schemas rather than the legacy implicit Tensor path.
  Graph graph;
  (void)graph.export_topic<schema::ImageT>("holomq://image", TopicOptions{}, std::move(tensors));

  // Key assertion: the singular contract retains its selector identity and finite byte bound in
  // the external topic's canonical Tensor-field inventory.
  const GraphView view = graph.view();
  ASSERT_EQ(view.external_topics().size(), 1U);
  ASSERT_EQ(view.ports().size(), 1U);
  const auto& topic = view.external_topics().front();
  EXPECT_EQ(topic.serialized_payload_capacity_bytes,
            detail::runtime_port_type_ops<schema::ImageT>()->wire_capacity);
  ASSERT_EQ(topic.tensor_fields.size(), 1U);
  EXPECT_EQ(topic.tensor_fields.front().field_id, kImageData.field_id);
  EXPECT_EQ(topic.tensor_fields.front().tensor_index, 0U);
  EXPECT_EQ(topic.tensor_fields.front().tensor.max_byte_span, 8U * 1024U * 1024U);
}

// REQ: GRAPH-012
// REQ: EXTERNAL-010
// Graph topologies:
//   holomq://missing -X-> incomplete imported descriptor port
//   duplicate exported descriptor port -X-> holomq://duplicate
TEST(GraphComposition, ExternalDescriptorTensorFieldsRejectIncompleteAndDuplicateInventories) {
  const TensorBoundaryContract contract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 64U},
  };

  // Key setup: author one inventory missing a required field and another repeating a field while
  // omitting its sibling, covering both completeness and uniqueness validation.
  Graph incomplete;
  DescriptorTensorTopicContract<MultiTensorGraphSchema> missing;
  missing.field(MultiTensorGraphFields::left, contract)
      .field(MultiTensorGraphFields::right, contract);
  const Error missing_error = graph_error_from([&] {
    (void)incomplete.import_topic<MultiTensorGraphSchema>(
        "holomq://missing", TopicOptions{}, std::move(missing));
  });
  // Key assertion: both invalid inventories fail with their specific diagnostic and leave no
  // external-topic record behind.
  EXPECT_EQ(missing_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(missing_error.message, "GRAPH_DESCRIPTOR_TENSOR_TOPIC_INVENTORY_INCOMPLETE");
  EXPECT_TRUE(incomplete.view().external_topics().empty());

  Graph duplicate;
  DescriptorTensorTopicContract<MultiTensorGraphSchema> repeated;
  repeated.field(MultiTensorGraphFields::left, contract)
      .field(MultiTensorGraphFields::left, contract)
      .field(MultiTensorGraphFields::calibration, contract);
  const Error duplicate_error = graph_error_from([&] {
    (void)duplicate.export_topic<MultiTensorGraphSchema>(
        "holomq://duplicate", TopicOptions{}, std::move(repeated));
  });
  EXPECT_EQ(duplicate_error.code, ErrorCode::kDuplicateName);
  EXPECT_EQ(duplicate_error.message,
            "GRAPH_DESCRIPTOR_TENSOR_TOPIC_FIELD_DUPLICATE (field_id=1, tensor_index=0)");
  EXPECT_TRUE(duplicate.view().external_topics().empty());
}

// REQ: GRAPH-025
//   producer(CommonTensorPassOp) --[output→input]--> consumer(CommonTensorPassOp)
TEST(GraphComposition, CommonTensorContractsNormalizeIntoOwnedPortRecords) {
  // Key setup: both ends use role-specific shared contracts whose maximum-shape views are local to
  //            Operator setup, then compose one ordinary Tensor edge.
  Graph graph;
  auto producer = graph.op<CommonTensorPassOp>("producer");
  auto consumer = graph.op<CommonTensorPassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);
  const GraphView accepted = graph.view();
  const auto [source, target] = single_edge_ports(accepted);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  ASSERT_TRUE(source->setup.tensor_contract);
  ASSERT_TRUE(target->setup.tensor_contract);

  // Key assertion: both role contracts own identical derived C-order bounds after setup returns,
  //                and a rejected common-contract override leaves the graph identity unchanged.
  EXPECT_EQ(source->setup.tensor_contract->max_byte_span, 128U);
  ASSERT_EQ(source->setup.tensor_contract->axes.size(), 2U);
  EXPECT_EQ(source->setup.tensor_contract->axes[0].max_abs_stride_bytes, 8U);
  EXPECT_EQ(source->setup.tensor_contract->axes[1].max_abs_stride_bytes, 1U);
  EXPECT_EQ(target->setup.tensor_contract->memory_conversion, MemoryConversionPolicy::kDisable);

  const auto accepted_hash = accepted.identity().hash;
  const Error conflict = graph_error_from([&] {
    graph.configure(consumer->input)
        .expects_tensor(TensorInputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kCudaDevice,
                    .dtype = DLDataType{kDLFloat, 32U, 1U},
                    .rank = 2U,
                },
        });
  });
  EXPECT_EQ(conflict.code, ErrorCode::kIncompatible);
  EXPECT_TRUE(conflict.message.starts_with("GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT"));
  EXPECT_EQ(graph.view().identity().hash, accepted_hash);
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: GRAPH-012
// Graph topology:
//   holomq://contiguous --> imported raw Tensor port
//   invalid rank or stride evidence -X-> additional imported ports
TEST(GraphComposition, TensorLogicalBoundsAcceptContiguousEvidenceAndRejectIncompleteBounds) {
  Graph graph;
  const std::array<TensorAxisBounds, 2> axes{
      TensorAxisBounds{.max_extent = 16, .max_abs_stride_bytes = 4},
      TensorAxisBounds{.max_extent = 8, .max_abs_stride_bytes = 64},
  };
  TopicOptions contiguous;
  contiguous.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .rank = 2U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 512U, .axes = axes},
  };
  EXPECT_NO_THROW((void)graph.import_topic<Tensor>("holomq://contiguous", contiguous));

  TopicOptions rank_too_large = contiguous;
  rank_too_large.tensor->representation.rank = 3;
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<Tensor>("holomq://rank", rank_too_large);
            }).message,
            "GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID");

  const std::array<TensorAxisBounds, 1> incomplete_axes{
      TensorAxisBounds{.max_extent = 16, .max_abs_stride_bytes = 4}};
  TopicOptions incomplete;
  incomplete.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .strides = TensorStridePolicy::kBoundedStrided},
      .bounds = TensorBounds{.max_rank = 2U, .max_byte_span = 512U, .axes = incomplete_axes},
  };
  EXPECT_EQ(graph_error_from([&] {
              (void)graph.import_topic<Tensor>("holomq://incomplete-strides", incomplete);
            }).message,
            "GRAPH_TENSOR_STRIDE_BOUNDS_INCOMPLETE");
}

// REQ: GRAPH-006
// REQ: GRAPH-012
// REQ: GRAPH-039
//   producer(TensorOp) --[output→boundary.input()]--> holomq://tensor-export
//   Boundary sets inline-host emit policy; conflicting producer policy is rejected.
TEST(GraphComposition, TensorPayloadPolicyIsNormalizedAcrossProducerAndBoundary) {
  Graph graph;
  auto producer = graph.op<TensorOp>("producer");
  static constexpr std::array axes{
      TensorAxisBounds{.max_extent = 128U, .max_abs_stride_bytes = 1U}};
  TopicOptions boundary_options;
  boundary_options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 128U, .axes = axes},
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
              .max_inline_bytes = 256U,
          },
  };
  auto boundary = graph.export_topic<Tensor>("holomq://tensor-export", boundary_options);
  auto flow = graph.add_flow(producer->output, boundary.input());
  const GraphView accepted = graph.view();
  ASSERT_EQ(accepted.edges().size(), 1);
  ASSERT_TRUE(accepted.edges()[0].effective_tensor_payload);
  // GoogleTest assertions establish these invariants, but clang-tidy does not model the macros.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(accepted.edges()[0].effective_tensor_payload.value().interface_tensor_emit_policy,
            InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded);
  EXPECT_EQ(accepted.edges()[0].effective_tensor_payload.value().max_inline_bytes, 256);
  // The raw Tensor's logical span bounds allocation/transport geometry; it is not the distinct
  // descriptor-schema staging option and therefore cannot conflict with producer payload policy.
  EXPECT_FALSE(
      accepted.edges()[0].effective_tensor_payload.value().max_tensor_byte_span.has_value());

  PayloadOptions conflicting;
  conflicting.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  const Error conflict =
      graph_error_from([&] { graph.configure(producer->output).payload_options(conflicting); });
  EXPECT_EQ(conflict.code, ErrorCode::kIncompatible);
  EXPECT_TRUE(conflict.message.starts_with("GRAPH_TENSOR_PAYLOAD_POLICY_CONFLICT"));
  EXPECT_NE(conflict.message.find("provided="), std::string::npos);
  EXPECT_NE(conflict.message.find("required="), std::string::npos);
  EXPECT_EQ(conflict.message.find("source spec="), std::string::npos);
  EXPECT_EQ(conflict.message.find("target spec="), std::string::npos);
  const GraphView after_conflict = graph.view();
  ASSERT_TRUE(after_conflict.edges()[0].effective_tensor_payload);
  EXPECT_EQ(after_conflict.edges()[0].effective_tensor_payload.value().interface_tensor_emit_policy,
            accepted.edges()[0].effective_tensor_payload.value().interface_tensor_emit_policy);
  EXPECT_EQ(after_conflict.edges()[0].effective_tensor_payload.value().max_inline_bytes,
            accepted.edges()[0].effective_tensor_payload.value().max_inline_bytes);
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: GRAPH-006
// REQ: GRAPH-018
// REQ: GRAPH-031
//   producer(TensorPassOp) --[output→boundary.input()]--> holomq://layout-export
//   scalar_source(PassOp) --[output→input]--> scalar_target(PassOp)
//   Non-Tensor layout/memory-conversion mutations on scalar/POD ports are rejected.
TEST(GraphComposition, TensorLayoutMutationRecomputesEdgesAndRejectsNonTensorMutation) {
  Graph graph;
  auto producer = graph.op<TensorPassOp>("producer");
  TopicOptions boundary_options;
  boundary_options.tensor = TensorBoundaryContract{
      .representation = TensorRepresentation{.memory_kind = MemoryKind::kPinnedHost},
      .bounds = TensorBounds{.max_rank = 4U, .max_byte_span = 4096U},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  graph.configure(producer->output)
      .produces_tensor(
          TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kPinnedHost}});
  auto boundary = graph.export_topic<Tensor>("holomq://layout-export", boundary_options);
  auto tensor_flow = graph.add_flow(producer->output, boundary.input());

  graph.configure(producer->output)
      .produces_tensor(TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kPinnedHost,
                                                           .dtype = DLDataType{kDLUInt, 8, 1},
                                                           .rank = 2}});
  const GraphView updated = graph.view();
  ASSERT_EQ(updated.edges().size(), 1);
  const auto& updated_payload = updated.edges()[0].effective_tensor_payload;
  if (!updated_payload) {
    FAIL() << "layout mutation removed the normalized edge payload";
    return;
  }
  EXPECT_EQ(updated_payload->interface_tensor_emit_policy,
            InterfaceTensorEmitPolicy::kCopyIfNeeded);

  const auto prior_hash = updated.identity().hash;
  const Error invalid_layout = graph_error_from([&] {
    graph.configure(producer->output)
        .produces_tensor(TensorOutputSpec{.representation = {.dtype = DLDataType{kDLUInt, 0, 1}}});
  });
  EXPECT_EQ(invalid_layout.message, "GRAPH_TENSOR_DTYPE_INVALID");
  const GraphView after_invalid = graph.view();
  EXPECT_EQ(after_invalid.identity().hash, prior_hash);
  const auto& retained_payload = after_invalid.edges()[0].effective_tensor_payload;
  if (!retained_payload) {
    FAIL() << "rejected layout mutation changed the normalized edge payload";
    return;
  }
  EXPECT_EQ(retained_payload->interface_tensor_emit_policy,
            InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_TRUE(tensor_flow.graph_valid());

  auto scalar_source = graph.op<PassOp>("scalar_source");
  auto scalar_target = graph.op<PassOp>("scalar_target");
  auto scalar_flow = graph.add_flow(scalar_source->output, scalar_target->input);
  EXPECT_EQ(graph_error_from([&] {
              graph.configure(scalar_source->output)
                  .produces_tensor(TensorOutputSpec{.representation = {.rank = 1U}});
            }).message,
            "GRAPH_NON_TENSOR_LAYOUT_MUTATION");
  EXPECT_EQ(graph_error_from([&] {
              graph.configure(scalar_target->input)
                  .expects_tensor(TensorInputSpec{.representation = {.rank = 1U}});
            }).message,
            "GRAPH_NON_TENSOR_LAYOUT_MUTATION");
  EXPECT_EQ(graph_error_from([&] {
              graph.configure(scalar_flow).memory_conversion(MemoryConversionPolicy::kDisable);
            }).message,
            "GRAPH_NON_TENSOR_MEMORY_CONVERSION");

  auto pod_source = graph.op<PodSourceOp>("pod_source");
  EXPECT_EQ(graph_error_from([&] {
              graph.configure(pod_source->output)
                  .produces_tensor(TensorOutputSpec{.representation = {.rank = 1U}});
            }).message,
            "GRAPH_NON_TENSOR_LAYOUT_MUTATION");
  auto carrier = graph.op<UnmarkedCarrierOp>("unmarked_carrier");
  EXPECT_EQ(graph_error_from([&] {
              graph.configure(carrier->output)
                  .produces_tensor(TensorOutputSpec{.representation = {.rank = 1U}});
            }).message,
            "GRAPH_NON_TENSOR_LAYOUT_MUTATION");
}

// Exact layout facts constrain one typed Tensor edge even when no output pool exists yet. Both
// initial authoring and a later override reject conflicts without mutating the accepted Graph.
// REQ: GRAPH-025
//   producer(TensorOp) --[output→input]--> consumer(TensorPassOp)
//   Conflicting layout on consumer input (memory/dtype/rank/strides) is rejected atomically; the
//   accepted graph identity and edge count are unchanged after the rejection.
// REQ: GRAPH-039
TEST(GraphComposition, TensorEdgeLayoutsRejectRepresentationConflictsAtomically) {
  constexpr std::array bounded_axes{
      TensorAxisBounds{.max_extent = 16U, .max_abs_stride_bytes = 1U}};
  const auto output_contract = [&](const TensorRepresentation& representation) {
    TensorOutputSpec contract{.representation = representation};
    if (representation.strides == TensorStridePolicy::kBoundedStrided) {
      contract.bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = bounded_axes};
    }
    return contract;
  };
  const auto input_contract = [&](const TensorRepresentation& representation) {
    TensorInputSpec contract{.representation = representation};
    if (representation.strides == TensorStridePolicy::kBoundedStrided) {
      contract.bounds = TensorBounds{.max_rank = 1U, .max_byte_span = 16U, .axes = bounded_axes};
    }
    return contract;
  };
  const auto expect_conflict = [](const TensorRepresentation& accepted,
                                  const TensorRepresentation& conflicting,
                                  const auto& output_contract,
                                  const auto& input_contract) {
    // Key setup: producer and consumer both declare the accepted layout before the flow is added.
    Graph graph;
    auto producer = graph.op<TensorOp>("producer");
    auto consumer = graph.op<TensorPassOp>("consumer");
    graph.configure(producer->output).produces_tensor(output_contract(accepted));
    graph.configure(consumer->input).expects_tensor(input_contract(accepted));
    auto flow = graph.add_flow(producer->output, consumer->input);

    const auto accepted_identity = graph.view().identity();
    const Error mutation_error = graph_error_from(
        [&] { graph.configure(consumer->input).expects_tensor(input_contract(conflicting)); });
    // Key assertion: conflicting layout is rejected with GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT
    //                and the graph identity is unchanged after the failed mutation.
    EXPECT_EQ(mutation_error.code, ErrorCode::kIncompatible);
    EXPECT_TRUE(mutation_error.message.starts_with("GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT"));
    EXPECT_NE(mutation_error.message.find("producer.output"), std::string::npos);
    EXPECT_NE(mutation_error.message.find("consumer.input"), std::string::npos);
    EXPECT_NE(mutation_error.message.find("source spec=graph override, target spec=graph override"),
              std::string::npos);
    EXPECT_EQ(graph.view().identity().hash, accepted_identity.hash);
    EXPECT_TRUE(flow.graph_valid());

    Graph rejected;
    auto rejected_producer = rejected.op<TensorOp>("producer");
    auto rejected_consumer = rejected.op<TensorPassOp>("consumer");
    rejected.configure(rejected_producer->output).produces_tensor(output_contract(accepted));
    rejected.configure(rejected_consumer->input).expects_tensor(input_contract(conflicting));
    const auto before_flow = rejected.view().identity();
    const Error flow_error = graph_error_from(
        [&] { (void)rejected.add_flow(rejected_producer->output, rejected_consumer->input); });
    EXPECT_EQ(flow_error.code, ErrorCode::kIncompatible);
    EXPECT_NE(flow_error.message.find("GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT"), std::string::npos);
    EXPECT_NE(flow_error.message.find("producer.output"), std::string::npos) << flow_error.message;
    EXPECT_NE(flow_error.message.find("consumer.input"), std::string::npos) << flow_error.message;
    EXPECT_NE(flow_error.message.find("provided"), std::string::npos) << flow_error.message;
    EXPECT_NE(flow_error.message.find("required"), std::string::npos) << flow_error.message;
    const GraphView after_flow = rejected.view();
    EXPECT_EQ(after_flow.identity().hash, before_flow.hash);
    EXPECT_TRUE(after_flow.edges().empty());
  };

  const TensorRepresentation accepted{
      .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 1U};
  expect_conflict(accepted,
                  TensorRepresentation{.memory_kind = MemoryKind::kCudaDevice},
                  output_contract,
                  input_contract);
  expect_conflict(accepted,
                  TensorRepresentation{.dtype = DLDataType{kDLFloat, 8U, 1U}},
                  output_contract,
                  input_contract);
  expect_conflict(accepted, TensorRepresentation{.rank = 2U}, output_contract, input_contract);
  expect_conflict(TensorRepresentation{.memory_kind = MemoryKind::kHost,
                                       .dtype = DLDataType{kDLUInt, 8U, 1U},
                                       .rank = 1U,
                                       .strides = TensorStridePolicy::kContiguousOnly},
                  TensorRepresentation{.strides = TensorStridePolicy::kBoundedStrided},
                  output_contract,
                  input_contract);
}

// REQ: OPERATOR-041
// REQ: OPERATOR-046
// REQ: GRAPH-030
// Graph topology for each case:
//   producer --[incompatible raw Tensor bounds]--X-> consumer
TEST(GraphComposition, TensorEdgesRejectProducerBoundsAboveConsumerRequirements) {
  const auto expect_conflict = [](const TensorOutputSpec& output_contract,
                                  const TensorInputSpec& input_contract) {
    Graph graph;
    auto producer = graph.op<TensorOp>("producer");
    auto consumer = graph.op<TensorPassOp>("consumer");
    graph.configure(producer->output).produces_tensor(output_contract);
    graph.configure(consumer->input).expects_tensor(input_contract);

    const Error error =
        graph_error_from([&] { (void)graph.add_flow(producer->output, consumer->input); });
    EXPECT_EQ(error.code, ErrorCode::kIncompatible);
    EXPECT_TRUE(error.message.starts_with("GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT"));
    EXPECT_TRUE(graph.view().edges().empty());
  };
  const TensorRepresentation rank_one{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 1U,
  };

  {
    SCOPED_TRACE("maximum byte span");
    expect_conflict(
        TensorOutputSpec{
            .representation = rank_one,
            .bounds = tensor_bounds(2048U),
            .storage = TensorOutputStorage::kRuntimePool,
        },
        TensorInputSpec{
            .representation = rank_one,
            .bounds = tensor_bounds(1024U),
        });
  }

  constexpr std::array<std::int64_t, 2U> producer_shape{3, 2};
  constexpr std::array<std::int64_t, 2U> consumer_shape{2, 3};
  const TensorRepresentation rank_two{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 2U,
  };
  {
    SCOPED_TRACE("maximum axis extent");
    expect_conflict(
        TensorOutputSpec{
            .representation = rank_two,
            .bounds = tensor_bounds(producer_shape),
            .storage = TensorOutputStorage::kRuntimePool,
        },
        TensorInputSpec{
            .representation = rank_two,
            .bounds = tensor_bounds(consumer_shape),
        });
  }

  constexpr std::array producer_axes{
      TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 3U},
      TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 1U},
  };
  constexpr std::array consumer_axes{
      TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 2U},
      TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 1U},
  };
  const TensorRepresentation bounded_rank_two{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLUInt, 8U, 1U},
      .rank = 2U,
      .strides = TensorStridePolicy::kBoundedStrided,
  };
  {
    SCOPED_TRACE("maximum per-axis stride");
    expect_conflict(
        TensorOutputSpec{
            .representation = bounded_rank_two,
            .bounds =
                TensorBounds{
                    .max_rank = 2U,
                    .max_byte_span = 5U,
                    .axes = producer_axes,
                },
            .storage = TensorOutputStorage::kRuntimePool,
        },
        TensorInputSpec{
            .representation = bounded_rank_two,
            .bounds =
                TensorBounds{
                    .max_rank = 2U,
                    .max_byte_span = 5U,
                    .axes = consumer_axes,
                },
        });
  }
}

// REQ: CHAN-075
//   invalid(InvalidSchemaPackageOp) --X [package validation]--> graph admission
TEST(GraphComposition, RejectsInvalidGeneratedPackageDuringOperatorSetup) {
  // Key setup: the operator's generated owning type advertises an incomplete String codec.
  Graph graph;
  const Error error = graph_error_from([&] { (void)graph.op<InvalidSchemaPackageOp>("invalid"); });

  // Key assertion: setup returns the precise package validation error and publishes no vertex.
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(error.message.starts_with("generated schema codec/profile contract is invalid"));
  EXPECT_NE(error.message.find("operator=invalid"), std::string::npos);
  EXPECT_TRUE(graph.view().vertices().empty());
  EXPECT_TRUE(graph.view().ports().empty());
}

// REQ: CHAN-099
//   producer(ImagePassOp) --[descriptor-backed ImageT]--> consumer(ImagePassOp)
TEST(GraphComposition, DescriptorTensorImageComposesAsTypedGraphFlow) {
  // Key setup: author a normal public Graph flow whose generated schema owns a Tensor handle.
  Graph graph;
  auto producer = graph.op<ImagePassOp>("producer");
  auto consumer = graph.op<ImagePassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  ASSERT_EQ(view.ports().size(), 4U);

  // Key assertion: both typed declarations preserve schema identity and descriptor transport.
  const auto [source, target] = single_edge_ports(view);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  for (const PortRecord* port : {source, target}) {
    EXPECT_EQ(port->payload_identity.kind, PayloadIdentityKind::kSchema);
    EXPECT_EQ(port->payload_identity.payload_kind, PayloadKind::kDescriptorBacked);
    const auto& identity = port->payload_identity.schema;
    if (!identity.has_value()) {
      ADD_FAILURE() << "typed Image port lost its schema identity";
      continue;
    }
    EXPECT_EQ(identity->family_id, schema_identity<schema::ImageT>().family_id);
  }
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: OPERATOR-043
// REQ: OPERATOR-047
//   producer(SingleTensorUnqualifiedPassOp)
//       --[descriptor-backed ImageT]--> consumer(SingleTensorUnqualifiedPassOp)
TEST(GraphComposition, SingleTensorSchemaPreservesUnqualifiedFieldAuthoring) {
  Graph graph;
  auto producer = graph.op<SingleTensorUnqualifiedPassOp>("producer");
  auto consumer = graph.op<SingleTensorUnqualifiedPassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  const auto [source, target] = single_edge_ports(view);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  // Key setup: N = 1 uses every pre-existing unqualified field authoring call.
  EXPECT_EQ(source->descriptor_tensor_field_count, 1U);
  EXPECT_EQ(target->descriptor_tensor_field_count, 1U);
  EXPECT_TRUE(source->setup.tensor_fields.empty());
  EXPECT_TRUE(target->setup.tensor_fields.empty());
  // Key assertion: the calls remain port-scoped and preserve their exact layout, allocation,
  // materialization, and input-conversion facts.
  ASSERT_TRUE(source->setup.tensor_layout);
  ASSERT_TRUE(source->setup.tensor_allocation);
  ASSERT_TRUE(source->setup.payload_options);
  ASSERT_TRUE(target->setup.tensor_layout);
  ASSERT_TRUE(target->setup.memory_conversion);
  // The assertions above establish that each optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(source->setup.tensor_layout->memory_kind, MemoryKind::kHost);
  EXPECT_EQ(source->setup.tensor_allocation->max_byte_span, 4096U);
  EXPECT_EQ(source->setup.payload_options->max_tensor_byte_span, 4096U);
  EXPECT_EQ(target->setup.memory_conversion, MemoryConversionPolicy::kDisable);
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: OPERATOR-043
// REQ: OPERATOR-047
//   producer(SingleTensorQualifiedPassOp)
//       --[descriptor-backed ImageT]--> consumer(SingleTensorQualifiedPassOp)
TEST(GraphComposition, SingleTensorQualifiedFieldNormalizesToLegacyPortContract) {
  // Key setup: author the sole Image Tensor through its typed generated field selector.
  Graph graph;
  auto producer = graph.op<SingleTensorQualifiedPassOp>("producer");
  auto consumer = graph.op<SingleTensorQualifiedPassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  const auto [source, target] = single_edge_ports(view);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);

  // Key assertion: Graph capture folds N=1 field facts into the established port-level form.
  EXPECT_TRUE(source->setup.tensor_fields.empty());
  EXPECT_TRUE(target->setup.tensor_fields.empty());
  ASSERT_TRUE(source->setup.tensor_allocation);
  ASSERT_TRUE(source->setup.payload_options);
  ASSERT_TRUE(target->setup.memory_conversion);
  EXPECT_EQ(source->setup.tensor_allocation->max_byte_span, 4096U);
  EXPECT_EQ(source->setup.payload_options->max_tensor_byte_span, 4096U);
  EXPECT_EQ(target->setup.memory_conversion, MemoryConversionPolicy::kDisable);
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: OPERATOR-043
// REQ: OPERATOR-047
//   invalid(InvalidSingleTensorMixedScopeOp) --X [mixed selector scopes]--> graph admission
TEST(GraphComposition, SingleTensorQualifiedFieldRejectsMixedScope) {
  // Key setup: combine the typed N=1 field selector with a legacy port-wide Tensor declaration.
  Graph graph;
  const Error error =
      graph_error_from([&] { (void)graph.op<InvalidSingleTensorMixedScopeOp>("invalid"); });

  // Key assertion: ambiguous scope fails before Graph state is published.
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_TENSOR_FIELD_SCOPE_CONFLICT");
  EXPECT_TRUE(graph.view().vertices().empty());
}

// REQ: OPERATOR-043
// REQ: OPERATOR-047
//   producer(MultiTensorFieldPassOp)
//       --[left, right, calibration]--> consumer(MultiTensorFieldPassOp)
TEST(GraphComposition, MultiTensorFieldsPreserveCanonicalTypedContracts) {
  Graph graph;
  auto producer = graph.op<MultiTensorFieldPassOp>("producer");
  auto consumer = graph.op<MultiTensorFieldPassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  const auto [source, target] = single_edge_ports(view);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  EXPECT_EQ(source->descriptor_tensor_field_count, 3U);
  EXPECT_EQ(target->descriptor_tensor_field_count, 3U);
  ASSERT_EQ(source->setup.tensor_fields.size(), 3U);
  ASSERT_EQ(target->setup.tensor_fields.size(), 3U);
  EXPECT_EQ(source->setup.tensor_fields[0].field_id, 1U);
  EXPECT_EQ(source->setup.tensor_fields[0].tensor_index, 0U);
  EXPECT_EQ(source->setup.tensor_fields[1].field_id, 3U);
  EXPECT_EQ(source->setup.tensor_fields[1].tensor_index, 1U);
  EXPECT_EQ(source->setup.tensor_fields[2].field_id, 7U);
  EXPECT_EQ(source->setup.tensor_fields[2].tensor_index, 2U);
  ASSERT_TRUE(source->setup.tensor_fields[0].tensor_allocation);
  ASSERT_TRUE(source->setup.tensor_fields[1].tensor_layout);
  // The outer optional was asserted immediately above.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  ASSERT_TRUE(source->setup.tensor_fields[1].tensor_layout->memory_kind);
  ASSERT_TRUE(source->setup.tensor_fields[2].payload_options);
  // The outer optional was asserted immediately above.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  ASSERT_TRUE(source->setup.tensor_fields[2].payload_options->max_tensor_byte_span);
  ASSERT_TRUE(target->setup.tensor_fields[2].memory_conversion);
  // The assertions above establish that each nested optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(source->setup.tensor_fields[0].tensor_allocation->max_byte_span, 4096U);
  EXPECT_EQ(source->setup.tensor_fields[1].tensor_layout->memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(source->setup.tensor_fields[2].payload_options->max_tensor_byte_span, 256U);
  EXPECT_EQ(target->setup.tensor_fields[2].memory_conversion, MemoryConversionPolicy::kDisable);
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: OPERATOR-036
//   producer(MultiTensorContractPassOp)
//       --[left, right, calibration]--> consumer(MultiTensorContractPassOp)
TEST(GraphComposition, MultiTensorContractsLowerToCanonicalFieldRecords) {
  // Key setup: author repeated image fields and one scalar calibration field through contracts.
  Graph graph;
  auto producer = graph.op<MultiTensorContractPassOp>("producer");
  auto consumer = graph.op<MultiTensorContractPassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  const auto [source, target] = single_edge_ports(view);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(source->setup.tensor_fields.size(), 3U);
  ASSERT_EQ(target->setup.tensor_fields.size(), 3U);
  ASSERT_TRUE(source->setup.tensor_fields[0].tensor_allocation);
  ASSERT_TRUE(source->setup.tensor_fields[0].payload_options);
  ASSERT_TRUE(source->setup.tensor_fields[0].payload_options->max_tensor_byte_span);
  ASSERT_TRUE(source->setup.tensor_fields[2].tensor_allocation);
  ASSERT_TRUE(source->setup.tensor_fields[2].tensor_layout);
  ASSERT_TRUE(source->setup.tensor_fields[2].tensor_layout->rank);
  ASSERT_TRUE(target->setup.tensor_fields[0].memory_conversion);

  // Key assertion: convenience defaults lower to the same explicit canonical field facts.
  // The assertions above establish that each referenced optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(source->setup.tensor_fields[0].field_id, 1U);
  EXPECT_EQ(source->setup.tensor_fields[1].field_id, 3U);
  EXPECT_EQ(source->setup.tensor_fields[0].tensor_allocation->max_rank, 2U);
  EXPECT_EQ(source->setup.tensor_fields[0].tensor_allocation->max_byte_span, 4096U);
  EXPECT_EQ(source->setup.tensor_fields[0].payload_options->max_tensor_byte_span, 4096U);
  EXPECT_EQ(source->setup.tensor_fields[2].tensor_allocation->max_rank, 1U);
  EXPECT_EQ(source->setup.tensor_fields[2].tensor_layout->rank, 0U);
  EXPECT_FALSE(source->setup.tensor_fields[2].payload_options);
  EXPECT_EQ(target->setup.tensor_fields[0].memory_conversion, MemoryConversionPolicy::kDisable);
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: OPERATOR-036
//   concise_producer(MultiTensorContractPassOp)
//       --[left, right, calibration]--> concise_consumer(MultiTensorContractPassOp)
//   fluent_producer(MultiTensorFluentContractPassOp)
//       --[left, right, calibration]--> fluent_consumer(MultiTensorFluentContractPassOp)
TEST(GraphComposition, ConciseMultiTensorContractsMatchFluentGraphIdentity) {
  // Key setup: author otherwise identical graphs through concise and original fluent contracts.
  Graph concise{"contract-equivalence"};
  auto concise_producer = concise.op<MultiTensorContractPassOp>("producer");
  auto concise_consumer = concise.op<MultiTensorContractPassOp>("consumer");
  auto concise_flow = concise.add_flow(concise_producer->output, concise_consumer->input);

  Graph fluent{"contract-equivalence"};
  auto fluent_producer = fluent.op<MultiTensorFluentContractPassOp>("producer");
  auto fluent_consumer = fluent.op<MultiTensorFluentContractPassOp>("consumer");
  auto fluent_flow = fluent.add_flow(fluent_producer->output, fluent_consumer->input);

  ASSERT_TRUE(concise_flow.graph_valid());
  ASSERT_TRUE(fluent_flow.graph_valid());

  // Key assertion: convenience syntax changes no canonical Graph fact or identity hash.
  EXPECT_EQ(concise.view().identity().hash, fluent.view().identity().hash);
}

// REQ: OPERATOR-036
//   producer(MultiTensorAggregateContractPassOp)
//       --[left, right, calibration]--> consumer(MultiTensorAggregateContractPassOp)
TEST(GraphComposition, MultiTensorAggregateContractsCanonicalizeAuthoredOrder) {
  // Key setup: declare a partial input and complete output in deliberately noncanonical order.
  Graph graph;
  auto producer = graph.op<MultiTensorAggregateContractPassOp>("producer");
  auto consumer = graph.op<MultiTensorAggregateContractPassOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  const auto [source, target] = single_edge_ports(view);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(source->setup.tensor_fields.size(), 3U);
  ASSERT_EQ(target->setup.tensor_fields.size(), 2U);
  ASSERT_TRUE(source->setup.tensor_fields[0].tensor_layout);
  ASSERT_TRUE(source->setup.tensor_fields[2].tensor_layout);
  ASSERT_TRUE(target->setup.tensor_fields[1].memory_conversion);

  // Key assertion: canonical Graph records preserve complete output and optional input facts.
  EXPECT_EQ(source->setup.tensor_fields[0].field_id, 1U);
  EXPECT_EQ(source->setup.tensor_fields[1].field_id, 3U);
  EXPECT_EQ(source->setup.tensor_fields[2].field_id, 7U);
  EXPECT_EQ(target->setup.tensor_fields[0].field_id, 3U);
  EXPECT_EQ(target->setup.tensor_fields[1].field_id, 7U);
  EXPECT_EQ(source->setup.tensor_fields[0].tensor_layout->memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(source->setup.tensor_fields[2].tensor_layout->memory_kind, MemoryKind::kHost);
  EXPECT_EQ(target->setup.tensor_fields[1].memory_conversion, MemoryConversionPolicy::kDisable);
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: GRAPH-039
//   producer(MultiTensorContractPassOp) --[rank-0 calibration]--X
//       consumer(MultiTensorAggregateContractPassOp) expects rank-1 calibration.
TEST(GraphComposition, TensorFieldConflictDoesNotClaimPortContractProvenance) {
  Graph graph;
  auto producer = graph.op<MultiTensorContractPassOp>("producer");
  auto consumer = graph.op<MultiTensorAggregateContractPassOp>("consumer");
  const auto before = graph.view().identity();

  const Error conflict =
      graph_error_from([&] { (void)graph.add_flow(producer->output, consumer->input); });

  EXPECT_EQ(conflict.code, ErrorCode::kIncompatible);
  EXPECT_TRUE(conflict.message.starts_with("GRAPH_TENSOR_FIELD_LAYOUT_EDGE_CONFLICT"));
  EXPECT_NE(conflict.message.find("producer.output"), std::string::npos);
  EXPECT_NE(conflict.message.find("consumer.input"), std::string::npos);
  EXPECT_NE(conflict.message.find("field[7].tensor[2].rank"), std::string::npos);
  EXPECT_EQ(conflict.message.find("source spec="), std::string::npos);
  EXPECT_EQ(conflict.message.find("target spec="), std::string::npos);
  EXPECT_EQ(graph.view().identity().hash, before.hash);
  EXPECT_TRUE(graph.view().edges().empty());
}

struct InvalidMultiTensorDeclarationCase {
  InvalidMultiTensorDeclarationKind kind;
  const char* name;
  const char* diagnostic;
  ErrorCode error_code{ErrorCode::kInvalidArgument};
};

// REQ: OPERATOR-036
class InvalidMultiTensorDeclarationTest
    : public testing::TestWithParam<InvalidMultiTensorDeclarationCase> {};

// REQ: OPERATOR-036
TEST_P(InvalidMultiTensorDeclarationTest, RejectsBeforeGraphPublication) {
  const auto& test_case = GetParam();

  // invalid(InvalidMultiTensorDeclarationOp)   (isolated; no edges)
  Graph graph;
  const Error error = graph_error_from(
      [&] { (void)graph.op<InvalidMultiTensorDeclarationOp>("invalid", test_case.kind); });

  // Each case retains its precise declaration-stage diagnostic and publishes no partial Graph.
  EXPECT_EQ(error.code, test_case.error_code);
  EXPECT_EQ(error.message, test_case.diagnostic);
  EXPECT_TRUE(graph.view().vertices().empty());
}

INSTANTIATE_TEST_SUITE_P(
    InvalidMultiTensorDeclarations, InvalidMultiTensorDeclarationTest,
    testing::Values(
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kDuplicateAggregateSelector,
            "DuplicateAggregateSelector",
            "GRAPH_TENSOR_FIELD_DUPLICATE (field_id=1, tensor_index=0)",
            ErrorCode::kDuplicateName},
        InvalidMultiTensorDeclarationCase{InvalidMultiTensorDeclarationKind::kUnqualifiedField,
                                          "UnqualifiedField",
                                          "GRAPH_TENSOR_FIELD_SCOPE_REQUIRED"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kForeignSelector,
            "ForeignSelector",
            "GRAPH_TENSOR_FIELD_SCHEMA_MISMATCH (field_id=4, tensor_index=0)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kDuplicateSelector,
            "DuplicateSelector",
            "GRAPH_TENSOR_FIELD_DUPLICATE (field_id=1, tensor_index=0)",
            ErrorCode::kDuplicateName},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kUnknownField,
            "UnknownField",
            "GRAPH_TENSOR_FIELD_UNKNOWN (field_id=9, tensor_index=1)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kInvalidDtype,
            "InvalidDtype",
            "GRAPH_TENSOR_DTYPE_INVALID (field_id=1, tensor_index=0)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kInvalidAllocation,
            "InvalidAllocation",
            "GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID (field_id=1, tensor_index=0)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kUnresolvedRank,
            "UnresolvedRank",
            "GRAPH_TENSOR_ALLOCATION_INVALID (field_id=1, tensor_index=0)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kZeroInlineBytes,
            "ZeroInlineBytes",
            "GRAPH_TENSOR_INLINE_POLICY_INVALID (field_id=1, tensor_index=0)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kZeroTensorSpan,
            "ZeroTensorSpan",
            "GRAPH_TENSOR_STAGING_POLICY_INVALID (field_id=1, tensor_index=0)"},
        InvalidMultiTensorDeclarationCase{
            InvalidMultiTensorDeclarationKind::kInconsistentInlineSpan,
            "InconsistentInlineSpan",
            "GRAPH_TENSOR_INLINE_POLICY_INVALID (field_id=1, tensor_index=0)"}),
    [](const testing::TestParamInfo<InvalidMultiTensorDeclarationCase>& info) {
      return info.param.name;
    });

// REQ: GRAPH-006
// REQ: GRAPH-018
//   producer(TensorCarrierOp) --[output→input]--> consumer(TensorCarrierOp)
//   Schema carrier with inline-host policy; effective_tensor_payload must survive snapshot.
TEST(GraphComposition, TensorBearingSchemaCarrierPreservesExactPayloadOptions) {
  Graph graph;
  auto producer = graph.op<TensorCarrierOp>("producer");
  auto consumer = graph.op<TensorCarrierOp>("consumer");
  auto flow = graph.add_flow(producer->output, consumer->input);

  PayloadOptions options;
  options.schema = schema_identity<GraphSchemaPayload>();
  options.interface_tensor_emit_policy =
      InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  options.max_inline_bytes = 256;
  graph.configure(producer->output).payload_options(options);

  const GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1);
  const auto& effective = view.edges()[0].effective_tensor_payload;
  if (!effective || !effective->schema) {
    FAIL() << "Tensor-bearing schema carrier lost its effective payload options";
    return;
  }
  EXPECT_EQ(effective->schema->canonical_name, "holoscan.schema.String");
  EXPECT_EQ(effective->max_inline_bytes, 256);
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: GRAPH-006
// REQ: GRAPH-018
// Graph topologies:
//   scalar | carrier | pod (each an isolated invalid Operator)
TEST(GraphComposition, PayloadOptionsRejectScalarAndUnmarkedCarrierProvenance) {
  Graph scalar;
  EXPECT_EQ(
      graph_error_from([&] { (void)scalar.op<InvalidScalarPayloadOptionsOp>("scalar"); }).message,
      "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS");

  Graph carrier;
  EXPECT_EQ(graph_error_from([&] {
              (void)carrier.op<InvalidUnmarkedCarrierPolicyOp>("carrier");
            }).message,
            "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS");

  Graph pod;
  EXPECT_EQ(graph_error_from([&] { (void)pod.op<InvalidPodPayloadOptionsOp>("pod"); }).message,
            "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS");
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorOperatorSpec, DescriptorOutputRejectsASecondTensorTransferSource) {
  OperatorSpec spec;
  Output<schema::ImageT> output;
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  spec.output(output, "output")
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      })
      .payload_options(options);

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  ASSERT_TRUE(metadata.payload_options);
  EXPECT_EQ(metadata.payload_options->max_tensor_byte_span, 2048U);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorOperatorSpec, DescriptorOutputRejectsTransferAuthoredBeforeTensorContract) {
  OperatorSpec spec;
  Output<schema::ImageT> output;
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  spec.output(output, "output")
      .payload_options(options)
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.tensor_contract_error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  EXPECT_FALSE(metadata.tensor_contract);
  ASSERT_TRUE(metadata.payload_options);
  EXPECT_EQ(metadata.payload_options->max_tensor_byte_span, 1024U);
}

TEST(TensorOperatorSpec, DescriptorOutputPreservesSchemaOptionsAcrossContractAuthoring) {
  OperatorSpec spec;
  Output<schema::ImageT> output;
  PayloadOptions options;
  options.schema = schema_identity<GraphSchemaPayload>();

  spec.output(output, "output")
      .payload_options(options)
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
      });

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(metadata.payload_options);
  ASSERT_TRUE(metadata.payload_options->schema);
  EXPECT_EQ(metadata.payload_options->schema->canonical_name,
            schema_identity<GraphSchemaPayload>().canonical_name);
  EXPECT_EQ(metadata.payload_options->schema->compatibility_epoch,
            schema_identity<GraphSchemaPayload>().compatibility_epoch);
}

TEST(TensorOperatorSpec, DescriptorOutputCombinesSchemaOptionsWithContractTransferPolicy) {
  OperatorSpec spec;
  Output<schema::ImageT> output;
  PayloadOptions options;
  options.schema = schema_identity<GraphSchemaPayload>();

  spec.output(output, "output")
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
          .bounds = tensor_bounds(2048U),
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              },
      })
      .payload_options(options);

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.tensor_contract_error.code, ErrorCode::kSuccess);
  ASSERT_TRUE(metadata.payload_options);
  ASSERT_TRUE(metadata.payload_options->schema);
  EXPECT_EQ(metadata.payload_options->schema->canonical_name,
            schema_identity<GraphSchemaPayload>().canonical_name);
  EXPECT_EQ(metadata.payload_options->interface_tensor_emit_policy,
            InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_EQ(metadata.payload_options->max_tensor_byte_span, 2048U);
}

// REQ: GRAPH-008
// REQ: GRAPH-031
// Graph topology:
//   tensor (isolated configurable raw Tensor output)
TEST(GraphComposition, TensorConfiguratorIsStableAndOnlyTightensDeclaredAllocation) {
  Graph graph{"root"};
  auto tensor = graph.op<TensorOp>("tensor");
  auto& first = graph.configure(tensor->output);
  auto& second = graph.configure(tensor->output);
  EXPECT_EQ(&first, &second);
  first.produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
      .bounds = tensor_bounds(1024U),
      .storage = TensorOutputStorage::kRuntimePool,
  });
  EXPECT_THROW(first.late_joiner_history_depth(0), RuntimeError);
  EXPECT_THROW(first.produces_tensor(TensorOutputSpec{
                   .representation = {.memory_kind = MemoryKind::kHost, .rank = 4U},
                   .bounds = tensor_bounds(8192U),
                   .storage = TensorOutputStorage::kRuntimePool,
               }),
               RuntimeError);

  first.produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
  });
  const GraphView pass_through = graph.view();
  ASSERT_EQ(pass_through.ports().size(), 1U);
  ASSERT_TRUE(pass_through.ports()[0].graph_override.tensor_contract);
  EXPECT_EQ(pass_through.ports()[0].graph_override.tensor_contract->storage,
            TensorOutputStorage::kPassThrough);
  EXPECT_FALSE(pass_through.ports()[0].graph_override.tensor_allocation);
}

// REQ: COMPILE-067
// Graph topology:
//   source (isolated derived raw Tensor output)
TEST(GraphComposition, TensorConfiguratorPreservesDerivedAllocationForPassThroughRefinement) {
  Graph graph{"derived-output-override"};
  auto source = graph.op<DerivedTensorOutputOp>("source");
  const GraphView before = graph.view();

  // Key setup: attempt one pool replacement and one compatible logical refinement.
  const Error error = graph_error_from([&] {
    graph.configure(source->output)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .rank = 1U},
            .bounds = tensor_bounds(2048U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  });

  graph.configure(source->output)
      .produces_tensor(TensorOutputSpec{
          .storage = TensorOutputStorage::kPassThrough,
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                  .max_staging_byte_span = 512U,
              },
      });

  // Key assertion: capacity replacement fails while the logical refinement retains derivation.
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_TENSOR_DERIVED_ALLOCATION_OVERRIDE_INVALID");
  const GraphView after = graph.view();
  EXPECT_NE(after.identity(), before.identity());
  ASSERT_EQ(after.ports().size(), 1U);
  EXPECT_TRUE(after.ports()[0].setup.tensor_byte_span_expression);
  ASSERT_TRUE(after.ports()[0].graph_override.tensor_contract);
  EXPECT_EQ(after.ports()[0].graph_override.tensor_contract->storage,
            TensorOutputStorage::kPassThrough);
  ASSERT_TRUE(after.ports()[0].graph_override.tensor_contract->transfer);
  EXPECT_EQ(after.ports()[0].graph_override.tensor_contract->transfer->max_staging_byte_span, 512U);
  EXPECT_FALSE(after.ports()[0].graph_override.tensor_allocation);
}

// REQ: COMPILE-076
// Graph topology:
//   source (isolated Parameter-derived memory-kind output)
TEST(GraphComposition, TensorConfiguratorCannotReplaceDerivedMemoryKind) {
  Graph graph{"derived-memory-kind-override"};
  const auto source = graph.op<DerivedMemoryKindOutputOp>("source");

  const Error error = graph_error_from([&] {
    graph.configure(source->output)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost},
            .storage = TensorOutputStorage::kPassThrough,
        });
  });

  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT");
  const GraphView view = graph.view();
  ASSERT_EQ(view.ports().size(), 1U);
  EXPECT_TRUE(view.ports().front().setup.tensor_memory_kind_expression);
  EXPECT_FALSE(view.ports().front().graph_override.tensor_contract);
}

// REQ: COMPILE-067
// Graph topology:
//   source (isolated derived raw Tensor output)
TEST(GraphComposition, DerivedOutputRejectsGraphPayloadTransferAuthority) {
  Graph graph{"derived-payload-transfer"};
  const auto source = graph.op<DerivedTensorOutputOp>("source");
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;

  // Key setup: generic Graph payload options try to become a second transfer authority.
  const Error error =
      graph_error_from([&] { graph.configure(source->output).payload_options(options); });

  // Key assertion: derived outputs require transfer refinement through produces_tensor().
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
  const GraphView view = graph.view();
  ASSERT_EQ(view.ports().size(), 1U);
  EXPECT_FALSE(view.ports().front().graph_override.payload_options);
}

// REQ: COMPILE-067
// Graph topology:
//   source (isolated two-field descriptor output)
TEST(GraphComposition, TensorConfiguratorRequiresFieldScopeForDerivedMultiTensorOutput) {
  Graph graph{"derived-multi-output-override"};
  const auto source = graph.op<DerivedMultiTensorOutputOp>("source");

  // Key setup: the public Graph configurator can address only the aggregate output port.
  const Error error = graph_error_from([&] {
    graph.configure(source->output)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost, .rank = 2U},
            .storage = TensorOutputStorage::kPassThrough,
        });
  });

  // Key assertion: field-derived authority cannot be silently replaced at aggregate scope.
  EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(error.message, "GRAPH_TENSOR_FIELD_SCOPE_REQUIRED");
  const GraphView view = graph.view();
  ASSERT_EQ(view.ports().size(), 1U);
  EXPECT_FALSE(view.ports().front().graph_override.tensor_contract);
}

// REQ: COMPILE-067
// Graph topologies:
//   first:  source (1024-byte derived expression)
//   second: source (2048-byte derived expression)
TEST(GraphComposition, TensorByteSpanExpressionStructureContributesToGraphIdentity) {
  Graph first{"derived-expression-identity"};
  (void)first.op<DerivedTensorOutputOp>("source", 1024U);
  Graph second{"derived-expression-identity"};
  (void)second.op<DerivedTensorOutputOp>("source", 2048U);

  // Key assertion: stable expression structure is graph identity, unlike its resolved values.
  EXPECT_NE(first.view().identity().hash, second.view().identity().hash);
}

// REQ: GRAPH-029
// REQ: GRAPH-031
// Graph topologies:
//   payload-first:  tensor (isolated configurable output)
//   contract-first: tensor (isolated configurable output)
TEST(GraphComposition, TensorConfiguratorsRejectTransferSourcesInEitherOrder) {
  PayloadOptions options;
  options.interface_tensor_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  options.max_tensor_byte_span = 1024U;
  const TensorOutputSpec contract{
      .representation = {.memory_kind = MemoryKind::kCudaDevice, .rank = 2U},
      .bounds = tensor_bounds(2048U),
      .storage = TensorOutputStorage::kRuntimePool,
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
          },
  };

  {
    Graph graph{"payload-first"};
    auto tensor = graph.op<TensorOp>("tensor");
    auto& configure = graph.configure(tensor->output);
    configure.payload_options(options);
    const Error error = graph_error_from([&] { configure.produces_tensor(contract); });
    EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");

    const GraphView view = graph.view();
    ASSERT_EQ(view.ports().size(), 1U);
    EXPECT_FALSE(view.ports()[0].graph_override.tensor_contract);
    ASSERT_TRUE(view.ports()[0].graph_override.payload_options);
    EXPECT_EQ(view.ports()[0].graph_override.payload_options->max_tensor_byte_span, 1024U);
  }

  {
    Graph graph{"contract-first"};
    auto tensor = graph.op<TensorOp>("tensor");
    auto& configure = graph.configure(tensor->output);
    configure.produces_tensor(contract);
    const Error error = graph_error_from([&] { configure.payload_options(options); });
    EXPECT_EQ(error.code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(error.message, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");

    const GraphView view = graph.view();
    ASSERT_EQ(view.ports().size(), 1U);
    ASSERT_TRUE(view.ports()[0].graph_override.tensor_contract);
    ASSERT_TRUE(view.ports()[0].graph_override.tensor_contract->transfer);
    EXPECT_EQ(view.ports()[0].graph_override.tensor_contract->transfer->max_staging_byte_span,
              2048U);
  }
}

// REQ: GRAPH-008
// REQ: GRAPH-031
// REQ: OPERATOR-036
// Graph topology:
//   producer --[descriptor-backed Image]--> consumer
TEST(GraphComposition, CommonContractTightensSingularDescriptorAllocation) {
  Graph graph{"root"};
  auto producer = graph.op<SingleTensorUnqualifiedPassOp>("producer");
  auto consumer = graph.op<SingleTensorUnqualifiedPassOp>("consumer");
  const auto flow = graph.add_flow(producer->output, consumer->input);
  const TensorOutputSpec contract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
          },
      .bounds = tensor_bounds(2048U),
      .storage = TensorOutputStorage::kRuntimePool,
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              .max_staging_byte_span = 1024U,
          },
  };

  // Key setup: tighten an ImageT output's setup-time pool and staging bounds through the common
  // Graph-level Tensor contract.
  graph.configure(producer->output).produces_tensor(contract);
  const GraphView view = graph.view();
  const auto [source, target] = single_edge_ports(view);

  // Key assertion: the N = 1 descriptor uses the same normalized allocation and transfer facts as
  // a raw Tensor output, while retaining its generated schema identity.
  ASSERT_NE(source, nullptr);
  ASSERT_NE(target, nullptr);
  ASSERT_TRUE(source->graph_override.tensor_contract);
  ASSERT_TRUE(source->graph_override.tensor_allocation);
  ASSERT_TRUE(source->graph_override.payload_options);
  ASSERT_TRUE(source->graph_override.payload_options->schema);
  EXPECT_EQ(source->graph_override.tensor_allocation->max_rank, 2U);
  EXPECT_EQ(source->graph_override.tensor_allocation->max_byte_span, 2048U);
  EXPECT_EQ(source->graph_override.payload_options->max_tensor_byte_span, 1024U);
  EXPECT_EQ(source->graph_override.payload_options->schema->family_id,
            schema_identity<schema::ImageT>().family_id);
  EXPECT_TRUE(flow.graph_valid());
}

// REQ: GRAPH-008
// REQ: GRAPH-020
// REQ: GRAPH-031
// Graph topology:
//   tensor (isolated configurable raw Tensor output)
TEST(GraphComposition, TensorTryConfigureMapsConfiguratorAllocationFailure) {
  Graph graph{"root"};
  auto tensor = graph.op<TensorOp>("tensor");
  const GraphView before = graph.view();

  detail::GraphCompilerAccess::fail_mutation_after(graph, 0);
  auto failed = graph.try_configure(tensor->output);
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(failed.error().message.empty());
  EXPECT_EQ(graph.view().identity(), before.identity());

  auto retry = graph.try_configure(tensor->output);
  EXPECT_TRUE(retry);
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
