/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "../support/optional_assertions.hpp"
#include "core/domain/tensor_layout_validation.hpp"
#include "core/plan_record_contracts.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] SchemaIdentityRecord schema_identity() {
  return SchemaIdentityRecord{
      .canonical_name = "holoscan.test.Schema",
      .compatibility_epoch = 1U,
      .family_id = derive_schema_family_id("holoscan.test.Schema", 1U),
      .file_identifier = make_flatbuffer_file_identifier("TPSC"),
  };
}

[[nodiscard]] PayloadIdentityRecord pod_payload_identity() {
  return PayloadIdentityRecord{
      .kind = PayloadIdentityKind::kPod,
      .payload_kind = PayloadKind::kInlineTyped,
      .pod =
          PodIdentityRecord{
              .canonical_name = "holoscan.test.Pod",
              .version = 2U,
              .object_size = 8U,
              .object_alignment = 4U,
              .fields = {{.name = "value",
                          .byte_offset = 0U,
                          .scalar_kind = PodScalarKind::kUInt32,
                          .element_count = 2U}},
          },
  };
}

TEST(PlanRecordContracts, ValidatesTheCompleteSupportedDLPackDataTypeCodeMatrix) {
  struct CodeExpectation {
    std::uint8_t code;
    bool valid;
  };
  constexpr std::array kExpectations{
      CodeExpectation{kDLInt, true},
      CodeExpectation{kDLUInt, true},
      CodeExpectation{kDLFloat, true},
      CodeExpectation{kDLOpaqueHandle, true},
      CodeExpectation{kDLBfloat, true},
      CodeExpectation{kDLComplex, true},
      CodeExpectation{kDLBool, true},
      CodeExpectation{7U, false},
      CodeExpectation{8U, false},
      CodeExpectation{std::numeric_limits<std::uint8_t>::max(), false},
  };

  // Key setup: evaluate every admitted DLPack code plus unsupported extension
  // and sentinel codes with nonzero width and the maximum lane count.
  for (const auto& expectation : kExpectations) {
    const DLDataType dtype{
        .code = expectation.code, .bits = 32U, .lanes = std::numeric_limits<std::uint16_t>::max()};
    EXPECT_EQ(valid_tensor_dtype(dtype), expectation.valid)
        << "code=" << static_cast<std::uint32_t>(expectation.code);
  }

  // Key assertion: code support alone is insufficient when width or lanes are zero.
  EXPECT_FALSE(valid_tensor_dtype(DLDataType{.code = kDLFloat, .bits = 0U, .lanes = 1U}));
  EXPECT_FALSE(valid_tensor_dtype(DLDataType{.code = kDLFloat, .bits = 32U, .lanes = 0U}));
}

TEST(PlanRecordContracts, ComparesCompletePayloadIdentity) {
  // Key setup: create exact schema and POD payload identities, then mutate one nested fact.
  const PayloadIdentityRecord schema{
      .kind = PayloadIdentityKind::kSchema,
      .payload_kind = PayloadKind::kSerializedBytes,
      .schema = schema_identity(),
  };
  // Preserve a separate copy to test identity equality.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const PayloadIdentityRecord same_schema = schema;
  PayloadIdentityRecord changed_schema = schema;
  HOLOSCAN_ASSERT_HAS_VALUE(changed_schema.schema);
  changed_schema.schema->compatibility_epoch += 1U;
  const PayloadIdentityRecord pod = pod_payload_identity();
  // Preserve a separate copy to test identity equality.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const PayloadIdentityRecord same_pod = pod;
  PayloadIdentityRecord changed_pod = pod;
  HOLOSCAN_ASSERT_HAS_VALUE(changed_pod.pod);
  changed_pod.pod->fields.front().element_count += 1U;

  // Key assertion: equality covers the complete tagged identity and ordered nested POD layout.
  EXPECT_TRUE(payload_identity_exactly_equal(schema, same_schema));
  EXPECT_FALSE(payload_identity_exactly_equal(schema, changed_schema));
  EXPECT_TRUE(payload_identity_exactly_equal(pod, same_pod));
  EXPECT_FALSE(payload_identity_exactly_equal(pod, changed_pod));
  EXPECT_FALSE(payload_identity_exactly_equal(schema, pod));
}

TEST(PlanRecordContracts, ComparesCompleteTensorContracts) {
  // Key setup: create exact optional layouts and logical requirements with one nested axis.
  const TensorRepresentation layout{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{.code = kDLUInt, .bits = 16U, .lanes = 2U},
      .rank = 2U,
      .strides = TensorStridePolicy::kBoundedStrided,
  };
  const std::optional<TensorRepresentation> same_layout = layout;
  std::optional<TensorRepresentation> changed_layout = layout;
  HOLOSCAN_ASSERT_HAS_VALUE(changed_layout);
  auto& changed_dtype = changed_layout->dtype;
  HOLOSCAN_ASSERT_HAS_VALUE(changed_dtype);
  changed_dtype->lanes += 1U;
  const TensorBoundaryContractRecord requirements{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = layout.dtype,
      .rank = 2U,
      .max_rank = 2U,
      .max_byte_span = 32768U,
      .axes = {{.max_extent = 128U, .max_abs_stride_bytes = 256U}},
      .strides = TensorStridePolicy::kBoundedStrided,
      .foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable,
      .max_inline_bytes = std::nullopt,
      .max_staging_byte_span = std::nullopt,
      .memory_conversion = MemoryConversionPolicy::kDisable,
  };
  // Preserve a separate copy to test identity equality.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const TensorBoundaryContractRecord same_requirements = requirements;
  TensorBoundaryContractRecord changed_requirements = requirements;
  changed_requirements.axes.front().max_abs_stride_bytes += 1U;

  // Key assertion: optional dtype/layout and ordered logical-axis facts all participate.
  EXPECT_TRUE(tensor_port_layout_exactly_equal(layout, same_layout));
  EXPECT_FALSE(tensor_port_layout_exactly_equal(layout, changed_layout));
  EXPECT_FALSE(tensor_port_layout_exactly_equal(layout, std::nullopt));
  EXPECT_TRUE(logical_tensor_requirements_exactly_equal(requirements, same_requirements));
  EXPECT_FALSE(logical_tensor_requirements_exactly_equal(requirements, changed_requirements));
}

}  // namespace
}  // namespace holoscan::detail
