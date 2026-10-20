/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/hosted_parameter_snapshot.hpp>

namespace holoscan {
namespace {

// Builds a snapshot whose builder and temporary field storage expire before the caller reads it.
// Representative test: HostedSnapshot.OwnsStableScalarAndSequenceViewsAcrossCopies.
HostedParameterSnapshot representative_snapshot() {
  HostedParameterSnapshotBuilder builder{"holoscan.test.HostedParams", "hosted test values"};
  builder.field(HostedParameterField{
      .name = "threshold",
      .description = "positive threshold",
      .type_shape =
          ParamFieldTypeShape{
              .type = ParamLogicalType::kSignedInteger, .bit_width = 64U, .has_stable_shape = true},
      .mutability = ParamMutability::kRecompile,
      .constraints =
          ParamConstraints{
              .has_range = true,
              .range_min = ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                                 .signed_value = 1},
              .range_max = ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                                 .signed_value = 16}},
      .values = {ParamScalarValue::from(std::int64_t{4})},
  });
  builder.field(HostedParameterField{
      .name = "shape",
      .type_shape =
          ParamFieldTypeShape{
              .type = ParamLogicalType::kSequence,
              .element_or_rep_type = ParamLogicalType::kSignedInteger,
              .element_or_rep_bit_width = 64U,
              .has_stable_shape = true,
          },
      .constraints = ParamConstraints{.max_elements = 3U},
      .values = {ParamScalarValue::from(std::int64_t{720}),
                 ParamScalarValue::from(std::int64_t{1280}),
                 ParamScalarValue::from(std::int64_t{4})},
  });
  return builder.build();
}

// A runtime-hosted schema owns strings, descriptor rows, and scalar values together.
// Representative test: HostedSnapshot.OwnsStableScalarAndSequenceViewsAcrossCopies.
// REQ: PY-OP-002
TEST(HostedSnapshot, OwnsStableScalarAndSequenceViewsAcrossCopies) {
  // Key setup: Copy a mixed scalar/sequence snapshot after its local builder has been destroyed.
  HostedParameterSnapshot copy;
  {
    const HostedParameterSnapshot original = representative_snapshot();
    copy = original;
  }
  ASSERT_TRUE(copy);
  ASSERT_NE(copy.schema(), nullptr);
  ASSERT_EQ(copy.schema()->fields.size(), 2U);

  ParamScalarValue threshold;
  const ParamFieldDescriptor& threshold_field = copy.schema()->fields[0];
  const auto shape = copy.read_scalar_sequence(copy.schema()->fields[1]);

  // Key assertion: Every descriptor and borrowed value remains valid through shared ownership.
  EXPECT_EQ(copy.schema()->stable_id, "holoscan.test.HostedParams");
  EXPECT_EQ(threshold_field.canonical_name, "threshold");
  EXPECT_EQ(threshold_field.description, "positive threshold");
  EXPECT_EQ(copy.schema()->fields[1].canonical_name, "shape");
  EXPECT_TRUE(copy.read_scalar(threshold_field, threshold));
  std::int64_t decoded{};
  EXPECT_TRUE(threshold.try_as(decoded));
  EXPECT_EQ(decoded, 4);
  ASSERT_EQ(shape.size(), 3U);
  EXPECT_TRUE(shape[1].try_as(decoded));
  EXPECT_EQ(decoded, 1280);
  EXPECT_EQ(threshold_field.mutability, ParamMutability::kRecompile);
  EXPECT_TRUE(threshold_field.validate(&copy, threshold_field).ok());
}

// REQ: PY-OP-002
TEST(HostedSnapshot, RejectsDescriptorsOwnedByAnotherSnapshot) {
  // Key setup: Build two snapshots with the same schema so their field indices and shapes match.
  const HostedParameterSnapshot first = representative_snapshot();
  const HostedParameterSnapshot second = representative_snapshot();
  ASSERT_NE(first.schema(), nullptr);
  ASSERT_NE(second.schema(), nullptr);

  ParamScalarValue scalar;
  // Key assertion: Descriptor identity, not only its hosted index, controls snapshot reads.
  EXPECT_FALSE(first.read_scalar(second.schema()->fields[0], scalar));
  EXPECT_TRUE(first.read_scalar_sequence(second.schema()->fields[1]).empty());
  EXPECT_TRUE(first.read_scalar(first.schema()->fields[0], scalar));
  EXPECT_EQ(first.read_scalar_sequence(first.schema()->fields[1]).size(), 3U);
}

// REQ: PY-OP-002
TEST(HostedSnapshot, RejectsInvalidOrDuplicateRuntimeDeclarations) {
  // Key setup: Declare duplicate field names and an out-of-range scalar value.
  HostedParameterSnapshotBuilder duplicate{"holoscan.test.Duplicate"};
  const HostedParameterField field{
      .name = "value",
      .type_shape =
          ParamFieldTypeShape{
              .type = ParamLogicalType::kSignedInteger, .bit_width = 64U, .has_stable_shape = true},
      .values = {ParamScalarValue::from(std::int64_t{2})},
  };
  duplicate.field(field).field(field);

  HostedParameterSnapshotBuilder invalid{"holoscan.test.Invalid"};
  invalid.field(HostedParameterField{
      .name = "value",
      .type_shape = field.type_shape,
      .constraints =
          ParamConstraints{
              .has_range = true,
              .range_min = ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                                 .signed_value = 3},
              .range_max = ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                                 .signed_value = 5}},
      .values = field.values,
  });

  // Key assertion: Malformed hosted schemas fail before exposing any descriptor.
  EXPECT_THROW((void)duplicate.build(), RuntimeError);
  EXPECT_THROW((void)invalid.build(), RuntimeError);
}

}  // namespace
}  // namespace holoscan
