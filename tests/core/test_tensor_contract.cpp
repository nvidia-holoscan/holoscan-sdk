/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

#include <holoscan/core/tensor_contract.hpp>

namespace holoscan {
namespace {

static_assert(std::is_aggregate_v<TensorRepresentation>);
static_assert(std::is_aggregate_v<TensorBounds>);
static_assert(std::is_aggregate_v<TensorTransferPolicy>);
static_assert(std::is_aggregate_v<TensorInputSpec>);
static_assert(std::is_aggregate_v<TensorOutputSpec>);
static_assert(std::is_aggregate_v<TensorBoundaryContract>);
static_assert(!std::same_as<TensorInputSpec, TensorOutputSpec>);
static_assert(!std::same_as<TensorOutputSpec, TensorBoundaryContract>);

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, DirectionalContractsComposeSharedRepresentationAndBounds) {
  // Key setup: one CUDA representation and finite bounds are reused by distinct input and output
  //            role specs without introducing a CUDA device-ordinal or schema field.
  constexpr TensorRepresentation representation{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = DLDataType{kDLFloat, 32U, 1U},
      .rank = 3U,
      .strides = TensorStridePolicy::kContiguousOnly,
  };
  constexpr TensorBounds bounds{.max_byte_span = 16U * 1024U * 1024U};
  constexpr TensorInputSpec input{
      .representation = representation,
      .bounds = bounds,
      .memory_conversion = MemoryConversionPolicy::kDisable,
  };
  constexpr TensorOutputSpec output{
      .representation = representation,
      .bounds = bounds,
      .storage = TensorOutputStorage::kRuntimePool,
  };

  // Key assertion: the shared facts remain identical while conversion stays consumer-side and
  //                runtime storage intent stays producer-side.
  ASSERT_TRUE(input.representation.memory_kind);
  EXPECT_EQ(*input.representation.memory_kind, MemoryKind::kCudaDevice);
  ASSERT_TRUE(output.bounds);
  ASSERT_TRUE(output.bounds->max_byte_span);
  EXPECT_EQ(*output.bounds->max_byte_span, 16U * 1024U * 1024U);
  EXPECT_EQ(input.memory_conversion, MemoryConversionPolicy::kDisable);
  EXPECT_EQ(output.storage, TensorOutputStorage::kRuntimePool);
  EXPECT_FALSE(output.transfer);
}

// REQ: GRAPH-012
TEST(TensorContract, BoundaryRetainsBorrowedAxisBoundsAndTransferPolicy) {
  constexpr std::array axes{
      TensorAxisBounds{.max_extent = 1920U, .max_abs_stride_bytes = 4U},
      TensorAxisBounds{.max_extent = 1080U, .max_abs_stride_bytes = 7680U},
  };
  // Key setup: a bounded-strided host boundary carries complete per-axis evidence and an explicit
  //            copy-capable transfer policy with a tighter staging bound.
  const TensorBoundaryContract boundary{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kPinnedHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
              .strides = TensorStridePolicy::kBoundedStrided,
          },
      .bounds =
          TensorBounds{
              .max_rank = 2U,
              .max_byte_span = 8U * 1024U * 1024U,
              .axes = axes,
          },
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              .max_staging_byte_span = 4U * 1024U * 1024U,
          },
  };

  // Key assertion: the role wrapper preserves all boundary-only interoperability evidence while
  //                the shared axis vocabulary remains a borrowed authoring view.
  EXPECT_EQ(boundary.bounds.axes.data(), axes.data());
  EXPECT_EQ(boundary.bounds.axes.size(), axes.size());
  ASSERT_TRUE(boundary.transfer);
  EXPECT_EQ(boundary.transfer->foreign_emit_policy, InterfaceTensorEmitPolicy::kCopyIfNeeded);
  EXPECT_EQ(boundary.transfer->max_staging_byte_span, 4U * 1024U * 1024U);
  EXPECT_FALSE(boundary.memory_conversion);
}

// REQ: GRAPH-012
TEST(TensorContract, DirectAxisBoundsUseExactRepresentationRank) {
  constexpr std::array axes{
      TensorAxisBounds{.max_extent = 1080U, .max_abs_stride_bytes = 7680U},
      TensorAxisBounds{.max_extent = 1920U, .max_abs_stride_bytes = 4U},
  };
  // Key setup: direct per-axis bounds omit max_rank while the representation supplies exact rank.
  const TensorBoundaryContract boundary{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kPinnedHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 2U,
              .strides = TensorStridePolicy::kBoundedStrided,
          },
      .bounds = TensorBounds{.max_byte_span = 8U * 1024U * 1024U, .axes = axes},
  };

  const auto normalized = detail::normalize_tensor_boundary_contract(boundary);

  // Key assertion: exact rank supplies the effective axis capacity before bounds are copied.
  ASSERT_TRUE(normalized) << normalized.error().message;
  EXPECT_EQ(normalized->max_rank, 2U);
  EXPECT_EQ(normalized->origins.max_rank, TensorContractValueOrigin::kDerivedExactRank);
  ASSERT_EQ(normalized->axes.size(), axes.size());
  EXPECT_EQ(normalized->axes[0], axes[0]);
  EXPECT_EQ(normalized->axes[1], axes[1]);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, ContiguousShapeHelperDerivesOwnedBounds) {
  // Key setup: a runtime-pool output supplies only exact dtype, maximum shape, and storage intent.
  std::array<std::int64_t, 3U> shape{2, 3, 4};
  const TensorOutputSpec spec{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{kDLFloat, 32U, 1U},
              .rank = 3U,
          },
      .bounds = tensor_bounds(shape),
      .storage = TensorOutputStorage::kRuntimePool,
  };
  auto normalized = detail::normalize_tensor_output_spec(spec);
  ASSERT_TRUE(normalized) << normalized.error().message;
  shape.fill(99);

  // Key assertion: normalization derives C-order strides and span, then owns the axis evidence
  //                independently of the caller's authoring arrays.
  ASSERT_EQ(normalized->max_rank, 3U);
  ASSERT_EQ(normalized->max_byte_span, 96U);
  ASSERT_EQ(normalized->representation.strides, TensorStridePolicy::kContiguousOnly);
  EXPECT_EQ(normalized->origins.max_rank, TensorContractValueOrigin::kDerivedShapeBounds);
  EXPECT_EQ(normalized->origins.max_byte_span, TensorContractValueOrigin::kDerivedShapeBounds);
  EXPECT_EQ(normalized->origins.strides, TensorContractValueOrigin::kDerivedShapeBounds);
  ASSERT_EQ(normalized->axes.size(), 3U);
  EXPECT_EQ(normalized->axes[0], (TensorAxisBounds{.max_extent = 2U, .max_abs_stride_bytes = 48U}));
  EXPECT_EQ(normalized->axes[1], (TensorAxisBounds{.max_extent = 3U, .max_abs_stride_bytes = 16U}));
  EXPECT_EQ(normalized->axes[2], (TensorAxisBounds{.max_extent = 4U, .max_abs_stride_bytes = 4U}));
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, BoundedShapeHelperDerivesAddressedSpan) {
  constexpr std::array<std::int64_t, 2U> shape{2, 3};
  std::array<std::uint64_t, 2U> strides{64U, 4U};
  // Key setup: explicit maximum absolute strides select bounded-strided admission without a
  //            redundant stride-family declaration or byte-span calculation.
  const TensorInputSpec spec{
      .representation = TensorRepresentation{.dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U},
      .bounds = tensor_bounds(shape, TensorBoundsOptions{.max_abs_byte_strides = strides}),
  };
  auto normalized = detail::normalize_tensor_input_spec(spec);
  ASSERT_TRUE(normalized) << normalized.error().message;
  strides.fill(1U);

  // Key assertion: the maximum addressed byte span is 1 + (2 - 1) * 64 + (3 - 1) * 4, and the
  //                normalized record owns the original stride evidence.
  EXPECT_EQ(normalized->max_rank, 2U);
  EXPECT_EQ(normalized->max_byte_span, 73U);
  EXPECT_EQ(normalized->representation.strides, TensorStridePolicy::kBoundedStrided);
  EXPECT_EQ(normalized->origins.max_rank, TensorContractValueOrigin::kDerivedShapeBounds);
  EXPECT_EQ(normalized->origins.max_byte_span, TensorContractValueOrigin::kDerivedShapeBounds);
  EXPECT_EQ(normalized->origins.strides, TensorContractValueOrigin::kDerivedShapeBounds);
  EXPECT_EQ(detail::tensor_representation(*normalized).strides,
            TensorStridePolicy::kBoundedStrided);
  ASSERT_EQ(normalized->axes.size(), 2U);
  EXPECT_EQ(normalized->axes[0].max_abs_stride_bytes, 64U);
  EXPECT_EQ(normalized->axes[1].max_abs_stride_bytes, 4U);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, ByteSpanOnlyBoundsDoNotInventShape) {
  // Key setup: an author intentionally bounds only storage while leaving shape and rank dynamic.
  const TensorOutputSpec spec{
      .representation = TensorRepresentation{.memory_kind = MemoryKind::kPinnedHost},
      .bounds = tensor_bounds(4096U),
  };
  const auto normalized = detail::normalize_tensor_output_spec(spec);

  // Key assertion: normalization preserves the finite span without fabricating rank, axes, dtype,
  //                or a stride family.
  ASSERT_TRUE(normalized) << normalized.error().message;
  EXPECT_EQ(normalized->max_byte_span, 4096U);
  EXPECT_FALSE(normalized->max_rank);
  EXPECT_TRUE(normalized->axes.empty());
  EXPECT_FALSE(normalized->representation.dtype);
  EXPECT_FALSE(normalized->representation.strides);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, CopyPolicyDefaultsStagingSpanFromOverallBound) {
  // Key setup: a finite external boundary permits fallback copies but omits a redundant staging
  //            byte-span value.
  const TensorBoundaryContract contract{
      .representation =
          TensorRepresentation{
              .memory_kind = MemoryKind::kHost,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 1U,
          },
      .bounds = tensor_bounds(8192U),
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  const auto normalized = detail::normalize_tensor_boundary_contract(contract);

  // Key assertion: boundary normalization supplies contiguous layout and reuses the overall span
  //                as the staging bound while preserving the authored role contract.
  ASSERT_TRUE(normalized) << normalized.error().message;
  ASSERT_TRUE(normalized->transfer);
  EXPECT_EQ(normalized->transfer->max_staging_byte_span, 8192U);
  EXPECT_EQ(normalized->representation.strides, TensorStridePolicy::kContiguousOnly);
  EXPECT_EQ(normalized->role, TensorContractRole::kExternalBoundary);
  EXPECT_EQ(normalized->origins.max_rank, TensorContractValueOrigin::kDerivedExactRank);
  EXPECT_EQ(normalized->origins.strides, TensorContractValueOrigin::kDefaultedBoundary);
  EXPECT_EQ(normalized->origins.max_staging_byte_span,
            TensorContractValueOrigin::kDerivedMaxByteSpan);
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, ShapeHelperRequiresExactDtype) {
  constexpr std::array<std::int64_t, 1U> shape{8};
  // Key setup: maximum-shape authoring omits the exact dtype needed to derive byte strides/span.
  const TensorOutputSpec spec{
      .representation = TensorRepresentation{.memory_kind = MemoryKind::kHost, .rank = 1U},
      .bounds = tensor_bounds(shape),
  };
  const auto normalized = detail::normalize_tensor_output_spec(spec);

  // Key assertion: the spec is rejected at authoring normalization instead of assuming an
  //                element size or retaining a partially derived record.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_BOUNDS_DTYPE_REQUIRED"});
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, EmptyRoleContractIsRejected) {
  // Key setup: a present role wrapper carries no representation, bounds, policy, or storage fact.
  const auto normalized = detail::normalize_tensor_input_spec(TensorInputSpec{});

  // Key assertion: callers must omit an unconstrained spec instead of supplying an ambiguous
  //                empty object.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_CONTRACT_EMPTY"});
}

// REQ: GRAPH-012
TEST(TensorContract, EmptyBoundaryIsRejectedBeforeContiguousDefault) {
  // Key setup: a present boundary wrapper carries no authored representation, bounds, policy, or
  //            conversion fact.
  const auto normalized = detail::normalize_tensor_boundary_contract(TensorBoundaryContract{});

  // Key assertion: the boundary convenience default cannot turn an empty contract into a valid
  //                declaration; callers must omit TopicOptions::tensor instead.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_CONTRACT_EMPTY"});
}

// REQ: GRAPH-012
TEST(TensorContract, BoundaryRequiresFinitePhysicalFacts) {
  // Key setup: an external boundary states an exact dtype but omits storage kind and finite
  //            rank/span capacity.
  const TensorBoundaryContract contract{
      .representation = TensorRepresentation{.dtype = DLDataType{kDLUInt, 8U, 1U}},
  };
  const auto normalized = detail::normalize_tensor_boundary_contract(contract);

  // Key assertion: a present external contract cannot defer physical provisioning facts.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_BOUNDARY_INVALID"});
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, CopyCapableOutputRequiresFiniteStagingSpan) {
  // Key setup: a pass-through output admits fallback copy but supplies no overall byte bound from
  //            which normalization could derive staging capacity.
  const TensorOutputSpec spec{
      .representation = TensorRepresentation{.memory_kind = MemoryKind::kHost},
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  const auto normalized = detail::normalize_tensor_output_spec(spec);

  // Key assertion: copy permission never implies an unbounded or runtime-grown staging pool.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_STAGING_INVALID"});
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, StagingOverrideRequiresCopyCapablePolicy) {
  // Key setup: exportable-only delivery incorrectly carries a staging-capacity override.
  const TensorOutputSpec spec{
      .representation = TensorRepresentation{.memory_kind = MemoryKind::kHost},
      .bounds = tensor_bounds(4096U),
      .transfer =
          TensorTransferPolicy{
              .foreign_emit_policy = InterfaceTensorEmitPolicy::kRequireExportable,
              .max_staging_byte_span = 4096U,
          },
  };
  const auto normalized = detail::normalize_tensor_output_spec(spec);

  // Key assertion: the diagnostic identifies the staging-policy contradiction rather than an
  //                unrelated inline-policy failure.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_STAGING_POLICY_INVALID"});
}

// REQ: OPERATOR-039
// REQ: OPERATOR-045
TEST(TensorContract, ExplicitStridePolicyMustMatchShapeStrideEvidence) {
  constexpr std::array<std::int64_t, 1U> shape{8};
  constexpr std::array<std::uint64_t, 1U> strides{2U};
  // Key setup: explicit bounded-stride evidence contradicts an explicit contiguous-only policy.
  const TensorInputSpec spec{
      .representation =
          TensorRepresentation{
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 1U,
              .strides = TensorStridePolicy::kContiguousOnly,
          },
      .bounds = tensor_bounds(shape, TensorBoundsOptions{.max_abs_byte_strides = strides}),
  };
  const auto normalized = detail::normalize_tensor_input_spec(spec);

  // Key assertion: the diagnostic reports contradictory stride policy, not missing bounds.
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().message, std::string_view{"GRAPH_TENSOR_STRIDE_POLICY_CONFLICT"});
}

}  // namespace
}  // namespace holoscan
