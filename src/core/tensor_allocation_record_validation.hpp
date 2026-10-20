/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include <holoscan/core/execution_plan.hpp>

#include "domain/tensor_layout_validation.hpp"

namespace holoscan::detail {

[[nodiscard]] inline bool validate_tensor_allocation_axes(
    const TensorPoolBounds& bounds, const std::optional<TensorRepresentation>& layout,
    const std::span<const TensorAxisBounds> axes) noexcept {
  if (axes.size() > bounds.max_rank ||
      (layout && layout->strides &&
       static_cast<std::uint32_t>(*layout->strides) >
           static_cast<std::uint32_t>(TensorStridePolicy::kBoundedStrided))) {
    return false;
  }
  if (!layout || layout->strides != TensorStridePolicy::kBoundedStrided) {
    return true;
  }
  return axes.size() == bounds.max_rank &&
         std::all_of(axes.begin(), axes.end(), [](const TensorAxisBounds& axis) {
           return axis.max_abs_stride_bytes != 0U;
         });
}

/// Return whether Tensor-field allocation records are in canonical output/field order.
[[nodiscard]] inline bool canonical_tensor_field_allocation_order(
    const TensorFieldAllocationPlanRecord& left,
    const TensorFieldAllocationPlanRecord& right) noexcept {
  return left.output == right.output ? left.field_id < right.field_id
                                     : left.output.value() < right.output.value();
}

/// Validate the ordering required by contiguous per-output Tensor-field plan lookups.
[[nodiscard]] inline bool validate_tensor_field_allocation_order(
    std::span<const TensorFieldAllocationPlanRecord> allocations) noexcept {
  return std::is_sorted(
      allocations.begin(), allocations.end(), canonical_tensor_field_allocation_order);
}

/// Validate the immutable, environment-independent portion of a direct Tensor pool record.
[[nodiscard]] inline bool validate_tensor_allocation_record(
    const TensorAllocationPlanRecord& allocation,
    const PlanGenerationId expected_plan_generation) noexcept {
  const bool memory_and_device_valid = [&]() noexcept {
    switch (allocation.memory_kind) {
      case MemoryKind::kHost:
        return !allocation.device.valid();
      case MemoryKind::kPinnedHost:
        // Cross-process pinned storage retains a private synchronization device; process-local
        // pinned storage does not need one.
        return true;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged:
        return allocation.device.valid();
      case MemoryKind::kUnknown:
        return false;
    }
    return false;
  }();

  const bool aggregate_valid =
      allocation.slot_count != 0U &&
      allocation.bounds.max_byte_span <=
          std::numeric_limits<std::uint64_t>::max() / allocation.slot_count &&
      allocation.aggregate_byte_capacity ==
          static_cast<std::uint64_t>(allocation.slot_count) * allocation.bounds.max_byte_span &&
      allocation.aggregate_byte_capacity <= std::numeric_limits<std::size_t>::max();

  return expected_plan_generation != 0U && allocation.plan_generation == expected_plan_generation &&
         allocation.run_generation == 0U && allocation.output && !allocation.pool_id.empty() &&
         !allocation.allocator_root_id.empty() && allocation.allocator_process_group &&
         allocation.bounds.max_rank != 0U && allocation.bounds.max_byte_span != 0U &&
         allocation.layout && allocation.layout->memory_kind &&
         *allocation.layout->memory_kind == allocation.memory_kind &&
         (!allocation.layout->dtype || valid_tensor_dtype(*allocation.layout->dtype)) &&
         (!allocation.layout->rank || *allocation.layout->rank <= allocation.bounds.max_rank) &&
         validate_tensor_allocation_axes(allocation.bounds, allocation.layout, allocation.axes) &&
         memory_and_device_valid && allocation.max_emits_per_compute != 0U &&
         allocation.max_outstanding_loans == allocation.max_emits_per_compute &&
         allocation.max_outstanding_loans <= allocation.slot_count &&
         allocation.control_capacity == allocation.slot_count &&
         allocation.readiness_capacity == allocation.slot_count &&
         allocation.release_capacity == allocation.slot_count &&
         allocation.retirement_capacity == allocation.slot_count &&
         allocation.quarantine_capacity == allocation.slot_count && aggregate_valid &&
         allocation.materialize_before == LifecycleStage::kArm &&
         allocation.disposition == PlanDisposition::kAccepted && allocation.fallback_rejected;
}

/// Validate the immutable portion of one multi-Tensor field pool record.
[[nodiscard]] inline bool validate_tensor_field_allocation_record(
    const TensorFieldAllocationPlanRecord& allocation,
    const PlanGenerationId expected_plan_generation) noexcept {
  const bool memory_and_device_valid = [&]() noexcept {
    switch (allocation.memory_kind) {
      case MemoryKind::kHost:
        return !allocation.device.valid();
      case MemoryKind::kPinnedHost:
        return true;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged:
        return allocation.device.valid();
      case MemoryKind::kUnknown:
        return false;
    }
    return false;
  }();

  const bool aggregate_valid =
      allocation.slot_count != 0U &&
      allocation.bounds.max_byte_span <=
          std::numeric_limits<std::uint64_t>::max() / allocation.slot_count &&
      allocation.aggregate_byte_capacity ==
          static_cast<std::uint64_t>(allocation.slot_count) * allocation.bounds.max_byte_span &&
      allocation.aggregate_byte_capacity <= std::numeric_limits<std::size_t>::max();

  return expected_plan_generation != 0U && allocation.plan_generation == expected_plan_generation &&
         allocation.run_generation == 0U && allocation.output && !allocation.pool_id.empty() &&
         !allocation.allocator_root_id.empty() && allocation.allocator_process_group &&
         allocation.bounds.max_rank != 0U && allocation.bounds.max_byte_span != 0U &&
         allocation.layout && allocation.layout->memory_kind &&
         *allocation.layout->memory_kind == allocation.memory_kind &&
         (!allocation.layout->dtype || valid_tensor_dtype(*allocation.layout->dtype)) &&
         (!allocation.layout->rank || *allocation.layout->rank <= allocation.bounds.max_rank) &&
         validate_tensor_allocation_axes(allocation.bounds, allocation.layout, allocation.axes) &&
         memory_and_device_valid && allocation.max_emits_per_compute != 0U &&
         allocation.max_outstanding_loans == allocation.max_emits_per_compute &&
         allocation.max_outstanding_loans <= allocation.slot_count &&
         allocation.control_capacity == allocation.slot_count &&
         allocation.readiness_capacity == allocation.slot_count &&
         allocation.release_capacity == allocation.slot_count &&
         allocation.retirement_capacity == allocation.slot_count &&
         allocation.quarantine_capacity == allocation.slot_count && aggregate_valid &&
         allocation.materialize_before == LifecycleStage::kArm &&
         allocation.disposition == PlanDisposition::kAccepted && allocation.fallback_rejected;
}

/// Validate pool-keyed field contributors without depending on their storage order.
[[nodiscard]] inline bool validate_tensor_field_capacity_contributors(
    const TensorFieldAllocationPlanRecord& allocation,
    std::span<const PoolCapacityContributorRecord> contributors) noexcept {
  constexpr std::array<std::string_view, 5U> kSuffixes{
      ".producer", ".delivery", ".transport-retention", ".retention", ".retirement"};
  std::array<bool, kSuffixes.size()> seen{};
  auto total = std::uint32_t{};
  for (const auto& contributor : contributors) {
    if (contributor.pool_id != allocation.pool_id) {
      continue;
    }
    const std::string_view contributor_id{contributor.contributor_id};
    const auto suffix = std::find_if(kSuffixes.begin(), kSuffixes.end(), [&](const auto candidate) {
      return contributor_id.size() == allocation.pool_id.size() + candidate.size() &&
             contributor_id.starts_with(allocation.pool_id) && contributor_id.ends_with(candidate);
    });
    const auto index = static_cast<std::size_t>(std::distance(kSuffixes.begin(), suffix));
    if (suffix == kSuffixes.end() || seen[index] || contributor.maximum_distinct_slots == 0U ||
        contributor.reasons.empty() ||
        contributor.maximum_distinct_slots > std::numeric_limits<std::uint32_t>::max() - total) {
      return false;
    }
    seen[index] = true;
    total += contributor.maximum_distinct_slots;
  }
  return seen[0] && seen[4] && total == allocation.slot_count;
}

}  // namespace holoscan::detail
