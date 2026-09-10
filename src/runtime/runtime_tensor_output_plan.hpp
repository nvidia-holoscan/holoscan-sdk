/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "runtime_tensor_resources.hpp"

namespace holoscan::detail {

struct ExecutionPlanState;

/// Return the dense slot-claim index used by bounded claim ledgers.
[[nodiscard]] constexpr std::size_t tensor_slot_claim_index(TensorSlotClaimReason reason) noexcept {
  return static_cast<std::size_t>(reason);
}

/// Retain the authenticated publication and slot-claim policy for one output pool.
struct OutputPlanContract {
  PublicationOutcomePlanRecord publication;
  TensorSlotClaimLedger::Admission admitted_claims{};
};

/// Derive the bounded set of downstream CUDA release streams.
[[nodiscard]] expected<std::uint32_t, Error> release_stream_capacity(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) noexcept;

/// Authenticate one output allocation against publication and claim-capacity records.
[[nodiscard]] expected<OutputPlanContract, Error> validate_output_plan_contract(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
    RuntimeRateMaterializationView materialization = {},
    const RuntimeTensorAllocationGeneration* generated_allocation = nullptr);

}  // namespace holoscan::detail
