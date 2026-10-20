/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>

#include "runtime_rate_state.hpp"
#include "runtime_tensor_resource_materialization.hpp"

namespace holoscan::detail {

struct ExecutionPlanState;
struct RuntimeFixedTensorAddress;
struct RuntimeOperatorInstance;
class RuntimeCudaExecutionResources;
class RuntimeTensorOutputTestControl;

/// Retain the pure, validated allocation inventory consumed by output-pool construction.
struct RuntimeTensorOutputMaterializationPlan {
  std::vector<TensorAllocationPlanRecord> allocations;
};

/// Validate lifecycle-selected output geometry before any device or Operator stream mutation.
[[nodiscard]] expected<RuntimeTensorOutputMaterializationPlan, Error>
plan_runtime_tensor_output_materialization(const ExecutionPlanState& plan,
                                           RunGenerationId run_generation,
                                           ProcessGroupId process_group,
                                           RuntimeRateMaterializationView materialization);

/// Own direct and schema Tensor output pools behind a cold-path lifecycle seam.
class RuntimeTensorOutputResources final {
 public:
  class Impl;

  RuntimeTensorOutputResources(const RuntimeTensorOutputResources&) = delete;
  RuntimeTensorOutputResources& operator=(const RuntimeTensorOutputResources&) = delete;
  RuntimeTensorOutputResources(RuntimeTensorOutputResources&&) = delete;
  RuntimeTensorOutputResources& operator=(RuntimeTensorOutputResources&&) = delete;
  ~RuntimeTensorOutputResources();

  /// Validate, allocate, and bind every output pool owned by this process.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeTensorOutputResources>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      const RuntimeCudaExecutionResources& cuda_resources,
      std::span<const RuntimeTensorDeviceResource> devices, RunGenerationId run_generation,
      ProcessGroupId process_group,
      const std::shared_ptr<RuntimeTensorOutputTestControl>& test_control,
      RuntimeRateMaterializationView materialization,
      RuntimeTensorOutputMaterializationPlan materialization_plan);

  /// Stop accepting output operations and release pending pool work.
  void close() noexcept;

  /// Check whether one Operator may consume Channel terminal authority.
  [[nodiscard]] expected<bool, Error> preflight_owner_terminal_close(
      OperatorInstanceId owner) noexcept;

  /// Advance nonblocking retirement for every output owned by one Operator.
  [[nodiscard]] expected<bool, Error> advance_owner_terminal_retirement(
      OperatorInstanceId owner) noexcept;

  /// Remove every output declaration binding owned by the pools.
  void detach_ports() noexcept;

  /// Apply lifecycle-selected logical limits to materialized startup backing.
  [[nodiscard]] expected<void, Error> prepare_rate_candidate_startup(
      RuntimeRateMaterializationView materialization) noexcept;

  /// Prepare affected output pools for one lifecycle restart transaction.
  [[nodiscard]] expected<void, Error> begin_restart(std::span<const OperatorInstanceId> members,
                                                    LifecycleStage from_stage) noexcept;

  /// Materialize lifecycle-selected replacement geometry for a prepared restart.
  [[nodiscard]] expected<void, Error> prepare_rate_candidate_restart(
      RuntimeRateMaterializationView materialization) noexcept;

  /// Publish every prepared replacement output pool.
  [[nodiscard]] expected<void, Error> publish_restart() noexcept;

  /// Check that every prepared replacement can be published atomically.
  [[nodiscard]] expected<void, Error> preflight_restart_publication() noexcept;

  /// Restore output bindings after a failed restart transaction.
  [[nodiscard]] expected<void, Error> rollback_restart() noexcept;

  /// Return the number of active direct Tensor output pools.
  [[nodiscard]] std::size_t output_pool_count() const noexcept;

  [[nodiscard]] expected<RuntimeFixedTensorAddress, Error> fixed_output_address(
      PortId output) noexcept;
  [[nodiscard]] expected<bool, Error> fixed_output_available(PortId output) noexcept;
  [[nodiscard]] bool restore_output_binding(PortId output) noexcept;

 private:
  explicit RuntimeTensorOutputResources(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::detail
