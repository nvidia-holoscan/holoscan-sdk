/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph_exec.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "lifecycle_executor.hpp"
#include "runtime_clock_reconstruction.hpp"
#include "runtime_operator_reconstruction.hpp"
#include "runtime_work_inventory.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] Error startup_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

}  // namespace

expected<std::unique_ptr<GraphExec>, Error> GraphExec::reconstruct(
    std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
    ProcessGroupId process_group, RuntimeReplayMismatchDetail* replay_mismatch) {
  if (replay_mismatch != nullptr) {
    *replay_mismatch = RuntimeReplayMismatchDetail{};
  }
  if (!plan) {
    return make_unexpected(
        startup_error(ErrorCode::kInvalidArgument, "RUNTIME_EXECUTION_PLAN_EMPTY"));
  }

  auto operators =
      reconstruct_runtime_operators(*plan, run_generation, process_group, replay_mismatch);
  if (!operators) {
    return make_unexpected(std::move(operators.error()));
  }

  auto result = std::unique_ptr<GraphExec>{new GraphExec{std::move(plan), process_group}};
  result->operators_ = std::move(operators->operators);
  result->receiver_by_vertex_ = std::move(operators->receiver_by_vertex);
  result->receiver_by_operator_id_ = std::move(operators->receiver_by_operator_id);
  result->package_capability_report_ = std::move(operators->package_capability_report);
  result->parameter_updates_ = std::move(operators->parameter_updates);

  // A compiler-accepted empty Graph retains authored clocks only as inert evidence. Once the
  // complete frozen inventory is authenticated, construct no run-owned clock that cannot be
  // observed by an Operator.
  if (!has_plan_callback_owners(*result->plan_)) {
    return result;
  }

  auto clock = reconstruct_runtime_clocks(*result->plan_);
  if (!clock) {
    return make_unexpected(std::move(clock.error()));
  }
  result->clock_registry_ = std::move(clock->registry);
  result->default_clock_ = std::move(clock->default_clock);
  return result;
}

GraphExec::~GraphExec() = default;

GraphExec::GraphExec(std::shared_ptr<const ExecutionPlanState> plan, ProcessGroupId process_group)
    : plan_(std::move(plan)), process_group_(process_group) {}

std::optional<std::size_t> GraphExec::receiver_index(VertexId vertex) const noexcept {
  if (!vertex || vertex.value >= receiver_by_vertex_.size()) {
    return std::nullopt;
  }
  const std::size_t receiver = receiver_by_vertex_[static_cast<std::size_t>(vertex.value)];
  return receiver != kNoReceiver && receiver < operators_.size() &&
                 operators_[receiver].vertex == vertex
             ? std::optional<std::size_t>{receiver}
             : std::nullopt;
}

std::optional<std::size_t> GraphExec::receiver_index(
    OperatorInstanceId operator_id) const noexcept {
  if (operator_id == 0U ||
      static_cast<std::size_t>(operator_id) >= receiver_by_operator_id_.size()) {
    return std::nullopt;
  }
  const std::size_t receiver = receiver_by_operator_id_[static_cast<std::size_t>(operator_id)];
  return receiver != kNoReceiver && receiver < operators_.size() &&
                 operators_[receiver].operator_id == operator_id
             ? std::optional<std::size_t>{receiver}
             : std::nullopt;
}

RuntimeOperatorInstance* GraphExec::operator_by_vertex(VertexId vertex) noexcept {
  const auto receiver = receiver_index(vertex);
  return receiver ? &operators_[*receiver] : nullptr;
}

const RuntimeOperatorInstance* GraphExec::operator_by_vertex(VertexId vertex) const noexcept {
  const auto receiver = receiver_index(vertex);
  return receiver ? &operators_[*receiver] : nullptr;
}

RuntimeOperatorInstance* GraphExec::operator_by_id(OperatorInstanceId operator_id) noexcept {
  const auto receiver = receiver_index(operator_id);
  return receiver ? &operators_[*receiver] : nullptr;
}

const RuntimeOperatorInstance* GraphExec::operator_by_id(
    OperatorInstanceId operator_id) const noexcept {
  const auto receiver = receiver_index(operator_id);
  return receiver ? &operators_[*receiver] : nullptr;
}

expected<ClockRef, Error> GraphExec::resolve_clock(ClockId id) const noexcept {
  return clock_registry_.resolve(id);
}

expected<void, Error> GraphExec::wait_until(ClockId id, std::int64_t timestamp_ns) const noexcept {
  return clock_registry_.wait_until(id, timestamp_ns);
}

expected<void, Error> GraphExec::initialize_lifecycle(
    RunGenerationId run_generation, const std::atomic_bool* cancellation, ExecutionLaneSet* lanes,
    RuntimeNotificationSourceRegistry* notifications, RuntimeRateState* rates,
    RuntimeRateMaterializationBinding rate_materialization) {
  if (!package_capability_report_ || !package_capability_report_->compatible() ||
      package_capability_report_->context.run_generation != run_generation) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_PACKAGE_CAPABILITY_STATE_INVALID"));
  }

  auto lifecycle = RuntimeLifecycleExecutor::create(*plan_,
                                                    operators_,
                                                    run_generation,
                                                    cancellation,
                                                    this,
                                                    lanes,
                                                    *package_capability_report_,
                                                    notifications,
                                                    rates,
                                                    rate_materialization);
  if (!lifecycle) {
    return make_unexpected(std::move(lifecycle.error()));
  }
  lifecycle_ = std::move(lifecycle).value();
  return {};
}

expected<void, Error> GraphExec::set_lifecycle_notifications(
    RuntimeNotificationSourceRegistry* notifications) noexcept {
  if (lifecycle_ == nullptr) {
    return make_unexpected(
        startup_error(ErrorCode::kLifecycleRunState, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }
  return lifecycle_->set_notifications(notifications);
}

bool GraphExec::lifecycle_startup() noexcept {
  return lifecycle_ != nullptr && lifecycle_->startup();
}

LifecycleRestartExecutionOutcome GraphExec::lifecycle_restart(
    const LifecycleRestartExecutionRequest& request) noexcept {
  return lifecycle_ == nullptr ? LifecycleRestartExecutionOutcome{} : lifecycle_->restart(request);
}

bool GraphExec::lifecycle_shutdown() noexcept {
  return lifecycle_ == nullptr || lifecycle_->shutdown();
}

std::string_view GraphExec::lifecycle_failure_token() const noexcept {
  return lifecycle_ == nullptr ? std::string_view{"RUNTIME_LIFECYCLE_CALLBACK_FAILED"}
                               : lifecycle_->failure_token();
}

}  // namespace holoscan::detail
