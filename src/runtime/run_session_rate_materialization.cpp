/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "run_session_internal.hpp"

namespace holoscan::detail {

// Startup/restart-shared rate-candidate resource materialization.

void RunSessionState::discard_rate_candidate_temporal_depths(
    std::span<ReadyRunState> states) noexcept {
  for (ReadyRunState& state : states) {
    for (auto& selector : state.input_selectors) {
      selector.selector.discard_rate_candidate_capacity();
    }
    for (auto& aligned : state.aligned_selectors) {
      aligned.selector.discard_rate_candidate_capacities();
    }
  }
}

// Validate every selector candidate before exposing any lifecycle-selected active depth.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RunSessionState::prepare_rate_candidate_temporal_depths(
    GraphExec& exec, std::span<ReadyRunState> states,
    const RuntimeRateMaterializationView& materialization) noexcept {
  const auto fail = [&](Error error) -> expected<void, Error> {
    discard_rate_candidate_temporal_depths(states);
    return make_unexpected(std::move(error));
  };
  for (const auto& depth : materialization.queue_depths) {
    if (!depth.active_depth_update_required) {
      continue;
    }
    const auto recipe = std::ranges::find(
        plan_->queue_sizing_recipes, depth.edge, &QueueSizingRecipePlanRecord::edge);
    if (recipe == plan_->queue_sizing_recipes.end() ||
        std::ranges::count(
            plan_->queue_sizing_recipes, depth.edge, &QueueSizingRecipePlanRecord::edge) != 1) {
      return fail(Error{ErrorCode::kInvalidArgument, "QUEUE_SIZING_RECIPE_MISSING"});
    }
    if (!recipe->latest_binding_stage || recipe->explicit_depth ||
        depth.active_logical_depth == 0U) {
      continue;
    }
    const auto edge =
        std::ranges::find(plan_->payload_edges, depth.edge, &PayloadEdgePlanRecord::edge);
    if (edge == plan_->payload_edges.end() ||
        std::ranges::count(plan_->payload_edges, depth.edge, &PayloadEdgePlanRecord::edge) != 1 ||
        !edge->target) {
      return fail(Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    const auto port = std::ranges::find_if(plan_->graph.ports(), [&](const PortRecord& row) {
      return row.id.value == edge->target.value() && row.direction == PortDirection::kInput;
    });
    if (port == plan_->graph.ports().end()) {
      return fail(Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    const bool local = worker_owns_vertex(*plan_, exec.process_group(), port->owner);
    const std::size_t planned_selector_count =
        std::ranges::count(plan_->input_selections, edge->target, &InputSelectionPlanRecord::input);
    if (planned_selector_count > 1U) {
      return fail(Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    std::size_t prepared = 0U;
    for (ReadyRunState& state : states) {
      for (auto& selector : state.input_selectors) {
        if (selector.input != edge->target || selector.aligned_member) {
          continue;
        }
        auto accepted =
            selector.selector.prepare_rate_candidate_capacity(depth.active_logical_depth);
        if (!accepted) {
          return fail(Error{accepted.error().code, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
        }
        ++prepared;
      }
      for (auto& aligned : state.aligned_selectors) {
        if (std::ranges::find(aligned.selector.member_inputs(), edge->target) ==
            aligned.selector.member_inputs().end()) {
          continue;
        }
        auto accepted = aligned.selector.prepare_rate_candidate_capacity(
            edge->target, depth.active_logical_depth);
        if (!accepted) {
          return fail(Error{accepted.error().code, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
        }
        ++prepared;
      }
    }
    if (local && prepared != planned_selector_count) {
      return fail(Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
  }
  return {};
}

// Keep cold lifecycle materialization and its cleanup precedence in one auditable callback.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RunSessionState::ensure_rate_candidate_resources(
    void* state, const RuntimeRateMaterializationView& materialization, LifecycleStage stage,
    LifecycleTransitionPhase phase) noexcept {
  auto* const context = static_cast<RateResourceMaterializationState*>(state);
  auto* const session = context == nullptr ? nullptr : context->session;
  const bool forward = phase == LifecycleTransitionPhase::kStartupForward ||
                       phase == LifecycleTransitionPhase::kRestartForward;
  if (session == nullptr || context->exec == nullptr || context->shared == nullptr ||
      context->cuda == nullptr || context->ready_states.empty() || !forward ||
      (stage != LifecycleStage::kAllocate && stage != LifecycleStage::kWarmUp &&
       stage != LifecycleStage::kArm)) {
    return make_unexpected(
        Error{ErrorCode::kLifecycleRunState, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  if (phase == LifecycleTransitionPhase::kStartupForward && context->defer_startup &&
      !context->resources_materialized) {
    if (stage != LifecycleStage::kAllocate) {
      return make_unexpected(Error{ErrorCode::kNotReady, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      struct CandidateQueueLookup {
        const ExecutionPlanState* plan{};
        const RuntimeRateMaterializationView* materialization{};
      } queue_lookup{.plan = session->plan_.get(), .materialization = &materialization};
      const RuntimeChannelSizingOverride sizing_override{
          .state = &queue_lookup,
          .queue_depth = [](void* opaque,
                            EdgeId edge) noexcept -> expected<std::optional<std::uint32_t>, Error> {
            const auto* const lookup = static_cast<const CandidateQueueLookup*>(opaque);
            if (lookup == nullptr || lookup->plan == nullptr ||
                lookup->materialization == nullptr) {
              return make_unexpected(
                  Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
            }
            const auto recipe = std::ranges::find(
                lookup->plan->queue_sizing_recipes, edge, &QueueSizingRecipePlanRecord::edge);
            if (recipe == lookup->plan->queue_sizing_recipes.end() ||
                recipe->geometry_mode != AllocationGeometryMode::kLifecycleCandidate) {
              return std::optional<std::uint32_t>{};
            }
            if (std::ranges::count(lookup->plan->queue_sizing_recipes,
                                   edge,
                                   &QueueSizingRecipePlanRecord::edge) != 1) {
              return make_unexpected(
                  Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
            }
            const auto depth = std::ranges::find(
                lookup->materialization->queue_depths, edge, &RuntimeQueueDepthGeneration::edge);
            if (depth == lookup->materialization->queue_depths.end() ||
                std::ranges::count(lookup->materialization->queue_depths,
                                   edge,
                                   &RuntimeQueueDepthGeneration::edge) != 1 ||
                depth->active_logical_depth == 0U) {
              return make_unexpected(
                  Error{ErrorCode::kNotReady, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
            }
            return std::optional<std::uint32_t>{depth->active_logical_depth};
          },
      };
      if (!session->initialize_channel_session(*context->exec,
                                               context->ready_states.front(),
                                               *context->shared,
                                               *context->cuda,
                                               sizing_override)) {
        return make_unexpected(Error{ErrorCode::kFailure, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
      }
      session->channel_session_ = context->shared->channels;
      for (ReadyRunState& ready : context->ready_states) {
        bind_shared_resources(ready, *context->shared);
      }
      auto tensor_resources = RuntimeTensorResources::create(*session->plan_,
                                                             context->exec->operators(),
                                                             session->run_generation_,
                                                             context->exec->process_group(),
                                                             *context->cuda,
                                                             materialization);
      if (!tensor_resources) {
        return make_unexpected(std::move(tensor_resources.error()));
      }
      session->tensor_resources_ = std::move(*tensor_resources);
      for (ReadyRunState& ready : context->ready_states) {
        if (!session->initialize_ready_state(
                *context->exec, ready, *context->shared, &materialization)) {
          return make_unexpected(
              Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
        }
      }
      if (!session->bind_shared_production_notifications(
              *context->exec, context->ready_states, *context->shared)) {
        return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
      }
      if (context->shared->notifications != nullptr) {
        auto installed =
            context->exec->set_lifecycle_notifications(context->shared->notifications.get());
        if (!installed) {
          return make_unexpected(std::move(installed.error()));
        }
      }
      context->resources_materialized = true;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
#endif
  }
  if (!context->resources_materialized) {
    return make_unexpected(Error{ErrorCode::kNotReady, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  const bool queue_update_required =
      std::ranges::any_of(materialization.queue_depths,
                          [](const auto& depth) { return depth.active_depth_update_required; });
  if (queue_update_required && session->channel_session_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotReady, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  for (const auto& depth : materialization.queue_depths) {
    if (!depth.active_depth_update_required || depth.active_logical_depth == 0U) {
      continue;
    }
    auto prepared = session->channel_session_->prepare_rate_candidate_active_depth(
        depth.edge, depth.active_logical_depth);
    if (!prepared) {
      Error error = std::move(prepared.error());
      session->channel_session_->discard_rate_candidate_active_depths();
      if (!depth.admitted_depth_ceiling &&
          (error.code == ErrorCode::kNotSupported || error.code == ErrorCode::kResourceExhausted)) {
        return make_unexpected(
            Error{ErrorCode::kResourceExhausted, "QUEUE_REPLACEMENT_NOT_ADMITTED"});
      }
      return make_unexpected(std::move(error));
    }
  }
  auto temporal = session->prepare_rate_candidate_temporal_depths(
      *context->exec, context->ready_states, materialization);
  if (!temporal) {
    session->channel_session_->discard_rate_candidate_active_depths();
    session->discard_rate_candidate_temporal_depths(context->ready_states);
    return temporal;
  }
  if (session->tensor_resources_ != nullptr) {
    auto prepared =
        phase == LifecycleTransitionPhase::kRestartForward
            ? session->tensor_resources_->prepare_rate_candidate_restart(materialization)
            : session->tensor_resources_->prepare_rate_candidate_startup(materialization);
    if (!prepared) {
      session->channel_session_->discard_rate_candidate_active_depths();
      session->discard_rate_candidate_temporal_depths(context->ready_states);
      return make_unexpected(std::move(prepared.error()));
    }
  }
  return {};
}

}  // namespace holoscan::detail
