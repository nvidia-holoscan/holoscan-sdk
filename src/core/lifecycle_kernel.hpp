/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle_types.hpp>

namespace holoscan::detail {

using LifecycleNodeId = std::uint32_t;
using LifecycleScopeId = std::uint64_t;

/// Canonical Graph-instance scope used to validate parent-authored cross-child edges.
struct LifecycleScopeRecord {
  LifecycleScopeId scope_id{};
  std::optional<LifecycleScopeId> parent_scope_id;
};

/// Compiler-local identity for one logical operator checkpoint owner.
struct LifecycleLogicalNodeRecord {
  LifecycleNodeId node_id{};
  LifecycleScopeId scope_id{};
  std::string stable_name;
  std::uint32_t stable_order{};
};

/// Pointer-free description of one registered sparse callback.
enum class LifecycleLogicalCallbackOrigin : std::uint8_t {
  kAdvancedMember,
  kLegacyStart,
  kLegacyStop,
};

struct LifecycleLogicalCallbackRecord {
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  bool required{};
  LifecycleMethodId method{};
  LifecycleLogicalCallbackOrigin origin{LifecycleLogicalCallbackOrigin::kAdvancedMember};
};

/// Complete identity of one stage-local ordering fact.
struct LifecycleLogicalDependencyKey {
  LifecycleScopeId scope_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::string stable_name;
};

/// Declare the unique checkpoint that produces an ordering fact.
struct LifecycleDependencyProducerRecord {
  LifecycleNodeId node_id{};
  LifecycleStage declared_stage{LifecycleStage::kConfigure};
  LifecycleLogicalDependencyKey key;
};

/// Declare that a checkpoint follows the producer of an ordering fact.
struct LifecycleDependencyRequirementRecord {
  LifecycleNodeId node_id{};
  LifecycleStage declared_stage{LifecycleStage::kConfigure};
  LifecycleLogicalDependencyKey key;
};

/// Declare a direct same-stage edge authored by one Graph scope over its descendants.
struct LifecycleDirectEdgeRecord {
  LifecycleScopeId declaration_scope_id{};
  LifecycleNodeId before_node_id{};
  LifecycleStage before_stage{LifecycleStage::kConfigure};
  LifecycleNodeId after_node_id{};
  LifecycleStage after_stage{LifecycleStage::kConfigure};
};

/// Finite cold-path bounds for normalized records and execution bookkeeping.
struct LifecycleKernelCapacity {
  std::size_t max_scopes{};
  std::size_t max_nodes{};
  std::size_t max_callbacks{};
  std::size_t max_dependency_records{};
  std::size_t max_direct_edges{};
  std::size_t max_completions{};
};

/// Untrusted logical records consumed by lifecycle normalization.
struct LifecycleStageGraphInput {
  std::vector<LifecycleScopeRecord> scopes;
  std::vector<LifecycleLogicalNodeRecord> nodes;
  std::vector<LifecycleLogicalCallbackRecord> callbacks;
  std::vector<LifecycleDependencyProducerRecord> producers;
  std::vector<LifecycleDependencyRequirementRecord> requirements;
  std::vector<LifecycleDirectEdgeRecord> direct_edges;
};

enum class LifecycleKernelErrorCode : std::uint8_t {
  kCapacityExceeded,
  kInvalidScope,
  kInvalidNode,
  kDuplicateNode,
  kAmbiguousNodeIdentity,
  kInvalidStage,
  kInvalidCallback,
  kInvalidDependencyKey,
  kMissingProducer,
  kDuplicateProducer,
  kCrossScopeDependency,
  kCrossStageDependency,
  kDuplicateCallback,
  kCycle,
};

/// Bounded structured failure returned by the private normalizer.
struct LifecycleKernelError {
  LifecycleKernelErrorCode code{LifecycleKernelErrorCode::kInvalidNode};
  LifecycleScopeId scope_id{};
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::string stable_name;
};

struct LifecycleStageLevel {
  std::vector<LifecycleNodeId> node_ids;
};

struct LifecycleStagePlan {
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::vector<LifecycleStageLevel> levels;
};

/// Private normalized stage graph. Every stage plan contains every logical node.
struct LifecycleStageGraph {
  LifecycleKernelCapacity capacity;
  std::vector<LifecycleLogicalNodeRecord> nodes;
  std::vector<LifecycleLogicalCallbackRecord> callbacks;
  std::array<LifecycleStagePlan, 8> stages;
};

[[nodiscard]] expected<LifecycleStageGraph, LifecycleKernelError> build_lifecycle_stage_graph(
    const LifecycleStageGraphInput& input, const LifecycleKernelCapacity& capacity);

using LifecycleKernelCallback = LifecycleStatus (*)(void* self, LifecycleStage stage,
                                                    LifecycleTransitionPhase phase) noexcept;

/// Session-local receiver binding for a pointer-free logical callback record.
struct LifecycleCallbackBinding {
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  void* self{};
  LifecycleKernelCallback callback{};
};

struct LifecycleCompletedCheckpoint {
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
};

struct LifecycleLogRecord;
using LifecycleLogSink = void (*)(void* state, const LifecycleLogRecord& record);

/// Mutable synthetic execution state retained across startup, restart, and stop.
struct LifecycleKernelState {
  bool steady_state{};
  std::size_t completion_capacity{};
  std::vector<LifecycleCompletedCheckpoint> completed;
  std::vector<bool> rollback_visited;
  LifecycleLogSink log_sink{};
  void* log_state{};
  bool defer_failure_report{};
};

[[nodiscard]] LifecycleKernelState make_lifecycle_kernel_state(const LifecycleStageGraph& graph);

enum class LifecycleExecutionDisposition : std::uint8_t {
  kCompleted,
  kRolledBack,
  kInvalidState,
  kInvalidRestartStage,
  kInvalidBinding,
  kStageFailed,
  kRollbackFailed,
  kCapacityExceeded,
};

struct LifecycleExecutionFailure {
  bool present{};
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
  LifecycleStatus status{LifecycleStatus::kFatalFailure};
};

/// Result preserving the forward/teardown failure independently from rollback failure.
struct LifecycleExecutionResult {
  LifecycleExecutionDisposition disposition{LifecycleExecutionDisposition::kCompleted};
  LifecycleExecutionFailure primary_failure;
  LifecycleExecutionFailure rollback_failure;
  LifecycleExecutionFailure restoration_failure;
  LifecycleExecutionFailure restoration_rollback_failure;
  bool steady_state{};
};

enum class LifecycleLogEventKind : std::uint8_t {
  kTransitionBegin,
  kCallbackBegin,
  kCallbackEnd,
  kTransitionEnd,
};

enum class LifecycleLogOperation : std::uint8_t {
  kStartup,
  kRestart,
  kShutdown,
};

enum class LifecycleLogSeverity : std::uint8_t {
  kTrace,
  kDebug,
  kError,
};

/// Fixed-size structural lifecycle evidence passed to the private logging seam.
struct LifecycleLogRecord {
  LifecycleLogEventKind kind{LifecycleLogEventKind::kTransitionBegin};
  LifecycleLogOperation operation{LifecycleLogOperation::kStartup};
  LifecycleExecutionDisposition disposition{LifecycleExecutionDisposition::kCompleted};
  LifecycleNodeId node_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
  LifecycleStatus status{LifecycleStatus::kOk};
};

/// Return the fixed severity assigned to one structural lifecycle record.
[[nodiscard]] LifecycleLogSeverity lifecycle_log_severity(
    const LifecycleLogRecord& record) noexcept;

/// Invoke a private log sink without allowing diagnostics to affect lifecycle state.
void emit_lifecycle_log_noexcept(LifecycleLogSink sink, void* state,
                                 const LifecycleLogRecord& record) noexcept;

[[nodiscard]] LifecycleExecutionResult execute_lifecycle_startup(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleKernelState& state) noexcept;

[[nodiscard]] LifecycleExecutionResult execute_lifecycle_restart(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleStage requested_stage, LifecycleKernelState& state) noexcept;

[[nodiscard]] LifecycleExecutionResult execute_lifecycle_shutdown(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleKernelState& state) noexcept;

}  // namespace holoscan::detail
