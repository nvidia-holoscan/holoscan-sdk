/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_control_state.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "parameter_catalog_projection.hpp"
#include "runtime_callback_context.hpp"

namespace holoscan::detail {

// RunSession owns one reference and every policy-specific handle state retains another. The shared
// link is created before asynchronous startup, attached once, and closed once for all capabilities.
struct ParamControlLink {
  std::atomic<std::size_t> references{1};
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  OperatorInstanceId operator_count{};
  std::vector<ParamAddress> fields;
  std::mutex mutex;
  std::condition_variable state_changed;
  void* coordinator{};
  const ParamControlOperations* operations{};
  std::size_t active_calls{};
  std::size_t detach_waiters{};
  bool closed{};
  bool detach_notified{};
  bool detach_hook_in_progress{};
};

// Copies of one public wrapper share this fixed policy state. Independently minted handles own
// distinct policy states but retain the same session link, so no handle can widen another.
struct ParamControlState {
  ParamControlState(ParamControlLink* link_value, ParamAccessPolicy policy_value) noexcept
      : link(link_value), policy(policy_value) {}

  std::atomic<std::size_t> references{1};
  ParamControlLink* link{};
  ParamAccessPolicy policy{};
  std::atomic<std::uint32_t> open_event_cursors{};
  ParameterCatalogView catalog;
};

struct ParamEventCursorState {
  ParamControlState* capability{};
  std::uint32_t cursor{};
};

namespace {

thread_local std::size_t g_general_control_forbidden_depth = 0U;

[[nodiscard]] bool run_scope(ParamScopeFilter scope) noexcept {
  const OperatorRef empty;
  return scope.operator_instance == empty && scope.field == 0U;
}

[[nodiscard]] bool scope_contains(ParamScopeFilter parent, ParamScopeFilter child) noexcept {
  if (run_scope(parent)) {
    return true;
  }
  if (run_scope(child) || parent.operator_instance != child.operator_instance) {
    return false;
  }
  return parent.field == 0U || parent.field == child.field;
}

[[nodiscard]] bool scope_belongs_to_link(const ParamControlLink& link,
                                         ParamScopeFilter scope) noexcept {
  if (run_scope(scope)) {
    return true;
  }
  const OperatorRef owner = scope.operator_instance;
  if (!owner.run_bound() || owner.plan_generation != link.plan_generation ||
      owner.run_generation != link.run_generation || owner.id > link.operator_count) {
    return false;
  }
  if (scope.field == 0U) {
    return true;
  }
  return std::find(link.fields.begin(),
                   link.fields.end(),
                   ParamAddress{.operator_instance = owner, .field = scope.field}) !=
         link.fields.end();
}

[[nodiscard]] bool policy_is_restriction(const ParamAccessPolicy& parent,
                                         const ParamAccessPolicy& child) noexcept {
  const auto parent_permissions = static_cast<std::uint8_t>(parent.permissions);
  const auto child_permissions = static_cast<std::uint8_t>(child.permissions);
  return parent.origin == child.origin &&
         (child_permissions & static_cast<std::uint8_t>(~parent_permissions)) == 0U &&
         scope_contains(parent.allowed_scope, child.allowed_scope) &&
         (!child.include_internal || parent.include_internal) &&
         (!child.include_sensitive || parent.include_sensitive) &&
         child.max_event_cursors <= parent.max_event_cursors;
}

void retain_link(ParamControlLink* link) noexcept {
  if (link != nullptr) {
    link->references.fetch_add(1, std::memory_order_relaxed);
  }
}

void release_link(ParamControlLink* link) noexcept {
  if (link != nullptr && link->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete link;
  }
}

class ParamControlLinkClaim {
 public:
  explicit ParamControlLinkClaim(ParamControlLink* link) noexcept : link_(link) {
    retain_link(link_);
  }

  ParamControlLinkClaim(const ParamControlLinkClaim&) = delete;
  ParamControlLinkClaim& operator=(const ParamControlLinkClaim&) = delete;

  ~ParamControlLinkClaim() noexcept { release_link(link_); }

 private:
  ParamControlLink* link_{};
};

void retain_state(ParamControlState* state) noexcept {
  if (state != nullptr) {
    state->references.fetch_add(1, std::memory_order_relaxed);
  }
}

void release_state(ParamControlState* state) noexcept {
  if (state != nullptr && state->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    release_link(state->link);
    delete state;
  }
}

void finish_call(ParamControlLink* link) noexcept {
  const std::lock_guard lock{link->mutex};
  --link->active_calls;
  if (link->closed && link->active_calls == 0) {
    link->state_changed.notify_all();
  }
}

class ParamControlCall {
 public:
  ParamControlCall() noexcept = default;

  ParamControlCall(ParamControlLink* link_value, void* coordinator_value,
                   const ParamControlOperations* operations_value,
                   ParamAccessPolicy policy_value) noexcept
      : link_(link_value),
        coordinator_(coordinator_value),
        operations_(operations_value),
        policy_(policy_value) {}

  ParamControlCall(const ParamControlCall&) = delete;
  ParamControlCall& operator=(const ParamControlCall&) = delete;

  ParamControlCall(ParamControlCall&& other) noexcept
      : link_(std::exchange(other.link_, nullptr)),
        coordinator_(other.coordinator_),
        operations_(other.operations_),
        policy_(other.policy_) {}

  ParamControlCall& operator=(ParamControlCall&&) = delete;

  ~ParamControlCall() noexcept {
    if (link_ != nullptr) {
      finish_call(link_);
    }
  }

  [[nodiscard]] void* coordinator() const noexcept { return coordinator_; }
  [[nodiscard]] const ParamControlOperations& operations() const noexcept { return *operations_; }
  [[nodiscard]] const ParamAccessPolicy& policy() const noexcept { return policy_; }

 private:
  ParamControlLink* link_{};
  void* coordinator_{};
  const ParamControlOperations* operations_{};
  ParamAccessPolicy policy_{};
};

[[nodiscard]] expected<ParamControlCall, Error> acquire_call(ParamControlState* state) {
  if (state == nullptr || state->link == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  ParamControlLink* const link = state->link;
  std::unique_lock lock{link->mutex};
  link->state_changed.wait(lock, [link] {
    return link->closed || (link->coordinator != nullptr && link->operations != nullptr);
  });
  if (link->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  ++link->active_calls;
  return ParamControlCall{link, link->coordinator, link->operations, state->policy};
}

[[nodiscard]] expected<ParamControlCall, Error> acquire_call_for(ParamControlState* state,
                                                                 std::chrono::nanoseconds timeout) {
  if (state == nullptr || state->link == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  ParamControlLink* const link = state->link;
  std::unique_lock lock{link->mutex};
  const bool ready = link->state_changed.wait_for(lock, timeout, [link] {
    return link->closed || (link->coordinator != nullptr && link->operations != nullptr);
  });
  if (!ready) {
    return make_unexpected(Error{ErrorCode::kTimeout});
  }
  if (link->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  ++link->active_calls;
  return ParamControlCall{link, link->coordinator, link->operations, state->policy};
}

[[nodiscard]] expected<ParamControlCall, Error> acquire_cursor_cleanup_call(
    ParamControlState* state) noexcept {
  if (state == nullptr || state->link == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  ParamControlLink* const link = state->link;
  const std::lock_guard lock{link->mutex};
  // Close is monotonic, but the coordinator remains alive until detach drains active calls.
  // Cursor release may therefore reclaim its finite slot during that close-to-detach window.
  if (link->coordinator == nullptr || link->operations == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  ++link->active_calls;
  return ParamControlCall{link, link->coordinator, link->operations, state->policy};
}

void release_event_cursor_state(ParamEventCursorState* state) noexcept {
  if (state == nullptr) {
    return;
  }
  auto call = acquire_cursor_cleanup_call(state->capability);
  if (call) {
    call->operations().unregister_event_cursor(call->coordinator(), state->cursor);
  }
  state->capability->open_event_cursors.fetch_sub(1U, std::memory_order_release);
  release_state(state->capability);
  delete state;
}

[[nodiscard]] ParamBatchUpdateResult closed_submit_result(std::size_t patch_count) noexcept {
  ParamBatchUpdateResult result{
      .status = ParamUpdateStatus::kClosed,
      .committed = false,
  };
  if (patch_count <= ParamBatchUpdateResult::kNoFailedPatch) {
    result.patch_count = static_cast<std::uint32_t>(patch_count);
  }
  return result;
}

[[nodiscard]] ParamBatchUpdateResult invalid_context_submit_result(
    std::size_t patch_count) noexcept {
  ParamBatchUpdateResult result{
      .status = ParamUpdateStatus::kInvalidCallingContext,
      .committed = false,
  };
  if (patch_count <= ParamBatchUpdateResult::kNoFailedPatch) {
    result.patch_count = static_cast<std::uint32_t>(patch_count);
  }
  return result;
}

void close_link(ParamControlLink* link) noexcept {
  if (link == nullptr) {
    return;
  }

  void* coordinator = nullptr;
  const ParamControlOperations* operations = nullptr;
  {
    const std::lock_guard lock{link->mutex};
    link->closed = true;
    link->state_changed.notify_all();
    if (!link->detach_notified && link->coordinator != nullptr && link->operations != nullptr) {
      link->detach_notified = true;
      coordinator = link->coordinator;
      operations = link->operations;
      link->detach_hook_in_progress = true;
    }
  }

  // The hook runs outside the link lock because it may wake threads that need the same lifetime
  // machinery. The explicit in-progress claim keeps detach from retiring the captured coordinator
  // until this out-of-lock call returns.
  if (coordinator != nullptr) {
    operations->begin_detach(coordinator);
    const std::lock_guard lock{link->mutex};
    link->detach_hook_in_progress = false;
    link->state_changed.notify_all();
  }
}

}  // namespace

ParamGeneralControlContextScope::ParamGeneralControlContextScope() noexcept {
  ++g_general_control_forbidden_depth;
}

ParamGeneralControlContextScope::~ParamGeneralControlContextScope() noexcept {
  if (g_general_control_forbidden_depth != 0U) {
    --g_general_control_forbidden_depth;
  }
}

bool general_parameter_control_forbidden() noexcept {
  return g_general_control_forbidden_depth != 0U || runtime_callback_context_restricted();
}

expected<ParamControlLink*, Error> ParamControlAccess::create_unattached_link(
    PlanGenerationId plan_generation, RunGenerationId run_generation,
    OperatorInstanceId operator_count, std::span<const ParamAddress> fields) noexcept {
  if (plan_generation == 0U || run_generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  for (const ParamAddress field : fields) {
    if (!field.operator_instance.run_bound() || field.operator_instance.id > operator_count ||
        field.operator_instance.plan_generation != plan_generation ||
        field.operator_instance.run_generation != run_generation || field.field == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }
  try {
    auto* link = new ParamControlLink{};
    try {
      link->plan_generation = plan_generation;
      link->run_generation = run_generation;
      link->operator_count = operator_count;
      link->fields.assign(fields.begin(), fields.end());
    } catch (...) {
      delete link;
      throw;
    }
    return link;
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

expected<void, Error> ParamControlAccess::attach(ParamControlLink* link, void* coordinator,
                                                 const ParamControlOperations* operations) {
  if (link == nullptr || coordinator == nullptr || operations == nullptr ||
      operations->catalog == nullptr || operations->max_event_cursors == nullptr ||
      operations->read == nullptr || operations->submit == nullptr ||
      operations->await == nullptr || operations->capture == nullptr ||
      operations->register_event_cursor == nullptr ||
      operations->unregister_event_cursor == nullptr || operations->drain_events == nullptr ||
      operations->begin_detach == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  const std::lock_guard lock{link->mutex};
  if (link->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (link->coordinator != nullptr || link->operations != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  link->coordinator = coordinator;
  link->operations = operations;
  link->state_changed.notify_all();
  return {};
}

expected<void, Error> ParamControlAccess::wait_until_attached(ParamControlLink* link) {
  if (link == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  const ParamControlLinkClaim claim{link};
  std::unique_lock lock{link->mutex};
  link->state_changed.wait(lock, [link] {
    return link->closed || (link->coordinator != nullptr && link->operations != nullptr);
  });
  if (link->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return {};
}

expected<ParamControlLink*, Error> ParamControlAccess::retain_for_handle(
    ParamControlLink* link, ParamAccessPolicy policy) noexcept {
  if (link == nullptr || !valid_policy_shape(policy)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  const std::lock_guard lock{link->mutex};
  if (link->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!scope_belongs_to_link(*link, policy.allowed_scope)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  retain_link(link);
  return link;
}

expected<ParamControlHandle, Error> ParamControlAccess::make_handle_from_retained(
    ParamControlLink* retained_link, ParamAccessPolicy policy) {
  if (retained_link == nullptr || !valid_policy_shape(policy)) {
    release_link(retained_link);
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ParamControlState* state = nullptr;
  try {
    state = new ParamControlState{retained_link, policy};
  } catch (...) {
    release_link(retained_link);
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }

  expected<ParameterCatalogView, Error> filtered = [&]() -> expected<ParameterCatalogView, Error> {
    if (!has_param_permission(policy.permissions, ParamPermission::kRead)) {
      return filter_catalog(ParameterCatalogView{},
                            policy,
                            retained_link->plan_generation,
                            retained_link->run_generation);
    }
    auto call = acquire_call(state);
    if (!call) {
      return expected<ParameterCatalogView, Error>{make_unexpected(call.error())};
    }
    return filter_catalog(call->operations().catalog(call->coordinator()),
                          policy,
                          retained_link->plan_generation,
                          retained_link->run_generation);
  }();
  if (!filtered) {
    release_state(state);
    return make_unexpected(filtered.error());
  }
  state->catalog = std::move(*filtered);
  return ParamControlHandle{state};
}

expected<ParamControlHandle, Error> ParamControlAccess::restrict_handle(
    ParamControlState* state, const ParamAccessPolicy& policy) {
  if (state == nullptr || state->link == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!valid_policy_shape(policy)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!policy_is_restriction(state->policy, policy)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  auto retained = retain_for_handle(state->link, policy);
  if (!retained) {
    return make_unexpected(retained.error());
  }
  return make_prepared_handle(*retained, policy, state->catalog);
}

expected<ParameterCatalogView, Error> ParamControlAccess::filter_catalog(
    const ParameterCatalogView& source, const ParamAccessPolicy& policy,
    PlanGenerationId plan_generation, RunGenerationId run_generation) noexcept {
  return filter_runtime_parameter_catalog(source, policy, plan_generation, run_generation);
}

expected<ParamControlHandle, Error> ParamControlAccess::make_prepared_handle(
    ParamControlLink* retained_link, ParamAccessPolicy policy,
    const ParameterCatalogView& source) noexcept {
  ParamControlState* state = nullptr;
  try {
    state = new ParamControlState{retained_link, policy};
  } catch (...) {
    release_link(retained_link);
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const ParameterCatalogView empty_source;
  const ParameterCatalogView& visible_source =
      has_param_permission(policy.permissions, ParamPermission::kRead) ? source : empty_source;
  auto filtered = filter_catalog(
      visible_source, policy, retained_link->plan_generation, retained_link->run_generation);
  if (!filtered) {
    release_state(state);
    return make_unexpected(filtered.error());
  }
  state->catalog = std::move(*filtered);
  return ParamControlHandle{state};
}

const ParameterCatalogView& ParamControlAccess::catalog(ParamControlState* state) noexcept {
  static const ParameterCatalogView kEmptyCatalog;
  return state == nullptr || general_parameter_control_forbidden() ? kEmptyCatalog : state->catalog;
}

expected<ParamEventCursor, Error> ParamControlAccess::open_event_cursor(
    ParamControlState* state, ParamEventCursorOptions options) {
  if (state == nullptr || state->link == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!has_param_permission(state->policy.permissions, ParamPermission::kObserve) ||
      state->policy.max_event_cursors == 0U) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  if (options.after_event_sequence != 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto call = acquire_call(state);
  if (!call) {
    return make_unexpected(call.error());
  }
  const std::uint32_t admitted = call->operations().max_event_cursors(call->coordinator());
  if (state->policy.max_event_cursors > admitted) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  std::uint32_t count = state->open_event_cursors.load(std::memory_order_relaxed);
  while (count < state->policy.max_event_cursors) {
    if (state->open_event_cursors.compare_exchange_weak(
            count, count + 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      const auto registered =
          call->operations().register_event_cursor(call->coordinator(), call->policy());
      if (!registered) {
        state->open_event_cursors.fetch_sub(1U, std::memory_order_release);
        return make_unexpected(registered.error());
      }
      retain_state(state);
      try {
        return ParamEventCursor{new ParamEventCursorState{
            .capability = state,
            .cursor = *registered,
        }};
      } catch (...) {
        call->operations().unregister_event_cursor(call->coordinator(), *registered);
        state->open_event_cursors.fetch_sub(1U, std::memory_order_release);
        release_state(state);
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
    }
  }
  return make_unexpected(Error{ErrorCode::kResourceExhausted});
}

bool ParamControlAccess::valid_policy_shape(const ParamAccessPolicy& policy) noexcept {
  if (policy.origin != ParamControlOrigin::kLocalApplication &&
      policy.origin != ParamControlOrigin::kExternalAdapter) {
    return false;
  }
  constexpr auto kKnownPermissions = static_cast<std::uint8_t>(ParamPermission::kRead) |
                                     static_cast<std::uint8_t>(ParamPermission::kUpdate) |
                                     static_cast<std::uint8_t>(ParamPermission::kAwait) |
                                     static_cast<std::uint8_t>(ParamPermission::kCapture) |
                                     static_cast<std::uint8_t>(ParamPermission::kObserve);
  if ((static_cast<std::uint8_t>(policy.permissions) &
       static_cast<std::uint8_t>(~kKnownPermissions)) != 0U ||
      (!has_param_permission(policy.permissions, ParamPermission::kObserve) &&
       policy.max_event_cursors != 0U)) {
    return false;
  }
  if (run_scope(policy.allowed_scope)) {
    return true;
  }
  return policy.allowed_scope.operator_instance.run_bound();
}

void ParamControlAccess::close(ParamControlLink* link) noexcept {
  close_link(link);
}

void ParamControlAccess::detach_and_release(ParamControlLink* link) noexcept {
  if (link == nullptr) {
    return;
  }

  close_link(link);

  {
    std::unique_lock lock{link->mutex};
    ++link->detach_waiters;
    link->state_changed.notify_all();
    link->state_changed.wait(
        lock, [link] { return link->active_calls == 0 && !link->detach_hook_in_progress; });
    --link->detach_waiters;
    link->coordinator = nullptr;
    link->operations = nullptr;
  }
  release_link(link);
}

void ParamControlTestAccess::wait_until_detach_waits(ParamControlLink* link) {
  if (link == nullptr) {
    return;
  }
  std::unique_lock lock{link->mutex};
  link->state_changed.wait(lock, [link] { return link->detach_waiters != 0U; });
}

}  // namespace holoscan::detail

namespace holoscan {
namespace {

[[nodiscard]] bool path_is_prefix(const OperatorPath& prefix, const OperatorPath& path) noexcept {
  const auto prefix_segments = prefix.segments();
  const auto path_segments = path.segments();
  return prefix_segments.size() <= path_segments.size() &&
         std::equal(prefix_segments.begin(), prefix_segments.end(), path_segments.begin());
}

[[nodiscard]] bool scope_contains(const ParamSnapshotScope& parent,
                                  const ParamSnapshotScope& child) noexcept {
  switch (parent.kind()) {
    case ParamSnapshotScopeKind::kRoot:
      return true;
    case ParamSnapshotScopeKind::kGraph:
      return child.kind() != ParamSnapshotScopeKind::kRoot &&
             path_is_prefix(parent.path(), child.path());
    case ParamSnapshotScopeKind::kOperator:
      return child.kind() == ParamSnapshotScopeKind::kOperator && parent.path() == child.path();
  }
  return false;
}

}  // namespace

expected<ParamSnapshot, Error> ParamSnapshot::slice(const ParamSnapshotScope& scope) const {
  if (scope.kind() != ParamSnapshotScopeKind::kRoot &&
      scope.kind() != ParamSnapshotScopeKind::kGraph &&
      scope.kind() != ParamSnapshotScopeKind::kOperator) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (scope.kind() != ParamSnapshotScopeKind::kRoot && scope.path().segments().empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!scope_contains(scope_, scope)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }

  try {
    ParamSnapshot result;
    result.plan_digest_ = plan_digest_;
    result.plan_generation_ = plan_generation_;
    result.run_generation_ = run_generation_;
    result.catalog_digest_ = catalog_digest_;
    result.view_ = view_;
    result.consistency_ = consistency_;
    result.scope_ = scope;
    result.accepted_through_ = accepted_through_;
    result.published_through_ = published_through_;
    if (scope.kind() == ParamSnapshotScopeKind::kRoot) {
      result.operators_ = operators_;
      return result;
    }

    const auto requested_segments = scope.path().segments();
    for (const auto& record : operators_) {
      const auto operator_segments = record.path.segments();
      const bool selected = scope.kind() == ParamSnapshotScopeKind::kOperator
                                ? record.path == scope.path()
                                : operator_segments.size() > requested_segments.size() &&
                                      std::equal(requested_segments.begin(),
                                                 requested_segments.end(),
                                                 operator_segments.begin());
      if (!selected) {
        continue;
      }
      result.operators_.push_back(record);
    }
    if (result.operators_.empty()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    return result;
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

expected<ParamSnapshot, Error> ParamSnapshot::slice(const OperatorPath& path) const {
  try {
    return slice(ParamSnapshotScope::operator_(path));
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

expected<ParamSnapshot, Error> ParamSnapshot::slice(std::string_view operator_path) const {
  auto parsed = OperatorPath::parse(operator_path);
  if (!parsed) {
    return make_unexpected(std::move(parsed.error()));
  }
  return slice(*parsed);
}

ParamControlHandle::ParamControlHandle(const ParamControlHandle& other) noexcept
    : state_(other.state_) {
  detail::retain_state(state_);
}

ParamControlHandle::ParamControlHandle(ParamControlHandle&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

ParamControlHandle& ParamControlHandle::operator=(const ParamControlHandle& other) noexcept {
  if (this != &other) {
    detail::ParamControlState* const replacement = other.state_;
    detail::retain_state(replacement);
    detail::release_state(std::exchange(state_, replacement));
  }
  return *this;
}

ParamControlHandle& ParamControlHandle::operator=(ParamControlHandle&& other) noexcept {
  if (this != &other) {
    detail::release_state(std::exchange(state_, std::exchange(other.state_, nullptr)));
  }
  return *this;
}

ParamControlHandle::~ParamControlHandle() noexcept {
  detail::release_state(state_);
}

const ParameterCatalogView& ParamControlHandle::catalog() const noexcept {
  return detail::ParamControlAccess::catalog(state_);
}

expected<ParamReadResult, Error> ParamControlHandle::read(ParamAddress address,
                                                          ParamReadView view) const {
  if (detail::general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (view != ParamReadView::kPublished && view != ParamReadView::kLatestAccepted &&
      view != ParamReadView::kPending) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto call = detail::acquire_call(state_);
  if (!call) {
    return make_unexpected(call.error());
  }
  return call->operations().read(call->coordinator(), call->policy(), address, view);
}

expected<ParamControlHandle, Error> ParamControlHandle::restrict(
    const ParamAccessPolicy& policy) const {
  if (detail::general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return detail::ParamControlAccess::restrict_handle(state_, policy);
}

ParamBatchUpdateResult ParamControlHandle::submit(const ParamPatch& patch,
                                                  ParamUpdateOptions options) {
  return submit(std::span<const ParamPatch>{&patch, 1U}, options);
}

ParamBatchUpdateResult ParamControlHandle::submit(std::span<const ParamPatch> patches,
                                                  ParamUpdateOptions options) {
  if (detail::general_parameter_control_forbidden()) {
    return detail::invalid_context_submit_result(patches.size());
  }
  auto call = detail::acquire_call(state_);
  if (!call) {
    return detail::closed_submit_result(patches.size());
  }
  return call->operations().submit(call->coordinator(), call->policy(), patches, options);
}

expected<ParamUpdateResult, Error> ParamControlHandle::await(
    ParamUpdateSequence update_sequence, std::chrono::nanoseconds timeout) const {
  if (detail::general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (update_sequence == 0 || timeout.count() < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  const auto wait_started = std::chrono::steady_clock::now();
  auto call = detail::acquire_call_for(state_, timeout);
  if (!call) {
    return make_unexpected(call.error());
  }
  const auto attach_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - wait_started);
  const std::chrono::nanoseconds remaining =
      attach_elapsed >= timeout ? std::chrono::nanoseconds::zero() : timeout - attach_elapsed;
  return call->operations().await(call->coordinator(), call->policy(), update_sequence, remaining);
}

expected<ParamSnapshot, Error> ParamControlHandle::capture(ParamCaptureOptions options) const {
  if (detail::general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto call = detail::acquire_call(state_);
  if (!call) {
    return make_unexpected(call.error());
  }
  if (!has_param_permission(call->policy().permissions, ParamPermission::kCapture)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  return call->operations().capture(
      call->coordinator(), call->policy(), state_->catalog.compatibility_digest(), options);
}

expected<ParamEventCursor, Error> ParamControlHandle::open_event_cursor(
    ParamEventCursorOptions options) const {
  if (detail::general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return detail::ParamControlAccess::open_event_cursor(state_, options);
}

ParamEventCursor::ParamEventCursor(ParamEventCursor&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

ParamEventCursor& ParamEventCursor::operator=(ParamEventCursor&& other) noexcept {
  if (this != &other) {
    detail::release_event_cursor_state(std::exchange(state_, std::exchange(other.state_, nullptr)));
  }
  return *this;
}

ParamEventCursor::~ParamEventCursor() noexcept {
  detail::release_event_cursor_state(std::exchange(state_, nullptr));
}

expected<std::size_t, Error> ParamEventCursor::drain(std::span<ParamEvent> output) noexcept {
  if (state_ == nullptr || state_->capability == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (detail::general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto call = detail::acquire_call(state_->capability);
  if (!call) {
    return make_unexpected(call.error());
  }
  return call->operations().drain_events(call->coordinator(), state_->cursor, output);
}

}  // namespace holoscan
