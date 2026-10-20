/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <span>

#include <holoscan/core/parameter_control.hpp>

namespace holoscan::detail {

// This callback table isolates the installed one-pointer handle from the concrete generation
// coordinator. Each function runs while ParamControlLink holds an active-call claim, so the
// session must keep `coordinator` alive until ParamControlAccess::detach_and_release returns.
struct ParamControlOperations {
  const ParameterCatalogView& (*catalog)(void* coordinator) noexcept = nullptr;
  std::uint32_t (*max_event_cursors)(void* coordinator) noexcept = nullptr;
  expected<ParamReadResult, Error> (*read)(void* coordinator, const ParamAccessPolicy& policy,
                                           ParamAddress address, ParamReadView view) = nullptr;
  ParamBatchUpdateResult (*submit)(void* coordinator, const ParamAccessPolicy& policy,
                                   std::span<const ParamPatch> patches,
                                   ParamUpdateOptions options) = nullptr;
  expected<ParamUpdateResult, Error> (*await)(void* coordinator, const ParamAccessPolicy& policy,
                                              ParamUpdateSequence update_sequence,
                                              std::chrono::nanoseconds timeout) = nullptr;
  expected<ParamSnapshot, Error> (*capture)(void* coordinator, const ParamAccessPolicy& policy,
                                            ParamCatalogDigest source_catalog_digest,
                                            ParamCaptureOptions options) = nullptr;
  expected<std::uint32_t, Error> (*register_event_cursor)(
      void* coordinator, const ParamAccessPolicy& policy) noexcept = nullptr;
  void (*unregister_event_cursor)(void* coordinator, std::uint32_t cursor) noexcept = nullptr;
  expected<std::size_t, Error> (*drain_events)(void* coordinator, std::uint32_t cursor,
                                               std::span<ParamEvent> output) noexcept = nullptr;

  // Detachment marks the link closed before invoking this hook outside the link mutex. The link
  // retains an explicit hook-in-progress claim, so coordinator retirement cannot pass the hook.
  // The coordinator uses it only to wake bounded waits; it must not destroy or reenter detach from
  // this callback.
  void (*begin_detach)(void* coordinator) noexcept = nullptr;
};

struct ParamControlLink;

/// @brief Mark one runtime callback scope where owning control calls are forbidden.
class ParamGeneralControlContextScope {
 public:
  ParamGeneralControlContextScope() noexcept;
  ParamGeneralControlContextScope(const ParamGeneralControlContextScope&) = delete;
  ParamGeneralControlContextScope& operator=(const ParamGeneralControlContextScope&) = delete;
  ~ParamGeneralControlContextScope() noexcept;
};

/// @brief Return whether the current thread is executing a runtime callback.
[[nodiscard]] bool general_parameter_control_forbidden() noexcept;

// RunSession owns one shared link for its complete lifetime. Any number of policy-specific public
// handles retain that link through their private ParamControlState, so the session tracks no list
// of minted capabilities and one detach closes them all.
class ParamControlAccess {
 public:
  // Create the session-owned tombstone and immutable scope catalog before asynchronous startup.
  // The returned pointer owns one intrusive reference that detach_and_release consumes exactly
  // once; copied addresses contain no runtime object pointers or authority.
  [[nodiscard]] static expected<ParamControlLink*, Error> create_unattached_link(
      PlanGenerationId plan_generation, RunGenerationId run_generation,
      OperatorInstanceId operator_count, std::span<const ParamAddress> fields) noexcept;

  // Attach exactly one live coordinator. A stop/close that won the race rejects attachment.
  [[nodiscard]] static expected<void, Error> attach(ParamControlLink* link, void* coordinator,
                                                    const ParamControlOperations* operations);

  // Cold-path RunSession binding waits through asynchronous startup and returns once operations
  // are available or the monotonic session close wins.
  [[nodiscard]] static expected<void, Error> wait_until_attached(ParamControlLink* link);

  // Validate one fixed policy against an open link and retain one reference while the caller
  // still owns its session-side lifetime lock. The returned reference must be transferred to
  // make_handle_from_retained or released by that function on every failure path.
  [[nodiscard]] static expected<ParamControlLink*, Error> retain_for_handle(
      ParamControlLink* link, ParamAccessPolicy policy) noexcept;

  // Mint one fixed-policy capability from a previously retained link reference. Read-capable
  // handles synchronously prepare their immutable filtered catalog before success is returned;
  // attachment, closure, or allocation failure remains a typed construction error.
  [[nodiscard]] static expected<ParamControlHandle, Error> make_handle_from_retained(
      ParamControlLink* retained_link, ParamAccessPolicy policy);

  // Derive an independently retained child state only when every requested policy coordinate is
  // equal to or narrower than the parent. Its immutable catalog is filtered from the already
  // prepared parent projection, so the operation never waits for coordinator attachment.
  [[nodiscard]] static expected<ParamControlHandle, Error> restrict_handle(
      ParamControlState* state, const ParamAccessPolicy& policy);

  [[nodiscard]] static const ParameterCatalogView& catalog(ParamControlState* state) noexcept;

  [[nodiscard]] static expected<ParamEventCursor, Error> open_event_cursor(
      ParamControlState* state, ParamEventCursorOptions options);

  // Validate closed enum values and the structural shape of a run/Operator/field scope. Session
  // code separately verifies that a nonzero run-bound scope belongs to that exact RunSession.
  [[nodiscard]] static bool valid_policy_shape(const ParamAccessPolicy& policy) noexcept;

  // Publish the monotonic close boundary without releasing the session reference. This lets a
  // stop request win before startup attachment while the worker still safely observes the link.
  static void close(ParamControlLink* link) noexcept;

  // Monotonically close the link, wake startup and bounded await waiters, retire active calls, and
  // consume the session's intrusive reference before the coordinator is destroyed.
  static void detach_and_release(ParamControlLink* link) noexcept;

 private:
  [[nodiscard]] static expected<ParameterCatalogView, Error> filter_catalog(
      const ParameterCatalogView& source, const ParamAccessPolicy& policy,
      PlanGenerationId plan_generation, RunGenerationId run_generation) noexcept;

  [[nodiscard]] static expected<ParamControlHandle, Error> make_prepared_handle(
      ParamControlLink* retained_link, ParamAccessPolicy policy,
      const ParameterCatalogView& source) noexcept;
};

/// @brief Expose deterministic source-tree-only synchronization for control-link race tests.
class ParamControlTestAccess {
 public:
  /// @brief Wait until one detach caller is parked behind live coordinator work.
  /// @param link Source-tree control link retained by the test.
  static void wait_until_detach_waits(ParamControlLink* link);
};

}  // namespace holoscan::detail
