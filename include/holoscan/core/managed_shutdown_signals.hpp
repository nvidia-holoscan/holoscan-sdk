/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

namespace detail {
class ManagedShutdownSignalsAccess;
}  // namespace detail

/// @brief Own one early process domain for managed SIGINT and SIGTERM acceptance.
///
/// On Linux, create this object on the initial application thread before other participant
/// threads are started. Threads created afterward inherit its managed-signal mask. The object is
/// creator-thread-affine, supports one managed run binding, and must outlive that binding
/// and every participant thread. On unsupported platforms, create() reports kNotSupported.
///
/// The type is move-constructible on its creator thread but is neither copyable nor
/// move-assignable. A moved-from or successfully closed object has no domain. Public operations on
/// one domain are not thread-safe and must be serialized on the creator thread. Establishment may
/// allocate, create descriptors, and start one ordinary broker thread. The domain installs no
/// signal handler and does not change the application's SIGINT or SIGTERM dispositions.
///
/// A domain that successfully reserves a managed run binding is spent after that attempt; create a
/// new domain for another managed run. HSDK use of an inherited domain after `fork()`, and host
/// mutation of the managed masks or dispositions, are currently unsupported. Domain descriptors
/// are close-on-exec, but the blocked signal mask is inherited by a child and survives `exec()`.
/// Before executing another program, reset the child mask with `sigprocmask()` or use
/// `POSIX_SPAWN_SETSIGMASK` with an application-chosen mask. HSDK does not intercept `exec()`;
/// detectable post-fork API use reports `kNotSupported`.
class HOLOSCAN_CORE_EXPORT ManagedShutdownSignals {
 public:
  /// @brief Establish the sole process managed-shutdown-signal domain.
  /// @return A creator-thread-affine domain, or a bounded setup, ownership, or platform error.
  /// @note On Linux, call this before starting any other participant thread. The call snapshots and
  /// blocks SIGINT and SIGTERM on the caller, verifies the controlled thread boundary, allocates
  /// fixed domain state and descriptors, and starts one broker thread. A competing domain reports
  /// `kResourceExhausted`; an unprepared multithreaded process reports `kInvalidArgument`; setup
  /// failures report `kFailure` or `kResourceExhausted`. Non-Linux platforms report
  /// `kNotSupported`. No session is created on failure.
  /// Setup is transactional: every safely acquired resource is retired and the exact creator mask
  /// and unchanged dispositions are preserved on failure. A rollback failure poisons the process
  /// domain so later managed admission fails instead of assuming clean ownership.
  [[nodiscard]] static expected<ManagedShutdownSignals, Error> create();

  /// @brief Close this domain and restore the creator thread's exact prior signal mask.
  /// @return Success, or a bounded wrong-thread, active-binding, or restoration error.
  /// @note Call this on the creator thread after the bound session and every participant thread
  /// have retired. An active binding reports `kNotReady` without consuming the domain and may be
  /// retried after `RunSession::wait()`. A successful close consumes the handle; a repeated close
  /// or close on a moved-from object reports `kClosed`. Wrong-thread use reports
  /// `kInvalidArgument`, post-fork use reports `kNotSupported`, disposition drift reports
  /// `kIncompatible`, and cleanup failures report `kFailure` and may poison later managed
  /// admission. The call synchronizes with and joins the broker and therefore is not a real-time or
  /// bounded-latency operation.
  /// After the broker retires, cleanup attempts continue after a primary join, descriptor,
  /// disposition, or mask error, including a best-effort exact creator-mask restoration. An error
  /// that prevents safe broker retirement instead retains a poisoned domain rather than racing its
  /// live state.
  [[nodiscard]] expected<void, Error> close() noexcept;

  /// @brief Transfer the domain owner without changing creator-thread affinity.
  /// @param other Domain handle to consume. The moved-from handle becomes closed-like and its
  /// destructor is a no-op.
  /// @note Move only on the creator thread; moving does not transfer thread affinity or permit
  /// concurrent access.
  ManagedShutdownSignals(ManagedShutdownSignals&&) noexcept;

  /// @brief Disable copying of unique process-domain ownership.
  ManagedShutdownSignals(const ManagedShutdownSignals&) = delete;

  /// @brief Disable replacement of an existing process-domain owner.
  ManagedShutdownSignals& operator=(ManagedShutdownSignals&&) = delete;

  /// @brief Disable copying of unique process-domain ownership.
  ManagedShutdownSignals& operator=(const ManagedShutdownSignals&) = delete;

  /// @brief Attempt best-effort containment for a domain not explicitly closed.
  /// @note Prefer `close()` so failures remain observable. On the creator thread, destruction may
  /// request cooperative stop for an active binding and wait without a whole-run bound before
  /// closing. Destruction on another thread or irreparable cleanup poisons later managed admission;
  /// it never throws.
  ~ManagedShutdownSignals() noexcept;

 private:
  class Impl;

  explicit ManagedShutdownSignals(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class detail::ManagedShutdownSignalsAccess;
};

}  // namespace holoscan
