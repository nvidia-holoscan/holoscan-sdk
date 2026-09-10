/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

#include <holoscan/core/run_session.hpp>

#include "runtime_diagnostic_sink.hpp"

namespace holoscan::detail {

/// Source-private callbacks exposed to a pre-freeze terminal participant.
struct RuntimeTerminalParticipantControl {
  void* state{};
  void (*request_stop)(void*) noexcept {};
  void (*append_managed_diagnostic)(void*, std::string_view, const RuntimeManagedSignalDiagnostic&,
                                    bool) noexcept {};
  void (*finish_start_failure)(void*) noexcept {};
};

/// Coordinate one private cleanup owner immediately before terminal-state publication.
class RuntimeTerminalParticipant {
 public:
  RuntimeTerminalParticipant() = default;
  RuntimeTerminalParticipant(const RuntimeTerminalParticipant&) = delete;
  RuntimeTerminalParticipant& operator=(const RuntimeTerminalParticipant&) = delete;
  virtual ~RuntimeTerminalParticipant() = default;

  virtual void attach(RuntimeTerminalParticipantControl control) noexcept = 0;

  /// Stage a worker-owned candidate and wait until external cleanup permits terminal publication.
  /// @return True when participant cleanup succeeded; false promotes the candidate to kFailed.
  [[nodiscard]] virtual bool before_terminal_freeze(RunTermination candidate) noexcept = 0;

  /// Observe that the terminal state was published after participant release.
  virtual void after_terminal_freeze() noexcept = 0;

  /// Wait until the external cleanup owner has retired this participant and released ownership.
  virtual void wait_for_retirement() noexcept = 0;

  /// Stage worker-construction failure without waiting on the invoking controller thread.
  virtual void stage_start_failure() noexcept = 0;

  /// Abandon a reserved binding when no RunSession state could be attached.
  virtual void abort_before_session() noexcept = 0;
};

}  // namespace holoscan::detail
