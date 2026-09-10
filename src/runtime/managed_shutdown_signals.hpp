/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string_view>

#include <holoscan/core/expected.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>

#include "runtime_diagnostic_sink.hpp"
#include "runtime_terminal_participant.hpp"

namespace holoscan::detail {

class RuntimeManagedSignalTestControl;

struct ManagedSignalAdmissionFailure {
  std::string_view token;
  RuntimeManagedSignalDiagnostic detail;
  Error error;
  std::string_view cleanup_token;
  RuntimeManagedSignalDiagnostic cleanup_detail;
  Error cleanup_error;
  bool cleanup_present{};
  bool poison_process{};
};

class HOLOSCAN_CORE_EXPORT ManagedShutdownSignalsAccess {
 public:
  [[nodiscard]] static ManagedShutdownSignals::Impl* get(ManagedShutdownSignals& signals) noexcept;
  [[nodiscard]] static expected<ManagedShutdownSignals, ManagedSignalAdmissionFailure>
  create_for_test(std::shared_ptr<RuntimeManagedSignalTestControl> test_control);
};

class ManagedSignalBindingController {
 public:
  ManagedSignalBindingController() = default;
  ManagedSignalBindingController(const ManagedSignalBindingController&) = delete;
  ManagedSignalBindingController& operator=(const ManagedSignalBindingController&) = delete;
  virtual ~ManagedSignalBindingController() = default;

  [[nodiscard]] virtual std::shared_ptr<RuntimeTerminalParticipant> terminal_participant() = 0;
  [[nodiscard]] virtual bool caller_drives() const noexcept = 0;
  virtual void drive() noexcept = 0;
  virtual void abort_before_session() noexcept = 0;
};

struct PreparedManagedSignalBinding {
  std::shared_ptr<ManagedSignalBindingController> controller;
};

/// Prepare default Linux managed admission, adopting a compatible public domain when present.
[[nodiscard]] expected<PreparedManagedSignalBinding, ManagedSignalAdmissionFailure>
prepare_default_managed_signal_binding(
    std::shared_ptr<RuntimeManagedSignalTestControl> test_control = {});

/// Reserve the single binding owned by an explicit public domain.
[[nodiscard]] expected<PreparedManagedSignalBinding, ManagedSignalAdmissionFailure>
prepare_explicit_managed_signal_binding(
    ManagedShutdownSignals& signals,
    std::shared_ptr<RuntimeManagedSignalTestControl> test_control = {});

}  // namespace holoscan::detail
