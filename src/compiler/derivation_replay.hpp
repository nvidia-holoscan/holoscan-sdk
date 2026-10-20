/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../core/diagnostic_context.hpp"
#include "../core/operator_spec_compiler_access.hpp"

namespace holoscan::detail {

/// Keep borrowed setup declarations alive until the owning derivation pass consumes them.
struct DerivationReplay {
  std::unique_ptr<OperatorBase> instance;
  std::unique_ptr<OperatorSpec> spec;
};

enum class DerivationReplayOutcome : std::uint8_t {
  kReady,
  kConstructionFailed,
  kSetupThrew,
  kDeclarationFailed,
};

/// Reconstruct one Operator without caching or combining independent derivation passes.
/// Callers retain their feature-specific structural tokens and skip/fatal decisions.
[[nodiscard]] inline DerivationReplayOutcome replay_derivation(const OperatorFactoryBase& factory,
                                                               DerivationReplay& replay,
                                                               DiagnosticCapture& capture) {
  capture.clear();
  auto created = factory.create(&capture);
  if (!created || !created.value()) {
    if (capture.count == 0U) {
      if (created) {
        capture.append(ErrorCode::kFailure, "factory returned no operator");
      } else {
        capture.append(created.error().code, created.error().message);
      }
    }
    return DerivationReplayOutcome::kConstructionFailed;
  }
  // Preserve the pre-existing allocation boundary: an exhausted OperatorSpec allocation
  // propagates before setup entry, and is not attributed to a user callback.
  replay = DerivationReplay{.instance = std::move(created.value()),
                            .spec = std::make_unique<OperatorSpec>()};
  try {
    replay.instance->setup(*replay.spec);
  } catch (...) {
    capture.capture_exception();
    return DerivationReplayOutcome::kSetupThrew;
  }
  const Error& error = OperatorSpecCompilerAccess::declaration_error(*replay.spec);
  if (error.code != ErrorCode::kSuccess) {
    capture.append(error.code, error.message);
    return DerivationReplayOutcome::kDeclarationFailed;
  }
  return DerivationReplayOutcome::kReady;
}

}  // namespace holoscan::detail
