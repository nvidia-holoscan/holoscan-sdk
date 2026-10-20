/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/holoscan_core_export.h>

namespace holoscan::detail {

/// Provide deterministic source-tree fault injection for artifact replacement tests.
///
/// This private header is not installed. Production callers cannot configure the
/// hooks, and the default path adds only relaxed atomic probes to cold emission.
class ArtifactEmitterTestAccess {
 public:
  /// Fail the next native replacement after its verified sibling is prepared.
  HOLOSCAN_CORE_EXPORT static void fail_next_replace() noexcept;

  /// Fail one replacement after the requested number of successful replacements.
  /// @param successful_replacements Number of replacements allowed before failure.
  HOLOSCAN_CORE_EXPORT static void fail_replace_after_successes(
      std::uint32_t successful_replacements) noexcept;

  /// Throw one injected filesystem exception at the public emitter boundary.
  HOLOSCAN_CORE_EXPORT static void throw_next_filesystem_error() noexcept;

  /// Append one sentinel byte to the next staged temporary before verification.
  HOLOSCAN_CORE_EXPORT static void oversize_next_temporary() noexcept;

  /// Return the fixed schema-v1 maximum artifact-manifest size.
  HOLOSCAN_CORE_EXPORT static std::size_t maximum_manifest_bytes();

  /// Clear a pending staged-temporary injection.
  HOLOSCAN_CORE_EXPORT static void reset_temporary_injection() noexcept;
};

}  // namespace holoscan::detail
