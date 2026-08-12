/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace holoscan::detail {

/// Classify the compile-target instruction-set identity without compiler text.
enum class CompileTargetArchitecture : std::uint8_t {
  kX86_64,
  kAArch64,
  kUnknown,
};

/// Classify the compile-target operating-system and userspace ABI.
enum class CompileTargetPlatform : std::uint8_t {
  kWindowsMsvc,
  kAppleDarwin,
  kLinuxGnu,
  kLinuxUnknown,
  kUnknown,
};

/// Return the canonical triple for an already classified target.
///
/// This private pure helper supports cross-platform conformance without
/// pretending that one host can redefine its compiler's predefined macros.
[[nodiscard]] std::string_view compile_target_triple_for(CompileTargetArchitecture architecture,
                                                         CompileTargetPlatform platform) noexcept;

/// Return the canonical target triple for the current Core binary.
///
/// The value describes the compile target, not the host discovered at runtime.
/// Unknown architecture/OS combinations fail closed to an explicit token so
/// plan provenance never claims a more specific target than the build proves.
[[nodiscard]] std::string_view compile_target_triple() noexcept;

}  // namespace holoscan::detail
