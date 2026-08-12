/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compile_platform.hpp"

#if defined(__linux__) && __has_include(<features.h>)
#include <features.h>
#endif

namespace holoscan::detail {

std::string_view compile_target_triple_for(CompileTargetArchitecture architecture,
                                           CompileTargetPlatform platform) noexcept {
  switch (architecture) {
    case CompileTargetArchitecture::kX86_64:
      switch (platform) {
        case CompileTargetPlatform::kWindowsMsvc:
          return "x86_64-pc-windows-msvc";
        case CompileTargetPlatform::kAppleDarwin:
          return "x86_64-apple-darwin";
        case CompileTargetPlatform::kLinuxGnu:
          return "x86_64-unknown-linux-gnu";
        case CompileTargetPlatform::kLinuxUnknown:
          return "x86_64-unknown-linux-unknown";
        case CompileTargetPlatform::kUnknown:
          return "x86_64-unknown-unknown";
      }
      break;
    case CompileTargetArchitecture::kAArch64:
      switch (platform) {
        case CompileTargetPlatform::kWindowsMsvc:
          return "aarch64-pc-windows-msvc";
        case CompileTargetPlatform::kAppleDarwin:
          return "aarch64-apple-darwin";
        case CompileTargetPlatform::kLinuxGnu:
          return "aarch64-unknown-linux-gnu";
        case CompileTargetPlatform::kLinuxUnknown:
          return "aarch64-unknown-linux-unknown";
        case CompileTargetPlatform::kUnknown:
          return "aarch64-unknown-unknown";
      }
      break;
    case CompileTargetArchitecture::kUnknown:
      break;
  }
  return "unknown-unknown-unknown";
}

std::string_view compile_target_triple() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  constexpr auto architecture = CompileTargetArchitecture::kX86_64;
#elif defined(_M_ARM64) || defined(__aarch64__)
  constexpr auto architecture = CompileTargetArchitecture::kAArch64;
#else
  constexpr auto architecture = CompileTargetArchitecture::kUnknown;
#endif

#if defined(_WIN32)
  constexpr auto platform = CompileTargetPlatform::kWindowsMsvc;
#elif defined(__APPLE__)
  constexpr auto platform = CompileTargetPlatform::kAppleDarwin;
#elif defined(__linux__) && defined(__GLIBC__)
  constexpr auto platform = CompileTargetPlatform::kLinuxGnu;
#elif defined(__linux__)
  constexpr auto platform = CompileTargetPlatform::kLinuxUnknown;
#else
  constexpr auto platform = CompileTargetPlatform::kUnknown;
#endif
  return compile_target_triple_for(architecture, platform);
}

}  // namespace holoscan::detail
