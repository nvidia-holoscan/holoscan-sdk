/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Helpers shared by the core examples; example code only, not part of the SDK API.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace holoviz::examples {

/// Return the path of the running executable. Prefer the OS-level query so the result is
/// correct when the program is invoked through PATH and \p argv0 has no directory component;
/// fall back to resolving \p argv0 against the current working directory.
inline std::filesystem::path current_executable_path(const char* argv0) {
#ifdef _WIN32
  wchar_t module_path[MAX_PATH]{};
  const DWORD length = GetModuleFileNameW(nullptr, module_path, MAX_PATH);
  if (length > 0 && length < MAX_PATH) {
    return std::filesystem::path(module_path);
  }
#else
  std::error_code read_link_error;
  const std::filesystem::path exe_path =
      std::filesystem::read_symlink("/proc/self/exe", read_link_error);
  if (!read_link_error) {
    return exe_path;
  }
#endif
  std::error_code canonical_error;
  const std::filesystem::path canonical_path = std::filesystem::canonical(argv0, canonical_error);
  if (!canonical_error) {
    return canonical_path;
  }
  return std::filesystem::path(argv0);
}

/// Result of parse_resource_path_args(): either the resolved resource path, or the exit code
/// the program should return immediately (after `--help`, `--print-resource-path`, or a usage
/// error).
struct ResourcePathArgs {
  std::string res_path;
  std::optional<int> exit_code;
};

/// Parse the command line shared by the resource-path examples: one optional positional
/// resource path plus `--help`/`-h` and `--print-resource-path`. When no path is given,
/// default to the \p assets_dir_name directory next to the running executable.
inline ResourcePathArgs parse_resource_path_args(int argc, char** argv,
                                                 const char* assets_dir_name) {
  const auto print_usage = [&](std::ostream& out) {
    out << "Usage: " << argv[0] << " [options] [<resource path>]\n"
        << "  <resource path> must contain nv_logo.png and shader.slang; defaults to the\n"
        << "  build-tree asset directory or the installed example directory\n"
        << "Options:\n"
        << "  -h, --help             print this message and exit\n"
        << "  --print-resource-path  resolve and check the resource path, print it and exit\n";
  };

  bool print_resource_path = false;
  std::string res_path;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout);
      return {.exit_code = EXIT_SUCCESS};
    }
    if (arg == "--print-resource-path") {
      print_resource_path = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::cerr << "Unknown option: " << arg << "\n";
      print_usage(std::cerr);
      return {.exit_code = EXIT_FAILURE};
    } else if (res_path.empty()) {
      res_path = std::string(arg);
    } else {
      print_usage(std::cerr);
      return {.exit_code = EXIT_FAILURE};
    }
  }

  if (res_path.empty()) {
    res_path = (current_executable_path(argv[0]).parent_path() / assets_dir_name).string();
  }

  if (print_resource_path) {
    const std::filesystem::path assets_path{res_path};
    for (const char* file_name : {"nv_logo.png", "shader.slang"}) {
      std::error_code exists_error;
      if (!std::filesystem::exists(assets_path / file_name, exists_error)) {
        std::cerr << "Missing " << (assets_path / file_name).string() << "\n";
        return {.exit_code = EXIT_FAILURE};
      }
    }
    std::cout << std::filesystem::absolute(assets_path).string() << "\n";
    return {.exit_code = EXIT_SUCCESS};
  }

  return {.res_path = std::move(res_path)};
}

}  // namespace holoviz::examples
