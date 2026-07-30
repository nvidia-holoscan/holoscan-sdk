/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string remove_ignored_errors(const std::string& captured_error) {
  HOLOSCAN_LOG_INFO("original_capure: {}", captured_error);
  std::vector<std::string> err_lines;
  std::string error_string = captured_error;
  size_t pos = 0;
  std::string delimiter = "\n";

  while ((pos = error_string.find(delimiter)) != std::string::npos) {
    std::string line = error_string.substr(0, pos);
    err_lines.push_back(std::move(line));
    error_string.erase(0, pos + delimiter.length());
  }

  std::vector<std::string> errors_to_ignore = {
      // some versions of the UCX extension print this error during application shutdown
      "Connection dropped with status -25"};

  for (const std::string& err : errors_to_ignore) {
    err_lines.erase(std::remove_if(err_lines.begin(),
                                   err_lines.end(),
                                   [&](const std::string& line) {
                                     return line.find(err) != std::string::npos;
                                   }),
                    err_lines.end());
  }

  std::string result;
  for (const std::string& line : err_lines) {
    result += line + "\n";
  }

  return result;
}

}  // namespace
