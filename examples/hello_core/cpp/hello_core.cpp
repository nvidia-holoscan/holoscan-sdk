/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for installed-SDK consumers: exercises the public expected<T, Error> headers and the
// compiled error_code_name() ABI. If find_package(holoscan) cannot resolve the public interface
// targets, headers, or core library, this example fails against the installed tree.

#include <iostream>
#include <string>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace {

holoscan::expected<int, holoscan::Error> checked_value(bool ok) {
  if (!ok) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "value was rejected"});
  }
  return 42;
}

}  // namespace

int main() {
  const auto good = checked_value(true);
  const auto bad = checked_value(false);

  if (!good.has_value() || bad.has_value()) {
    std::cerr << "holoscan core sample FAILED: unexpected expected<> state\n";
    return 1;
  }

  // error_code_name() verifies the linked core error-name implementation through its public ABI.
  const std::string_view code_name = holoscan::error_code_name(bad.error().code);
  if (code_name != "kInvalidArgument") {
    std::cerr << "holoscan core sample FAILED: error_code_name was '" << code_name << "'\n";
    return 1;
  }

  std::cout << "value=" << good.value() << ", error=" << code_name << ": " << bad.error().message
            << "\n";
  std::cout << "holoscan core sample complete\n";
  return 0;
}
