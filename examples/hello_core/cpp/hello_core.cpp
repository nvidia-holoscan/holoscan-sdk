/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for installed-SDK consumers: exercises the holoscan::core expected<T, Error> API,
// which pulls in the header-only PUBLIC dependencies tl::expected (expected.hpp) and magic_enum
// (errors.hpp / error_code_name). If find_package(holoscan) cannot resolve those interface targets,
// or their headers are not installed, this example fails to configure/compile against the install
// tree -- which is exactly what we want CI to catch.

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

  // error_code_name() is backed by magic_enum.
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
