/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_TESTS_UTILS_HOLOINFER_BACKEND_TEST_UTILS_HPP
#define HOLOSCAN_TESTS_UTILS_HOLOINFER_BACKEND_TEST_UTILS_HPP

#include <gtest/gtest.h>

#include <cstdlib>
#include <string_view>

namespace holoscan::tests {

inline bool env_var_enabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (value == nullptr) {
    return false;
  }
  const std::string_view env_value(value);
  return !env_value.empty() && env_value != "0" && env_value != "false" && env_value != "FALSE" &&
         env_value != "off" && env_value != "OFF";
}

}  // namespace holoscan::tests

#if defined(HOLOINFER_ORT_ENABLED)
// Missing ONNX Runtime is a failure by default. Package/container jobs that intentionally omit
// it can opt out explicitly with HOLOSCAN_TEST_SKIP_ONNX_RUNTIME.
#define HOLOSCAN_TEST_SKIP_IF_ONNX_RUNTIME_BACKEND_DISABLED()                    \
  do {                                                                           \
    if (::holoscan::tests::env_var_enabled("HOLOSCAN_TEST_SKIP_ONNX_RUNTIME")) { \
      GTEST_SKIP() << "ONNX Runtime backend tests disabled by "                  \
                      "HOLOSCAN_TEST_SKIP_ONNX_RUNTIME";                         \
    }                                                                            \
  } while (0)
#else
#define HOLOSCAN_TEST_SKIP_IF_ONNX_RUNTIME_BACKEND_DISABLED() \
  do {                                                        \
    GTEST_SKIP() << "ONNX Runtime backend not enabled";       \
  } while (0)
#endif

#endif /* HOLOSCAN_TESTS_UTILS_HOLOINFER_BACKEND_TEST_UTILS_HPP */
