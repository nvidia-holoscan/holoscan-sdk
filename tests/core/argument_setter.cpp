/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <complex>
#include <cstdint>
#include <string>
#include <typeinfo>

// clang-format off
#include "holoscan/core/component_spec.hpp"  // must be before argument_setter import
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/arg.hpp"
// clang-format on

namespace holoscan {

TEST(ArgumentSetter, TestArgumentSetterInstance) {
  ArgumentSetter instance = ArgumentSetter::get_instance();

  // call static ensure_type method
  ArgumentSetter::ensure_type<float>;  // NOLINT(clang-diagnostic-unused-value)

  // get the setter corresponding to float
  float f = 1.0;
  auto func = instance.get_argument_setter(typeid(f));
}

TEST(ArgumentSetter, SetParamThrowsOnBadAnyCast) {
  // Prepare an int parameter
  Parameter<int32_t> param_int;
  ParameterWrapper param_wrap(param_int);

  // Provide a mismatched native type (std::string) to trigger bad_any_cast path
  Arg arg("beta");
  arg = std::string("not_an_int");

  // Expect ArgumentSetter::set_param to throw with a descriptive message
  EXPECT_THROW(
      {
        try {
          ArgumentSetter::set_param(param_wrap, arg);
        } catch (const std::runtime_error& e) {
          std::string msg(e.what());
          EXPECT_NE(msg.find("Failed to set parameter 'beta'"), std::string::npos);
          EXPECT_NE(msg.find("arg_type"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

}  // namespace holoscan
