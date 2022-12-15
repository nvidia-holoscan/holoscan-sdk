/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// clang-format on

namespace holoscan {

TEST(ArgumentSetter, TestArgumentSetterInstance) {
  ArgumentSetter instance = ArgumentSetter::get_instance();

  ArgumentSetter::ensure_type<float>;
  // ArgumentSetter::ensure_type<std::complex<float>>;  // will fail to compile
}

}  // namespace holoscan
