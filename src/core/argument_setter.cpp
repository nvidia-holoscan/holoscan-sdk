/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/argument_setter.hpp>

namespace holoscan {

ArgumentSetter& ArgumentSetter::get_instance() {
  static ArgumentSetter instance;
  return instance;
}

}  // namespace holoscan
