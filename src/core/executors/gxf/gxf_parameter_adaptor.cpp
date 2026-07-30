/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/executors/gxf/gxf_parameter_adaptor.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace holoscan::gxf {

GXFParameterAdaptor& GXFParameterAdaptor::get_instance() {
  static GXFParameterAdaptor instance;
  return instance;
}

}  // namespace holoscan::gxf
