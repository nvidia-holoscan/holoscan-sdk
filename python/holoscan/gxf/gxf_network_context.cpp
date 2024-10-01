/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>

#include <memory>

#include "gxf_network_context_pydoc.hpp"

#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_network_context.hpp"

#include "gxf/core/gxf.h"

namespace py = pybind11;

namespace holoscan {

void init_gxf_network_context(py::module_& m) {
  py::class_<gxf::GXFNetworkContext,
             NetworkContext,
             gxf::GXFComponent,
             std::shared_ptr<gxf::GXFNetworkContext>>(
      m, "GXFNetworkContext", doc::GXFNetworkContext::doc_GXFNetworkContext);
}  // PYBIND11_MODULE

}  // namespace holoscan
