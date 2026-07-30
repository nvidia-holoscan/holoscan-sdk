/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <memory>

#include "gxf_network_context_pydoc.hpp"

#include <holoscan/core/gxf/gxf_component.hpp>
#include <holoscan/core/gxf/gxf_network_context.hpp>

#include <gxf/core/gxf.h>

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
