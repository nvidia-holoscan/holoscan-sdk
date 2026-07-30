/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <memory>

#include <holoscan/core/executor.hpp>
#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/core/flow_graphs/flow_graph.hpp>
#include <holoscan/core/fragment.hpp>
#include "./executors_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

PYBIND11_MODULE(_executors, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Executor Python Bindings
        -------------------------------------
        .. currentmodule:: _executors
    )pbdoc";

  py::class_<gxf::GXFExecutor, Executor, std::shared_ptr<gxf::GXFExecutor>>(
      m, "GXFExecutor", doc::GXFExecutor::doc_GXFExecutor)
      .def(py::init<Fragment*>(), "app"_a, doc::GXFExecutor::doc_GXFExecutor_app);
  // Note: context property and run method are inherited from Executor
}  // PYBIND11_MODULE
}  // namespace holoscan
