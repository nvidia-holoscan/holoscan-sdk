/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <variant>

#include "../../core/component_util.hpp"
#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp>

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyPingTensorRxOp : public holoscan::ops::PingTensorRxOp {
 public:
  /* Inherit the constructors */
  using PingTensorRxOp::PingTensorRxOp;

  // Define a constructor that fully initializes the object.
  PyPingTensorRxOp(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                   const py::args& args, bool receive_as_tensormap = true,
                   const std::string& name = operator_default_name_v<ops::PingTensorRxOp>)
      : PingTensorRxOp(Arg{"receive_as_tensormap", receive_as_tensormap}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

/* The python module */

PYBIND11_MODULE(_ping_tensor_rx, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK PingTensorRxOp Python Bindings
        --------------------------------------------------
        .. currentmodule:: _ping_tensor_rx
    )pbdoc";

  py::class_<PingTensorRxOp, PyPingTensorRxOp, Operator, std::shared_ptr<PingTensorRxOp>>(
      m, "PingTensorRxOp", doc::PingTensorRxOp::doc_PingTensorRxOp)
      .def(
          py::init<std::variant<Fragment*, Subgraph*>, const py::args&, bool, const std::string&>(),
          "fragment"_a,
          "receive_as_tensormap"_a = true,
          "name"_a = std::string(operator_default_name_v<ops::PingTensorRxOp>),
          doc::PingTensorRxOp::doc_PingTensorRxOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
