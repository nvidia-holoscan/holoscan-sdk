/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>  // for std::optional

#include <memory>
#include <string>
#include <variant>

#include "../../core/component_util.hpp"
#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/operators/test_ops/rx_dtype_test.hpp"
#include "holoscan/operators/test_ops/tx_dtype_test.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyDataTypeTxTestOp : public holoscan::ops::DataTypeTxTestOp {
 public:
  /* Inherit the constructors */
  using DataTypeTxTestOp::DataTypeTxTestOp;

  // Define a constructor that fully initializes the object.
  PyDataTypeTxTestOp(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                     const py::args& args, const std::string& data_type = "double"s,
                     const std::string& name = "data_type_tx_test_op"s)
      : DataTypeTxTestOp(ArgList{Arg{"data_type", data_type}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

class PyDataTypeRxTestOp : public holoscan::ops::DataTypeRxTestOp {
 public:
  /* Inherit the constructors */
  using DataTypeRxTestOp::DataTypeRxTestOp;

  // Define a constructor that fully initializes the object.
  PyDataTypeRxTestOp(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                     const py::args& args, const std::string& name = "data_type_rx_test_op"s)
      : DataTypeRxTestOp() {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

/* The python module */

PYBIND11_MODULE(_test_ops, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Test Operators Python Bindings
        --------------------------------------------------
        .. currentmodule:: _test_ops
    )pbdoc";

  py::class_<DataTypeTxTestOp, PyDataTypeTxTestOp, Operator, std::shared_ptr<DataTypeTxTestOp>>(
      m, "DataTypeTxTestOp", doc::DataTypeTxTestOp::doc_DataTypeTxTestOp)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    const py::args&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "data_type"_a = "double"s,
           "name"_a = "data_type_tx_test_op"s,
           doc::DataTypeTxTestOp::doc_DataTypeTxTestOp);

  py::class_<DataTypeRxTestOp, PyDataTypeRxTestOp, Operator, std::shared_ptr<DataTypeRxTestOp>>(
      m, "DataTypeRxTestOp", doc::DataTypeRxTestOp::doc_DataTypeRxTestOp)
      .def(py::init<std::variant<Fragment*, Subgraph*>, const py::args&, const std::string&>(),
           "fragment"_a,
           "name"_a = "data_type_rx_test_op"s,
           doc::DataTypeRxTestOp::doc_DataTypeRxTestOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
