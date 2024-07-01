/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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

class PyGXFCodeletOp : public GXFCodeletOp {
 public:
  /* Inherit the constructors */
  using GXFCodeletOp::GXFCodeletOp;

  // Define a constructor that fully initializes the object.
  PyGXFCodeletOp(py::object op, Fragment* fragment, const std::string& gxf_typename,
                 const py::args& args, const std::string& name, const py::kwargs& kwargs)
      : GXFCodeletOp(gxf_typename.c_str()) {
    py_op_ = op;
    py_initialize_ = py::getattr(op, "initialize");  // cache the initialize method

    add_positional_condition_and_resource_args(this, args);
    add_kwargs(this, kwargs);

    name_ = name;
    fragment_ = fragment;
  }

  void initialize() override {
    // Get the initialize method of the Python Operator class and call it
    py::gil_scoped_acquire scope_guard;

    // TODO(gbae): setup tracing
    // // set_py_tracing();

    // Call the initialize method of the Python Operator class
    py_initialize_.operator()();

    // Call the parent class's initialize method after invoking the Python Operator's initialize
    // method.
    GXFCodeletOp::initialize();
  }

 protected:
  py::object py_op_ = py::none();
  py::object py_initialize_ = py::none();
};

/* The python module */

PYBIND11_MODULE(_gxf_codelet, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _gxf_codelet
    )pbdoc";

  py::class_<GXFCodeletOp, PyGXFCodeletOp, Operator, std::shared_ptr<GXFCodeletOp>>(
      m, "GXFCodeletOp", doc::GXFCodeletOp::doc_GXFCodeletOp)
      .def(py::init<>())
      .def(py::init<py::object,
                    Fragment*,
                    const std::string&,
                    const py::args&,
                    const std::string&,
                    const py::kwargs&>(),
           "op"_a,
           "fragment"_a,
           "gxf_typename"_a,
           "name"_a = "gxf_codelet"s,
           doc::GXFCodeletOp::doc_GXFCodeletOp)
      .def_property_readonly(
          "gxf_typename", &GXFCodeletOp::gxf_typename, doc::GXFCodeletOp::doc_gxf_typename)
      .def("initialize", &GXFCodeletOp::initialize, doc::GXFCodeletOp::doc_initialize)
      .def("setup", &GXFCodeletOp::setup, doc::GXFCodeletOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
