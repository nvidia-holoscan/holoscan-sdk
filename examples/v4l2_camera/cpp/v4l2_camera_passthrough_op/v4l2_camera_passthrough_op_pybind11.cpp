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

#include <memory>
#include <string>

#include "./v4l2_camera_passthrough_op_pydoc.hpp"

#include "holoscan/core/condition.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"

// Include the C++ operator header
#include "v4l2_camera_passthrough_op.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::ops {

// Utility function to handle positional condition and resource arguments
void add_positional_condition_and_resource_args(Operator* op, const py::args& args) {
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (py::isinstance<Condition>(*it)) {
      op->add_arg(it->cast<std::shared_ptr<Condition>>());
    } else if (py::isinstance<Resource>(*it)) {
      op->add_arg(it->cast<std::shared_ptr<Resource>>());
    } else {
      HOLOSCAN_LOG_WARN(
          "Unhandled positional argument detected (only Condition and Resource objects can be "
          "parsed positionally)");
    }
  }
}

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyV4L2CameraPassthroughOp : public V4L2CameraPassthroughOp {
 public:
  /* Inherit the constructors */
  using V4L2CameraPassthroughOp::V4L2CameraPassthroughOp;

  // Define a constructor that fully initializes the object.
  // This operator has no parameters, so we only need fragment and name
  PyV4L2CameraPassthroughOp(Fragment* fragment, const py::args& args,
                            const std::string& name = "v4l2_camera_passthrough")
      : V4L2CameraPassthroughOp() {
    // Handle any positional condition or resource arguments
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_v4l2_camera_passthrough, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK V4L2CameraPassthroughOp Python Bindings
        ----------------------------------------------------
        .. currentmodule:: _v4l2_camera_passthrough
    )pbdoc";

  py::class_<V4L2CameraPassthroughOp,
             PyV4L2CameraPassthroughOp,
             Operator,
             std::shared_ptr<V4L2CameraPassthroughOp>>(
      m, "V4L2CameraPassthroughOp", doc::V4L2CameraPassthroughOp::doc_V4L2CameraPassthroughOp)
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           "fragment"_a,
           "name"_a = "v4l2_camera_passthrough"s,
           doc::V4L2CameraPassthroughOp::doc_V4L2CameraPassthroughOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
