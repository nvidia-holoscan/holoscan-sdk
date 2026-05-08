/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/raw_image_processor/raw_image_processor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "../../core/component_util.hpp"
#include "../operator_util.hpp"

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/subgraph.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

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
class PyRawImageProcessorOp : public RawImageProcessorOp {
 public:
  /* Inherit the constructors */
  using RawImageProcessorOp::RawImageProcessorOp;

  // Define a constructor that fully initializes the object.
  PyRawImageProcessorOp(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                        const py::args& args, int pixel_format, int bayer_format,
                        int32_t optical_black = 0, int cuda_device_ordinal = 0,
                        const std::string& name = operator_default_name_v<ops::RawImageProcessorOp>)
      : RawImageProcessorOp(ArgList{Arg{"pixel_format", pixel_format},
                                    Arg{"bayer_format", bayer_format},
                                    Arg{"optical_black", optical_black},
                                    Arg{"cuda_device_ordinal", cuda_device_ordinal}}) {
    add_positional_condition_and_resource_args(this, args);
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

PYBIND11_MODULE(_raw_image_processor, m) {
  py::class_<RawImageProcessorOp,
             PyRawImageProcessorOp,
             Operator,
             std::shared_ptr<RawImageProcessorOp>>(m, "RawImageProcessorOp")
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    const py::args&,
                    int,
                    int,
                    int32_t,
                    int,
                    const std::string&>(),
           "fragment"_a,
           "pixel_format"_a,
           "bayer_format"_a,
           "optical_black"_a = 0,
           "cuda_device_ordinal"_a = 0,
           "name"_a = std::string(operator_default_name_v<ops::RawImageProcessorOp>))
      .def("setup", &RawImageProcessorOp::setup, "spec"_a);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
