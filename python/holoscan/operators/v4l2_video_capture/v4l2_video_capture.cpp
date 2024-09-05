/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>  // for std::optional support

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp"
// #include "holoscan/core/gxf/gxf_operator.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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

class PyV4L2VideoCaptureOp : public V4L2VideoCaptureOp {
 public:
  /* Inherit the constructors */
  using V4L2VideoCaptureOp::V4L2VideoCaptureOp;

  // Define a constructor that fully initializes the object.
  PyV4L2VideoCaptureOp(Fragment* fragment, const py::args& args,
                       std::shared_ptr<::holoscan::Allocator> allocator,
                       const std::string& device = "/dev/video0"s, uint32_t width = 0,
                       uint32_t height = 0, uint32_t num_buffers = 4,
                       const std::string& pixel_format = "auto",
                       bool pass_through = false,
                       const std::string& name = "v4l2_video_capture",
                       std::optional<uint32_t> exposure_time = std::nullopt,
                       std::optional<uint32_t> gain = std::nullopt)
      : V4L2VideoCaptureOp(ArgList{Arg{"allocator", allocator},
                                   Arg{"device", device},
                                   Arg{"width", width},
                                   Arg{"height", height},
                                   Arg{"numBuffers", num_buffers},
                                   Arg{"pixel_format", pixel_format},
                                   Arg{"pass_through", pass_through}}) {
    if (exposure_time.has_value()) { this->add_arg(Arg{"exposure_time", exposure_time.value()}); }
    if (gain.has_value()) { this->add_arg(Arg{"gain", gain.value()}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_v4l2_video_capture, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK V4L2VideoCaptureOp Python Bindings
        -----------------------------------------------
        .. currentmodule:: _v4l2_video_capture
    )pbdoc";

  py::class_<V4L2VideoCaptureOp,
             PyV4L2VideoCaptureOp,
             Operator,
             std::shared_ptr<V4L2VideoCaptureOp>>(
      m, "V4L2VideoCaptureOp", doc::V4L2VideoCaptureOp::doc_V4L2VideoCaptureOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    bool,
                    const std::string&,
                    std::optional<uint32_t>,
                    std::optional<uint32_t>>(),
           "fragment"_a,
           "allocator"_a,
           "device"_a = "0"s,
           "width"_a = 0,
           "height"_a = 0,
           "num_buffers"_a = 4,
           "pixel_format"_a = "auto"s,
           "pass_through"_a = false,
           "name"_a = "v4l2_video_capture"s,
           "exposure_time"_a = py::none(),
           "gain"_a = py::none(),
           doc::V4L2VideoCaptureOp::doc_V4L2VideoCaptureOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
