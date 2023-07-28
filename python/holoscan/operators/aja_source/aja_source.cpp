/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <string>

#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/aja_source/aja_source.hpp"

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

class PyAJASourceOp : public AJASourceOp {
 public:
  /* Inherit the constructors */
  using AJASourceOp::AJASourceOp;

  // Define a constructor that fully initializes the object.
  PyAJASourceOp(Fragment* fragment, const std::string& device = "0"s,
                NTV2Channel channel = NTV2Channel::NTV2_CHANNEL1, uint32_t width = 1920,
                uint32_t height = 1080, uint32_t framerate = 60, bool rdma = false,
                bool enable_overlay = false,
                NTV2Channel overlay_channel = NTV2Channel::NTV2_CHANNEL2, bool overlay_rdma = true,
                const std::string& name = "aja_source")
      : AJASourceOp(ArgList{Arg{"device", device},
                            Arg{"channel", channel},
                            Arg{"width", width},
                            Arg{"height", height},
                            Arg{"framerate", framerate},
                            Arg{"rdma", rdma},
                            Arg{"enable_overlay", enable_overlay},
                            Arg{"overlay_channel", overlay_channel},
                            Arg{"overlay_rdma", overlay_rdma}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_aja_source, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _aja_source
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::enum_<NTV2Channel>(m, "NTV2Channel")
      .value("NTV2_CHANNEL1", NTV2Channel::NTV2_CHANNEL1)
      .value("NTV2_CHANNEL2", NTV2Channel::NTV2_CHANNEL2)
      .value("NTV2_CHANNEL3", NTV2Channel::NTV2_CHANNEL3)
      .value("NTV2_CHANNEL4", NTV2Channel::NTV2_CHANNEL4)
      .value("NTV2_CHANNEL5", NTV2Channel::NTV2_CHANNEL5)
      .value("NTV2_CHANNEL6", NTV2Channel::NTV2_CHANNEL6)
      .value("NTV2_CHANNEL7", NTV2Channel::NTV2_CHANNEL7)
      .value("NTV2_CHANNEL8", NTV2Channel::NTV2_CHANNEL8)
      .value("NTV2_MAX_NUM_CHANNELS", NTV2Channel::NTV2_MAX_NUM_CHANNELS)
      .value("NTV2_CHANNEL_INVALID", NTV2Channel::NTV2_CHANNEL_INVALID);

  py::class_<AJASourceOp, PyAJASourceOp, Operator, std::shared_ptr<AJASourceOp>>(
      m, "AJASourceOp", doc::AJASourceOp::doc_AJASourceOp)
      .def(py::init<Fragment*,
                    const std::string&,
                    NTV2Channel,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    bool,
                    NTV2Channel,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "device"_a = "0"s,
           "channel"_a = NTV2Channel::NTV2_CHANNEL1,
           "width"_a = 1920,
           "height"_a = 1080,
           "framerate"_a = 60,
           "rdma"_a = false,
           "enable_overlay"_a = false,
           "overlay_channel"_a = NTV2Channel::NTV2_CHANNEL2,
           "overlay_rdma"_a = true,
           "name"_a = "aja_source"s,
           doc::AJASourceOp::doc_AJASourceOp_python)
      .def("initialize", &AJASourceOp::initialize, doc::AJASourceOp::doc_initialize)
      .def("setup", &AJASourceOp::setup, "spec"_a, doc::AJASourceOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
