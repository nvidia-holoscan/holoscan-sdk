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

#include <memory>
#include <string>

#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/video_stream_recorder/video_stream_recorder.hpp"

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

class PyVideoStreamRecorderOp : public VideoStreamRecorderOp {
 public:
  /* Inherit the constructors */
  using VideoStreamRecorderOp::VideoStreamRecorderOp;

  // Define a constructor that fully initializes the object.
  PyVideoStreamRecorderOp(Fragment* fragment, const std::string& directory,
                          const std::string& basename, bool flush_on_tick_ = false,
                          const std::string& name = "video_stream_recorder")
      : VideoStreamRecorderOp(ArgList{Arg{"directory", directory},
                                      Arg{"basename", basename},
                                      Arg{"flush_on_tick", flush_on_tick_}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_video_stream_recorder, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _video_stream_recorder
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

  py::class_<VideoStreamRecorderOp,
             PyVideoStreamRecorderOp,
             Operator,
             std::shared_ptr<VideoStreamRecorderOp>>(
      m, "VideoStreamRecorderOp", doc::VideoStreamRecorderOp::doc_VideoStreamRecorderOp)
      .def(py::init<Fragment*, const std::string&, const std::string&, bool, const std::string&>(),
           "fragment"_a,
           "directory"_a,
           "basename"_a,
           "flush_on_tick"_a = false,
           "name"_a = "recorder"s,
           doc::VideoStreamRecorderOp::doc_VideoStreamRecorderOp)
      .def("initialize",
           &VideoStreamRecorderOp::initialize,
           doc::VideoStreamRecorderOp::doc_initialize)
      .def("setup", &VideoStreamRecorderOp::setup, "spec"_a, doc::VideoStreamRecorderOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
