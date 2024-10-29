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

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/operators/video_stream_recorder/video_stream_recorder.hpp"

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

class PyVideoStreamRecorderOp : public VideoStreamRecorderOp {
 public:
  /* Inherit the constructors */
  using VideoStreamRecorderOp::VideoStreamRecorderOp;

  // Define a constructor that fully initializes the object.
  PyVideoStreamRecorderOp(Fragment* fragment, const py::args& args, const std::string& directory,
                          const std::string& basename, bool flush_on_tick_ = false,
                          const std::string& name = "video_stream_recorder")
      : VideoStreamRecorderOp(ArgList{Arg{"directory", directory},
                                      Arg{"basename", basename},
                                      Arg{"flush_on_tick", flush_on_tick_}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_video_stream_recorder, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK VideoStreamRecorderOp Python Bindings
        --------------------------------------------------
        .. currentmodule:: _video_stream_recorder
    )pbdoc";

  py::class_<VideoStreamRecorderOp,
             PyVideoStreamRecorderOp,
             Operator,
             std::shared_ptr<VideoStreamRecorderOp>>(
      m, "VideoStreamRecorderOp", doc::VideoStreamRecorderOp::doc_VideoStreamRecorderOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "directory"_a,
           "basename"_a,
           "flush_on_tick"_a = false,
           "name"_a = "video_stream_recorder"s,
           doc::VideoStreamRecorderOp::doc_VideoStreamRecorderOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
