/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp"

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

class PySegmentationPostprocessorOp : public SegmentationPostprocessorOp {
 public:
  /* Inherit the constructors */
  using SegmentationPostprocessorOp::SegmentationPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PySegmentationPostprocessorOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<::holoscan::Allocator> allocator,
      const std::string& in_tensor_name = "", const std::string& network_output_type = "softmax"s,
      const std::string& data_format = "hwc"s,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "segmentation_postprocessor"s)
      : SegmentationPostprocessorOp(ArgList{Arg{"in_tensor_name", in_tensor_name},
                                            Arg{"network_output_type", network_output_type},
                                            Arg{"data_format", data_format},
                                            Arg{"allocator", allocator}}) {
    if (cuda_stream_pool) {
      this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_segmentation_postprocessor, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK SegmentationPostprocessorOp Bindings
        -------------------------------------------------
        .. currentmodule:: _segmentation_postprocessor
    )pbdoc";

  py::class_<SegmentationPostprocessorOp,
             PySegmentationPostprocessorOp,
             Operator,
             std::shared_ptr<SegmentationPostprocessorOp>>(
      m,
      "SegmentationPostprocessorOp",
      doc::SegmentationPostprocessorOp::doc_SegmentationPostprocessorOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_name"_a = ""s,
           "network_output_type"_a = "softmax"s,
           "data_format"_a = "hwc"s,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "segmentation_postprocessor"s,
           doc::SegmentationPostprocessorOp::doc_SegmentationPostprocessorOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
