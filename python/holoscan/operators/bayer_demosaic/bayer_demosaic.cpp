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
#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"

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

class PyBayerDemosaicOp : public BayerDemosaicOp {
 public:
  /* Inherit the constructors */
  using BayerDemosaicOp::BayerDemosaicOp;

  // Define a constructor that fully initializes the object.
  PyBayerDemosaicOp(Fragment* fragment, const py::args& args,
                    std::shared_ptr<holoscan::Allocator> pool,
                    std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                    const std::string& in_tensor_name = "", const std::string& out_tensor_name = "",
                    int interpolation_mode = 0, int bayer_grid_pos = 2, bool generate_alpha = false,
                    int alpha_value = 255, const std::string& name = "bayer_demosaic")
      : BayerDemosaicOp(ArgList{Arg{"pool", pool},
                                Arg{"in_tensor_name", in_tensor_name},
                                Arg{"out_tensor_name", out_tensor_name},
                                Arg{"interpolation_mode", interpolation_mode},
                                Arg{"bayer_grid_pos", bayer_grid_pos},
                                Arg{"generate_alpha", generate_alpha},
                                Arg{"alpha_value", alpha_value}}) {
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

PYBIND11_MODULE(_bayer_demosaic, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK BayerDemosaicOp Python Bindings
        --------------------------------------------
        .. currentmodule:: _bayer_demosaic
    )pbdoc";

  py::class_<BayerDemosaicOp, PyBayerDemosaicOp, Operator, std::shared_ptr<BayerDemosaicOp>>(
      m, "BayerDemosaicOp", doc::BayerDemosaicOp::doc_BayerDemosaicOp)
      .def(py::init<>())
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<holoscan::Allocator>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&,
                    const std::string&,
                    int,
                    int,
                    bool,
                    int,
                    const std::string&>(),
           "fragment"_a,
           "pool"_a,
           "cuda_stream_pool"_a = py::none(),
           "in_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "interpolation_mode"_a = 0,
           "bayer_grid_pos"_a = 2,
           "generate_alpha"_a = false,
           "alpha_value"_a = 255,
           "name"_a = "bayer_demosaic"s,
           doc::BayerDemosaicOp::doc_BayerDemosaicOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
