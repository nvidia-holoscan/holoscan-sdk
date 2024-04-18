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
#include <pybind11/stl.h>  // for dict & vector

#include <memory>
#include <string>
#include <vector>

#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/inference/inference.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

InferenceOp::DataMap _dict_to_inference_datamap(py::dict dict) {
  InferenceOp::DataMap data_map;
  for (auto& [key, value] : dict) {
    data_map.insert(key.cast<std::string>(), value.cast<std::string>());
  }
  return data_map;
}

InferenceOp::DataVecMap _dict_to_inference_datavecmap(py::dict dict) {
  InferenceOp::DataVecMap data_vec_map;
  for (auto& [key, value] : dict) {
    data_vec_map.insert(key.cast<std::string>(), value.cast<std::vector<std::string>>());
  }
  return data_vec_map;
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

class PyInferenceOp : public InferenceOp {
 public:
  /* Inherit the constructors */
  using InferenceOp::InferenceOp;

  // Define a constructor that fully initializes the object.
  PyInferenceOp(Fragment* fragment, const std::string& backend,
                std::shared_ptr<::holoscan::Allocator> allocator,
                py::dict inference_map,      // InferenceOp::DataVecMap
                py::dict model_path_map,     // InferenceOp::DataMap
                py::dict pre_processor_map,  // InferenceOp::DataVecMap
                py::dict device_map,         // InferenceOp::DataMap
                py::dict backend_map,        // InferenceOp::DataMap
                const std::vector<std::string>& in_tensor_names,
                const std::vector<std::string>& out_tensor_names, bool infer_on_cpu = false,
                bool parallel_inference = true, bool input_on_cuda = true,
                bool output_on_cuda = true, bool transmit_on_cuda = true, bool enable_fp16 = false,
                bool is_engine_path = false,
                std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                // TODO(grelee): handle receivers similarly to HolovizOp?  (default: {})
                // TODO(grelee): handle transmitter similarly to HolovizOp?
                const std::string& name = "inference")
      : InferenceOp(ArgList{Arg{"backend", backend},
                            Arg{"allocator", allocator},
                            Arg{"in_tensor_names", in_tensor_names},
                            Arg{"out_tensor_names", out_tensor_names},
                            Arg{"infer_on_cpu", infer_on_cpu},
                            Arg{"parallel_inference", parallel_inference},
                            Arg{"input_on_cuda", input_on_cuda},
                            Arg{"output_on_cuda", output_on_cuda},
                            Arg{"transmit_on_cuda", transmit_on_cuda},
                            Arg{"enable_fp16", enable_fp16},
                            Arg{"is_engine_path", is_engine_path}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    name_ = name;
    fragment_ = fragment;

    // Workaround to maintain backwards compatibility with the v0.5 API:
    // convert any single str values to List[str].
    py::dict inference_map_dict = inference_map.cast<py::dict>();
    for (auto& [key, value] : inference_map_dict) {
      if (py::isinstance<py::str>(value)) {
        // warn about deprecated non-list input
        auto key_str = key.cast<std::string>();
        HOLOSCAN_LOG_WARN("Values for model {} not in list form.", key_str);
        HOLOSCAN_LOG_INFO(
            "HoloInfer in Holoscan SDK 0.6 onwards expects a list of tensor names for models "
            "in the parameter set.");
        HOLOSCAN_LOG_INFO(
            "Converting input tensor names for model {} to list form for backward "
            "compatibility.",
            key_str);

        py::list value_list;
        value_list.append(value);
        inference_map_dict[key] = value_list;
      }
    }

    // convert from Python dict to InferenceOp::DataVecMap
    auto inference_map_datavecmap = _dict_to_inference_datavecmap(inference_map_dict);
    this->add_arg(Arg("inference_map", inference_map_datavecmap));

    auto model_path_datamap = _dict_to_inference_datamap(model_path_map.cast<py::dict>());
    this->add_arg(Arg("model_path_map", model_path_datamap));

    auto device_datamap = _dict_to_inference_datamap(device_map.cast<py::dict>());
    this->add_arg(Arg("device_map", device_datamap));

    auto backend_datamap = _dict_to_inference_datamap(backend_map.cast<py::dict>());
    this->add_arg(Arg("backend_map", backend_datamap));

    // convert from Python dict to InferenceOp::DataVecMap
    auto pre_processor_datamap = _dict_to_inference_datavecmap(pre_processor_map.cast<py::dict>());
    this->add_arg(Arg("pre_processor_map", pre_processor_datamap));

    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_inference, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _inference
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

  py::class_<InferenceOp, PyInferenceOp, Operator, std::shared_ptr<InferenceOp>> inference_op(
      m, "InferenceOp", doc::InferenceOp::doc_InferenceOp);

  inference_op
      .def(py::init<Fragment*,
                    const std::string&,
                    std::shared_ptr<::holoscan::Allocator>,
                    py::dict,
                    py::dict,
                    py::dict,
                    py::dict,
                    py::dict,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "backend"_a,
           "allocator"_a,
           "inference_map"_a,
           "model_path_map"_a,
           "pre_processor_map"_a,
           "device_map"_a = py::dict(),
           "backend_map"_a = py::dict(),
           "in_tensor_names"_a = std::vector<std::string>{},
           "out_tensor_names"_a = std::vector<std::string>{},
           "infer_on_cpu"_a = false,
           "parallel_inference"_a = true,
           "input_on_cuda"_a = true,
           "output_on_cuda"_a = true,
           "transmit_on_cuda"_a = true,
           "enable_fp16"_a = false,
           "is_engine_path"_a = false,
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "inference"s,
           doc::InferenceOp::doc_InferenceOp)
      .def("initialize", &InferenceOp::initialize, doc::InferenceOp::doc_initialize)
      .def("setup", &InferenceOp::setup, "spec"_a, doc::InferenceOp::doc_setup);

  py::class_<InferenceOp::DataMap>(inference_op, "DataMap")
      .def(py::init<>())
      .def("insert", &InferenceOp::DataMap::get_map)
      .def("get_map", &InferenceOp::DataMap::get_map);

  py::class_<InferenceOp::DataVecMap>(inference_op, "DataVecMap")
      .def(py::init<>())
      .def("insert", &InferenceOp::DataVecMap::get_map)
      .def("get_map", &InferenceOp::DataVecMap::get_map);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
