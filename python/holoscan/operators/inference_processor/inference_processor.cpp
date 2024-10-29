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

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/inference_processor/inference_processor.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::ops {

InferenceProcessorOp::DataMap _dict_to_processor_datamap(const py::dict& dict) {
  InferenceProcessorOp::DataMap data_map;
  for (const auto& [key, value] : dict) {
    data_map.insert(key.cast<std::string>(), value.cast<std::string>());
  }
  return data_map;
}

InferenceProcessorOp::DataVecMap _dict_to_processor_datavecmap(const py::dict& dict) {
  InferenceProcessorOp::DataVecMap data_vec_map;
  for (const auto& [key, value] : dict) {
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

class PyInferenceProcessorOp : public InferenceProcessorOp {
 public:
  /* Inherit the constructors */
  using InferenceProcessorOp::InferenceProcessorOp;

  // Define a constructor that fully initializes the object.
  PyInferenceProcessorOp(Fragment* fragment, const py::args& args,
                         std::shared_ptr<::holoscan::Allocator> allocator,
                         const py::dict& process_operations,  // InferenceProcessorOp::DataVecMap
                         const py::dict& processed_map,       // InferenceProcessorOp::DataVecMap
                         const std::vector<std::string>& in_tensor_names,
                         const std::vector<std::string>& out_tensor_names,
                         bool input_on_cuda = false, bool output_on_cuda = false,
                         bool transmit_on_cuda = false, bool disable_transmitter = false,
                         std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                         const std::string& config_path = ""s,
                         const std::string& name = "postprocessor"s)
      : InferenceProcessorOp(ArgList{Arg{"allocator", allocator},
                                     Arg{"in_tensor_names", in_tensor_names},
                                     Arg{"out_tensor_names", out_tensor_names},
                                     Arg{"input_on_cuda", input_on_cuda},
                                     Arg{"output_on_cuda", output_on_cuda},
                                     Arg{"transmit_on_cuda", transmit_on_cuda},
                                     Arg{"config_path", config_path},
                                     Arg{"disable_transmitter", disable_transmitter}}) {
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;

    // convert from Python dict to InferenceProcessorOp::DataVecMap
    auto process_operations_datavecmap =
        _dict_to_processor_datavecmap(process_operations.cast<py::dict>());
    this->add_arg(Arg("process_operations", process_operations_datavecmap));

    // Workaround to maintain backwards compatibility with the v0.5 API:
    // convert any single str values to List[str].
    auto processed_map_dict = processed_map.cast<py::dict>();
    for (const auto& [key, value] : processed_map_dict) {
      if (py::isinstance<py::str>(value)) {
        // warn about deprecated non-list input
        auto key_str = key.cast<std::string>();
        HOLOSCAN_LOG_WARN("Values for tensor {} not in list form.", key_str);
        HOLOSCAN_LOG_INFO(
            "HoloInfer in Holoscan SDK 0.6 onwards expects a list of tensor names "
            "in the parameter set.");
        HOLOSCAN_LOG_INFO(
            "Converting processed tensor mappings for tensor {} to list form for backward "
            "compatibility.",
            key_str);

        py::list value_list;
        value_list.append(value);
        processed_map_dict[key] = value_list;
      }
    }

    // convert from Python dict to InferenceProcessorOp::DataVecMap
    auto processed_map_datavecmap = _dict_to_processor_datavecmap(processed_map_dict);
    this->add_arg(Arg("processed_map", processed_map_datavecmap));

    spec_ = std::make_shared<OperatorSpec>(fragment);

    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_inference_processor, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK InferenceProcessorOp Python Bindings
        -------------------------------------------------
        .. currentmodule:: _inference_processor
    )pbdoc";

  py::class_<InferenceProcessorOp,
             PyInferenceProcessorOp,
             Operator,
             std::shared_ptr<InferenceProcessorOp>>
      inference_processor_op(
          m, "InferenceProcessorOp", doc::InferenceProcessorOp::doc_InferenceProcessorOp);

  inference_processor_op.def(py::init<Fragment*,
                                      const py::args&,
                                      std::shared_ptr<::holoscan::Allocator>,
                                      py::dict,
                                      py::dict,
                                      const std::vector<std::string>&,
                                      const std::vector<std::string>&,
                                      bool,
                                      bool,
                                      bool,
                                      bool,
                                      std::shared_ptr<holoscan::CudaStreamPool>,
                                      const std::string&,
                                      const std::string&>(),
                             "fragment"_a,
                             "allocator"_a,
                             "process_operations"_a = py::dict(),
                             "processed_map"_a = py::dict(),
                             "in_tensor_names"_a = std::vector<std::string>{},
                             "out_tensor_names"_a = std::vector<std::string>{},
                             "input_on_cuda"_a = false,
                             "output_on_cuda"_a = false,
                             "transmit_on_cuda"_a = false,
                             "disable_transmitter"_a = false,
                             "cuda_stream_pool"_a = py::none(),
                             "config_path"_a = ""s,
                             "name"_a = "postprocessor"s,
                             doc::InferenceProcessorOp::doc_InferenceProcessorOp);

  py::class_<InferenceProcessorOp::DataMap>(inference_processor_op, "DataMap")
      .def(py::init<>())
      .def("insert", &InferenceProcessorOp::DataMap::get_map)
      .def("get_map", &InferenceProcessorOp::DataMap::get_map);

  py::class_<InferenceProcessorOp::DataVecMap>(inference_processor_op, "DataVecMap")
      .def(py::init<>())
      .def("insert", &InferenceProcessorOp::DataVecMap::get_map)
      .def("get_map", &InferenceProcessorOp::DataVecMap::get_map);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
