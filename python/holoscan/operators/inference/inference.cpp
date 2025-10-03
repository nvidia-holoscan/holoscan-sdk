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
#include <pybind11/stl.h>  // for dict & vector

#include <memory>
#include <string>
#include <vector>

#include "../../core/emitter_receiver_registry.hpp"  // EmitterReceiverRegistry
#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/codec_registry.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/inference/codecs.hpp"
#include "holoscan/operators/inference/inference.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;
namespace holoscan::ops {

InferenceOp::DataMap _dict_to_inference_datamap(const py::dict& dict) {
  InferenceOp::DataMap data_map;
  for (const auto& [key, value] : dict) {
    data_map.insert(key.cast<std::string>(), value.cast<std::string>());
  }
  return data_map;
}

InferenceOp::DataVecMap _dict_to_inference_datavecmap(const py::dict& dict) {
  InferenceOp::DataVecMap data_vec_map;
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

class PyInferenceOp : public InferenceOp {
 public:
  /* Inherit the constructors */
  using InferenceOp::InferenceOp;

  // Define a constructor that fully initializes the object.
  PyInferenceOp(Fragment* fragment, const py::args& args, const std::string& backend,
                std::shared_ptr<::holoscan::Allocator> allocator,
                const py::dict& inference_map,      // InferenceOp::DataVecMap
                const py::dict& model_path_map,     // InferenceOp::DataMap
                const py::dict& pre_processor_map,  // InferenceOp::DataVecMap
                const py::dict& device_map,         // InferenceOp::DataMap
                const py::dict& dla_core_map,       // InferenceOp::DataMap
                const py::dict& temporal_map,       // InferenceOp::DataMap
                const py::dict& activation_map,     // InferenceOp::DataMap
                const py::dict& backend_map,        // InferenceOp::DataMap
                const std::vector<std::string>& in_tensor_names,
                const std::vector<std::string>& out_tensor_names,
                const std::vector<std::vector<int32_t>>& trt_opt_profile, bool infer_on_cpu = false,
                bool parallel_inference = true, bool input_on_cuda = true,
                bool output_on_cuda = true, bool transmit_on_cuda = true,
                bool dynamic_input_dims = false, bool enable_fp16 = false,
                bool enable_cuda_graphs = true, int32_t dla_core = -1, bool dla_gpu_fallback = true,
                bool is_engine_path = false,
                std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
                // TODO(grelee): handle receivers similarly to HolovizOp?  (default: {})
                // TODO(grelee): handle transmitter similarly to HolovizOp?
                const std::string& name = "inference")
      : InferenceOp(ArgList{Arg{"backend", backend},
                            Arg{"allocator", allocator},
                            Arg{"in_tensor_names", in_tensor_names},
                            Arg{"out_tensor_names", out_tensor_names},
                            Arg{"trt_opt_profile", trt_opt_profile},
                            Arg{"infer_on_cpu", infer_on_cpu},
                            Arg{"parallel_inference", parallel_inference},
                            Arg{"input_on_cuda", input_on_cuda},
                            Arg{"output_on_cuda", output_on_cuda},
                            Arg{"transmit_on_cuda", transmit_on_cuda},
                            Arg{"dynamic_input_dims", dynamic_input_dims},
                            Arg{"enable_fp16", enable_fp16},
                            Arg{"enable_cuda_graphs", enable_cuda_graphs},
                            Arg{"dla_core", dla_core},
                            Arg{"dla_gpu_fallback", dla_gpu_fallback},
                            Arg{"is_engine_path", is_engine_path}}) {
    if (cuda_stream_pool) {
      this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;

    // Workaround to maintain backwards compatibility with the v0.5 API:
    // convert any single str values to List[str].
    auto inference_map_dict = inference_map.cast<py::dict>();
    for (const auto& [key, value] : inference_map_dict) {
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

    auto temporal_map_infer = temporal_map.cast<py::dict>();
    for (const auto& [key, value] : temporal_map_infer) {
      if (!py::isinstance<py::str>(value)) {
        temporal_map_infer[key] = py::str(value);
      }
    }

    auto activation_map_infer = activation_map.cast<py::dict>();
    for (const auto& [key, value] : activation_map_infer) {
      if (!py::isinstance<py::str>(value)) {
        activation_map_infer[key] = py::str(value);
      }
    }

    auto device_map_infer = device_map.cast<py::dict>();
    for (const auto& [key, value] : device_map_infer) {
      if (!py::isinstance<py::str>(value)) {
        device_map_infer[key] = py::str(value);
      }
    }

    auto dla_core_map_infer = dla_core_map.cast<py::dict>();
    for (const auto& [key, value] : dla_core_map_infer) {
      if (!py::isinstance<py::str>(value)) {
        dla_core_map_infer[key] = py::str(value);
      }
    }

    // convert from Python dict to InferenceOp::DataVecMap
    auto inference_map_datavecmap = _dict_to_inference_datavecmap(inference_map_dict);
    this->add_arg(Arg("inference_map", inference_map_datavecmap));

    auto model_path_datamap = _dict_to_inference_datamap(model_path_map.cast<py::dict>());
    this->add_arg(Arg("model_path_map", model_path_datamap));

    auto device_datamap = _dict_to_inference_datamap(device_map_infer);
    this->add_arg(Arg("device_map", device_datamap));

    auto dla_core_datamap = _dict_to_inference_datamap(dla_core_map_infer);
    this->add_arg(Arg("dla_core_map", dla_core_datamap));

    auto temporal_datamap = _dict_to_inference_datamap(temporal_map_infer);
    this->add_arg(Arg("temporal_map", temporal_datamap));

    auto activation_datamap = _dict_to_inference_datamap(activation_map_infer);
    this->add_arg(Arg("activation_map", activation_datamap));

    auto backend_datamap = _dict_to_inference_datamap(backend_map.cast<py::dict>());
    this->add_arg(Arg("backend_map", backend_datamap));

    // convert from Python dict to InferenceOp::DataVecMap
    auto pre_processor_datamap = _dict_to_inference_datavecmap(pre_processor_map.cast<py::dict>());
    this->add_arg(Arg("pre_processor_map", pre_processor_datamap));

    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

/* The python module */

PYBIND11_MODULE(_inference, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK InferenceOp Python Bindings
        ----------------------------------------
        .. currentmodule:: _inference
    )pbdoc";

  py::class_<InferenceOp, PyInferenceOp, Operator, std::shared_ptr<InferenceOp>> inference_op(
      m, "InferenceOp", doc::InferenceOp::doc_InferenceOp);

  inference_op.def(py::init<Fragment*,
                            const py::args&,
                            const std::string&,
                            std::shared_ptr<::holoscan::Allocator>,
                            py::dict,
                            py::dict,
                            py::dict,
                            py::dict,
                            py::dict,
                            py::dict,
                            py::dict,
                            py::dict,
                            const std::vector<std::string>&,
                            const std::vector<std::string>&,
                            const std::vector<std::vector<int32_t>>&,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            bool,
                            int32_t,
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
                   "dla_core_map"_a = py::dict(),
                   "temporal_map"_a = py::dict(),
                   "activation_map"_a = py::dict(),
                   "backend_map"_a = py::dict(),
                   "in_tensor_names"_a = std::vector<std::string>{},
                   "out_tensor_names"_a = std::vector<std::string>{},
                   "trt_opt_profile"_a = std::vector<std::vector<int32_t>>{{1, 1, 1}},
                   "infer_on_cpu"_a = false,
                   "parallel_inference"_a = true,
                   "input_on_cuda"_a = true,
                   "output_on_cuda"_a = true,
                   "transmit_on_cuda"_a = true,
                   "dynamic_input_dims"_a = false,
                   "enable_fp16"_a = false,
                   "enable_cuda_graphs"_a = true,
                   "dla_core"_a = -1,
                   "dla_gpu_fallback"_a = true,
                   "is_engine_path"_a = false,
                   "cuda_stream_pool"_a = py::none(),
                   "name"_a = "inference"s,
                   doc::InferenceOp::doc_InferenceOp);

  py::class_<InferenceOp::DataMap>(inference_op, "DataMap")
      .def(py::init<>())
      .def("insert", &InferenceOp::DataMap::get_map)
      .def("get_map", &InferenceOp::DataMap::get_map);

  py::class_<InferenceOp::DataVecMap>(inference_op, "DataVecMap")
      .def(py::init<>())
      .def("insert", &InferenceOp::DataVecMap::get_map)
      .def("get_map", &InferenceOp::DataVecMap::get_map);

  py::class_<InferenceOp::ActivationSpec>(inference_op, "ActivationSpec")
      .def(py::init<const std::string&, bool>())
      .def("is_active", &InferenceOp::ActivationSpec::is_active, "Get model active flag")
      .def("model", &InferenceOp::ActivationSpec::model, "Get model name")
      .def("set_active",
           &InferenceOp::ActivationSpec::set_active,
           "Set model active flag",
           py::arg("active") = true);

  gxf::CodecRegistry::get_instance()
      .add_codec<std::vector<holoscan::ops::InferenceOp::ActivationSpec>>(
          "std::vector<std::vector<holoscan::ops::InferenceOp::ActivationSpec>>", true);
  // See python bindings for holoviz operator
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    HOLOSCAN_LOG_INFO("Call in register types for ActivationSpec");
    registry.add_emitter_receiver<std::vector<holoscan::ops::InferenceOp::ActivationSpec>>(
        "std::vector<InferenceOp::ActivationSpec>"s);
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
