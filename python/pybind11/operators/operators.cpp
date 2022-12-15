/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "./operators_pydoc.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/block_memory_pool.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/std_ops.hpp"

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

static const std::vector<std::vector<float>> VIZ_TOOL_DEFAULT_COLORS = {{0.12f, 0.47f, 0.71f},
                                                                        {0.20f, 0.63f, 0.17f},
                                                                        {0.89f, 0.10f, 0.11f},
                                                                        {1.00f, 0.50f, 0.00f},
                                                                        {0.42f, 0.24f, 0.60f},
                                                                        {0.69f, 0.35f, 0.16f},
                                                                        {0.65f, 0.81f, 0.89f},
                                                                        {0.70f, 0.87f, 0.54f},
                                                                        {0.98f, 0.60f, 0.60f},
                                                                        {0.99f, 0.75f, 0.44f},
                                                                        {0.79f, 0.70f, 0.84f},
                                                                        {1.00f, 1.00f, 0.60f}};

class PyFormatConverterOp : public FormatConverterOp {
 public:
  /* Inherit the constructors */
  using FormatConverterOp::FormatConverterOp;

  // Define a constructor that fully initializes the object.
  PyFormatConverterOp(Fragment* fragment, std::shared_ptr<holoscan::Allocator> pool,
                      const std::string& out_dtype, const std::string& in_dtype = "",
                      const std::string& in_tensor_name = "",
                      const std::string& out_tensor_name = "", float scale_min = 0.f,
                      float scale_max = 1.f, uint8_t alpha_value = static_cast<uint8_t>(255),
                      int32_t resize_height = 0, int32_t resize_width = 0, int32_t resize_mode = 0,
                      const std::vector<int> out_channel_order = std::vector<int>{},
                      const std::string& name = "format_converter")
      : FormatConverterOp(ArgList{Arg{"in_tensor_name", in_tensor_name},
                                  Arg{"in_dtype", in_dtype},
                                  Arg{"out_tensor_name", out_tensor_name},
                                  Arg{"out_dtype", out_dtype},
                                  Arg{"scale_min", scale_min},
                                  Arg{"scale_max", scale_max},
                                  Arg{"alpha_value", alpha_value},
                                  Arg{"resize_width", resize_width},
                                  Arg{"resize_height", resize_height},
                                  Arg{"resize_mode", resize_mode},
                                  Arg{"out_channel_order", out_channel_order},
                                  Arg{"pool", pool}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyVideoStreamReplayerOp : public VideoStreamReplayerOp {
 public:
  /* Inherit the constructors */
  using VideoStreamReplayerOp::VideoStreamReplayerOp;

  // Define a constructor that fully initializes the object.
  PyVideoStreamReplayerOp(Fragment* fragment, const std::string& directory,
                          const std::string& basename, size_t batch_size = 1UL,
                          bool ignore_corrupted_entities = true, float frame_rate = 0.f,
                          bool realtime = true, bool repeat = false, uint64_t count = 0UL,
                          const std::string& name = "video_stream_replayer")
      : VideoStreamReplayerOp(ArgList{Arg{"directory", directory},
                                      Arg{"basename", basename},
                                      Arg{"batch_size", batch_size},
                                      Arg{"ignore_corrupted_entities", ignore_corrupted_entities},
                                      Arg{"frame_rate", frame_rate},
                                      Arg{"realtime", realtime},
                                      Arg{"repeat", repeat},
                                      Arg{"count", count}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

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
    initialize();
  }
};

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
    initialize();
  }
};

class PyLSTMTensorRTInferenceOp : public LSTMTensorRTInferenceOp {
 public:
  /* Inherit the constructors */
  using LSTMTensorRTInferenceOp::LSTMTensorRTInferenceOp;

  // Define a constructor that fully initializes the object.
  PyLSTMTensorRTInferenceOp(
      Fragment* fragment, const std::vector<std::string>& input_tensor_names,
      const std::vector<std::string>& output_tensor_names,
      const std::vector<std::string>& input_binding_names,
      const std::vector<std::string>& output_binding_names, const std::string& model_file_path,
      const std::string& engine_cache_dir,
      // int64_t dla_core,
      std::shared_ptr<holoscan::Allocator> pool,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool,
      // std::shared_ptr<holoscan::Resource> clock,
      const std::string& plugins_lib_namespace = "",
      const std::vector<std::string>& input_state_tensor_names = std::vector<std::string>{},
      const std::vector<std::string>& output_state_tensor_names = std::vector<std::string>{},
      bool force_engine_update = false, bool enable_fp16_ = false, bool verbose = false,
      bool relaxed_dimension_check = true, int64_t max_workspace_size = 67108864l,
      int32_t max_batch_size = 1, const std::string& name = "lstm_tensor_rt_inference")
      : LSTMTensorRTInferenceOp(ArgList{Arg{"input_tensor_names", input_tensor_names},
                                        Arg{"output_tensor_names", output_tensor_names},
                                        Arg{"input_binding_names", input_binding_names},
                                        Arg{"output_binding_names", output_binding_names},
                                        Arg{"model_file_path", model_file_path},
                                        Arg{"engine_cache_dir", engine_cache_dir},
                                        // Arg{"dla_core", dla_core},
                                        Arg{"pool", pool},
                                        Arg{"cuda_stream_pool", cuda_stream_pool},
                                        // Arg{"clock", clock},
                                        Arg{"plugins_lib_namespace", plugins_lib_namespace},
                                        Arg{"input_state_tensor_names", input_state_tensor_names},
                                        Arg{"output_state_tensor_names", output_state_tensor_names},
                                        Arg{"force_engine_update", force_engine_update},
                                        Arg{"enable_fp16_", enable_fp16_},
                                        Arg{"verbose", verbose},
                                        Arg{"relaxed_dimension_check", relaxed_dimension_check},
                                        Arg{"max_workspace_size", max_workspace_size},
                                        Arg{"max_batch_size", max_batch_size}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyTensorRTInferenceOp : public TensorRTInferenceOp {
 public:
  /* Inherit the constructors */
  using TensorRTInferenceOp::TensorRTInferenceOp;

  // Define a constructor that fully initializes the object.
  PyTensorRTInferenceOp(Fragment* fragment, const std::string& model_file_path,
                        const std::string& engine_cache_dir,
                        const std::vector<std::string>& input_tensor_names,
                        const std::vector<std::string>& input_binding_names,
                        const std::vector<std::string>& output_tensor_names,
                        const std::vector<std::string>& output_binding_names,
                        std::shared_ptr<holoscan::Allocator> pool,
                        std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool,
                        const std::string& plugins_lib_namespace = "",
                        bool force_engine_update = false, int64_t max_workspace_size = 67108864l,
                        // int64_t dla_core,
                        int32_t max_batch_size = 1, bool enable_fp16_ = false,
                        bool relaxed_dimension_check = true, bool verbose = false,
                        // std::shared_ptr<holoscan::Resource> clock,
                        const std::string& name = "tensor_rt_inference")
      : TensorRTInferenceOp(ArgList{Arg{"model_file_path", model_file_path},
                                    Arg{"engine_cache_dir", engine_cache_dir},
                                    Arg{"input_tensor_names", input_tensor_names},
                                    Arg{"output_tensor_names", output_tensor_names},
                                    Arg{"input_binding_names", input_binding_names},
                                    Arg{"output_binding_names", output_binding_names},
                                    Arg{"pool", pool},
                                    Arg{"cuda_stream_pool", cuda_stream_pool},
                                    Arg{"force_engine_update", force_engine_update},
                                    Arg{"plugins_lib_namespace", plugins_lib_namespace},
                                    Arg{"max_workspace_size", max_workspace_size},
                                    // Arg{"dla_core", dla_core},
                                    Arg{"max_batch_size", max_batch_size},
                                    // Arg{"clock", clock},
                                    Arg{"enable_fp16_", enable_fp16_},
                                    Arg{"relaxed_dimension_check", relaxed_dimension_check},
                                    Arg{"verbose", verbose}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyHolovizOp : public HolovizOp {
 public:
  /* Inherit the constructors */
  using HolovizOp::HolovizOp;

  // Define a constructor that fully initializes the object.
  PyHolovizOp(
      Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
      std::vector<holoscan::IOSpec*> receivers = std::vector<holoscan::IOSpec*>(),
      const std::vector<HolovizOp::InputSpec>& tensors = std::vector<HolovizOp::InputSpec>(),
      const std::vector<std::vector<float>>& color_lut = std::vector<std::vector<float>>(),
      const std::string& window_title = "Holoviz", const std::string& display_name = "DP-0",
      uint32_t width = 1920, uint32_t height = 1080, uint32_t framerate = 60,
      bool use_exclusive_display = false, bool fullscreen = false, bool headless = false,
      bool enable_render_buffer_input = false, bool enable_render_buffer_output = false,
      const std::string& name = "holoviz_op")
      : HolovizOp(ArgList{Arg{"allocator", allocator},
                          Arg{"color_lut", color_lut},
                          Arg{"window_title", window_title},
                          Arg{"display_name", display_name},
                          Arg{"width", width},
                          Arg{"height", height},
                          Arg{"framerate", framerate},
                          Arg{"use_exclusive_display", use_exclusive_display},
                          Arg{"fullscreen", fullscreen},
                          Arg{"headless", headless},
                          Arg{"enable_render_buffer_input", enable_render_buffer_input},
                          Arg{"enable_render_buffer_output", enable_render_buffer_output}}) {
    // only append tensors argument if it is not empty
    //     avoids [holoscan] [error] [gxf_operator.hpp:126] Unable to handle parameter 'tensors'
    if (tensors.size() > 0) { this->add_arg(Arg{"tensors", tensors}); }
    if (receivers.size() > 0) { this->add_arg(Arg{"receivers", receivers}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PySegmentationPostprocessorOp : public SegmentationPostprocessorOp {
 public:
  /* Inherit the constructors */
  using SegmentationPostprocessorOp::SegmentationPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PySegmentationPostprocessorOp(Fragment* fragment,
                                std::shared_ptr<::holoscan::Allocator> allocator,
                                const std::string& in_tensor_name = "",
                                const std::string& network_output_type = "softmax"s,
                                const std::string& data_format = "hwc"s,
                                const std::string& name = "segmentation_postprocessor"s)
      : SegmentationPostprocessorOp(ArgList{Arg{"in_tensor_name", in_tensor_name},
                                            Arg{"network_output_type", network_output_type},
                                            Arg{"data_format", data_format},
                                            Arg{"allocator", allocator}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyToolTrackingPostprocessorOp : public ToolTrackingPostprocessorOp {
 public:
  /* Inherit the constructors */
  using ToolTrackingPostprocessorOp::ToolTrackingPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PyToolTrackingPostprocessorOp(
      Fragment* fragment, std::shared_ptr<Allocator> device_allocator,
      std::shared_ptr<Allocator> host_allocator, float min_prob = 0.5f,
      std::vector<std::vector<float>> overlay_img_colors = VIZ_TOOL_DEFAULT_COLORS,
      const std::string& name = "tool_tracking_postprocessor")
      : ToolTrackingPostprocessorOp(ArgList{Arg{"device_allocator", device_allocator},
                                            Arg{"host_allocator", host_allocator},
                                            Arg{"min_prob", min_prob},
                                            Arg{"overlay_img_colors", overlay_img_colors}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

MultiAIInferenceOp::DataMap _dict_to_multiai_inference_datamap(py::dict dict) {
  MultiAIInferenceOp::DataMap data_map;
  for (auto item : dict) {
    data_map.insert(item.first.cast<std::string>(), item.second.cast<std::string>());
  }
  return data_map;
}

MultiAIInferenceOp::DataVecMap _dict_to_multiai_inference_datavecmap(py::dict dict) {
  MultiAIInferenceOp::DataVecMap data_vec_map;
  for (auto item : dict) {
    data_vec_map.insert(item.first.cast<std::string>(),
                        item.second.cast<std::vector<std::string>>());
  }
  return data_vec_map;
}

class PyMultiAIInferenceOp : public MultiAIInferenceOp {
 public:
  /* Inherit the constructors */
  using MultiAIInferenceOp::MultiAIInferenceOp;

  // Define a constructor that fully initializes the object.
  PyMultiAIInferenceOp(Fragment* fragment, const std::string& backend,
                       std::shared_ptr<::holoscan::Allocator> allocator,
                       py::dict inference_map,      // MultiAIPostprocessorOp::DataMap
                       py::dict model_path_map,     // MultiAIPostprocessorOp::DataMap
                       py::dict pre_processor_map,  // MultiAIPostprocessorOp::DataVecMap
                       const std::vector<std::string>& in_tensor_names,
                       const std::vector<std::string>& out_tensor_names, bool infer_on_cpu = false,
                       bool parallel_inference = true, bool input_on_cuda = true,
                       bool output_on_cuda = true, bool transmit_on_cuda = true,
                       bool enable_fp16 = false, bool is_engine_path = false,
                       // TODO(grelee): handle receivers similarly to HolovizOp?  (default: {})
                       // TODO(grelee): handle transmitter similarly to HolovizOp?
                       const std::string& name = "multi_ai_inference")
      : MultiAIInferenceOp(ArgList{Arg{"backend", backend},
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
    name_ = name;
    fragment_ = fragment;

    // convert from Python dict to MultiAIPostprocessorOp::DataMap
    auto inference_map_datamap = _dict_to_multiai_inference_datamap(inference_map.cast<py::dict>());
    this->add_arg(Arg("inference_map", inference_map_datamap));

    auto model_path_datamap = _dict_to_multiai_inference_datamap(model_path_map.cast<py::dict>());
    this->add_arg(Arg("model_path_map", model_path_datamap));

    // convert from Python dict to MultiAIPostprocessorOp::DataVecMap
    auto pre_processor_datamap =
        _dict_to_multiai_inference_datavecmap(pre_processor_map.cast<py::dict>());
    this->add_arg(Arg("pre_processor_map", pre_processor_datamap));

    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

MultiAIPostprocessorOp::DataMap _dict_to_multiai_postproc_datamap(py::dict dict) {
  MultiAIPostprocessorOp::DataMap data_map;
  for (auto item : dict) {
    data_map.insert(item.first.cast<std::string>(), item.second.cast<std::string>());
  }
  return data_map;
}

MultiAIPostprocessorOp::DataVecMap _dict_to_multiai_postproc_datavecmap(py::dict dict) {
  MultiAIPostprocessorOp::DataVecMap data_vec_map;
  for (auto item : dict) {
    data_vec_map.insert(item.first.cast<std::string>(),
                        item.second.cast<std::vector<std::string>>());
  }
  return data_vec_map;
}

class PyMultiAIPostprocessorOp : public MultiAIPostprocessorOp {
 public:
  /* Inherit the constructors */
  using MultiAIPostprocessorOp::MultiAIPostprocessorOp;

  // Define a constructor that fully initializes the object.
  PyMultiAIPostprocessorOp(
      Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
      py::dict process_operations,                       // MultiAIPostprocessorOp::DataVecMap
      py::dict processed_map,                            // MultiAIPostprocessorOp::DataMap
      const std::vector<std::string>& in_tensor_names,   // = {std::string("")},
      const std::vector<std::string>& out_tensor_names,  // = {std::string("")},
      bool input_on_cuda = false, bool output_on_cuda = false, bool transmit_on_cuda = false,
      // TODO(grelee): handle receivers similarly to HolovizOp?  (default: {})
      // TODO(grelee): handle transmitter similarly to HolovizOp?
      const std::string& name = "multi_ai_postprocessor")
      : MultiAIPostprocessorOp(ArgList{Arg{"allocator", allocator},
                                       Arg{"in_tensor_names", in_tensor_names},
                                       Arg{"out_tensor_names", out_tensor_names},
                                       Arg{"input_on_cuda", input_on_cuda},
                                       Arg{"output_on_cuda", output_on_cuda},
                                       Arg{"transmit_on_cuda", transmit_on_cuda}}) {
    name_ = name;
    fragment_ = fragment;

    // convert from Python dict to MultiAIPostprocessorOp::DataVecMap
    auto process_operations_datavecmap =
        _dict_to_multiai_postproc_datavecmap(process_operations.cast<py::dict>());
    this->add_arg(Arg("process_operations", process_operations_datavecmap));

    // convert from Python dict to MultiAIPostprocessorOp::DataMap
    auto processed_map_datamap = _dict_to_multiai_postproc_datamap(processed_map.cast<py::dict>());
    this->add_arg(Arg("processed_map", processed_map_datamap));

    spec_ = std::make_shared<OperatorSpec>(fragment);

    setup(*spec_.get());
    initialize();
  }
};

class PyVisualizerICardioOp : public VisualizerICardioOp {
 public:
  /* Inherit the constructors */
  using VisualizerICardioOp::VisualizerICardioOp;

  // Define a constructor that fully initializes the object.
  PyVisualizerICardioOp(Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
                        const std::vector<std::string>& in_tensor_names = {std::string("")},
                        const std::vector<std::string>& out_tensor_names = {std::string("")},
                        bool input_on_cuda = false,
                        // TODO(grelee): handle receivers similarly to HolovizOp?  (default: {})
                        // TODO(grelee): handle transmitter similarly to HolovizOp?
                        const std::string& name = "visualizer_icardio")
      : VisualizerICardioOp(ArgList{Arg{"allocator", allocator},
                                    Arg{"in_tensor_names", in_tensor_names},
                                    Arg{"out_tensor_names", out_tensor_names},
                                    Arg{"input_on_cuda", input_on_cuda}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

class PyBayerDemosaicOp : public BayerDemosaicOp {
 public:
  /* Inherit the constructors */
  using BayerDemosaicOp::BayerDemosaicOp;

  // Define a constructor that fully initializes the object.
  PyBayerDemosaicOp(Fragment* fragment, std::shared_ptr<holoscan::Allocator> pool,
                    std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool,
                    const std::string& in_tensor_name = "", const std::string& out_tensor_name = "",
                    int interpolation_mode = 0, int bayer_grid_pos = 2, bool generate_alpha = false,
                    int alpha_value = 255, const std::string& name = "bayer_demosaic")
      : BayerDemosaicOp(ArgList{Arg{"pool", pool},
                                Arg{"cuda_stream_pool", cuda_stream_pool},
                                Arg{"in_tensor_name", in_tensor_name},
                                Arg{"out_tensor_name", out_tensor_name},
                                Arg{"interpolation_mode", interpolation_mode},
                                Arg{"bayer_grid_pos", bayer_grid_pos},
                                Arg{"generate_alpha", generate_alpha},
                                Arg{"alpha_value", alpha_value}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

#if HOLOSCAN_BUILD_EMERGENT == 1

class PyEmergentSourceOp : public EmergentSourceOp {
 public:
  /* Inherit the constructors */
  using EmergentSourceOp::EmergentSourceOp;

  // Define a constructor that fully initializes the object.
  PyEmergentSourceOp(Fragment* fragment,
                     // defaults here should match constexpr values in EmergentSourceOp::Setup
                     uint32_t width = 4200, uint32_t height = 2160, uint32_t framerate = 240,
                     bool rdma = false, const std::string& name = "emergent_source")
      : EmergentSourceOp(ArgList{Arg{"width", width},
                                 Arg{"height", height},
                                 Arg{"framerate", framerate},
                                 Arg{"rdma", rdma}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

#endif  // HOLOSCAN_BUILD_EMERGENT == 1

PYBIND11_MODULE(_operators, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _operators
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

  /*****************************************
   * operators inheriting from GXFOperator *
   *****************************************/

  py::class_<FormatConverterOp,
             PyFormatConverterOp,
             GXFOperator,
             std::shared_ptr<FormatConverterOp>>(
      m, "FormatConverterOp", doc::FormatConverterOp::doc_FormatConverterOp)
      .def(py::init<Fragment*,
                    std::shared_ptr<holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    float,
                    float,
                    uint8_t,
                    int32_t,
                    int32_t,
                    int32_t,
                    const std::vector<int>,
                    const std::string&>(),
           "fragment"_a,
           "pool"_a,
           "out_dtype"_a,
           "in_dtype"_a = ""s,
           "in_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "scale_min"_a = 0.f,
           "scale_max"_a = 1.f,
           "alpha_value"_a = static_cast<uint8_t>(255),
           "resize_height"_a = 0,
           "resize_width"_a = 0,
           "resize_mode"_a = 0,
           "out_channel_order"_a = std::vector<int>{},
           "name"_a = "format_converter"s,
           doc::FormatConverterOp::doc_FormatConverterOp_python)
      .def_property_readonly("gxf_typename",
                             &FormatConverterOp::gxf_typename,
                             doc::FormatConverterOp::doc_gxf_typename)
      .def("initialize", &FormatConverterOp::initialize, doc::FormatConverterOp::doc_initialize)
      .def("setup", &FormatConverterOp::setup, "spec"_a, doc::FormatConverterOp::doc_setup);

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

  py::class_<AJASourceOp, PyAJASourceOp, GXFOperator, std::shared_ptr<AJASourceOp>>(
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
      .def_property_readonly(
          "gxf_typename", &AJASourceOp::gxf_typename, doc::AJASourceOp::doc_gxf_typename)
      .def("initialize", &AJASourceOp::initialize, doc::AJASourceOp::doc_initialize)
      .def("setup", &AJASourceOp::setup, "spec"_a, doc::AJASourceOp::doc_setup);

  py::class_<LSTMTensorRTInferenceOp,
             PyLSTMTensorRTInferenceOp,
             GXFOperator,
             std::shared_ptr<LSTMTensorRTInferenceOp>>(
      m, "LSTMTensorRTInferenceOp", doc::LSTMTensorRTInferenceOp::doc_LSTMTensorRTInferenceOp)
      .def(py::init<Fragment*,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::string&,
                    const std::string&,
                    // int64_t,  // dla_core
                    std::shared_ptr<holoscan::Allocator>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    // std::shared_ptr<holoscan::Resource>,  // clock
                    const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    bool,
                    bool,
                    bool,
                    int64_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "input_tensor_names"_a,
           "output_tensor_names"_a,
           "input_binding_names"_a,
           "output_binding_names"_a,
           "model_file_path"_a,
           "engine_cache_dir"_a,
           // "dla_core"_a,
           "pool"_a,
           "cuda_stream_pool"_a,
           // "clock"_a,
           "plugins_lib_namespace"_a = "",
           "input_state_tensor_names"_a = std::vector<std::string>{},
           "output_state_tensor_names"_a = std::vector<std::string>{},
           "force_engine_update"_a = false,
           "enable_fp16_"_a = false,
           "verbose"_a = false,
           "relaxed_dimension_check"_a = true,
           "max_workspace_size"_a = 67108864l,
           "max_batch_size"_a = 1,
           "name"_a = "lstm_tensor_rt_inference"s,
           doc::LSTMTensorRTInferenceOp::doc_LSTMTensorRTInferenceOp_python)
      .def_property_readonly("gxf_typename",
                             &LSTMTensorRTInferenceOp::gxf_typename,
                             doc::LSTMTensorRTInferenceOp::doc_gxf_typename)
      .def("initialize",
           &LSTMTensorRTInferenceOp::initialize,
           doc::LSTMTensorRTInferenceOp::doc_initialize)
      .def("setup",
           &LSTMTensorRTInferenceOp::setup,
           "spec"_a,
           doc::LSTMTensorRTInferenceOp::doc_setup);

  py::class_<TensorRTInferenceOp,
             PyTensorRTInferenceOp,
             GXFOperator,
             std::shared_ptr<TensorRTInferenceOp>>(
      m, "TensorRTInferenceOp", doc::TensorRTInferenceOp::doc_TensorRTInferenceOp)
      .def(py::init<Fragment*,
                    const std::string&,
                    const std::string&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    std::shared_ptr<holoscan::Allocator>,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&,
                    bool,
                    int64_t,
                    // int64_t dla_core,
                    int64_t,
                    bool,
                    bool,
                    bool,
                    // std::shared_ptr<holoscan::Resource> clock,
                    const std::string&>(),
           "fragment"_a,
           "model_file_path"_a,
           "engine_cache_dir"_a,
           "input_tensor_names"_a,
           "input_binding_names"_a,
           "output_tensor_names"_a,
           "output_binding_names"_a,
           "pool"_a,
           "cuda_stream_pool"_a,
           "plugins_lib_namespace"_a = "",
           "force_engine_update"_a = false,
           "max_workspace_size"_a = 67108864l,
           // "dla_core"_a,
           "max_batch_size"_a = 1,
           "enable_fp16_"_a = false,
           "relaxed_dimension_check"_a = true,
           "verbose"_a = false,
           // "clock"_a,
           "name"_a = "tensor_rt_inference"s,
           doc::TensorRTInferenceOp::doc_TensorRTInferenceOp_python)
      .def_property_readonly("gxf_typename",
                             &TensorRTInferenceOp::gxf_typename,
                             doc::TensorRTInferenceOp::doc_gxf_typename)
      .def("initialize", &TensorRTInferenceOp::initialize, doc::TensorRTInferenceOp::doc_initialize)
      .def("setup", &TensorRTInferenceOp::setup, "spec"_a, doc::TensorRTInferenceOp::doc_setup);

  py::class_<VideoStreamRecorderOp,
             PyVideoStreamRecorderOp,
             GXFOperator,
             std::shared_ptr<VideoStreamRecorderOp>>(
      m, "VideoStreamRecorderOp", doc::VideoStreamRecorderOp::doc_VideoStreamRecorderOp)
      .def(py::init<Fragment*, const std::string&, const std::string&, bool, const std::string&>(),
           "fragment"_a,
           "directory"_a,
           "basename"_a,
           "flush_on_tick"_a = false,
           "name"_a = "recorder"s,
           doc::VideoStreamRecorderOp::doc_VideoStreamRecorderOp_python)
      .def_property_readonly("gxf_typename",
                             &VideoStreamRecorderOp::gxf_typename,
                             doc::VideoStreamRecorderOp::doc_gxf_typename)
      .def("initialize",
           &VideoStreamRecorderOp::initialize,
           doc::VideoStreamRecorderOp::doc_initialize)
      .def("setup", &VideoStreamRecorderOp::setup, "spec"_a, doc::VideoStreamRecorderOp::doc_setup);

  py::class_<VideoStreamReplayerOp,
             PyVideoStreamReplayerOp,
             GXFOperator,
             std::shared_ptr<VideoStreamReplayerOp>>(
      m, "VideoStreamReplayerOp", doc::VideoStreamReplayerOp::doc_VideoStreamReplayerOp)
      .def(py::init<Fragment*,
                    const std::string&,
                    const std::string&,
                    size_t,
                    bool,
                    float,
                    bool,
                    bool,
                    uint64_t,
                    const std::string&>(),
           "fragment"_a,
           "directory"_a,
           "basename"_a,
           "batch_size"_a = 1UL,
           "ignore_corrupted_entities"_a = true,
           "frame_rate"_a = 1.f,
           "realtime"_a = true,
           "repeat"_a = false,
           "count"_a = 0UL,
           "name"_a = "format_converter"s,
           doc::VideoStreamReplayerOp::doc_VideoStreamReplayerOp_python)
      .def_property_readonly("gxf_typename",
                             &VideoStreamReplayerOp::gxf_typename,
                             doc::VideoStreamReplayerOp::doc_gxf_typename)
      .def("initialize",
           &VideoStreamReplayerOp::initialize,
           doc::VideoStreamReplayerOp::doc_initialize)
      .def("setup", &VideoStreamReplayerOp::setup, "spec"_a, doc::VideoStreamReplayerOp::doc_setup);

  py::class_<HolovizOp, PyHolovizOp, GXFOperator, std::shared_ptr<HolovizOp>> holoviz_op(
      m, "HolovizOp", doc::HolovizOp::doc_HolovizOp);
  holoviz_op
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    std::vector<holoscan::IOSpec*>,
                    const std::vector<HolovizOp::InputSpec>&,
                    const std::vector<std::vector<float>>&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "receivers"_a = std::vector<holoscan::IOSpec*>(),
           "tensors"_a = std::vector<HolovizOp::InputSpec>(),
           "color_lut"_a = std::vector<std::vector<float>>(),
           "window_title"_a = "Holoviz",
           "display_name"_a = "DP-0",
           "width"_a = 1920,
           "height"_a = 1080,
           "framerate"_a = 60,
           "use_exclusive_display"_a = false,
           "fullscreen"_a = false,
           "headless"_a = false,
           "enable_render_buffer_input"_a = false,
           "enable_render_buffer_output"_a = false,
           "name"_a = "holoviz_op"s,
           doc::HolovizOp::doc_HolovizOp_python)
      .def_property_readonly(
          "gxf_typename", &HolovizOp::gxf_typename, doc::HolovizOp::doc_gxf_typename)
      .def("initialize", &HolovizOp::initialize, doc::HolovizOp::doc_initialize)
      .def("setup", &HolovizOp::setup, "spec"_a, doc::HolovizOp::doc_setup);

  py::enum_<HolovizOp::InputType>(holoviz_op, "InputType")
      .value("UNKNOWN", HolovizOp::InputType::UNKNOWN)
      .value("COLOR", HolovizOp::InputType::COLOR)
      .value("COLOR_LUT", HolovizOp::InputType::COLOR_LUT)
      .value("POINTS", HolovizOp::InputType::POINTS)
      .value("LINES", HolovizOp::InputType::LINES)
      .value("LINE_STRIP", HolovizOp::InputType::LINE_STRIP)
      .value("TRIANGLES", HolovizOp::InputType::TRIANGLES)
      .value("CROSSES", HolovizOp::InputType::CROSSES)
      .value("RECTANGLES", HolovizOp::InputType::RECTANGLES)
      .value("OVALS", HolovizOp::InputType::OVALS)
      .value("TEXT", HolovizOp::InputType::TEXT);

  py::class_<HolovizOp::InputSpec>(holoviz_op, "InputSpec")
      .def(py::init<const std::string, HolovizOp::InputType>())
      .def_readwrite("_type", &HolovizOp::InputSpec::type_)
      .def_readwrite("_color", &HolovizOp::InputSpec::color_)
      .def_readwrite("_opacity", &HolovizOp::InputSpec::opacity_)
      .def_readwrite("_priority", &HolovizOp::InputSpec::priority_)
      .def_readwrite("_line_width", &HolovizOp::InputSpec::line_width_)
      .def_readwrite("_point_size", &HolovizOp::InputSpec::point_size_)
      .def_readwrite("_text", &HolovizOp::InputSpec::text_);

  py::class_<SegmentationPostprocessorOp,
             PySegmentationPostprocessorOp,
             GXFOperator,
             std::shared_ptr<SegmentationPostprocessorOp>>(
      m,
      "SegmentationPostprocessorOp",
      doc::SegmentationPostprocessorOp::doc_SegmentationPostprocessorOp)
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::string&,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_name"_a = ""s,
           "network_output_type"_a = "softmax"s,
           "data_format"_a = "hwc"s,
           "name"_a = "segmentation_postprocessor"s,
           doc::SegmentationPostprocessorOp::doc_SegmentationPostprocessorOp_python)
      .def_property_readonly("gxf_typename",
                             &SegmentationPostprocessorOp::gxf_typename,
                             doc::SegmentationPostprocessorOp::doc_gxf_typename)
      .def("setup",
           &SegmentationPostprocessorOp::setup,
           "spec"_a,
           doc::SegmentationPostprocessorOp::doc_setup);

  py::class_<ToolTrackingPostprocessorOp,
             PyToolTrackingPostprocessorOp,
             GXFOperator,
             std::shared_ptr<ToolTrackingPostprocessorOp>>(
      m,
      "ToolTrackingPostprocessorOp",
      doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp)
      .def(py::init<Fragment*,
                    std::shared_ptr<Allocator>,
                    std::shared_ptr<Allocator>,
                    float,
                    std::vector<std::vector<float>>,
                    const std::string&>(),
           "fragment"_a,
           "device_allocator"_a,
           "host_allocator"_a,
           "min_prob"_a = 0.5f,
           "overlay_img_colors"_a = VIZ_TOOL_DEFAULT_COLORS,
           "name"_a = "tool_tracking_postprocessor"s,
           doc::ToolTrackingPostprocessorOp::doc_ToolTrackingPostprocessorOp_python)
      .def_property_readonly("gxf_typename",
                             &ToolTrackingPostprocessorOp::gxf_typename,
                             doc::ToolTrackingPostprocessorOp::doc_gxf_typename)
      .def("setup",
           &ToolTrackingPostprocessorOp::setup,
           "spec"_a,
           doc::ToolTrackingPostprocessorOp::doc_setup);

  py::class_<MultiAIInferenceOp,
             PyMultiAIInferenceOp,
             GXFOperator,
             std::shared_ptr<MultiAIInferenceOp>>
      multiai_infererence_op(
          m, "MultiAIInferenceOp", doc::MultiAIInferenceOp::doc_MultiAIInferenceOp);

  multiai_infererence_op
      .def(py::init<Fragment*,
                    const std::string&,
                    std::shared_ptr<::holoscan::Allocator>,
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
                    const std::string&>(),
           "fragment"_a,
           "backend"_a,
           "allocator"_a,
           "inference_map"_a,
           "model_path_map"_a,
           "pre_processor_map"_a,
           "in_tensor_names"_a = std::vector<std::string>{{std::string("")}},
           "out_tensor_names"_a = std::vector<std::string>{{std::string("")}},
           "infer_on_cpu"_a = false,
           "parallel_inference"_a = true,
           "input_on_cuda"_a = true,
           "output_on_cuda"_a = true,
           "transmit_on_cuda"_a = true,
           "enable_fp16"_a = false,
           "is_engine_path"_a = false,
           "name"_a = "multi_ai_inference"s,
           doc::MultiAIInferenceOp::doc_MultiAIInferenceOp_python)
      .def_property_readonly("gxf_typename",
                             &MultiAIInferenceOp::gxf_typename,
                             doc::MultiAIInferenceOp::doc_gxf_typename)
      .def("initialize", &MultiAIInferenceOp::initialize, doc::MultiAIInferenceOp::doc_initialize)
      .def("setup", &MultiAIInferenceOp::setup, "spec"_a, doc::MultiAIInferenceOp::doc_setup);

  py::class_<MultiAIInferenceOp::DataMap>(multiai_infererence_op, "DataMap")
      .def(py::init<>())
      .def("insert", &MultiAIInferenceOp::DataMap::get_map)
      .def("get_map", &MultiAIInferenceOp::DataMap::get_map);

  py::class_<MultiAIInferenceOp::DataVecMap>(multiai_infererence_op, "DataVecMap")
      .def(py::init<>())
      .def("insert", &MultiAIInferenceOp::DataVecMap::get_map)
      .def("get_map", &MultiAIInferenceOp::DataVecMap::get_map);

  py::class_<MultiAIPostprocessorOp,
             PyMultiAIPostprocessorOp,
             GXFOperator,
             std::shared_ptr<MultiAIPostprocessorOp>>
      multiai_postprocessor_op(
          m, "MultiAIPostprocessorOp", doc::MultiAIPostprocessorOp::doc_MultiAIPostprocessorOp);

  multiai_postprocessor_op
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    py::dict,
                    py::dict,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "process_operations"_a,
           "processed_map"_a,
           "in_tensor_names"_a = std::vector<std::string>{{std::string("")}},
           "out_tensor_names"_a = std::vector<std::string>{{std::string("")}},
           "input_on_cuda"_a = false,
           "output_on_cuda"_a = false,
           "transmit_on_cuda"_a = false,
           "name"_a = "multi_ai_postprocessor"s,
           doc::MultiAIPostprocessorOp::doc_MultiAIPostprocessorOp_python)
      .def_property_readonly("gxf_typename",
                             &MultiAIPostprocessorOp::gxf_typename,
                             doc::MultiAIPostprocessorOp::doc_gxf_typename)
      .def("initialize",
           &MultiAIPostprocessorOp::initialize,
           doc::MultiAIPostprocessorOp::doc_initialize)
      .def("setup",
           &MultiAIPostprocessorOp::setup,
           "spec"_a,
           doc::MultiAIPostprocessorOp::doc_setup);

  py::class_<MultiAIPostprocessorOp::DataMap>(multiai_postprocessor_op, "DataMap")
      .def(py::init<>())
      .def("insert", &MultiAIPostprocessorOp::DataMap::get_map)
      .def("get_map", &MultiAIPostprocessorOp::DataMap::get_map);

  py::class_<MultiAIPostprocessorOp::DataVecMap>(multiai_postprocessor_op, "DataVecMap")
      .def(py::init<>())
      .def("insert", &MultiAIPostprocessorOp::DataVecMap::get_map)
      .def("get_map", &MultiAIPostprocessorOp::DataVecMap::get_map);

  py::class_<VisualizerICardioOp,
             PyVisualizerICardioOp,
             GXFOperator,
             std::shared_ptr<VisualizerICardioOp>>(
      m, "VisualizerICardioOp", doc::VisualizerICardioOp::doc_VisualizerICardioOp)
      .def(py::init<Fragment*,
                    std::shared_ptr<::holoscan::Allocator>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "in_tensor_names"_a,   // = {std::string("")},
           "out_tensor_names"_a,  // = {std::string("")},
           "input_on_cuda"_a = false,
           "name"_a = "visualizer_icardio"s,
           doc::VisualizerICardioOp::doc_VisualizerICardioOp_python)
      .def_property_readonly("gxf_typename",
                             &VisualizerICardioOp::gxf_typename,
                             doc::VisualizerICardioOp::doc_gxf_typename)
      .def("initialize", &VisualizerICardioOp::initialize, doc::VisualizerICardioOp::doc_initialize)
      .def("setup", &VisualizerICardioOp::setup, "spec"_a, doc::VisualizerICardioOp::doc_setup);

  py::class_<BayerDemosaicOp, PyBayerDemosaicOp, GXFOperator, std::shared_ptr<BayerDemosaicOp>>(
      m, "BayerDemosaicOp", doc::BayerDemosaicOp::doc_BayerDemosaicOp)
      .def(py::init<>(), doc::BayerDemosaicOp::doc_BayerDemosaicOp)
      .def(py::init<Fragment*,
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
           "cuda_stream_pool"_a,
           "in_tensor_name"_a = ""s,
           "out_tensor_name"_a = ""s,
           "interpolation_mode"_a = 0,
           "bayer_grid_pos"_a = 2,
           "generate_alpha"_a = false,
           "alpha_value"_a = 255,
           "name"_a = "format_converter"s,
           doc::BayerDemosaicOp::doc_BayerDemosaicOp_python)
      .def_property_readonly(
          "gxf_typename", &BayerDemosaicOp::gxf_typename, doc::BayerDemosaicOp::doc_gxf_typename)
      .def("initialize", &BayerDemosaicOp::initialize, doc::BayerDemosaicOp::doc_initialize)
      .def("setup", &BayerDemosaicOp::setup, "spec"_a, doc::BayerDemosaicOp::doc_setup);

#if HOLOSCAN_BUILD_EMERGENT == 1

  py::class_<EmergentSourceOp, PyEmergentSourceOp, GXFOperator, std::shared_ptr<EmergentSourceOp>>(
      m, "EmergentSourceOp", doc::EmergentSourceOp::doc_EmergentSourceOp)
      .def(py::init<Fragment*, uint32_t, uint32_t, uint32_t, bool, const std::string&>(),
           "fragment"_a,
           // defaults values here should match constexpr values in C++ EmergentSourceOp::Setup
           "width"_a = 4200,
           "height"_a = 2160,
           "framerate"_a = 240,
           "rdma"_a = false,
           "name"_a = "emergent_source"s,
           doc::EmergentSourceOp::doc_EmergentSourceOp_python)
      .def_property_readonly(
          "gxf_typename", &EmergentSourceOp::gxf_typename, doc::EmergentSourceOp::doc_gxf_typename)
      .def("initialize", &EmergentSourceOp::initialize, doc::EmergentSourceOp::doc_initialize)
      .def("setup", &EmergentSourceOp::setup, "spec"_a, doc::EmergentSourceOp::doc_setup);
#endif  // HOLOSCAN_BUILD_EMERGENT == 1
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
