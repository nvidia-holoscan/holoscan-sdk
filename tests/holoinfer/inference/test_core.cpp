/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_core.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

void HoloInferTests::clear_specs() {
  inference_specs_.reset();
  holoscan_infer_context_.reset();
}

HoloInfer::InferStatus HoloInferTests::setup_inference() {
  holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();
  auto status = holoscan_infer_context_->set_inference_params(inference_specs_);
  return status;
}

void HoloInferTests::setup_specifications() {
  // Create inference specification structure
  inference_specs_ =
      std::make_shared<HoloInfer::InferenceSpecs>(backend,
                                                  backend_map,
                                                  model_path_map,
                                                  pre_processor_map,
                                                  inference_map,
                                                  device_map,
                                                  dla_core_map,
                                                  temporal_map,
                                                  activation_map,
                                                  batch_sizes,
                                                  dynamic_inputs,
                                                  is_engine_path,
                                                  infer_on_cpu,
                                                  parallel_inference,
                                                  enable_fp16,
                                                  input_on_cuda,
                                                  output_on_cuda,
                                                  use_cuda_graphs,
                                                  dla_core,
                                                  dla_gpu_fallback,
                                                  false,
                                                  std::function<cudaStream_t(int32_t device_id)>());
}

HoloInfer::InferStatus HoloInferTests::create_specifications() {
  HoloInfer::InferStatus status = HoloInfer::InferStatus(HoloInfer::holoinfer_code::H_ERROR);

  try {
    setup_specifications();
    status = setup_inference();
  } catch (const std::runtime_error& rt) {
    return status;
  }
  return status;
}

HoloInfer::InferStatus HoloInferTests::call_parameter_check_inference() {
  return HoloInfer::inference_validity_check(
      model_path_map, pre_processor_map, inference_map, in_tensor_names, out_tensor_names);
}

HoloInfer::InferStatus HoloInferTests::prepare_for_inference() {
  clear_specs();

  auto status = create_specifications();

  for (const auto& td : in_tensor_dimensions) {
    auto db = std::make_shared<HoloInfer::DataBuffer>();
    size_t buffer_size =
        std::accumulate(td.second.begin(), td.second.end(), 1, std::multiplies<size_t>());

    db->device_buffer_->resize(buffer_size);
    db->host_buffer_->resize(buffer_size);
    inference_specs_->data_per_tensor_.insert({td.first, std::move(db)});
    auto dims = td.second;
    dims.insert(dims.begin(), 1);
    inference_specs_->dims_per_tensor_.insert({td.first, dims});
  }

  return status;
}

HoloInfer::InferStatus HoloInferTests::do_inference() {
  HoloInfer::InferStatus status = HoloInfer::InferStatus(HoloInfer::holoinfer_code::H_ERROR);

  try {
    if (!holoscan_infer_context_) {
      return status;
    }
    return holoscan_infer_context_->execute_inference(inference_specs_);
  } catch (...) {
    std::cout << "Exception occurred in inference.\n";
    return status;
  }
}

void HoloInferTests::cleanup_engines() {
  // cleaning engine files
  for (const auto& file : std::filesystem::directory_iterator(model_folder)) {
    if (file.is_regular_file()) {
      const auto filename = file.path().filename().string();
      if (filename.find(".engine.") != std::string::npos) {
        std::filesystem::remove(file.path());
        HOLOSCAN_LOG_INFO("Cleaning up engine file: {}", filename);
      }
    }
  }
}
