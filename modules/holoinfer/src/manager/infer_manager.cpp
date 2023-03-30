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
#include "infer_manager.hpp"

namespace holoscan {
namespace inference {

ManagerInfer::ManagerInfer() {}

void ManagerInfer::print_dimensions() {
  for (const auto& model_map : models_input_dims_) {
    std::cout << model_map.first << " Input Size: [";
    for (auto& d : model_map.second) std::cout << d << ", ";
    std::cout << "]\n";
  }

  for (const auto& model_map : models_output_dims_) {
    std::cout << model_map.first << " Output Size: [";
    for (auto& d : model_map.second) std::cout << d << ", ";
    std::cout << "]\n";
  }
}

InferStatus ManagerInfer::set_inference_params(std::shared_ptr<MultiAISpecs>& multiai_specs) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  auto multi_model_map = multiai_specs->get_path_map();
  auto backend_type = multiai_specs->backend_type_;
  cuda_buffer_in_ = multiai_specs->cuda_buffer_in_;
  cuda_buffer_out_ = multiai_specs->cuda_buffer_out_;

  if (multi_model_map.size() <= 0) {
    status.set_message("Inference manager, Empty model map for setup");
    return status;
  }

  if (supported_backend_.find(backend_type) == supported_backend_.end()) {
    status.set_message("Inference manager, " + backend_type +
                       " does not exist in inference toolkit");
    return status;
  }
  if (!supported_backend_.at(backend_type)) {
    status.set_message("Inference Manager " + backend_type + " : backend not supported.");
    return status;
  }

  try {
    for (auto& model_map : multi_model_map) {
      if (infer_param_.find(model_map.first) != infer_param_.end()) {
        status.set_message("Duplicate entry in settings for " + model_map.first);
        return status;
      }

      infer_param_.insert({model_map.first, std::make_unique<Params>()});

      infer_param_.at(model_map.first)->set_cuda_flag(multiai_specs->oncuda_);
      infer_param_.at(model_map.first)->set_instance_name(model_map.first);
      infer_param_.at(model_map.first)->set_model_path(model_map.second);

      if (multiai_specs->inference_map_.find(model_map.first) ==
          multiai_specs->inference_map_.end()) {
        status.set_message("Inference Map not found for " + model_map.first);
        return status;
      }
      auto tensor_name = multiai_specs->inference_map_.at(model_map.first);

      inference_map_[model_map.first] = tensor_name;

      if (backend_type.compare("trt") == 0) {
        if (multiai_specs->use_fp16_ && multiai_specs->is_engine_path_) {
          status.set_message(
              "WARNING: Engine files are the input, fp16 check/conversion is ignored");
          status.display_message();
        }
        if (!multiai_specs->oncuda_) {
          status.set_message("WARNING: TRT backend supports infernce on GPU, CPU flag is ignored");
          status.display_message();
        }
        holo_infer_context_.insert({model_map.first,
                                    std::make_unique<TrtInfer>(model_map.second,
                                                               model_map.first,
                                                               multiai_specs->use_fp16_,
                                                               multiai_specs->is_engine_path_,
                                                               cuda_buffer_in_,
                                                               cuda_buffer_out_)});

        std::vector<int64_t> dims = holo_infer_context_.at(model_map.first)->get_output_dims();
        allocate_host_device_buffers(multiai_specs->output_per_model_, dims, tensor_name);
      } else if (backend_type.compare("onnxrt") == 0) {
        if (cuda_buffer_in_ || cuda_buffer_out_) {
          status.set_message(
              "Inference manager, Cuda based in and out buffer not supported in onnxrt");
          return status;
        }
        if (multiai_specs->is_engine_path_) {
          status.set_message(
              "Inference manager, Engine path cannot be true with onnx runtime backend");
          return status;
        }

        bool is_aarch64 = is_platform_aarch64();
        if (is_aarch64 && multiai_specs->oncuda_) {
          status.set_message("Onnxruntime with CUDA not supported on aarch64.");
          return status;
        }

        holo_infer_context_.insert(
            {model_map.first,
             std::make_unique<OnnxInfer>(model_map.second, multiai_specs->oncuda_)});

        std::vector<int64_t> dims = holo_infer_context_.at(model_map.first)->get_output_dims();
        allocate_host_buffers(multiai_specs->output_per_model_, dims, tensor_name);
      } else {
        status.set_message("Inference manager, following backend not supported: " + backend_type);
        return status;
      }
      models_input_dims_.insert(
          {model_map.first, holo_infer_context_[model_map.first]->get_input_dims()});
      models_output_dims_.insert(
          {model_map.first, holo_infer_context_[model_map.first]->get_output_dims()});
    }
  } catch (const std::runtime_error& rt) {
    status.set_message("Inference manager" + std::string(rt.what()));
    return status;
  } catch (...) {
    status.set_message("Inference manager, Exception occurred.");
    return status;
  }
  parallel_processing_ = multiai_specs->parallel_processing_;
  return InferStatus();
}

void ManagerInfer::cleanup() {
  for (auto& infer_map : holo_infer_context_) {
    infer_map.second->cleanup();
    infer_map.second.reset();
  }

  for (auto& infer_p : infer_param_) { infer_p.second.reset(); }
}

InferStatus ManagerInfer::run_core_inference(const std::string& model_name,
                                             DataMap& permodel_preprocess_data,
                                             DataMap& permodel_output_data) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (permodel_preprocess_data.find(model_name) != permodel_preprocess_data.end()) {
    auto indat = permodel_preprocess_data.at(model_name);
    if (permodel_output_data.find(inference_map_.at(model_name)) == permodel_output_data.end()) {
      status.set_message("Infer Manager core, no output data mapping for " + model_name);
      return status;
    }
    auto outdat = permodel_output_data.at(inference_map_.at(model_name));

    if (holo_infer_context_.find(model_name) != holo_infer_context_.end()) {
      auto i_status = holo_infer_context_.at(model_name)->do_inference(indat, outdat);
      if (i_status.get_code() == holoinfer_code::H_ERROR) {
        i_status.display_message();
        status.set_message("Inference manager, Inference failed in core for " + model_name);
        return status;
      }

      return InferStatus();
    }
    status.set_message("Inference manager, Inference context for model " + model_name +
                       " is invalid.");
    return status;
  }
  status.set_message("Inference manager, Preprocessed data for model " + model_name +
                     " does not exist.");
  return status;
}

InferStatus ManagerInfer::execute_inference(DataMap& permodel_preprocess_data,
                                            DataMap& permodel_output_data) {
  InferStatus status = InferStatus();

  std::chrono::steady_clock::time_point s_time;
  std::chrono::steady_clock::time_point e_time;

  std::map<std::string, std::future<InferStatus>> inference_futures;
  s_time = std::chrono::steady_clock::now();
  for (auto& model_param : infer_param_) {
    std::string model_instance{model_param.first};

    if (!parallel_processing_) {
      run_core_inference(model_param.first, permodel_preprocess_data, permodel_output_data);
    } else {
      inference_futures.insert({model_instance,
                                std::async(std::launch::async,
                                           std::bind(&ManagerInfer::run_core_inference,
                                                     this,
                                                     model_param.first,
                                                     permodel_preprocess_data,
                                                     permodel_output_data))});
    }
  }

  if (parallel_processing_) {
    for (auto& inf_fut : inference_futures) {
      InferStatus infer_status = inf_fut.second.get();
      if (infer_status.get_code() != holoinfer_code::H_SUCCESS) {
        status.set_code(holoinfer_code::H_ERROR);
        infer_status.display_message();
        status.set_message("Inference manager, Inference failed in execution for " + inf_fut.first);
        return status;
      }
    }
  }

  e_time = std::chrono::steady_clock::now();
  int64_t current_infer_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time).count();

  status.set_message("Multi Model Inference Latency: " + std::to_string(current_infer_time) +
                     " ms");

  return status;
}

DimType ManagerInfer::get_input_dimensions() const {
  return models_input_dims_;
}

DimType ManagerInfer::get_output_dimensions() const {
  return models_output_dims_;
}

InferContext::InferContext() {
  try {
    manager = std::make_unique<ManagerInfer>();
  } catch (const std::bad_alloc&) { throw; }
}

InferStatus InferContext::execute_inference(DataMap& data_map, DataMap& output_data_map) {
  InferStatus status = InferStatus();
  try {
    if (data_map.size() == 0) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message("Inference manager, Error: Data map empty for inferencing");
      return status;
    }
    status = manager->execute_inference(data_map, output_data_map);
  } catch (...) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Inference manager, Error in inference");
    return status;
  }
  return status;
}

InferStatus InferContext::set_inference_params(std::shared_ptr<MultiAISpecs>& multiai_specs) {
  return manager->set_inference_params(multiai_specs);
}

InferContext::~InferContext() {
  manager->cleanup();
  manager.reset();
}

DimType InferContext::get_output_dimensions() const {
  return manager->get_output_dimensions();
}

}  // namespace inference
}  // namespace holoscan
