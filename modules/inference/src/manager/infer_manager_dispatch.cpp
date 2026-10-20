/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "infer_manager.hpp"

#include <chrono>
#include <climits>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace holoscan {
namespace inference {

InferStatus ManagerInfer::run_core_gr_inference(const std::string& model_name, void* in_buffer,
                                                void* out_buffer, cudaStream_t cuda_stream) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);
  try {
    holo_infer_context_.at(model_name)->do_gr_inference(in_buffer, out_buffer, cuda_stream);
  } catch (const std::runtime_error& e) {
    status.set_message("ERROR: " + std::string(e.what()));
    return status;
  }
  return InferStatus();
}

InferStatus ManagerInfer::run_core_inference(const std::string& model_name,
                                             const DataMap& input_preprocess_data,
                                             const DataMap& output_inferred_data,
                                             cudaStream_t cuda_stream) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  // Find if the current model exists in infer_param_
  if (infer_param_.find(model_name) == infer_param_.end()) {
    status.set_message("Infer Manager core, no parameter mapping for " + model_name);
    return status;
  }

  auto input_tensors = infer_param_.at(model_name)->get_input_tensor_names();
  auto output_tensors = infer_param_.at(model_name)->get_output_tensor_names();

  if (input_tensors.size() == 0 || output_tensors.size() == 0) {
    status.set_message("Infer Manager core, Incorrect input or output tensor size.");
    return status;
  }

  if (holo_infer_context_.find(model_name) == holo_infer_context_.end()) {
    status.set_message("Inference manager, Inference context for model " + model_name +
                       " is invalid.");
    return status;
  }

  // input and output buffer for current inference
  std::vector<std::shared_ptr<DataBuffer>> indata, outdata;

  for (const auto& in_tensor : input_tensors) {
    bool found_in_input = false;
    if (input_preprocess_data.find(in_tensor) != input_preprocess_data.end()) {
      found_in_input = true;
    } else {
      if (output_inferred_data.find(in_tensor) == output_inferred_data.end()) {
        status.set_message("Inference manager, Data for input tensor " + in_tensor +
                           " does not exist.");
        return status;
      }
    }
    auto preprocessed_data =
        found_in_input ? input_preprocess_data.at(in_tensor) : output_inferred_data.at(in_tensor);
    indata.push_back(std::move(preprocessed_data));
  }

  for (const auto& out_tensor : output_tensors) {
    if (output_inferred_data.find(out_tensor) == output_inferred_data.end()) {
      status.set_message("Infer Manager core, no output data mapping for " + out_tensor);
      return status;
    }
    outdata.push_back(output_inferred_data.at(out_tensor));
  }

  check_cuda(cudaEventRecord(cuda_event_, cuda_stream));

  // Record the inference event.
  cudaEvent_t cuda_event_inference = nullptr;

  const InferStatus i_status =
      holo_infer_context_.at(model_name)
          ->do_inference(indata, outdata, cuda_event_, &cuda_event_inference);

  if (i_status.get_code() == holoinfer_code::H_ERROR) {
    i_status.display_message();
    status.set_message("Inference manager, Inference failed in core for " + model_name);
    return status;
  }

  if (cuda_event_inference != nullptr) {
    check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_inference));
  }

  return InferStatus();
}

// Streamlined per-compute() call inference path for canonical InferenceOp.
// Keep each dependency level's submission and future collection together; failures must be
// collected before dependent models run and output dimensions are refreshed.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
InferStatus ManagerInfer::execute_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                            cudaStream_t cuda_stream) {
  InferStatus status = InferStatus();

  const auto& permodel_preprocess_data = inference_specs->data_per_tensor_;
  const auto& permodel_output_data = inference_specs->output_per_model_;

  if (permodel_preprocess_data.size() == 0) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Inference manager, Error: Data map empty for inferencing");
    return status;
  }

  if (frame_counter_++ == UINT_MAX - 1) {
    frame_counter_ = 0;
  }

  if (infer_param_.size() == 0) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(
        "Infer Manager core, inference parameters not set. Maybe setup is incomplete for inference "
        "contexts.");
    return status;
  }

  const bool latency_log_enabled = holoscan::log_level() <= holoscan::LogLevel::DEBUG;
  std::chrono::steady_clock::time_point s_time;
  std::chrono::steady_clock::time_point e_time;
  if (latency_log_enabled) {
    s_time = std::chrono::steady_clock::now();
  }

  for (const auto& level : execution_plan_) {
    // Futures map used by the parallel-dispatch branch below to hold one
    // packaged_task per model in this execution-plan level. Default-constructed
    // empty on every outer-loop iteration; std::map performs no allocation until
    // the first insert(), so sequential runs (which never call insert) pay only
    // the stack cost of the empty container.
    std::map<std::string, std::shared_ptr<std::packaged_task<InferStatus()>>> inference_futures;

    for (const auto& model_instance : level) {
      if (inference_specs->dynamic_input_dims_) {
        auto input_tensors = inference_specs->input_map_.at(model_instance);

        const bool set_dynamic_input =
            holo_infer_context_.at(model_instance)
                ->set_dynamic_input_dimension(input_tensors, inference_specs->dims_per_tensor_);

        if (!set_dynamic_input) {
          HOLOSCAN_LOG_ERROR("Setting up of dynamic input failed for model {}", model_instance);
          status.set_code(holoinfer_code::H_ERROR);
          return status;
        }
      }

      if (!parallel_processing_) {
        const InferStatus infer_status = run_core_inference(
            model_instance, permodel_preprocess_data, permodel_output_data, cuda_stream);
        if (infer_status.get_code() != holoinfer_code::H_SUCCESS) {
          status.set_code(holoinfer_code::H_ERROR);
          infer_status.display_message();
          status.set_message("Inference manager, Inference failed in execution for " +
                             model_instance);
          return status;
        }
      } else {
        size_t idx = 0;
        bool found = false;
        for (; idx < all_model_names_.size(); ++idx) {
          if (all_model_names_[idx] == model_instance) {
            found = true;
            break;
          }
        }

        if (found) {
          inference_futures.insert(
              {model_instance,
               work_queue_->async(std::bind(&ManagerInfer::run_core_inference_indexed,
                                            this,
                                            idx,
                                            permodel_preprocess_data,
                                            permodel_output_data,
                                            cuda_stream))});
        } else {
          inference_futures.insert({model_instance,
                                    work_queue_->async(std::bind(&ManagerInfer::run_core_inference,
                                                                 this,
                                                                 model_instance,
                                                                 permodel_preprocess_data,
                                                                 permodel_output_data,
                                                                 cuda_stream))});
        }
      }
    }

    if (parallel_processing_ && !inference_futures.empty()) {
      std::string failed_models;
      for (auto& inf_fut : inference_futures) {
        const InferStatus infer_status = inf_fut.second->get_future().get();
        if (infer_status.get_code() != holoinfer_code::H_SUCCESS) {
          status.set_code(holoinfer_code::H_ERROR);
          infer_status.display_message();
          failed_models += " " + inf_fut.first;
        }
      }
      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        status.set_message("Inference manager, Inference failed in execution for" + failed_models);
        return status;
      }
    }
  }
  // update output dimensions here for dynamic outputs
  for (const auto& [model_instance, _] : infer_param_) {
    models_output_dims_[model_instance] = holo_infer_context_.at(model_instance)->get_output_dims();
  }

  if (latency_log_enabled) {
    e_time = std::chrono::steady_clock::now();
    const int64_t current_infer_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time).count();
    status.set_message("Inference Latency: " + std::to_string(current_infer_time) + " ms");
  }

  return status;
}

InferStatus ManagerInfer::execute_gr_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                               cudaStream_t cuda_stream) {
  std::chrono::steady_clock::time_point s_time;
  std::chrono::steady_clock::time_point e_time;
  s_time = std::chrono::steady_clock::now();
  for (const auto& [model_instance, _] : infer_param_) {
    try {
      auto in_buffer = inference_specs->gpu_resident_input_;
      auto out_buffer = inference_specs->gpu_resident_output_;
      auto status = run_core_gr_inference(model_instance, in_buffer, out_buffer, cuda_stream);
      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        return status;
      }
    } catch (const std::runtime_error& e) {
      HOLOSCAN_LOG_ERROR("ERROR: " + std::string(e.what()));
      return InferStatus(holoinfer_code::H_ERROR, std::string(e.what()));
    } catch (...) {
      HOLOSCAN_LOG_ERROR("ERROR: Unknown exception occurred in execute_gr_inference.");
      return InferStatus(holoinfer_code::H_ERROR,
                         "Unknown exception occurred in execute_gr_inference.");
    }
  }
  e_time = std::chrono::steady_clock::now();
  const int64_t current_infer_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time).count();

  HOLOSCAN_LOG_DEBUG("First-time Inference Latency for GPU-resident inference: {} ms",
                     std::to_string(current_infer_time));

  return InferStatus();
}

DimType ManagerInfer::get_input_dimensions() const {
  return models_input_dims_;
}

DimType ManagerInfer::get_output_dimensions() const {
  return models_output_dims_;
}

}  // namespace inference
}  // namespace holoscan
