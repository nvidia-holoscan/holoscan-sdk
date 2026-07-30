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

#include <sys/sysinfo.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
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

  const auto device_id = infer_param_.at(model_name)->get_device_id();

  // input and output buffer for current inference
  std::vector<std::shared_ptr<DataBuffer>> indata, outdata;

  for (const auto& in_tensor : input_tensors) {
    if (input_preprocess_data.find(in_tensor) == input_preprocess_data.end()) {
      // if the input tensor is not present in the input_preprocess_data, it may be an
      // internal tensor that is part of output_inferred_data. This is the case of model
      // dependencies where the output of one model is the input to another model.
      if (output_inferred_data.find(in_tensor) == output_inferred_data.end()) {
        status.set_message("Inference manager, Data for input tensor " + in_tensor +
                           " does not exist.");
        return status;
      }
    }
  }

  // Transfer memory from data transfer GPU to inference device. This is using a separate stream
  // for each tensor and synchronizes the copies with the CUDA stream passed in as a parameter.
  if (device_id != device_gpu_dt_) {
    if (mgpu_input_buffer_.find(model_name) == mgpu_input_buffer_.end()) {
      HOLOSCAN_LOG_ERROR("Mapping for model {} not found on device {}.", model_name, device_id);
      status.set_message("Inference manager, Mapping not found for " + model_name +
                         " in multi gpu inference.");
      return status;
    }

    // [PERF T1.2] The multi-GPU input-transfer branch already guarantees device_id !=
    // device_gpu_dt_ (we're inside that conditional), so the cudaSetDevice is required here.
    // Documented for clarity — single-GPU paths skip the equivalent call (see below).
    check_cuda(cudaSetDevice(device_gpu_dt_));

    const DataMap& in_preprocess_data = mgpu_input_buffer_.at(model_name);

    const auto& input_streams_dev = input_streams_device_.at(model_name);
    const auto& in_streams_gpudt = input_streams_gpudt_.at(model_name);
    const cudaEvent_t cuda_event_d = mgpu_cuda_event_.at(model_name).at(device_id);
    const cudaEvent_t cuda_event_dt = mgpu_cuda_event_.at(model_name).at(device_gpu_dt_);

    for (const auto& in_tensor : input_tensors) {
      bool found_in_input = false;
      if (in_preprocess_data.find(in_tensor) != in_preprocess_data.end()) {
        found_in_input = true;
      } else {
        if (output_inferred_data.find(in_tensor) == output_inferred_data.end()) {
          status.set_message("Inference manager, Data for input tensor " + in_tensor +
                             " does not exist.");
          return status;
        }
      }

      auto preprocessed_data =
          found_in_input ? in_preprocess_data.at(in_tensor) : output_inferred_data.at(in_tensor);

      const auto device_buff = preprocessed_data->device_buffer_->data();
      const auto buffsize = preprocessed_data->device_buffer_->get_bytes();

      const auto device_gpu_dt_buff_in = preprocessed_data->device_buffer_->data();

      const cudaStream_t stream_d = input_streams_dev.at(in_tensor);
      const cudaStream_t stream_dt = in_streams_gpudt.at(in_tensor);
      check_cuda(cudaEventRecord(cuda_event_dt, cuda_stream));
      check_cuda(cudaStreamWaitEvent(stream_dt, cuda_event_dt));

      if (mgpu_p2p_transfer_) {
        // direct p2p transfer
        check_cuda(cudaMemcpyPeerAsync(
            device_buff, device_id, device_gpu_dt_buff_in, device_gpu_dt_, buffsize, stream_dt));
        check_cuda(cudaEventRecord(cuda_event_dt, stream_dt));
        check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_dt));
      } else {
        // transfer from gpu-dt to host
        /// @todo check if using pinned memory is faster
        input_preprocess_data.at(in_tensor)->host_buffer_->resize(buffsize);
        auto host_buff_in = input_preprocess_data.at(in_tensor)->host_buffer_->data();
        check_cuda(cudaMemcpyAsync(
            host_buff_in, device_gpu_dt_buff_in, buffsize, cudaMemcpyDeviceToHost, stream_dt));
        check_cuda(cudaEventRecord(cuda_event_dt, stream_dt));

        // transfer from host to device_id
        check_cuda(cudaSetDevice(device_id));
        check_cuda(cudaStreamWaitEvent(stream_d, cuda_event_dt));
        check_cuda(
            cudaMemcpyAsync(device_buff, host_buff_in, buffsize, cudaMemcpyHostToDevice, stream_d));
        check_cuda(cudaEventRecord(cuda_event_d, stream_d));
        check_cuda(cudaSetDevice(device_gpu_dt_));
        check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_d));
      }

      indata.push_back(preprocessed_data);
    }
  } else {
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
  }

  for (const auto& out_tensor : output_tensors) {
    if (output_inferred_data.find(out_tensor) == output_inferred_data.end()) {
      status.set_message("Infer Manager core, no output data mapping for " + out_tensor);
      return status;
    }

    if (device_id != device_gpu_dt_) {
      const DataMap& out_inferred_data = mgpu_output_buffer_.at(model_name);
      outdata.push_back(out_inferred_data.at(out_tensor));
    } else {
      outdata.push_back(output_inferred_data.at(out_tensor));
    }
  }

  check_cuda(cudaEventRecord(cuda_event_, cuda_stream));

  // Only switch CUDA device when the inference device differs from the data-transfer
  // device.
  const bool device_switch_needed = (device_id != device_gpu_dt_);
  if (device_switch_needed) {
    check_cuda(cudaSetDevice(device_id));
  }

  // Record the inference event.
  cudaEvent_t cuda_event_inference = nullptr;

  InferStatus i_status = holo_infer_context_.at(model_name)
                             ->do_inference(indata, outdata, cuda_event_, &cuda_event_inference);

  if (device_switch_needed) {
    check_cuda(cudaSetDevice(device_gpu_dt_));
  }

  if (i_status.get_code() == holoinfer_code::H_ERROR) {
    i_status.display_message();
    status.set_message("Inference manager, Inference failed in core for " + model_name);
    return status;
  }

  if (cuda_event_inference) {
    check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_inference));
  }

  // Output data setup after inference
  // by default memory mapped for all backends
  if ((device_id != device_gpu_dt_) && cuda_buffer_out_) {
    const DataMap& out_inferred_data = mgpu_output_buffer_.at(model_name);
    const auto& out_streams = output_streams_device_.at(model_name);
    const auto& out_streams_gpudt = output_streams_gpudt_.at(model_name);
    const cudaEvent_t cuda_event_d = mgpu_cuda_event_.at(model_name).at(device_id);
    const cudaEvent_t cuda_event_dt = mgpu_cuda_event_.at(model_name).at(device_gpu_dt_);

    for (auto& out_tensor : output_tensors) {
      auto buffsize = out_inferred_data.at(out_tensor)->device_buffer_->get_bytes();

      auto buffer_size_gpu_dt = output_inferred_data.at(out_tensor)->device_buffer_->get_bytes();
      if (buffer_size_gpu_dt != buffsize) {
        output_inferred_data.at(out_tensor)->device_buffer_->resize(buffsize);
      }
      auto device_gpu_dt_buff = output_inferred_data.at(out_tensor)->device_buffer_->data();

      auto device_buff = out_inferred_data.at(out_tensor)->device_buffer_->data();
      buffsize = out_inferred_data.at(out_tensor)->device_buffer_->get_bytes();

      const cudaStream_t stream_d = out_streams.at(out_tensor);
      const cudaStream_t stream_dt = out_streams_gpudt.at(out_tensor);
      check_cuda(cudaEventRecord(cuda_event_dt, cuda_stream));
      if (mgpu_p2p_transfer_) {
        // direct p2p transfer
        check_cuda(cudaStreamWaitEvent(stream_dt, cuda_event_dt));
        check_cuda(cudaMemcpyPeerAsync(
            device_gpu_dt_buff, device_gpu_dt_, device_buff, device_id, buffsize, stream_dt));
        check_cuda(cudaEventRecord(cuda_event_dt, stream_dt));
        check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_dt));
      } else {
        // transfer from device to host
        /// @todo check if using pinned memory is faster
        out_inferred_data.at(out_tensor)->host_buffer_->resize(buffsize);
        auto host_buff_out = out_inferred_data.at(out_tensor)->host_buffer_->data();
        check_cuda(cudaSetDevice(device_id));
        check_cuda(cudaStreamWaitEvent(stream_d, cuda_event_dt));
        check_cuda(cudaMemcpyAsync(
            host_buff_out, device_buff, buffsize, cudaMemcpyDeviceToHost, stream_d));
        check_cuda(cudaEventRecord(cuda_event_d, stream_d));

        // transfer from host to gpu-dt
        check_cuda(cudaSetDevice(device_gpu_dt_));
        check_cuda(cudaStreamWaitEvent(stream_dt, cuda_event_d));
        check_cuda(cudaMemcpyAsync(
            device_buff, host_buff_out, buffsize, cudaMemcpyHostToDevice, stream_dt));
        check_cuda(cudaEventRecord(cuda_event_dt, stream_dt));
        check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_dt));
      }
    }
  }

  return InferStatus();
}

// Streamlined per-compute() call inference path for canonical InferenceOp.
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

  auto activation_map = inference_specs->get_activation_map();

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
      bool process_model = true;

      if (inference_specs->dynamic_input_dims_) {
        auto input_tensors = inference_specs->pre_processor_map_.at(model_instance);

        bool set_dynamic_input =
            holo_infer_context_.at(model_instance)
                ->set_dynamic_input_dimension(input_tensors, inference_specs->dims_per_tensor_);

        if (!set_dynamic_input) {
          HOLOSCAN_LOG_ERROR("Setting up of dynamic input failed for model {}", model_instance);
          status.set_code(holoinfer_code::H_ERROR);
          return status;
        }
      }

      if (activation_map.find(model_instance) != activation_map.end()) {
        try {
          auto activation_value = std::stoul(activation_map.at(model_instance));
          HOLOSCAN_LOG_DEBUG(
              "Activation value: {} for Model: {}", activation_value, model_instance);
          if (activation_value > 1) {
            HOLOSCAN_LOG_WARN("Activation map can have either a value of 0 or 1 for a model.");
            HOLOSCAN_LOG_WARN("Activation map value is ignored for model {}", model_instance);
          }
          if (activation_value == 0) {
            process_model = false;
          }
        } catch (std::invalid_argument const& ex) {
          HOLOSCAN_LOG_WARN("Invalid argument in activation map: {}", ex.what());
          HOLOSCAN_LOG_WARN("Activation map value is ignored for model {}", model_instance);
        } catch (std::out_of_range const& ex) {
          HOLOSCAN_LOG_WARN("Invalid range in activation map: {}", ex.what());
          HOLOSCAN_LOG_WARN("Activation map value is ignored for model {}", model_instance);
        }
      }

      auto temporal_id = infer_param_.at(model_instance)->get_temporal_id();
      if (process_model && (frame_counter_ % temporal_id == 0)) {
        if (!parallel_processing_) {
          InferStatus infer_status = run_core_inference(
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

          const auto model_device_id = infer_param_.at(model_instance)->get_device_id();
          const bool on_gpu_dt = (model_device_id == device_gpu_dt_);
          if (found && on_gpu_dt) {
            inference_futures.insert(
                {model_instance,
                 work_queue_->async(std::bind(&ManagerInfer::run_core_inference_indexed,
                                              this,
                                              idx,
                                              permodel_preprocess_data,
                                              permodel_output_data,
                                              cuda_stream))});
          } else {
            inference_futures.insert(
                {model_instance,
                 work_queue_->async(std::bind(&ManagerInfer::run_core_inference,
                                              this,
                                              model_instance,
                                              permodel_preprocess_data,
                                              permodel_output_data,
                                              cuda_stream))});
          }
        }
      }
    }

    if (parallel_processing_ && !inference_futures.empty()) {
      std::string failed_models;
      for (auto& inf_fut : inference_futures) {
        InferStatus infer_status = inf_fut.second->get_future().get();
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
    int64_t current_infer_time =
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
  int64_t current_infer_time =
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
