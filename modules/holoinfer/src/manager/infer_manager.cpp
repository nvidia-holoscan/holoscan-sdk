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
#include "infer_manager.hpp"

#include <dlfcn.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

ManagerInfer::ManagerInfer() {}

InferStatus ManagerInfer::set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  auto multi_model_map = inference_specs->get_path_map();
  auto device_map = inference_specs->get_device_map();
  auto backend_type = inference_specs->backend_type_;
  auto backend_map = inference_specs->get_backend_map();
  cuda_buffer_in_ = inference_specs->cuda_buffer_in_;
  cuda_buffer_out_ = inference_specs->cuda_buffer_out_;

  if (multi_model_map.size() <= 0) {
    status.set_message("Inference manager, Empty model map for setup");
    return status;
  }

  if (backend_type.length() != 0 && backend_map.size() != 0) {
    status.set_message(
        "Inference manager, Backend type for all models and Backend map for each models specified. "
        "Either Backend type or Backend map is allowed.");
    return status;
  }

  if (backend_type.length() != 0) {
    if (supported_backend_.find(backend_type) == supported_backend_.end()) {
      status.set_message("Inference manager, " + backend_type +
                         " does not exist in inference toolkit");
      return status;
    }
  } else {
    if (backend_map.size() != 0) {
      for (auto const& [_, backend_] : backend_map) {
        if (supported_backend_.find(backend_) == supported_backend_.end()) {
          status.set_message("Inference manager, " + backend_ +
                             " does not exist in inference toolkit");
          return status;
        }
      }
    } else {
      status.set_message("Inference manager, neither backend or backend map specified");
      return status;
    }
  }

  // set up gpu-dt
  if (device_map.find("gpu-dt") != device_map.end()) {
    auto dev_id = std::stoi(device_map.at("gpu-dt"));
    device_gpu_dt = dev_id;
    HOLOSCAN_LOG_INFO("ID of data transfer GPU updated to: {}", device_gpu_dt);
  }

  std::set<int> unique_gpu_ids;
  unique_gpu_ids.insert(device_gpu_dt);

  for (auto const& [_, gpu_id] : device_map) {
    auto dev_id = std::stoi(gpu_id);
    cudaDeviceProp device_prop;
    auto cstatus = cudaGetDeviceProperties(&device_prop, dev_id);
    if (cstatus != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Error in getting device properties for gpu id: {}.", dev_id);
      HOLOSCAN_LOG_INFO("Use integer id's displayed after the GPU after executing: nvidia-smi -L");
      status.set_message("Incorrect gpu id in the configuration.");
      return status;
    }
    unique_gpu_ids.insert(dev_id);
  }

  auto vec_unique_gpu_ids = std::vector<int>(unique_gpu_ids.begin(), unique_gpu_ids.end());

  if (vec_unique_gpu_ids.size() > 1) {
    for (auto gid = 1; gid < vec_unique_gpu_ids.size(); ++gid) {
      int gpu_access_from_gpudt = 0, gpu_access_to_gpudt = 0;
      check_cuda(
          cudaDeviceCanAccessPeer(&gpu_access_from_gpudt, device_gpu_dt, vec_unique_gpu_ids[gid]));
      check_cuda(
          cudaDeviceCanAccessPeer(&gpu_access_to_gpudt, vec_unique_gpu_ids[gid], device_gpu_dt));

      if (gpu_access_from_gpudt == 1 && gpu_access_to_gpudt == 1) {
        HOLOSCAN_LOG_INFO("Setting GPU P2P access between GPU {} and GPU {}",
                          device_gpu_dt,
                          vec_unique_gpu_ids[gid]);
        check_cuda(cudaSetDevice(device_gpu_dt));
        cudaError_t cstatus = cudaDeviceEnablePeerAccess(vec_unique_gpu_ids[gid], 0);
        if (cstatus != cudaSuccess && cstatus != cudaErrorPeerAccessAlreadyEnabled) {
          HOLOSCAN_LOG_ERROR("Cuda error, {}", cudaGetErrorString(cstatus));
          HOLOSCAN_LOG_ERROR("Error enabling P2P access from GPU {} and GPU {}.",
                             device_gpu_dt,
                             vec_unique_gpu_ids[gid]);
          status.set_message("Enabling P2P access failed.");
          return status;
        }
        check_cuda(cudaSetDevice(vec_unique_gpu_ids[gid]));
        cstatus = cudaDeviceEnablePeerAccess(device_gpu_dt, 0);
        if (cstatus != cudaSuccess && cstatus != cudaErrorPeerAccessAlreadyEnabled) {
          HOLOSCAN_LOG_ERROR("Cuda error, {}", cudaGetErrorString(cstatus));
          HOLOSCAN_LOG_ERROR("Error enabling P2P access from GPU {} and GPU {}.",
                             vec_unique_gpu_ids[gid],
                             device_gpu_dt);
          status.set_message("Enabling P2P access failed.");
          return status;
        }
      } else {
        HOLOSCAN_LOG_WARN("P2P access between GPU {} and GPU {} is not available.",
                          device_gpu_dt,
                          vec_unique_gpu_ids[gid]);
        HOLOSCAN_LOG_INFO(
            "There can be any reason related to GPU type, GPU family or system setup (PCIE "
            "configuration).");
        HOLOSCAN_LOG_INFO("May be GPU {} and GPU {} are not in the same PCIE configuration.",
                          device_gpu_dt,
                          vec_unique_gpu_ids[gid]);
        HOLOSCAN_LOG_WARN(
            "Multi GPU inference feature will use Host (CPU memory) to transfer data across GPUs."
            "This may result in an additional latency.");
        mgpu_p2p_transfer = false;
      }
    }
  }

  try {
    // create inference contexts and memory allocations for each model
    for (auto& [model_name, model_path] : multi_model_map) {
      if (infer_param_.find(model_name) != infer_param_.end()) {
        status.set_message("Duplicate entry in settings for " + model_name);
        return status;
      }

      infer_param_.insert({model_name, std::make_unique<Params>()});

      infer_param_.at(model_name)->set_cuda_flag(inference_specs->oncuda_);
      infer_param_.at(model_name)->set_instance_name(model_name);
      infer_param_.at(model_name)->set_model_path(model_path);

      if (inference_specs->inference_map_.find(model_name) ==
          inference_specs->inference_map_.end()) {
        status.set_message("Inference Map not found for " + model_name);
        return status;
      }

      auto device_id = device_gpu_dt;
      if (device_map.find(model_name) != device_map.end()) {
        device_id = std::stoi(device_map.at(model_name));
        HOLOSCAN_LOG_INFO("Device id: {} for Model: {}", device_id, model_name);
      }

      infer_param_.at(model_name)->set_device_id(device_id);

      // Get input and output tensor maps of the model from inference_specs
      auto out_tensor_names = inference_specs->inference_map_.at(model_name);
      auto in_tensor_names = inference_specs->pre_processor_map_.at(model_name);

      // assign the input and output tensor names to the infer_param object
      infer_param_.at(model_name)->set_tensor_names(in_tensor_names, true);
      infer_param_.at(model_name)->set_tensor_names(out_tensor_names, false);

      check_cuda(cudaSetDevice(device_id));

      auto current_backend = holoinfer_backend::h_trt;
      if (backend_type.length() != 0) { current_backend = supported_backend_.at(backend_type); }

      if (backend_map.size() != 0) {
        if (backend_map.find(model_name) == backend_map.end()) {
          status.set_message("ERROR: Backend not found for model " + model_name);
          return status;
        }
        auto backend_ = backend_map.at(model_name);
        current_backend = supported_backend_.at(backend_);
      }

      switch (current_backend) {
        case holoinfer_backend::h_trt: {
          if (inference_specs->use_fp16_ && inference_specs->is_engine_path_) {
            status.set_message(
                "WARNING: Engine files are the input, fp16 check/conversion is ignored");
            status.display_message();
          }
          if (!inference_specs->oncuda_) {
            status.set_message("ERROR: TRT backend supports inference on GPU only");
            return status;
          }

          holo_infer_context_.insert({model_name,
                                      std::make_unique<TrtInfer>(model_path,
                                                                 model_name,
                                                                 device_id,
                                                                 inference_specs->use_fp16_,
                                                                 inference_specs->is_engine_path_,
                                                                 cuda_buffer_in_,
                                                                 cuda_buffer_out_)});
          break;
        }

        case holoinfer_backend::h_onnx: {
          if (cuda_buffer_in_ || cuda_buffer_out_) {
            status.set_message(
                "Inference manager, Cuda based in and out buffer not supported in onnxrt");
            return status;
          }
          if (inference_specs->is_engine_path_) {
            status.set_message(
                "Inference manager, Engine path cannot be true with onnx runtime backend");
            return status;
          }

          if (std::filesystem::path(model_path).extension() != ".onnx") {
            HOLOSCAN_LOG_ERROR("Onnx model must be in .onnx format.");
            status.set_message("Inference manager, model path must have .onnx extension.");
            return status;
          }

          bool is_aarch64 = is_platform_aarch64();
          if (is_aarch64 && inference_specs->oncuda_) {
            status.set_message("Onnxruntime with CUDA not supported on aarch64.");
            return status;
          }

#if use_onnxruntime
          HOLOSCAN_LOG_INFO("Searching for ONNX Runtime libraries");
          void* handle = dlopen("libholoscan_infer_onnx_runtime.so", RTLD_NOW);
          if (handle == nullptr) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("ONNX Runtime context setup failure.");
            return status;
          }
          HOLOSCAN_LOG_INFO("Found ONNX Runtime libraries");
          using NewOnnxInfer = OnnxInfer* (*)(const std::string&, bool);
          auto new_ort_infer = reinterpret_cast<NewOnnxInfer>(dlsym(handle, "NewOnnxInfer"));
          if (!new_ort_infer) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("ONNX Runtime context setup failure.");
            return status;
          }
          dlclose(handle);
          auto context = new_ort_infer(model_path, inference_specs->oncuda_);
          holo_infer_context_[model_name] = std::unique_ptr<OnnxInfer>(context);
#else
          HOLOSCAN_LOG_ERROR("Onnxruntime backend not supported or incorrectly installed.");
          status.set_message("Onnxruntime context setup failure.");
          return status;
#endif
          break;
        }

        case holoinfer_backend::h_torch: {
          if (std::filesystem::path(model_path).extension() != ".pt" &&
              std::filesystem::path(model_path).extension() != ".pth") {
            HOLOSCAN_LOG_ERROR("Torch model must be in torchsript format (.pt or .pth).");
            status.set_message("Inference manager, model path must have .pt or .pth extension.");
            return status;
          }
#if use_torch
          HOLOSCAN_LOG_INFO("Searching for libtorch libraries");
          void* handle = dlopen("libholoscan_infer_torch.so", RTLD_NOW);
          if (handle == nullptr) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("Torch context setup failure.");
            return status;
          }
          HOLOSCAN_LOG_INFO("Found libtorch libraries");
          using NewTorchInfer = TorchInfer* (*)(const std::string&, bool, bool, bool);
          auto new_torch_infer = reinterpret_cast<NewTorchInfer>(dlsym(handle, "NewTorchInfer"));
          if (!new_torch_infer) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("Torch context setup failure.");
            return status;
          }
          dlclose(handle);
          auto context = new_torch_infer(
              model_path, inference_specs->oncuda_, cuda_buffer_in_, cuda_buffer_out_);
          holo_infer_context_[model_name] = std::unique_ptr<TorchInfer>(context);
#else
          HOLOSCAN_LOG_ERROR("Torch backend not supported.");
          status.set_message("Torch context setup failure.");
          return status;
#endif
          break;
        }
        default: {
          status.set_message("ERROR: Backend not supported");
          return status;
        }
      }
      check_cuda(cudaSetDevice(device_gpu_dt));

      auto output_node_size = holo_infer_context_.at(model_name)->get_output_dims().size();
      auto input_node_size = holo_infer_context_.at(model_name)->get_input_dims().size();

      if (out_tensor_names.size() != output_node_size) {
        HOLOSCAN_LOG_ERROR("Size mismatch. Out tensor names: {}, Output node size: {}",
                           out_tensor_names.size(),
                           output_node_size);
        status.set_message("Output tensor size not equal to output nodes in model.");
        return status;
      }

      if (in_tensor_names.size() != input_node_size) {
        status.set_message("Input tensor size not equal to input nodes in model.");
        return status;
      }

      // Allocate output buffers for multi-gpu inference use-case
      // dm will contain databuffer for all tensors for the current model, dm will be populated only
      // if the GPU for inference is not same as GPU-dt
      DataMap dm;

      // in and out buffer on cuda not supported for onnxrt.
      bool allocate_cuda = (backend_type.compare("onnxrt") == 0) ? false : true;

      for (unsigned int d = 0; d < out_tensor_names.size(); d++) {
        std::vector<int64_t> dims = holo_infer_context_.at(model_name)->get_output_dims()[d];
        auto datatype = holo_infer_context_.at(model_name)->get_output_datatype()[d];
        if (datatype == holoinfer_datatype::h_Unsupported) {
          status.set_message("Unsupported datatype for tensor" + out_tensor_names[d]);
          return status;
        }

        auto astatus = allocate_buffers(inference_specs->output_per_model_,
                                        dims,
                                        datatype,
                                        out_tensor_names[d],
                                        allocate_cuda,
                                        device_id);
        if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
          astatus.display_message();
          status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
          return status;
        }
        HOLOSCAN_LOG_INFO("HoloInfer buffer created for {}", out_tensor_names[d]);

        if (device_id != device_gpu_dt) {
          check_cuda(cudaSetDevice(device_id));

          auto astatus =
              allocate_buffers(dm, dims, datatype, out_tensor_names[d], allocate_cuda, device_id);
          if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
            astatus.display_message();
            status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
            return status;
          }

          check_cuda(cudaSetDevice(device_gpu_dt));
        }
      }
      mgpu_output_buffer_.insert({model_name, std::move(dm)});

      if (device_id != device_gpu_dt) {
        // For Multi-GPU feature: allocate input and output cuda streams
        check_cuda(cudaSetDevice(device_gpu_dt));
        std::vector<cudaStream_t> in_streams_gpudt(in_tensor_names.size());
        std::map<std::string, cudaStream_t> in_streams_map_gpudt, out_streams_map_gpudt;

        // cuda stream creation per tensor and populating input_streams_gpudt map
        for (auto in = 0; in < in_tensor_names.size(); in++) {
          check_cuda(cudaStreamCreate(&in_streams_gpudt[in]));
          in_streams_map_gpudt.insert({in_tensor_names[in], in_streams_gpudt[in]});
        }
        input_streams_gpudt.insert({model_name, std::move(in_streams_map_gpudt)});

        std::vector<cudaStream_t> out_streams_gpudt(out_tensor_names.size());
        // cuda stream creation per output tensor and populating out_streams_map_gpudt map
        for (auto out = 0; out < out_tensor_names.size(); out++) {
          check_cuda(cudaStreamCreate(&out_streams_gpudt[out]));
          out_streams_map_gpudt.insert({out_tensor_names[out], out_streams_gpudt[out]});
        }
        output_streams_gpudt.insert({model_name, std::move(out_streams_map_gpudt)});

        check_cuda(cudaSetDevice(device_id));
        std::vector<cudaStream_t> in_streams_dev(in_tensor_names.size());
        std::map<std::string, cudaStream_t> in_streams_map_dev, out_streams_map_dev;

        // cuda stream creation per tensor and populating in_streams_map_dev
        for (auto in = 0; in < in_tensor_names.size(); in++) {
          check_cuda(cudaStreamCreate(&in_streams_dev[in]));
          in_streams_map_dev.insert({in_tensor_names[in], in_streams_dev[in]});
        }
        input_streams_device.insert({model_name, std::move(in_streams_map_dev)});

        std::vector<cudaStream_t> out_streams(out_tensor_names.size());

        // cuda stream creation per output tensor and populating output_streams map
        for (auto out = 0; out < out_tensor_names.size(); out++) {
          check_cuda(cudaStreamCreate(&out_streams[out]));
          out_streams_map_dev.insert({out_tensor_names[out], out_streams[out]});
        }

        output_streams_device.insert({model_name, std::move(out_streams_map_dev)});
        // stream allocation ends

        // allocate input buffers only for multi-gpu inference use case for allocation on GPUs other
        // than GPU-dt. Allocation on GPU-dt happens during data extraction from the incoming
        // messages
        DataMap dm_in;

        for (unsigned int d = 0; d < in_tensor_names.size(); d++) {
          std::vector<int64_t> dims = holo_infer_context_.at(model_name)->get_input_dims()[d];
          auto datatype = holo_infer_context_.at(model_name)->get_input_datatype()[d];
          if (datatype == holoinfer_datatype::h_Unsupported) {
            status.set_message("Unsupported datatype for tensor" + in_tensor_names[d]);
            return status;
          }

          auto astatus =
              allocate_buffers(dm_in, dims, datatype, in_tensor_names[d], allocate_cuda, device_id);
          if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
            astatus.display_message();
            status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
            return status;
          }
        }
        mgpu_input_buffer_.insert({model_name, std::move(dm_in)});
        check_cuda(cudaSetDevice(device_gpu_dt));
      }

      models_input_dims_.insert({model_name, holo_infer_context_.at(model_name)->get_input_dims()});
    }
  } catch (const std::runtime_error& rt) {
    raise_error("Inference Manager", "Setting Inference parameters: " + std::string(rt.what()));
  } catch (...) {
    raise_error("Inference Manager", "Setting Inference parameters: unknown exception occurred.");
  }

  parallel_processing_ = inference_specs->parallel_processing_;
  return InferStatus();
}

void ManagerInfer::cleanup() {
  for (auto& [_, context] : holo_infer_context_) {
    context->cleanup();
    context.reset();
  }

  for (auto& [_, infer_p] : infer_param_) { infer_p.reset(); }
}

ManagerInfer::~ManagerInfer() {
  cleanup();
}

InferStatus ManagerInfer::run_core_inference(const std::string& model_name,
                                             DataMap& input_preprocess_data,
                                             DataMap& output_inferred_data) {
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

  auto device_id = infer_param_.at(model_name)->get_device_id();
  check_cuda(cudaSetDevice(device_id));

  // input and output buffer for current inference
  std::vector<std::shared_ptr<DataBuffer>> indata, outdata;

  DataMap in_preprocess_data;
  if (device_id != device_gpu_dt) {
    if (mgpu_input_buffer_.find(model_name) == mgpu_input_buffer_.end()) {
      HOLOSCAN_LOG_ERROR("Mapping for model {} not found on device {}.", model_name, device_id);
      status.set_message("Inference manager, Mapping not found for " + model_name +
                         " in multi gpu inference.");
      return status;
    }
    in_preprocess_data = mgpu_input_buffer_.at(model_name);
  }

  for (const auto& in_tensor : input_tensors) {
    if (input_preprocess_data.find(in_tensor) == input_preprocess_data.end()) {
      status.set_message("Inference manager, Preprocessed data for tensor " + in_tensor +
                         " does not exist.");
      return status;
    }

    //  by default memory mapped for all backends
    if (device_id != device_gpu_dt) {
      check_cuda(cudaSetDevice(device_id));
      auto device_buff = in_preprocess_data.at(in_tensor)->device_buffer->data();
      auto buffsize = in_preprocess_data.at(in_tensor)->device_buffer->get_bytes();

      check_cuda(cudaSetDevice(device_gpu_dt));
      auto in_streams_gpudt = input_streams_gpudt.at(model_name);

      auto device_gpu_dt_buff_in = input_preprocess_data.at(in_tensor)->device_buffer->data();
      auto stream = in_streams_gpudt.at(in_tensor);
      if (mgpu_p2p_transfer) {
        check_cuda(cudaMemcpyPeerAsync(
            device_buff, device_id, device_gpu_dt_buff_in, device_gpu_dt, buffsize, stream));
      } else {
        // transfer from gpu-dt to host
        auto host_buff_in = input_preprocess_data.at(in_tensor)->host_buffer.data();
        check_cuda(cudaMemcpyAsync(
            host_buff_in, device_gpu_dt_buff_in, buffsize, cudaMemcpyDeviceToHost, stream));
      }
    } else {
      indata.push_back(input_preprocess_data.at(in_tensor));
    }
  }

  if (device_id != device_gpu_dt) {
    check_cuda(cudaSetDevice(device_gpu_dt));
    auto in_streams_gpudt = input_streams_gpudt.at(model_name);

    for (auto& [_, stream] : in_streams_gpudt) { check_cuda(cudaStreamSynchronize(stream)); }

    // If P2P is disabled, transfer data from host to device_id
    if (!mgpu_p2p_transfer) {
      check_cuda(cudaSetDevice(device_id));

      // transfer from host to device_id
      auto input_streams_dev = input_streams_device.at(model_name);
      for (const auto& in_tensor : input_tensors) {
        auto device_buff = in_preprocess_data.at(in_tensor)->device_buffer->data();
        auto host_buff_in = input_preprocess_data.at(in_tensor)->host_buffer.data();
        auto buffsize = in_preprocess_data.at(in_tensor)->device_buffer->get_bytes();
        auto dstream = input_streams_dev.at(in_tensor);

        check_cuda(
            cudaMemcpyAsync(device_buff, host_buff_in, buffsize, cudaMemcpyHostToDevice, dstream));
      }

      for (auto& [_, dstream] : input_streams_dev) { check_cuda(cudaStreamSynchronize(dstream)); }
    }

    for (const auto& in_tensor : input_tensors) {
      indata.push_back(in_preprocess_data.at(in_tensor));
    }
  }

  for (const auto& out_tensor : output_tensors) {
    if (output_inferred_data.find(out_tensor) == output_inferred_data.end()) {
      status.set_message("Infer Manager core, no output data mapping for " + out_tensor);
      return status;
    }

    if (device_id != device_gpu_dt) {
      check_cuda(cudaSetDevice(device_id));
      auto out_inferred_data = mgpu_output_buffer_.at(model_name);
      outdata.push_back(out_inferred_data.at(out_tensor));
    } else {
      outdata.push_back(output_inferred_data.at(out_tensor));
    }
  }

  check_cuda(cudaSetDevice(device_id));
  auto i_status = holo_infer_context_.at(model_name)->do_inference(indata, outdata);

  if (i_status.get_code() == holoinfer_code::H_ERROR) {
    i_status.display_message();
    status.set_message("Inference manager, Inference failed in core for " + model_name);
    return status;
  }

  // Output data setup after inference
  // by default memory mapped for all backends
  if (device_id != device_gpu_dt && cuda_buffer_out_) {
    auto out_inferred_data = mgpu_output_buffer_.at(model_name);
    auto out_streams = output_streams_device.at(model_name);

    for (auto& out_tensor : output_tensors) {
      check_cuda(cudaSetDevice(device_id));
      auto buffsize = out_inferred_data.at(out_tensor)->device_buffer->get_bytes();

      check_cuda(cudaSetDevice(device_gpu_dt));
      auto buffer_size_gpu_dt = output_inferred_data.at(out_tensor)->device_buffer->get_bytes();
      if (buffer_size_gpu_dt != buffsize) {
        output_inferred_data.at(out_tensor)->device_buffer->resize(buffsize);
      }
      auto device_gpu_dt_buff = output_inferred_data.at(out_tensor)->device_buffer->data();

      check_cuda(cudaSetDevice(device_id));
      auto device_buff = out_inferred_data.at(out_tensor)->device_buffer->data();
      buffsize = out_inferred_data.at(out_tensor)->device_buffer->get_bytes();

      auto stream = out_streams.at(out_tensor);
      if (mgpu_p2p_transfer) {
        check_cuda(cudaMemcpyPeerAsync(
            device_gpu_dt_buff, device_gpu_dt, device_buff, device_id, buffsize, stream));
      } else {
        // transfer from device to host
        auto host_buff_out = out_inferred_data.at(out_tensor)->host_buffer.data();
        check_cuda(
            cudaMemcpyAsync(host_buff_out, device_buff, buffsize, cudaMemcpyDeviceToHost, stream));
      }
    }

    for (auto& [_, stream] : out_streams) { check_cuda(cudaStreamSynchronize(stream)); }

    // if p2p is disabled, then move the data from host to gpu-dt
    if (!mgpu_p2p_transfer) {
      check_cuda(cudaSetDevice(device_gpu_dt));
      auto out_streams_gpudt = output_streams_gpudt.at(model_name);

      // transfer from host to gpu-dt
      for (auto& out_tensor : output_tensors) {
        auto device_gpu_dt_buff = output_inferred_data.at(out_tensor)->device_buffer->data();
        auto host_buff_out = out_inferred_data.at(out_tensor)->host_buffer.data();
        auto buffsize = output_inferred_data.at(out_tensor)->device_buffer->get_bytes();
        auto stream = out_streams_gpudt.at(out_tensor);

        check_cuda(cudaMemcpyAsync(
            device_gpu_dt_buff, host_buff_out, buffsize, cudaMemcpyHostToDevice, stream));
      }

      for (auto& [_, stream] : out_streams_gpudt) { check_cuda(cudaStreamSynchronize(stream)); }
    }
  }

  check_cuda(cudaSetDevice(device_gpu_dt));
  return InferStatus();
}

InferStatus ManagerInfer::execute_inference(DataMap& permodel_preprocess_data,
                                            DataMap& permodel_output_data) {
  InferStatus status = InferStatus();

  if (infer_param_.size() == 0) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(
        "Infer Manager core, inference parameters not set. Maybe setup is incomplete for inference "
        "contexts.");
    return status;
  }

  std::chrono::steady_clock::time_point s_time;
  std::chrono::steady_clock::time_point e_time;

  std::map<std::string, std::future<InferStatus>> inference_futures;
  s_time = std::chrono::steady_clock::now();
  for (const auto& [model_instance, _] : infer_param_) {
    if (!parallel_processing_) {
      InferStatus infer_status =
          run_core_inference(model_instance, permodel_preprocess_data, permodel_output_data);
      if (infer_status.get_code() != holoinfer_code::H_SUCCESS) {
        status.set_code(holoinfer_code::H_ERROR);
        infer_status.display_message();
        status.set_message("Inference manager, Inference failed in execution for " +
                           model_instance);
        return status;
      }
    } else {
      inference_futures.insert({model_instance,
                                std::async(std::launch::async,
                                           std::bind(&ManagerInfer::run_core_inference,
                                                     this,
                                                     model_instance,
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

  // update output dimensions here for dynamic outputs
  for (const auto& [model_instance, _] : infer_param_) {
    models_output_dims_[model_instance] = holo_infer_context_.at(model_instance)->get_output_dims();
  }
  e_time = std::chrono::steady_clock::now();
  int64_t current_infer_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(e_time - s_time).count();

  status.set_message("Inference Latency: " + std::to_string(current_infer_time) + " ms");

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
    if (g_managers.find("current_manager") != g_managers.end()) {
      HOLOSCAN_LOG_WARN("Inference context exists, cleaning up");
      g_managers.at("current_manager").reset();
      g_managers.erase("current_manager");
    }
    g_managers.insert({"current_manager", std::make_shared<ManagerInfer>()});
  } catch (const std::bad_alloc&) { throw; }
}

InferStatus InferContext::execute_inference(DataMap& data_map, DataMap& output_data_map) {
  InferStatus status = InferStatus();

  if (g_managers.find(unique_id_) == g_managers.end()) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Inference manager, Error: Inference manager not created or is not set up.");
    return status;
  }

  try {
    g_manager = g_managers.at(unique_id_);

    if (data_map.size() == 0) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message("Inference manager, Error: Data map empty for inferencing");
      return status;
    }
    status = g_manager->execute_inference(data_map, output_data_map);
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("Inference manager, Error in inference setup: ") + e.what());
    return status;
  }

  return status;
}

InferStatus InferContext::set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs) {
  InferStatus status = InferStatus();
  if (g_managers.size() == 0) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Inference manager, Error: Inference Manager not initiated");
    return status;
  }

  try {
    auto multi_model_map = inference_specs->get_path_map();

    if (multi_model_map.size() == 0) {
      if (g_managers.find("current_manager") != g_managers.end()) {
        g_managers.at("current_manager").reset();
        g_managers.erase("current_manager");
      }

      status.set_code(holoinfer_code::H_ERROR);
      status.set_message("Inference manager, Error: Multi modal map cannot be empty in setup.");
      return status;
    }

    std::string unique_id_name("");
    for (auto& [model_name, _] : multi_model_map) { unique_id_name += model_name + "_[]_"; }

    unique_id_ = unique_id_name;
    HOLOSCAN_LOG_INFO("Inference context ID: {}", unique_id_);

    if (g_managers.find(unique_id_name) != g_managers.end()) {
      if (g_managers.find("current_manager") != g_managers.end()) {
        g_managers.erase("current_manager");
      }

      status.set_code(holoinfer_code::H_ERROR);
      status.set_message(
          "Inference manager, Error: A manager with the same unique ID already exists.");
      HOLOSCAN_LOG_ERROR(
          "Inference manager setup error: model keywords are repeated in multiple instances of "
          "inference. All model instances must have unique keyword in the configuration file.");
      return status;
    }

    if (g_managers.find("current_manager") == g_managers.end()) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message("Inference manager, Error: Current Manager not initialized.");
      HOLOSCAN_LOG_ERROR("Inference manager setup error: Inference context not initialized.");
      return status;
    }

    g_managers.insert({unique_id_name, std::move(g_managers.at("current_manager"))});
    g_managers.erase("current_manager");

    g_manager = g_managers.at(unique_id_);
    status = g_manager->set_inference_params(inference_specs);
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("Inference manager, Error in inference setup: ") + e.what());
    return status;
  }

  return status;
}

InferContext::~InferContext() {
  if (g_managers.find(unique_id_) != g_managers.end()) {
    g_manager = g_managers.at(unique_id_);
    g_manager.reset();
    g_managers.erase(unique_id_);
  }
}

DimType InferContext::get_output_dimensions() const {
  g_manager = g_managers.at(unique_id_);
  return g_manager->get_output_dimensions();
}

}  // namespace inference
}  // namespace holoscan
