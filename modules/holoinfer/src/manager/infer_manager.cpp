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
#include "infer_manager.hpp"

#include <dlfcn.h>
#include <sys/sysinfo.h>

#include <algorithm>
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
  auto dla_core_map = inference_specs->get_dla_core_map();
  auto temporal_map = inference_specs->get_temporal_map();
  auto backend_type = inference_specs->backend_type_;
  auto backend_map = inference_specs->get_backend_map();
  auto trt_opt_profile = inference_specs->trt_opt_profile_;
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
  std::set<int> unique_gpu_ids;

  try {
    if (device_map.find("gpu-dt") != device_map.end()) {
      auto dev_id = std::stoi(device_map.at("gpu-dt"));
      device_gpu_dt_ = dev_id;
      HOLOSCAN_LOG_INFO("ID of data transfer GPU updated to: {}", device_gpu_dt_);
    }

    unique_gpu_ids.insert(device_gpu_dt_);

    for (auto const& [_, gpu_id] : device_map) {
      auto dev_id = std::stoi(gpu_id);
      cudaDeviceProp device_prop;
      auto cstatus = cudaGetDeviceProperties(&device_prop, dev_id);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Error in getting device properties for gpu id: {}.", dev_id);
        HOLOSCAN_LOG_INFO(
            "Use integer id's displayed after the GPU after executing: nvidia-smi -L");
        status.set_message("Incorrect gpu id in the configuration.");
        return status;
      }
      unique_gpu_ids.insert(dev_id);
    }
  } catch (std::invalid_argument const& ex) {
    HOLOSCAN_LOG_ERROR("Invalid argument in Device map: {}", ex.what());
    raise_error("Inference Manager", "Error in Device map.");
  } catch (std::out_of_range const& ex) {
    HOLOSCAN_LOG_ERROR("Invalid range in Device map: {}", ex.what());
    raise_error("Inference Manager", "Error in Device map.");
  } catch (...) {
    raise_error("Inference Manager", "Error in Device map.");
  }

  auto vec_unique_gpu_ids = std::vector<int>(unique_gpu_ids.begin(), unique_gpu_ids.end());

  if (vec_unique_gpu_ids.size() > 1) {
    for (auto gid = 1; gid < vec_unique_gpu_ids.size(); ++gid) {
      int gpu_access_from_gpudt = 0, gpu_access_to_gpudt = 0;
      check_cuda(
          cudaDeviceCanAccessPeer(&gpu_access_from_gpudt, device_gpu_dt_, vec_unique_gpu_ids[gid]));
      check_cuda(
          cudaDeviceCanAccessPeer(&gpu_access_to_gpudt, vec_unique_gpu_ids[gid], device_gpu_dt_));

      if (gpu_access_from_gpudt == 1 && gpu_access_to_gpudt == 1) {
        HOLOSCAN_LOG_INFO("Setting GPU P2P access between GPU {} and GPU {}",
                          device_gpu_dt_,
                          vec_unique_gpu_ids[gid]);
        check_cuda(cudaSetDevice(device_gpu_dt_));
        cudaError_t cstatus = cudaDeviceEnablePeerAccess(vec_unique_gpu_ids[gid], 0);
        if (cstatus != cudaSuccess && cstatus != cudaErrorPeerAccessAlreadyEnabled) {
          HOLOSCAN_LOG_ERROR("Cuda error, {}", cudaGetErrorString(cstatus));
          HOLOSCAN_LOG_ERROR("Error enabling P2P access from GPU {} and GPU {}.",
                             device_gpu_dt_,
                             vec_unique_gpu_ids[gid]);
          status.set_message("Enabling P2P access failed.");
          return status;
        }
        check_cuda(cudaSetDevice(vec_unique_gpu_ids[gid]));
        cstatus = cudaDeviceEnablePeerAccess(device_gpu_dt_, 0);
        if (cstatus != cudaSuccess && cstatus != cudaErrorPeerAccessAlreadyEnabled) {
          HOLOSCAN_LOG_ERROR("Cuda error, {}", cudaGetErrorString(cstatus));
          HOLOSCAN_LOG_ERROR("Error enabling P2P access from GPU {} and GPU {}.",
                             vec_unique_gpu_ids[gid],
                             device_gpu_dt_);
          status.set_message("Enabling P2P access failed.");
          return status;
        }
      } else {
        HOLOSCAN_LOG_WARN("P2P access between GPU {} and GPU {} is not available.",
                          device_gpu_dt_,
                          vec_unique_gpu_ids[gid]);
        HOLOSCAN_LOG_INFO(
            "There can be any reason related to GPU type, GPU family or system setup (PCIE "
            "configuration).");
        HOLOSCAN_LOG_INFO("May be GPU {} and GPU {} are not in the same PCIE configuration.",
                          device_gpu_dt_,
                          vec_unique_gpu_ids[gid]);
        HOLOSCAN_LOG_WARN(
            "Multi GPU inference feature will use Host (CPU memory) to transfer data across GPUs."
            "This may result in an additional latency.");
        mgpu_p2p_transfer_ = false;
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

      auto device_id = device_gpu_dt_;
      if (device_map.find(model_name) != device_map.end()) {
        device_id = std::stoi(device_map.at(model_name));
        HOLOSCAN_LOG_INFO("Device id: {} for Model: {}", device_id, model_name);
      }

      infer_param_.at(model_name)->set_device_id(device_id);

      auto dla_core = inference_specs->dla_core_;
      if (dla_core_map.find(model_name) != dla_core_map.end()) {
        dla_core = std::stoi(dla_core_map.at(model_name));
        HOLOSCAN_LOG_INFO("DLA core: {} for Model: {}", dla_core, model_name);
      }

      unsigned int temporal_id = 1;
      if (temporal_map.find(model_name) != temporal_map.end()) {
        try {
          temporal_id = std::stoul(temporal_map.at(model_name));
          HOLOSCAN_LOG_INFO("Temporal id: {} for Model: {}", temporal_id, model_name);
        } catch (std::invalid_argument const& ex) {
          HOLOSCAN_LOG_ERROR("Invalid argument in Temporal map: {}", ex.what());
          throw;
        } catch (std::out_of_range const& ex) {
          HOLOSCAN_LOG_ERROR("Invalid range in Temporal map: {}", ex.what());
          throw;
        }
      }
      infer_param_.at(model_name)->set_temporal_id(temporal_id);

      // Get input and output tensor maps of the model from inference_specs
      auto out_tensor_names = inference_specs->inference_map_.at(model_name);
      auto in_tensor_names = inference_specs->pre_processor_map_.at(model_name);

      // assign the input and output tensor names to the infer_param object
      infer_param_.at(model_name)->set_tensor_names(in_tensor_names, true);
      infer_param_.at(model_name)->set_tensor_names(out_tensor_names, false);

      check_cuda(cudaSetDevice(device_id));

      auto current_backend = holoinfer_backend::h_trt;
      if (backend_type.length() != 0) {
        current_backend = supported_backend_.at(backend_type);
      }

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
          if (inference_specs->is_engine_path_) {
            if (inference_specs->use_fp16_) {
              HOLOSCAN_LOG_WARN("Engine files are the input, fp16 check/conversion is ignored");
            }
            if (dla_core > -1) {
              HOLOSCAN_LOG_WARN("Engine files are the input, DLA core is ignored");
            }
          }
          if (!inference_specs->oncuda_) {
            status.set_message("ERROR: TRT backend supports inference on GPU only");
            return status;
          }

          holo_infer_context_.insert(
              {model_name,
               std::make_unique<TrtInfer>(model_path,
                                          model_name,
                                          trt_opt_profile,
                                          device_id,
                                          device_gpu_dt_,
                                          inference_specs->use_fp16_,
                                          inference_specs->use_cuda_graphs_,
                                          dla_core,
                                          inference_specs->dla_gpu_fallback_,
                                          inference_specs->is_engine_path_,
                                          cuda_buffer_in_,
                                          cuda_buffer_out_,
                                          inference_specs->allocate_cuda_stream_)});
          break;
        }

        case holoinfer_backend::h_onnx: {
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

#if defined(HOLOINFER_ORT_ENABLED)
          HOLOSCAN_LOG_INFO("Searching for ONNX Runtime libraries");
          void* handle = dlopen("libholoscan_infer_onnx_runtime.so", RTLD_NOW);
          if (handle == nullptr) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("ONNX Runtime context setup failure.");
            return status;
          }
          HOLOSCAN_LOG_INFO("Found ONNX Runtime libraries");
          using NewOnnxInfer = OnnxInfer* (*)(const std::string&,
                                              bool,
                                              int32_t,
                                              bool,
                                              bool,
                                              bool,
                                              bool,
                                              std::function<cudaStream_t(int32_t device_id)>);
          auto new_ort_infer = reinterpret_cast<NewOnnxInfer>(dlsym(handle, "NewOnnxInfer"));
          if (!new_ort_infer) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("ONNX Runtime context setup failure.");
            dlclose(handle);
            return status;
          }
          dlclose(handle);
          // The ONNX backend is not supporting CUDA Graphs in multi-treaded scenarios and also
          // requires that addresses of inputs are not changing. Since we need both features we
          // dont support CUDA Graphs for the ONNX backend.
          // See
          // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview
          // for more information.
          auto context = new_ort_infer(model_path,
                                       inference_specs->use_fp16_,
                                       dla_core,
                                       inference_specs->dla_gpu_fallback_,
                                       inference_specs->oncuda_,
                                       cuda_buffer_in_,
                                       cuda_buffer_out_,
                                       inference_specs->allocate_cuda_stream_);
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
#if defined(HOLOINFER_TORCH_ENABLED)
          HOLOSCAN_LOG_INFO("Searching for libtorch libraries");
          void* handle = dlopen("libholoscan_infer_torch.so", RTLD_NOW);
          if (handle == nullptr) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("Torch context setup failure.");
            return status;
          }
          HOLOSCAN_LOG_INFO("Found libtorch libraries");
          using NewTorchInfer = TorchInfer* (*)(const std::string&,
                                                bool,
                                                bool,
                                                bool,
                                                int,
                                                std::function<cudaStream_t(int32_t device_id)>);
          auto new_torch_infer = reinterpret_cast<NewTorchInfer>(dlsym(handle, "NewTorchInfer"));
          if (!new_torch_infer) {
            HOLOSCAN_LOG_ERROR(dlerror());
            status.set_message("Torch context setup failure.");
            dlclose(handle);
            return status;
          }
          dlclose(handle);
          auto context = new_torch_infer(model_path,
                                         inference_specs->oncuda_,
                                         cuda_buffer_in_,
                                         cuda_buffer_out_,
                                         device_id,
                                         inference_specs->allocate_cuda_stream_);
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
      check_cuda(cudaSetDevice(device_gpu_dt_));

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

      for (unsigned int d = 0; d < out_tensor_names.size(); d++) {
        std::vector<int64_t> dims = holo_infer_context_.at(model_name)->get_output_dims()[d];
        for (int td = 0; td < dims.size(); td++) {
          if (dims[td] < 1) {
            dims[td] = 1;
          }
        }
        auto datatype = holo_infer_context_.at(model_name)->get_output_datatype()[d];
        if (datatype == holoinfer_datatype::h_Unsupported) {
          status.set_message("Unsupported datatype for tensor" + out_tensor_names[d]);
          return status;
        }

        auto astatus = allocate_buffers(inference_specs->output_per_model_,
                                        dims,
                                        datatype,
                                        out_tensor_names[d],
                                        true /* allocate_cuda */,
                                        device_id);
        if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
          astatus.display_message();
          status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
          return status;
        }
        HOLOSCAN_LOG_INFO("HoloInfer buffer created for {}", out_tensor_names[d]);

        if (device_id != device_gpu_dt_) {
          check_cuda(cudaSetDevice(device_id));

          auto astatus = allocate_buffers(
              dm, dims, datatype, out_tensor_names[d], true /* allocate_cuda */, device_id);
          if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
            astatus.display_message();
            status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
            return status;
          }

          check_cuda(cudaSetDevice(device_gpu_dt_));
        }
      }
      mgpu_output_buffer_.insert({model_name, std::move(dm)});

      if (device_id != device_gpu_dt_) {
        // For Multi-GPU feature: allocate input and output cuda streams
        check_cuda(cudaSetDevice(device_gpu_dt_));
        std::vector<cudaStream_t> in_streams_gpudt(in_tensor_names.size());
        std::map<std::string, cudaStream_t> in_streams_map_gpudt, out_streams_map_gpudt;

        // cuda stream creation per tensor and populating input_streams_gpudt_ map
        for (auto in = 0; in < in_tensor_names.size(); in++) {
          check_cuda(cudaStreamCreate(&in_streams_gpudt[in]));
          in_streams_map_gpudt.insert({in_tensor_names[in], in_streams_gpudt[in]});
        }
        input_streams_gpudt_.insert({model_name, std::move(in_streams_map_gpudt)});

        std::vector<cudaStream_t> out_streams_gpudt(out_tensor_names.size());
        // cuda stream creation per output tensor and populating out_streams_map_gpudt map
        for (auto out = 0; out < out_tensor_names.size(); out++) {
          check_cuda(cudaStreamCreate(&out_streams_gpudt[out]));
          out_streams_map_gpudt.insert({out_tensor_names[out], out_streams_gpudt[out]});
        }
        output_streams_gpudt_.insert({model_name, std::move(out_streams_map_gpudt)});

        check_cuda(cudaSetDevice(device_id));
        std::vector<cudaStream_t> in_streams_dev(in_tensor_names.size());
        std::map<std::string, cudaStream_t> in_streams_map_dev, out_streams_map_dev;

        // cuda stream creation per tensor and populating in_streams_map_dev
        for (auto in = 0; in < in_tensor_names.size(); in++) {
          check_cuda(cudaStreamCreate(&in_streams_dev[in]));
          in_streams_map_dev.insert({in_tensor_names[in], in_streams_dev[in]});
        }
        input_streams_device_.insert({model_name, std::move(in_streams_map_dev)});

        std::vector<cudaStream_t> out_streams(out_tensor_names.size());

        // cuda stream creation per output tensor and populating output_streams map
        for (auto out = 0; out < out_tensor_names.size(); out++) {
          check_cuda(cudaStreamCreate(&out_streams[out]));
          out_streams_map_dev.insert({out_tensor_names[out], out_streams[out]});
        }

        output_streams_device_.insert({model_name, std::move(out_streams_map_dev)});
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

          auto astatus = allocate_buffers(
              dm_in, dims, datatype, in_tensor_names[d], true /* allocate_cuda */, device_id);
          if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
            astatus.display_message();
            status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
            return status;
          }
        }
        mgpu_input_buffer_.insert({model_name, std::move(dm_in)});
        check_cuda(cudaSetDevice(device_gpu_dt_));
      }

      models_input_dims_.insert({model_name, holo_infer_context_.at(model_name)->get_input_dims()});

      if (vec_unique_gpu_ids.size() > 1) {
        // create the CUDA event used to synchronize the streams
        auto event_per_gpu = mgpu_cuda_event_.insert({model_name, {}}).first;
        cudaEvent_t cuda_event;
        for (auto&& gid : vec_unique_gpu_ids) {
          check_cuda(cudaSetDevice(gid));
          check_cuda(cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming));
          event_per_gpu->second.insert({gid, cuda_event});
        }
        check_cuda(cudaSetDevice(device_gpu_dt_));
      }
    }

    check_cuda(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));

    if (inference_specs->parallel_processing_) {
      // create the work queue for parallel processing, limit the worker count the available core
      // count
      work_queue_ = std::make_unique<WorkQueue>(
          std::min(infer_param_.size(), static_cast<size_t>(get_nprocs())));
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

  for (auto& [_, infer_p] : infer_param_) {
    infer_p.reset();
  }

  if (cuda_event_) {
    cudaEventDestroy(cuda_event_);
  }
}

ManagerInfer::~ManagerInfer() {
  cleanup();
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
      status.set_message("Inference manager, Preprocessed data for tensor " + in_tensor +
                         " does not exist.");
      return status;
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

    check_cuda(cudaSetDevice(device_gpu_dt_));

    const DataMap& in_preprocess_data = mgpu_input_buffer_.at(model_name);

    const auto& input_streams_dev = input_streams_device_.at(model_name);
    const auto& in_streams_gpudt = input_streams_gpudt_.at(model_name);
    const cudaEvent_t cuda_event_d = mgpu_cuda_event_.at(model_name).at(device_id);
    const cudaEvent_t cuda_event_dt = mgpu_cuda_event_.at(model_name).at(device_gpu_dt_);

    for (const auto& in_tensor : input_tensors) {
      const auto device_buff = in_preprocess_data.at(in_tensor)->device_buffer_->data();
      const auto buffsize = in_preprocess_data.at(in_tensor)->device_buffer_->get_bytes();

      const auto device_gpu_dt_buff_in =
          input_preprocess_data.at(in_tensor)->device_buffer_->data();

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

      indata.push_back(in_preprocess_data.at(in_tensor));
    }
  } else {
    for (const auto& in_tensor : input_tensors) {
      indata.push_back(input_preprocess_data.at(in_tensor));
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

  check_cuda(cudaSetDevice(device_id));
  cudaEvent_t cuda_event_inference = nullptr;
  check_cuda(cudaEventCreate(&cuda_event_inference));

  InferStatus i_status = holo_infer_context_.at(model_name)
                             ->do_inference(indata, outdata, cuda_event_, &cuda_event_inference);

  check_cuda(cudaSetDevice(device_gpu_dt_));

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

  std::chrono::steady_clock::time_point s_time;
  std::chrono::steady_clock::time_point e_time;
  std::map<std::string, std::shared_ptr<std::packaged_task<InferStatus()>>> inference_futures;
  s_time = std::chrono::steady_clock::now();
  for (const auto& [model_instance, _] : infer_param_) {
    bool process_model = true;

    // for this particular model, set the input dimensions (for dynamic inputs)
    // get the sequence of holoscan tensors for this particular model

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
        HOLOSCAN_LOG_DEBUG("Activation value: {} for Model: {}", activation_value, model_instance);
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

  if (parallel_processing_) {
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
  } catch (const std::bad_alloc&) {
    throw;
  }
}

InferStatus InferContext::execute_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                            cudaStream_t cuda_stream) {
  InferStatus status = InferStatus();

  if (g_managers.find(unique_id_) == g_managers.end()) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Inference manager, Error: Inference manager not created or is not set up.");
    return status;
  }

  try {
    g_manager = g_managers.at(unique_id_);

    status = g_manager->execute_inference(inference_specs, cuda_stream);
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("Inference manager, Error in inference execution: ") + e.what());
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
    for (auto& [model_name, _] : multi_model_map) {
      unique_id_name += model_name + "_[]_";
    }

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

DimType InferContext::get_input_dimensions() const {
  g_manager = g_managers.at(unique_id_);
  return g_manager->get_input_dimensions();
}

}  // namespace inference
}  // namespace holoscan
