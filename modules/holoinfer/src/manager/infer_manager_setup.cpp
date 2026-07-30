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

#include <dlfcn.h>
#include <sys/sysinfo.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
namespace holoscan {
namespace inference {

// [Refactor] File-local helpers for backend plugin loading (dlopen/dlsym).
// Previously in infer_manager.cpp; moved here because they are only used by
// set_inference_params.
namespace {

std::string dlerror_to_string() {
  const char* dlerror_message = dlerror();
  return dlerror_message ? dlerror_message : "unknown linker error";
}

std::string backend_plugin_load_failure_message(const std::string& backend_name,
                                                const std::string& plugin_library,
                                                const std::string& dependency_hint,
                                                const std::string& linker_error) {
  return backend_name + " context setup failure. Failed to load backend '" + plugin_library +
         "' or one of its runtime dependencies. " + dependency_hint +
         " Dynamic linker error: " + linker_error;
}

std::string backend_plugin_symbol_failure_message(const std::string& backend_name,
                                                  const std::string& plugin_library,
                                                  const std::string& symbol_name,
                                                  const std::string& linker_error) {
  return backend_name + " context setup failure. Failed to resolve factory symbol '" + symbol_name +
         "' from backend plugin '" + plugin_library + "'. Dynamic linker error: " + linker_error;
}

struct DlHandleCloser {
  void operator()(void* handle) const noexcept {
    if (handle) {
      dlclose(handle);
    }
  }
};

using ScopedDlHandle = std::unique_ptr<void, DlHandleCloser>;

}  // namespace

InferStatus ManagerInfer::set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  auto multi_model_map = inference_specs->get_path_map();
  auto device_map = inference_specs->get_device_map();
  auto dla_core_map = inference_specs->get_dla_core_map();
  auto temporal_map = inference_specs->get_temporal_map();
  const auto& backend_type = inference_specs->backend_type_;
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
        const auto& backend_ = backend_map.at(model_name);
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

          // check here for valid entry
          if (trt_opt_profile.find(model_name) == trt_opt_profile.end()) {
            trt_opt_profile[model_name] = {};
          }
          auto current_trt_opt_profile = trt_opt_profile.at(model_name);

          holo_infer_context_.insert(
              {model_name,
               std::make_unique<TrtInfer>(model_path,
                                          model_name,
                                          current_trt_opt_profile,
                                          device_id,
                                          device_gpu_dt_,
                                          inference_specs->use_fp16_,
                                          inference_specs->use_cuda_graphs_,
                                          dla_core,
                                          inference_specs->dla_gpu_fallback_,
                                          inference_specs->is_engine_path_,
                                          cuda_buffer_in_,
                                          cuda_buffer_out_,
                                          inference_specs->allocate_cuda_stream_,
                                          inference_specs->build_cuda_context_,
                                          inference_specs->build_sm_count_)});

          if (inference_specs->gpu_resident_inference_) {
            try {
              holo_infer_context_.at(model_name)
                  ->init_gr_inference(inference_specs->gpu_resident_input_,
                                      inference_specs->gpu_resident_output_);
            } catch (const std::runtime_error& e) {
              status.set_message("ERROR: " + std::string(e.what()));
              return status;
            }
          }
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
          HOLOSCAN_LOG_INFO("Loading ONNX Runtime backend");
          const std::string onnxrt_backend_plugin =
              "libholoscan_infer_onnx_runtime.so." TOSTRING(PROJECT_VERSION_MAJOR);
          ScopedDlHandle handle(dlopen(onnxrt_backend_plugin.c_str(), RTLD_NOW));
          if (handle == nullptr) {
            std::string linker_error = dlerror_to_string();
            HOLOSCAN_LOG_ERROR("{}", linker_error);
            status.set_message(backend_plugin_load_failure_message(
                "ONNX Runtime",
                onnxrt_backend_plugin,
                "Install ONNX Runtime backend dependencies as described in the Holoscan SDK "
                "documentation installation page, and make the libraries visible to the dynamic "
                "linker; alternatively, use the TensorRT backend (trt) for ONNX models.",
                linker_error));
            return status;
          }
          HOLOSCAN_LOG_INFO("Loaded ONNX Runtime backend");
          using NewOnnxInfer = OnnxInfer* (*)(const std::string&,
                                              bool,
                                              int32_t,
                                              bool,
                                              bool,
                                              bool,
                                              bool,
                                              std::function<cudaStream_t(int32_t device_id)>);
          (void)dlerror();
          auto new_ort_infer = reinterpret_cast<NewOnnxInfer>(dlsym(handle.get(), "NewOnnxInfer"));
          if (!new_ort_infer) {
            std::string linker_error = dlerror_to_string();
            HOLOSCAN_LOG_ERROR("{}", linker_error);
            status.set_message(backend_plugin_symbol_failure_message(
                "ONNX Runtime", onnxrt_backend_plugin, "NewOnnxInfer", linker_error));
            return status;
          }
          // The ONNX backend is not supporting CUDA Graphs in multi-treaded scenarios and also
          // requires that addresses of inputs are not changing. Since we need both features we
          // dont support CUDA Graphs for the ONNX backend.
          // See
          // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview
          // for more information.
          backend_plugin_handles_.push_back(handle.get());
          (void)handle.release();
          auto context =
              std::unique_ptr<OnnxInfer>(new_ort_infer(model_path,
                                                       inference_specs->use_fp16_,
                                                       dla_core,
                                                       inference_specs->dla_gpu_fallback_,
                                                       inference_specs->oncuda_,
                                                       cuda_buffer_in_,
                                                       cuda_buffer_out_,
                                                       inference_specs->allocate_cuda_stream_));
          holo_infer_context_[model_name] = std::move(context);
#else
          HOLOSCAN_LOG_ERROR("Onnxruntime backend not supported or incorrectly installed.");
          status.set_message("Onnxruntime context setup failure.");
          return status;
#endif
          break;
        }

        case holoinfer_backend::h_torch: {
          // TODO(FAQ-FIX): This acceptance of .pth should align with the documentation.
          // The torch backend requires TorchScript format (.pt), not PyTorch state dict (.pth).
          // Consider rejecting .pth with a specific error instructing conversion to TorchScript.
          // Refs: public/docs/hsdk_faq.md (Development Q7), public/docs/inference.md
          if (std::filesystem::path(model_path).extension() != ".pt" &&
              std::filesystem::path(model_path).extension() != ".pth") {
            HOLOSCAN_LOG_ERROR("Torch model must be in torchsript format (.pt or .pth).");
            status.set_message("Inference manager, model path must have .pt or .pth extension.");
            return status;
          }
#if defined(HOLOINFER_TORCH_ENABLED)
          HOLOSCAN_LOG_INFO("Loading Torch backend");
          const std::string torch_backend_plugin =
              "libholoscan_infer_torch.so." TOSTRING(PROJECT_VERSION_MAJOR);
          ScopedDlHandle handle(dlopen(torch_backend_plugin.c_str(), RTLD_NOW));
          if (handle == nullptr) {
            std::string linker_error = dlerror_to_string();
            HOLOSCAN_LOG_ERROR("{}", linker_error);
            status.set_message(backend_plugin_load_failure_message(
                "Torch",
                torch_backend_plugin,
                "Install Torch backend dependencies as described in the Holoscan SDK "
                "documentation installation page, and make the libraries visible to the dynamic "
                "linker.",
                linker_error));
            return status;
          }
          HOLOSCAN_LOG_INFO("Loaded Torch backend");
          using NewTorchInfer = TorchInfer* (*)(const std::string&,
                                                bool,
                                                bool,
                                                bool,
                                                int,
                                                std::function<cudaStream_t(int32_t device_id)>);
          (void)dlerror();
          auto new_torch_infer =
              reinterpret_cast<NewTorchInfer>(dlsym(handle.get(), "NewTorchInfer"));
          if (!new_torch_infer) {
            std::string linker_error = dlerror_to_string();
            HOLOSCAN_LOG_ERROR("{}", linker_error);
            status.set_message(backend_plugin_symbol_failure_message(
                "Torch", torch_backend_plugin, "NewTorchInfer", linker_error));
            return status;
          }
          backend_plugin_handles_.push_back(handle.get());
          (void)handle.release();
          auto context =
              std::unique_ptr<TorchInfer>(new_torch_infer(model_path,
                                                          inference_specs->oncuda_,
                                                          cuda_buffer_in_,
                                                          cuda_buffer_out_,
                                                          device_id,
                                                          inference_specs->allocate_cuda_stream_));
          holo_infer_context_[model_name] = std::move(context);
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
      models_output_dims_.insert(
          {model_name, holo_infer_context_.at(model_name)->get_output_dims()});

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

    // Cache the single-model raw pointers used by run_core_inference_fast.
    if (infer_param_.size() == 1) {
      fast_model_name_ = infer_param_.begin()->first;
      fast_param_ = infer_param_.at(fast_model_name_).get();
      fast_ctx_ = holo_infer_context_.at(fast_model_name_).get();
    }

    // default-path optimization
    // ALWAYS populate the flat per-model pointer vectors, regardless of fast_multi_model_.
    // These pointers reference model-state objects that are immutable after
    // set_inference_params, so caching them is safe even for dynamic_input_dims pipelines
    // and for configs with activation_map/temporal_map/device_map (they read the same
    // underlying state, just with extra bookkeeping that the canonical path still does).
    // This makes the work_queue parallel branch's run_core_inference_indexed reachable
    // even when fast_multi_model is off.
    all_params_.reserve(infer_param_.size());
    all_ctx_.reserve(infer_param_.size());
    all_model_names_.reserve(infer_param_.size());
    for (auto& [name, param_uptr] : infer_param_) {
      all_params_.push_back(param_uptr.get());
      all_ctx_.push_back(holo_infer_context_.at(name).get());
      all_model_names_.push_back(name);
    }

    // Set up the streamlined dispatch caches and per-model parallel
    // streams ONLY when:
    //   - the caller has explicitly opted in (fast_multi_model_ == true), AND
    //   - none of the canonical operator's advanced features are in use:
    //       activation_map, temporal_map, multi-GPU device_map, per-model DLA, GR inference,
    //       dynamic input dims. These features have their own bookkeeping that the fast
    //       paths bypass; turning them on together would silently skip required work.
    fast_multi_model_ = inference_specs->fast_multi_model_;

    // Static-mode blockers — these force kStandard even when fast_multi_model is on.
    // DLA (dla_core / dla_core_map) is NOT a blocker: DLA is a build-time device
    // assignment; per-compute() call dispatch (enqueueV3) is identical for GPU- and DLA-bound
    // execution contexts. The DLA engine is built at set_inference_params; runtime
    // routing is transparent.
    const bool static_blockers =
        !inference_specs->activation_map_.empty() || !inference_specs->temporal_map_.empty() ||
        !inference_specs->device_map_.empty() || inference_specs->gpu_resident_inference_;

    // dynamic_input_dims no longer blocks par-fast stream allocation —
    // the DynParFast path uses the same per-model streams but skips the prewarm and
    // pre-built I/O cache. Still blocked for the static cache (seq_fast_data_cached_),
    // which the dyn paths don't use.
    if (fast_multi_model_ && !static_blockers) {
      // Per-model parallel streams + events
      bool all_trt = true;
      if (!inference_specs->backend_type_.empty()) {
        all_trt = (inference_specs->backend_type_ == "trt");
      } else {
        for (const auto& [m, b] : inference_specs->backend_map_) {
          if (b != "trt") {
            all_trt = false;
            break;
          }
        }
      }
      const int par_fast_max_n = []() {
        const char* env = std::getenv("HOLOINFER_PAR_FAST_MAX_N");
        return env ? std::atoi(env) : std::numeric_limits<int>::max();
      }();
      if (inference_specs->parallel_processing_ && infer_param_.size() >= 2 && all_trt &&
          static_cast<int>(infer_param_.size()) <= par_fast_max_n &&
          inference_specs->allocate_cuda_stream_) {
        std::vector<cudaStream_t> tentative(infer_param_.size(), nullptr);
        bool ok = true;
        for (size_t m = 0; m < infer_param_.size() && ok; ++m) {
          try {
            tentative[m] = inference_specs->allocate_cuda_stream_(device_gpu_dt_);
          } catch (const std::exception&) {
            ok = false;
            break;
          }
          if (!tentative[m])
            ok = false;
        }
        if (ok) {
          std::set<cudaStream_t> uniq(tentative.begin(), tentative.end());
          if (uniq.size() != tentative.size())
            ok = false;
        }
        if (ok) {
          par_streams_ = std::move(tentative);
          par_done_events_.assign(par_streams_.size(), nullptr);
          for (auto& e : par_done_events_) {
            check_cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
          }
          check_cuda(cudaEventCreateWithFlags(&par_outer_event_, cudaEventDisableTiming));

          // Prewarm CUDA-graph captures during setup — STATIC SHAPES ONLY.
          // Under dynamic dims, the captured graphs would be invalidated on the first
          // shape change anyway; skip prewarm in that case.
          if (!inference_specs->dynamic_input_dims_) {
            auto pw = prewarm_par_fast(/*passes=*/3);
            if (pw.get_code() == holoinfer_code::H_ERROR) {
              pw.display_message();
              // Tear down on prewarm failure → demote to work_queue path.
              if (par_outer_event_) {
                cudaEventDestroy(par_outer_event_);
                par_outer_event_ = nullptr;
              }
              for (auto& e : par_done_events_)
                if (e)
                  cudaEventDestroy(e);
              par_done_events_.clear();
              par_streams_.clear();
            }
          }
        } else {
          fprintf(stderr,
                  "[holoinfer] fast_multi_model par-fast demoted: executor's CudaStreamPool "
                  "could not supply %zu distinct streams. Increase cuda_stream_pool "
                  "max_size to at least %zu to enable inline parallel dispatch.\n",
                  infer_param_.size(),
                  infer_param_.size());
        }
      }
    }
  } catch (const std::runtime_error& rt) {
    raise_error("Inference Manager", "Setting Inference parameters: " + std::string(rt.what()));
  } catch (...) {
    raise_error("Inference Manager", "Setting Inference parameters: unknown exception occurred.");
  }

  parallel_processing_ = inference_specs->parallel_processing_;

  HOLOSCAN_LOG_INFO("Building execution plan...");
  auto plan_status = build_execution_plan(
      inference_specs->pre_processor_map_, inference_specs->inference_map_, execution_plan_);

  if (plan_status.get_code() == holoinfer_code::H_ERROR) {
    HOLOSCAN_LOG_ERROR("Execution plan build failed: {}", plan_status.get_message());
    return plan_status;
  }

  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
