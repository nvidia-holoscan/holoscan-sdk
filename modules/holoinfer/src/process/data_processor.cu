/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>
#include <cub/device/device_reduce.cuh>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <iostream>

#include "data_processor.hpp"

namespace holoscan {
namespace inference {

/**
 * This class implements an iterator which skips `step` elements between each iteration.
 */
class step_iterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = float;
  using difference_type = ptrdiff_t;
  using pointer = float*;
  using reference = float&;

  explicit step_iterator(pointer cur, size_t step) : cur_(cur), step_(step) {}
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance offset) const {
    return cur_[offset * step_];
  }

 private:
  pointer cur_;
  size_t step_;
};

/**
 * CUDA kernel normalizing the coordinates stored in the `key` member.
 *
 * @param rows
 * @param cols
 * @param channels
 * @param d_argmax
 * @param out
 */
static __global__ void normalize(size_t rows, size_t cols, size_t channels,
                                 cub::KeyValuePair<int, float>* d_argmax, float* out) {
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= channels) {
    return;
  }

  const int src_index = d_argmax[index].key;
  int row = src_index / cols;
  int col = src_index - (row * cols);
  out[index * 2 + 0] = (float)row / (float)rows;
  out[index * 2 + 1] = (float)col / (float)cols;
}

void DataProcessor::max_per_channel_scaled_cuda(size_t rows, size_t cols, size_t channels,
                                                const float* indata, float* outdata,
                                                cudaStream_t cuda_stream) {
  /// @todo This algorithm needs temporary storage, currently data processors are just functions
  /// without state. This should be an object with state so we can avoid re-allocating the temporary
  /// storage at each invocation.

  // Allocate result storage
  cub::KeyValuePair<int, float>* d_argmax = nullptr;
  check_cuda(
      cudaMallocAsync(&d_argmax, sizeof(cub::KeyValuePair<int, float>) * channels, cuda_stream));

  // get temp storage size
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, indata, d_argmax, rows * cols);

  // Allocate temporary storage
  check_cuda(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));

  for (size_t channel = 0; channel < channels; ++channel) {
    step_iterator iterator((float*)(indata + channel), channels);
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes, iterator, &d_argmax[channel], rows * cols, cuda_stream);
  }

  check_cuda(cudaFreeAsync(d_temp_storage, cuda_stream));

  dim3 block(32, 1, 1);
  dim3 grid((channels + block.x - 1) / block.x, 1, 1);
  normalize<<<grid, block, 0, cuda_stream>>>(rows, cols, channels, d_argmax, outdata);
  check_cuda(cudaPeekAtLastError());

  check_cuda(cudaFreeAsync(d_argmax, cuda_stream));
}

InferStatus DataProcessor::launchCustomKernel(const std::vector<std::string>& ids,
                                              const std::vector<int>& dimensions, const void* input,
                                              std::vector<int64_t>& processed_dims,
                                              DataMap& processed_data_map,
                                              const std::vector<std::string>& output_tensors,
                                              bool process_with_cuda, cudaStream_t cuda_stream) {
  if (output_tensors.size() == 0) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Output tensor size 0 in launchCustomKernel.");
  }

  if (!process_with_cuda) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, launching custom kernel must have I/O on cuda.");
  }

  auto out_tensor_name = output_tensors[0];

  if (cuda_graph_created_.find(out_tensor_name) == cuda_graph_created_.end()) {
    cuda_graph_created_[out_tensor_name] = false;
    cuda_graph_instantiated_[out_tensor_name] = false;
  }
  size_t dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
  size_t dimensionX = dimensions[0];
  size_t dimensionY = dimensions[1];
  size_t dimensionZ = dimensions[2];

  int kernel_count = ids.size();

  CUresult result = cuCtxPushCurrent(context_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Cuda Context push failed in launchKernel.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Cuda context push failed.");
  }

  if (first_time_kernel_launch_map_.find(out_tensor_name) == first_time_kernel_launch_map_.end()) {
    first_time_kernel_launch_map_[out_tensor_name] = true;
    intermediate_inputs_[out_tensor_name].push_back(const_cast<void*>(input));

    for (int i = 1; i < kernel_count; i++) {
      auto intermediate_buffer = std::make_shared<DataBuffer>(output_dtype_.at(ids[i - 1]));
      if (dynamic_output_dim_) {
        auto dyn_dimensions = custom_kernel_output_dimensions_.at(ids[i - 1]);
        dsize =
            accumulate(dyn_dimensions.begin(), dyn_dimensions.end(), 1, std::multiplies<size_t>());
      }
      intermediate_buffer->device_buffer_->resize(dsize);

      intermediate_buffers_[out_tensor_name].push_back(std::move(intermediate_buffer));
      intermediate_inputs_[out_tensor_name].push_back(
          intermediate_buffers_[out_tensor_name].back()->device_buffer_->data());
    }

    // create the output data

    if (processed_data_map.find(out_tensor_name) == processed_data_map.end()) {
      HOLOSCAN_LOG_INFO("Allocating memory for {} in launchGenericKernel", out_tensor_name);
      const auto [db, success] = processed_data_map.insert(
          {out_tensor_name, std::make_shared<DataBuffer>(output_dtype_.at(ids[kernel_count - 1]))});

      if (dynamic_output_dim_) {
        auto dyn_dimensions = custom_kernel_output_dimensions_.at(ids[kernel_count - 1]);
        dsize =
            accumulate(dyn_dimensions.begin(), dyn_dimensions.end(), 1, std::multiplies<size_t>());
        processed_dims.insert(processed_dims.begin(), dyn_dimensions.begin(), dyn_dimensions.end());
      } else {
        processed_dims.insert(processed_dims.begin(), dimensions.begin(), dimensions.end());
      }
      db->second->device_buffer_->resize(dsize);
      db->second->host_buffer_->resize(dsize);
    }

    intermediate_inputs_[out_tensor_name].push_back(
        processed_data_map.at(out_tensor_name)->device_buffer_->data());
  } else {
    intermediate_inputs_[out_tensor_name][0] = const_cast<void*>(input);
  }

  int buffer_count = 0;
  const char* error_string;

  for (auto id : ids) {
    if (dynamic_output_dim_ && buffer_count > 0) {
      // output of the previous kernel is the input to the current kernel
      auto dyn_output_dimensions = custom_kernel_output_dimensions_.at(ids[buffer_count - 1]);
      if (dyn_output_dimensions.size() == 1) {
        dimensionX = dyn_output_dimensions[0];
        dimensionY = 1;
        dimensionZ = 1;
      } else if (dyn_output_dimensions.size() == 2) {
        dimensionX = dyn_output_dimensions[0];
        dimensionY = dyn_output_dimensions[1];
        dimensionZ = 1;
      } else if (dyn_output_dimensions.size() == 3) {
        dimensionX = dyn_output_dimensions[0];
        dimensionY = dyn_output_dimensions[1];
        dimensionZ = dyn_output_dimensions[2];
      }
      dsize = accumulate(
          dyn_output_dimensions.begin(), dyn_output_dimensions.end(), 1, std::multiplies<size_t>());
    } else {
      dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
    }

    std::vector<void*> args = {&intermediate_inputs_[out_tensor_name][buffer_count],
                               &intermediate_inputs_[out_tensor_name][buffer_count + 1],
                               &dsize};

    // Find the dimensionality
    std::vector<std::string> threads_per_block;
    string_split(custom_kernel_thread_per_block_.at(id), threads_per_block, ',');

    int threadsPerBlockx = 1, threadsPerBlocky = 1, threadsPerBlockz = 1;
    int blocksPerGridx = 1, blocksPerGridy = 1, blocksPerGridz = 1;

    // compute appropriate grid and block size
    switch (threads_per_block.size()) {
      case 1:
      default: {
        threadsPerBlockx = std::atoi(threads_per_block[0].c_str());
        blocksPerGridx = (dsize + threadsPerBlockx - 1) / threadsPerBlockx;
        break;
      }

      case 2: {
        threadsPerBlockx = std::atoi(threads_per_block[0].c_str());
        threadsPerBlocky = std::atoi(threads_per_block[1].c_str());
        blocksPerGridx = (dimensions[0] + threadsPerBlockx - 1) / threadsPerBlockx;
        blocksPerGridy = (dimensions[1] + threadsPerBlocky - 1) / threadsPerBlocky;

        std::vector<void*> newargs = {&intermediate_inputs_[out_tensor_name][buffer_count],
                                      &intermediate_inputs_[out_tensor_name][buffer_count + 1],
                                      &dimensionX,
                                      &dimensionY};
        args = std::move(newargs);
        break;
      }

      case 3: {
        threadsPerBlockx = std::atoi(threads_per_block[0].c_str());
        threadsPerBlocky = std::atoi(threads_per_block[1].c_str());
        threadsPerBlockz = std::atoi(threads_per_block[2].c_str());
        blocksPerGridx = (dimensions[0] + threadsPerBlockx - 1) / threadsPerBlockx;
        blocksPerGridy = (dimensions[1] + threadsPerBlocky - 1) / threadsPerBlocky;
        blocksPerGridz = (dimensions[2] + threadsPerBlockz - 1) / threadsPerBlockz;
        std::vector<void*> newargs = {&intermediate_inputs_[out_tensor_name][buffer_count],
                                      &intermediate_inputs_[out_tensor_name][buffer_count + 1],
                                      &dimensionX,
                                      &dimensionY,
                                      &dimensionZ};
        args = std::move(newargs);
        break;
      }
    }

    dim3 gridDim(blocksPerGridx, blocksPerGridy, blocksPerGridz);
    dim3 blockDim(threadsPerBlockx, threadsPerBlocky, threadsPerBlockz);

    if (use_cuda_graph_) {
      if (!cuda_graph_created_[out_tensor_name]) {
        cuda_graph_created_[out_tensor_name] = true;
        CUgraph l_graph;
        result = cuGraphCreate(&l_graph, 0);
        if (result != CUDA_SUCCESS) {
          cuGetErrorString(result, &error_string);

          HOLOSCAN_LOG_ERROR("CUDA graph creation failed in launchKernel: {}", error_string);
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, CUDA graph creation failed.");
        }
        graph_[out_tensor_name] = l_graph;
      }

      if (!cuda_graph_instantiated_[out_tensor_name]) {
        CUDA_KERNEL_NODE_PARAMS kernelNodeParam = {0};
        kernelNodeParam.func = kernel_.at(id);
        kernelNodeParam.gridDimX = blocksPerGridx;
        kernelNodeParam.gridDimY = blocksPerGridy;
        kernelNodeParam.gridDimZ = blocksPerGridz;
        kernelNodeParam.blockDimX = threadsPerBlockx;
        kernelNodeParam.blockDimY = threadsPerBlocky;
        kernelNodeParam.blockDimZ = threadsPerBlockz;
        kernelNodeParam.sharedMemBytes = 0;
        kernelNodeParam.kernelParams = args.data();
        kernel_node_params_[out_tensor_name].push_back(std::move(kernelNodeParam));
        CUgraphNode kernelNode;
        if (buffer_count == 0) {
          result = cuGraphAddKernelNode(&kernelNode,
                                        graph_[out_tensor_name],
                                        nullptr,
                                        0,
                                        &kernel_node_params_[out_tensor_name][buffer_count]);
        } else {
          CUgraphNode dependencies[] = {kernel_nodes_[out_tensor_name][buffer_count - 1]};
          result = cuGraphAddKernelNode(&kernelNode,
                                        graph_[out_tensor_name],
                                        dependencies,
                                        1,
                                        &kernel_node_params_[out_tensor_name][buffer_count]);
        }
        if (result != CUDA_SUCCESS) {
          cuGetErrorString(result, &error_string);

          HOLOSCAN_LOG_ERROR("CUDA graph node addition failed in launchKernel: {}", error_string);
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, CUDA graph node creation failed.");
        }
        kernel_nodes_[out_tensor_name].push_back(kernelNode);
      }

      if (cuda_graph_instantiated_[out_tensor_name]) {
        CUDA_KERNEL_NODE_PARAMS updatedParams = kernel_node_params_[out_tensor_name][buffer_count];
        updatedParams.kernelParams = args.data();
        result = cuGraphExecKernelNodeSetParams(cuda_graph_instance_[out_tensor_name],
                                                kernel_nodes_[out_tensor_name][buffer_count],
                                                &updatedParams);
        if (result != CUDA_SUCCESS) {
          cuGetErrorString(result, &error_string);

          HOLOSCAN_LOG_ERROR("CUDA graph node setting failed in launchKernel: {}", error_string);
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, CUDA graph node setting failed.");
        }
      }
    } else {
      result = cuLaunchKernel(kernel_.at(id),
                              blocksPerGridx,
                              blocksPerGridy,
                              blocksPerGridz,
                              threadsPerBlockx,
                              threadsPerBlocky,
                              threadsPerBlockz,
                              0,
                              cuda_stream,
                              args.data(),
                              0);

      if (result != CUDA_SUCCESS) {
        cuGetErrorString(result, &error_string);
        HOLOSCAN_LOG_ERROR("CUDA error in launching custom kernel: {}", error_string);
        return InferStatus(holoinfer_code::H_ERROR,
                           "Data processor, error in launching custom kernel.");
      }
    }

    buffer_count++;
  }

  if (use_cuda_graph_) {
    if (!cuda_graph_instantiated_[out_tensor_name]) {
      result =
          cuGraphInstantiate(&cuda_graph_instance_[out_tensor_name], graph_[out_tensor_name], 0);
      if (result != CUDA_SUCCESS) {
        cuGetErrorString(result, &error_string);

        HOLOSCAN_LOG_ERROR("CUDA graph instantiation failed in launchKernel: {}", error_string);
        return InferStatus(holoinfer_code::H_ERROR,
                           "Data processor, CUDA graph instantiation failed.");
      }
    }
    result = cuGraphLaunch(cuda_graph_instance_[out_tensor_name], cuda_stream);
    if (result != CUDA_SUCCESS) {
      cuGetErrorString(result, &error_string);

      HOLOSCAN_LOG_ERROR("CUDA graph launch failed in launchKernel: {}", error_string);
      return InferStatus(holoinfer_code::H_ERROR, "Data processor, CUDA graph launch failed.");
    }

    cuda_graph_instantiated_[out_tensor_name] = true;
  }

  result = cuCtxPopCurrent(&context_);
  if (result != CUDA_SUCCESS) {
    cuGetErrorString(result, &error_string);

    HOLOSCAN_LOG_ERROR("Cuda context pop failed in launchKernel: {}", error_string);
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Cuda context setting failed.");
  }

  return InferStatus();
}

InferStatus DataProcessor::prepareCustomKernel() {
  nvrtcProgram prog;
  nvrtcResult nvResult =
      nvrtcCreateProgram(&prog, custom_cuda_src_.c_str(), "customKernel.cu", 0, NULL, NULL);

  if (nvResult != NVRTC_SUCCESS) {
    HOLOSCAN_LOG_INFO("Error in NVRTC program creation {}", nvrtcGetErrorString(nvResult));
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, NVRTC program creation failed.");
  }

  if (prog == NULL) {
    HOLOSCAN_LOG_ERROR("Created program is NULL from nvrtc");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, NVRTC created NULL program.");
  }

  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Failed to get CUDA device id.");
  }

  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Failed to get CUDA device properties.");
  }

  HOLOSCAN_LOG_DEBUG("Device {}: {}", device, prop.name);
  HOLOSCAN_LOG_DEBUG("GPU Compute Capability: {}.{}", prop.major, prop.minor);

  std::string arch_compile_string = "--gpu-architecture=sm_";
  arch_compile_string.append(std::to_string(prop.major)).append(std::to_string(prop.minor));
  HOLOSCAN_LOG_DEBUG("GPU architecture compilation flag: {}", arch_compile_string);

  const char* opts[] = {arch_compile_string.c_str()};

  nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);
  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    char* log = new char[logSize];
    nvrtcGetProgramLog(prog, log);
    HOLOSCAN_LOG_ERROR("Cuda kernel compilation error: {}", log);
    delete[] log;
    HOLOSCAN_LOG_ERROR("NVRTC compilation failed. Please review the custom cuda kernel.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, NVRTC compilation failed.");
  }

  size_t ptxSize;
  nvResult = nvrtcGetPTXSize(prog, &ptxSize);
  if (nvResult != NVRTC_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Error in NVRTC get ptx size {}", nvrtcGetErrorString(nvResult));
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, NVRTC get ptx size failed.");
  }

  HOLOSCAN_LOG_DEBUG("PTX size: {}", ptxSize);
  std::vector<char> ptx(ptxSize);
  nvResult = nvrtcGetPTX(prog, ptx.data());
  if (nvResult != NVRTC_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Error in NVRTC ptx data {}", nvrtcGetErrorString(nvResult));
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, NVRTC get ptx failed.");
  }
  HOLOSCAN_LOG_DEBUG("PTX file: {}", ptx.data());
  HOLOSCAN_LOG_INFO("NVRTC kernel compilation succeeded.");

  nvResult = nvrtcDestroyProgram(&prog);
  if (nvResult != NVRTC_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Error in NVRTC program destruction. {}", nvrtcGetErrorString(nvResult));
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, NVRTC prog destruction failed.");
  }

  CUresult result;
  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Cuda Init failed.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Cuda Init failed.");
  }

  result = cuDeviceGet(&device_, 0);
  if (result == CUDA_SUCCESS) {
    result = cuCtxCreate(&context_, 0, device_);
  }
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Cuda Context creation failed.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Cuda context creation failed.");
  }

  result = cuModuleLoadData(&module_, ptx.data());
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(result, &error_string);
    HOLOSCAN_LOG_ERROR("Cuda module loading failed. {}", error_string);
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Cuda module loading failed.");
  }

  for (const auto& item : custom_kernel_thread_per_block_) {
    auto kernel_id = item.first;
    auto kernel_name = "customKernel" + kernel_id;

    result = cuModuleGetFunction(&kernel_[kernel_id], module_, kernel_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char* error_string;
      cuGetErrorString(result, &error_string);
      HOLOSCAN_LOG_ERROR("Cuda module get function failed. {}", error_string);
      return InferStatus(holoinfer_code::H_ERROR,
                         "Data processor, Cuda module get function failed.");
    }
  }

  result = cuCtxPopCurrent(&context_);
  if (result != CUDA_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Cuda Context setting failed in prepareKernel.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Cuda context setting failed.");
  }

  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
