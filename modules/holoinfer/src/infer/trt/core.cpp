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

#include "core.hpp"

#include <NvInferPlugin.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoinfer_utils.hpp"

namespace holoscan {
namespace inference {

TrtInfer::TrtInfer(const std::string& model_path, const std::string& model_name,
                   const std::vector<int32_t>& trt_opt_profile, int device_id, int device_id_dt,
                   bool enable_fp16, bool enable_cuda_graphs, int32_t dla_core,
                   bool dla_gpu_fallback, bool is_engine_path, bool cuda_buf_in, bool cuda_buf_out)
    : model_path_(model_path),
      model_name_(model_name),
      trt_opt_profile_(trt_opt_profile),
      device_id_(device_id),
      enable_fp16_(enable_fp16),
      enable_cuda_graphs_(enable_cuda_graphs),
      dla_core_(dla_core),
      dla_gpu_fallback_(dla_gpu_fallback),
      is_engine_path_(is_engine_path),
      cuda_buf_in_(cuda_buf_in),
      cuda_buf_out_(cuda_buf_out) {
  if (trt_opt_profile.size() != 3) {
    HOLOSCAN_LOG_WARN(
        "TRT Inference: Optimization profile must of of size 3. Size from inference parameters: "
        "{}",
        trt_opt_profile.size());
    HOLOSCAN_LOG_INFO("Input optimization profile ignored. Using default optimization profile");
  } else {
    // set the network optimization profile for dynamic inputs
    network_options_.batch_sizes[0] = trt_opt_profile_[0];
    network_options_.batch_sizes[1] = trt_opt_profile_[1];
    network_options_.batch_sizes[2] = trt_opt_profile_[2];
  }

  // Set the device index
  network_options_.device_index = device_id_;

  network_options_.use_fp16 = enable_fp16_;
  network_options_.dla_core = dla_core_;
  network_options_.dla_gpu_fallback = dla_gpu_fallback_;
  initLibNvInferPlugins(nullptr, "");

  if (!is_engine_path_) {
    HOLOSCAN_LOG_INFO("TRT Inference: converting ONNX model at {}", model_path_);

    bool status = generate_engine_path(network_options_, model_path_, engine_path_);
    if (!status) { throw std::runtime_error("TRT Inference: could not generate TRT engine path."); }

    status = build_engine(model_path_, engine_path_, network_options_, logger_);
    if (!status) {
      HOLOSCAN_LOG_ERROR("Engine file creation failed for {}", model_path_);
      HOLOSCAN_LOG_INFO(
          "If the input path {} is an engine file, set 'is_engine_path' parameter to true in "
          "the inference settings in the application config.",
          model_path_);
      throw std::runtime_error("TRT Inference: failed to build TRT engine file.");
    }
  } else {
    engine_path_ = model_path_;
  }

  check_cuda(cudaSetDevice(device_id_));
  // Create the CUDA stream with the non-blocking flags set. This is needed for CUDA stream
  // capturing since capturing fails if another thread is scheduling work to stream '0' while
  // we capture in this thread. We explicitly synchronize with the caller using events so stream
  // '0' does not need to sync with us.
  check_cuda(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
  // create the CUDA event used to synchronize with the caller
  check_cuda(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));

  bool status = load_engine();
  if (!status) { throw std::runtime_error("TRT Inference: failed to load TRT engine file."); }

  status = initialize_parameters();
  if (!status) { throw std::runtime_error("TRT Inference: Initialization error."); }
}

TrtInfer::~TrtInfer() {
  if (context_) { context_.reset(); }
  if (engine_) { engine_.reset(); }
  if (cuda_stream_) { cudaStreamDestroy(cuda_stream_); }
  if (cuda_event_) { cudaEventDestroy(cuda_event_); }
  if (cuda_graph_instance_) { cudaGraphExecDestroy(cuda_graph_instance_); }
  if (infer_runtime_) { infer_runtime_.reset(); }
}

bool TrtInfer::load_engine() {
  HOLOSCAN_LOG_INFO("Loading Engine: {}", engine_path_);
  std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    HOLOSCAN_LOG_ERROR("Load Engine: File read error: {}", engine_path_);
    return false;
  }

  infer_runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!infer_runtime_) {
    HOLOSCAN_LOG_ERROR("Load Engine: Error in creating inference runtime.");
    return false;
  }

  // Set the device index
  auto status = cudaSetDevice(network_options_.device_index);
  if (status != 0) {
    HOLOSCAN_LOG_ERROR("Load Engine: Setting cuda device failed.");
    throw std::runtime_error("Error setting cuda device in load engine.");
  }

  if (network_options_.dla_core > -1) {
    // set the DLA core
    const int32_t available_dla_cores = infer_runtime_->getNbDLACores();
    if (network_options_.dla_core > available_dla_cores - 1) {
      HOLOSCAN_LOG_ERROR("DLA core '{}' is requested but max DLA core index is '{}'",
                         network_options_.dla_core,
                         available_dla_cores - 1);
      throw std::runtime_error("Error setting DLA core in load engine.");
    }
    infer_runtime_->setDLACore(network_options_.dla_core);
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
      infer_runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!engine_) {
    HOLOSCAN_LOG_ERROR("Load Engine: Error in deserializing cuda engine.");
    return false;
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    HOLOSCAN_LOG_ERROR("Load Engine: Error in creating execution context.");
    return false;
  }

  HOLOSCAN_LOG_INFO("Engine loaded: {}", engine_path_);
  return true;
}

std::vector<std::vector<int64_t>> TrtInfer::get_input_dims() const {
  return input_dims_;
}

std::vector<std::vector<int64_t>> TrtInfer::get_output_dims() const {
  return output_dims_;
}

std::vector<holoinfer_datatype> TrtInfer::get_input_datatype() const {
  return in_data_types_;
}

std::vector<holoinfer_datatype> TrtInfer::get_output_datatype() const {
  return out_data_types_;
}

bool TrtInfer::initialize_parameters() {
  if (!engine_) {
    HOLOSCAN_LOG_ERROR("Engine is Null.");
    return false;
  }
  const int num_bindings = engine_->getNbIOTensors();

  for (int i = 0; i < num_bindings; i++) {
    auto tensor_name = engine_->getIOTensorName(i);
    auto dims = engine_->getTensorShape(tensor_name);
    auto data_type = engine_->getTensorDataType(tensor_name);

    holoinfer_datatype holoinfer_type = holoinfer_datatype::h_Float32;
    switch (data_type) {
      case nvinfer1::DataType::kFLOAT: {
        holoinfer_type = holoinfer_datatype::h_Float32;
        break;
      }
      case nvinfer1::DataType::kINT32: {
        holoinfer_type = holoinfer_datatype::h_Int32;
        break;
      }
      case nvinfer1::DataType::kINT8: {
        holoinfer_type = holoinfer_datatype::h_Int8;
        break;
      }
      case nvinfer1::DataType::kUINT8: {
        holoinfer_type = holoinfer_datatype::h_UInt8;
        break;
      }
      case nvinfer1::DataType::kHALF: {
        holoinfer_type = holoinfer_datatype::h_Float16;
        break;
      }
      default: {
        HOLOSCAN_LOG_INFO(
            "TensorRT backend supports float, float16, int8, int32, uint8 data types.");
        HOLOSCAN_LOG_ERROR("Data type not supported.");
        return false;
      }
    }

    switch (engine_->getTensorIOMode(tensor_name)) {
      case nvinfer1::TensorIOMode::kINPUT: {
        if (dims.nbDims > 8) {
          HOLOSCAN_LOG_INFO("All tensors must have dimension size less than or equal to 8.");
          return false;
        }
        nvinfer1::Dims in_dimensions;
        in_dimensions.nbDims = dims.nbDims;

        for (size_t in = 0; in < dims.nbDims; in++) { in_dimensions.d[in] = dims.d[in]; }

        auto set_status = context_->setInputShape(tensor_name, in_dimensions);
        if (!set_status) {
          HOLOSCAN_LOG_ERROR("Dimension setup for input tensor {} failed.", tensor_name);
          return false;
        }

        std::vector<int64_t> indim;
        for (size_t in = 0; in < dims.nbDims; in++) { indim.push_back(dims.d[in]); }
        input_dims_.push_back(std::move(indim));

        in_data_types_.push_back(holoinfer_type);
      } break;
      case nvinfer1::TensorIOMode::kOUTPUT: {
        std::vector<int64_t> outdim;
        for (size_t in = 0; in < dims.nbDims; in++) { outdim.push_back(dims.d[in]); }

        output_dims_.push_back(outdim);
        out_data_types_.push_back(holoinfer_type);
      } break;
      default: {
        HOLOSCAN_LOG_ERROR("Input index {} is neither input nor output.", i);
        return false;
      }
    }
  }

  if (!context_->allInputDimensionsSpecified()) {
    HOLOSCAN_LOG_ERROR("Error, not all input dimensions specified.");
    return false;
  }

  return true;
}

InferStatus TrtInfer::do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffers,
                                   std::vector<std::shared_ptr<DataBuffer>>& output_buffers,
                                   cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);
  auto io_index = 0;

  // synchronize the CUDA stream used for inference with the CUDA event recorded when preparing
  // the input data
  check_cuda(cudaStreamWaitEvent(cuda_stream_, cuda_event_data));

  for (auto& input_buffer : input_buffers) {
    if (input_buffer->device_buffer_ == nullptr) {
      status.set_message(" TRT inference core: Input Device buffer is null.");
      return status;
    }

    if (cuda_buf_in_) {
      if (input_buffer->device_buffer_->data() == nullptr) {
        status.set_message(" TRT inference core: Data in Input Device buffer is null.");
        return status;
      }
    } else {
      // Host to Device transfer
      if (input_buffer->host_buffer_->size() == 0) {
        status.set_message(" TRT inference core: Empty input host buffer.");
        return status;
      }

      input_buffer->device_buffer_->resize(input_buffer->host_buffer_->size());

      auto cstatus = cudaMemcpyAsync(input_buffer->device_buffer_->data(),
                                     input_buffer->host_buffer_->data(),
                                     input_buffer->host_buffer_->get_bytes(),
                                     cudaMemcpyHostToDevice,
                                     cuda_stream_);
      if (cstatus != cudaSuccess) {
        status.set_message(" TRT inference core: Host to device transfer failed.");
        return status;
      }
      // When copying from pagable memory to device memory cudaMemcpyAsync() is copying the memory
      // to staging memory first and therefore is synchronous with the host execution. No need to
      // synchronize here.
    }
    auto tensor_name = engine_->getIOTensorName(io_index++);
    if (engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kINPUT) {
      HOLOSCAN_LOG_ERROR("Tensor name {} not an input binding.", tensor_name);
      status.set_message(" TRT inference core: Incorrect input tensor name.");
      return status;
    }
    auto set_flag = context_->setTensorAddress(tensor_name, input_buffer->device_buffer_->data());

    if (!set_flag) {
      HOLOSCAN_LOG_ERROR("Buffer binding failed for {} in inference core.", tensor_name);
      status.set_message(" TRT inference core: Error binding input buffer.");
      return status;
    }
  }

  for (auto& output_buffer : output_buffers) {
    if (output_buffer->device_buffer_ == nullptr) {
      status.set_message(" TRT inference core: Output Device buffer is null.");
      return status;
    }
    if (output_buffer->device_buffer_->data() == nullptr) {
      status.set_message(" TRT inference core: Data in Output Device buffer is null.");
      return status;
    }

    if (output_buffer->device_buffer_->size() == 0) {
      status.set_message(" TRT inference core: Output Device buffer size is 0.");
      return status;
    }
    auto tensor_name = engine_->getIOTensorName(io_index++);
    if (engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kOUTPUT) {
      HOLOSCAN_LOG_ERROR("Tensor name {} not an output binding.", tensor_name);
      status.set_message(" TRT inference core: Incorrect output tensor name.");
      return status;
    }

    auto set_flag = context_->setTensorAddress(tensor_name, output_buffer->device_buffer_->data());

    if (!set_flag) {
      HOLOSCAN_LOG_ERROR("Buffer binding failed for {} in inference core.", tensor_name);
      status.set_message(" TRT inference core: Error binding output buffer.");
      return status;
    }
  }

  bool capturing_graph = false;
  if (enable_cuda_graphs_) {
    // TRT works in two phases, the first phase can't be capture. Start capturing after the
    // first phase.
    // See https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#cuda-graphs for
    // more information.
    if (!first_phase_) {
      check_cuda(cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeThreadLocal));
      capturing_graph = true;
    }
    first_phase_ = false;
  }

  bool infer_status = context_->enqueueV3(cuda_stream_);
  if (!infer_status) {
    if (capturing_graph) {
      // end the capture and destroy the graph when inference failed and we are capturing
      cudaGraph_t cuda_graph = nullptr;
      check_cuda(cudaStreamEndCapture(cuda_stream_, &cuda_graph));
      check_cuda(cudaGraphDestroy(cuda_graph));
      capturing_graph = false;

      // if inference failed with CUDA Graphs enabled, retry without
      HOLOSCAN_LOG_WARN(
          "TRT inference failed with CUDA Graphs enabled, retrying with without CUDA Graphs. "
          "Consider disabling CUDA Graphs by setting the `enable_cuda_graphs` parameter to "
          "`false`.");
      enable_cuda_graphs_ = false;
      infer_status = context_->enqueueV3(cuda_stream_);
    }

    if (!infer_status) {
      status.set_message(" TRT inference core: Inference failure.");
      return status;
    }
  }

  if (capturing_graph) {
    cudaGraph_t cuda_graph = nullptr;
    check_cuda(cudaStreamEndCapture(cuda_stream_, &cuda_graph));

    // If we've already instantiated the graph, try to update it directly and avoid the
    // instantiation overhead
    cudaGraphExecUpdateResultInfo update_result;
    if (cuda_graph_instance_) {
      check_cuda(cudaGraphExecUpdate(cuda_graph_instance_, cuda_graph, &update_result));
    }

    // Instantiate during the first iteration or whenever the update fails for any reason
    if (!cuda_graph_instance_ || (update_result.result != cudaGraphExecUpdateSuccess)) {
      // If a previous update failed, destroy the cudaGraphExec_t before re-instantiating it
      if (cuda_graph_instance_ != NULL) { check_cuda(cudaGraphExecDestroy(cuda_graph_instance_)); }

      // Instantiate graphExec from graph. The error node and error message parameters are unused
      // here.
      check_cuda(cudaGraphInstantiate(&cuda_graph_instance_, cuda_graph, 0));
    }

    check_cuda(cudaGraphDestroy(cuda_graph));

    // now launch the graph
    check_cuda(cudaGraphLaunch(cuda_graph_instance_, cuda_stream_));
  }

  if (!cuda_buf_out_) {
    for (auto& output_buffer : output_buffers) {
      if (output_buffer->host_buffer_->size() == 0) {
        status.set_message(" TRT inference core: Empty output host buffer.");
        return status;
      }
      if (output_buffer->device_buffer_->size() != output_buffer->host_buffer_->size()) {
        status.set_message(" TRT inference core: Output Host and Device buffer size mismatch.");
        return status;
      }
      // Copy the results back to CPU memory
      auto cstatus = cudaMemcpyAsync(output_buffer->host_buffer_->data(),
                                     output_buffer->device_buffer_->data(),
                                     output_buffer->device_buffer_->get_bytes(),
                                     cudaMemcpyDeviceToHost,
                                     cuda_stream_);
      if (cstatus != cudaSuccess) {
        status.set_message(" TRT: Device to host transfer failed");
        return status;
      }
      // When copying from device memory to pagable memory the call is synchronous with the host
      // execution. No need to synchronize here.
    }
  }

  // record a CUDA event and pass it back to the caller
  check_cuda(cudaEventRecord(cuda_event_, cuda_stream_));
  *cuda_event_inference = cuda_event_;

  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
