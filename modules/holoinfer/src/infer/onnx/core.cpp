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

#include <onnxruntime_cxx_api.h>

#include <cassert>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer_utils.hpp>

namespace holoscan {
namespace inference {

// Pimpl
class OnnxInferImpl {
 public:
  // Internal only
  OnnxInferImpl(const std::string& model_file_path, bool enable_fp16, int32_t dla_core,
                bool dla_gpu_fallback, bool cuda_flag, bool cuda_buf_in, bool cuda_buf_out,
                std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream);
  ~OnnxInferImpl();

  const std::string model_path_;
  const bool enable_fp16_;
  const int32_t dla_core_;
  const bool dla_gpu_fallback_;
  const bool use_cuda_;
  const bool cuda_buf_in_;
  const bool cuda_buf_out_;
  const std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream_;

  std::unique_ptr<Ort::Env> env_;

  Ort::SessionOptions session_options_;
  OrtCUDAProviderOptionsV2* cuda_options_ = nullptr;
  OrtTensorRTProviderOptionsV2* tensor_rt_options_ = nullptr;

  std::unique_ptr<Ort::Session> session_;

  Ort::AllocatorWithDefaultOptions allocator_;

  size_t input_nodes_{0}, output_nodes_{0};

  std::vector<std::vector<int64_t>> input_dims_{};
  std::vector<std::vector<int64_t>> output_dims_{};

  std::vector<holoinfer_datatype> input_type_, output_type_;

  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;

  std::vector<Ort::AllocatedStringPtr> input_allocated_strings_;
  std::vector<Ort::AllocatedStringPtr> output_allocated_strings_;

  std::vector<Ort::Value> input_tensors_;
  std::vector<Ort::Value> output_tensors_;

  Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                            OrtMemType::OrtMemTypeDefault);

  Ort::MemoryInfo memory_info_cuda_ =
      Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  cudaStream_t cuda_stream_ = nullptr;
  cudaEvent_t cuda_event_ = nullptr;

  Ort::Value create_tensor(const std::shared_ptr<DataBuffer>& data_buffer,
                           const std::vector<int64_t>& dims, bool cuda_buf);

  // Wrapped Public APIs
  InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
                           std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                           cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference);
  void populate_model_details();
  void print_model_details();
  int set_holoscan_inf_onnx_session_options();
  std::vector<std::vector<int64_t>> get_input_dims() const;
  std::vector<std::vector<int64_t>> get_output_dims() const;
  std::vector<holoinfer_datatype> get_input_datatype() const;
  std::vector<holoinfer_datatype> get_output_datatype() const;
  void cleanup();
};

void OnnxInfer::print_model_details() {
  impl_->print_model_details();
}

void OnnxInferImpl::print_model_details() {
  HOLOSCAN_LOG_INFO("Input node count: {}", input_nodes_);
  HOLOSCAN_LOG_INFO("Output node count: {}", output_nodes_);

  HOLOSCAN_LOG_INFO("Input names: [{}]", fmt::join(input_names_, ", "));
  for (size_t a = 0; a < input_dims_.size(); a++) {
    HOLOSCAN_LOG_INFO(
        "Input Dimension for {}: [{}]", input_names_[a], fmt::join(input_dims_[a], ", "));
  }
  HOLOSCAN_LOG_INFO("Output names: [{}]", fmt::join(output_names_, ", "));
  for (size_t a = 0; a < output_dims_.size(); a++) {
    HOLOSCAN_LOG_INFO(
        "Output Dimension for {}: [{}]", output_names_[a], fmt::join(output_dims_[a], ", "));
  }
}

static holoinfer_datatype get_holoinfer_datatype(ONNXTensorElementDataType data_type) {
  switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return holoinfer_datatype::h_Float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return holoinfer_datatype::h_Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return holoinfer_datatype::h_Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return holoinfer_datatype::h_Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return holoinfer_datatype::h_UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return holoinfer_datatype::h_Float16;
    default:
      return holoinfer_datatype::h_Unsupported;
  }
}

static ONNXTensorElementDataType get_onnx_datatype(holoinfer_datatype data_type) {
  switch (data_type) {
    case holoinfer_datatype::h_Float32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case holoinfer_datatype::h_Int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case holoinfer_datatype::h_Int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case holoinfer_datatype::h_Int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case holoinfer_datatype::h_UInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case holoinfer_datatype::h_Float16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    default:
      HOLOSCAN_LOG_INFO(
          "Onnxruntime backend is supported with following input data types: float, float16, int8, "
          "int32, int64, uint8");
      HOLOSCAN_LOG_ERROR("Unsupported datatype in Onnx backend tensor creation.");
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

void OnnxInfer::populate_model_details() {
  impl_->populate_model_details();
}

void OnnxInferImpl::populate_model_details() {
  input_nodes_ = session_->GetInputCount();
  output_nodes_ = session_->GetOutputCount();

  for (size_t a = 0; a < input_nodes_; a++) {
    auto input_name_ptr = session_->GetInputNameAllocated(a, allocator_);
    input_allocated_strings_.push_back(std::move(input_name_ptr));
    input_names_.push_back(input_allocated_strings_.back().get());

    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(a);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType tensor_element_type = input_tensor_info.GetElementType();
    input_type_.push_back(get_holoinfer_datatype(tensor_element_type));
    auto indim = input_tensor_info.GetShape();
    if (indim[0] <= 0) {
      indim[0] = 1;
    }
    input_dims_.push_back(indim);
  }

  for (size_t a = 0; a < output_nodes_; a++) {
    auto output_name_ptr = session_->GetOutputNameAllocated(a, allocator_);
    output_allocated_strings_.push_back(std::move(output_name_ptr));
    output_names_.push_back(output_allocated_strings_.back().get());

    Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(a);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType tensor_element_type = output_tensor_info.GetElementType();

    output_type_.push_back(get_holoinfer_datatype(tensor_element_type));
    auto outdim = output_tensor_info.GetShape();
    if (outdim[0] <= 0) {
      outdim[0] = 1;
    }
    output_dims_.push_back(outdim);
  }

  print_model_details();
}

int OnnxInfer::set_holoscan_inf_onnx_session_options() {
  return impl_->set_holoscan_inf_onnx_session_options();
}

int OnnxInferImpl::set_holoscan_inf_onnx_session_options() {
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetInterOpNumThreads(1);
  if (use_cuda_) {
    // create and initialize TensorRT provider options
    Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&tensor_rt_options_));

    const std::filesystem::path path(model_path_);
    std::filesystem::path trt_engine_cache_path(model_path_);
    trt_engine_cache_path.replace_extension("");
    trt_engine_cache_path += "_onnx_cache_" + Ort::GetVersionString();

    if (!dla_gpu_fallback_) {
      HOLOSCAN_LOG_WARN("Onnxruntime backend DLA GPU fallback can't be disabled");
    }

    const std::vector<const char*> option_keys = {"trt_fp16_enable",
                                                  "trt_engine_cache_enable",
                                                  "trt_engine_cache_path",
                                                  "trt_timing_cache_enable",
                                                  "trt_timing_cache_path",
                                                  "trt_dla_enable",
                                                  "trt_dla_core"};
    const std::string dla_core_string = std::to_string(dla_core_);
    const std::vector<const char*> option_values = {
        enable_fp16_ ? "1" : "0",       // trt_fp16_enable
        "1",                            // trt_engine_cache_enable
        trt_engine_cache_path.c_str(),  // trt_engine_cache_path
        "1",                            // trt_timing_cache_enable
        trt_engine_cache_path.c_str(),  // trt_timing_cache_path
        dla_core_ > -1 ? "1" : "0",     // trt_dla_enable
        dla_core_string.c_str(),        // trt_dla_core
    };
    assert(option_keys.size() == option_values.size());
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
        tensor_rt_options_, option_keys.data(), option_values.data(), option_keys.size()));
    Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptionsWithValue(
        tensor_rt_options_, "user_compute_stream", cuda_stream_));

    // add the TensoRT provider
    session_options_.AppendExecutionProvider_TensorRT_V2(*tensor_rt_options_);

    // create and initialize CUDA provider options
    Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&cuda_options_));
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptionsWithValue(
        cuda_options_, "user_compute_stream", cuda_stream_));

    // add the CUDA provider
    session_options_.AppendExecutionProvider_CUDA_V2(*cuda_options_);
  }
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return 0;
}

extern "C" OnnxInfer* NewOnnxInfer(
    const std::string& model_file_path, bool enable_fp16, int32_t dla_core, bool dla_gpu_fallback,
    bool cuda_flag, bool cuda_buf_in, bool cuda_buf_out,
    std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream) {
  return new OnnxInfer(model_file_path,
                       enable_fp16,
                       dla_core,
                       dla_gpu_fallback,
                       cuda_flag,
                       cuda_buf_in,
                       cuda_buf_out,
                       std::move(allocate_cuda_stream));
}

OnnxInfer::OnnxInfer(const std::string& model_file_path, bool enable_fp16, int32_t dla_core,
                     bool dla_gpu_fallback, bool cuda_flag, bool cuda_buf_in, bool cuda_buf_out,
                     std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream)
    : impl_(new OnnxInferImpl(model_file_path, enable_fp16, dla_core, dla_gpu_fallback, cuda_flag,
                              cuda_buf_in, cuda_buf_out, std::move(allocate_cuda_stream))) {}

OnnxInfer::~OnnxInfer() {
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

static void logging_function(void* param, OrtLoggingLevel severity, const char* category,
                             const char* logid, const char* code_location, const char* message) {
  LogLevel log_level;
  switch (severity) {
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL:
      log_level = LogLevel::CRITICAL;
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR:
      log_level = LogLevel::ERROR;
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING:
      log_level = LogLevel::WARN;
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO:
      log_level = LogLevel::INFO;
      break;
    case OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE:
      log_level = LogLevel::DEBUG;
      break;
  }
  HOLOSCAN_LOG_CALL(log_level, "Onnxruntime {} {}", code_location, message);
}

OnnxInferImpl::OnnxInferImpl(const std::string& model_file_path, bool enable_fp16, int32_t dla_core,
                             bool dla_gpu_fallback, bool cuda_flag, bool cuda_buf_in,
                             bool cuda_buf_out,
                             std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream)
    : model_path_(model_file_path),
      enable_fp16_(enable_fp16),
      dla_core_(dla_core),
      dla_gpu_fallback_(dla_gpu_fallback),
      use_cuda_(cuda_flag),
      cuda_buf_in_(cuda_buf_in),
      cuda_buf_out_(cuda_buf_out),
      allocate_cuda_stream_(std::move(allocate_cuda_stream)) {
  try {
    OrtLoggingLevel logging_level;
    switch (log_level()) {
      case LogLevel::OFF:
      case LogLevel::CRITICAL:
        logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL;
        break;
      case LogLevel::ERROR:
        logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
        break;
      case LogLevel::WARN:
        logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
        break;
      case LogLevel::INFO:
        logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
        break;
      case LogLevel::DEBUG:
      case LogLevel::TRACE:
        logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
        break;
    }
    env_ = std::make_unique<Ort::Env>(logging_level, "onnx", logging_function, nullptr);
    if (!env_) {
      HOLOSCAN_LOG_ERROR("Env creation failed in Onnx inference constructor");
      throw std::runtime_error("Onnxruntime env creation failed");
    }

    // If a CUDA stream pool is provided, try to allocate a stream from it
    if (allocate_cuda_stream_) {
      cuda_stream_ = allocate_cuda_stream_(0);
    }

    // If no stream could be allocated, create a new one
    if (!cuda_stream_) {
      check_cuda(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
    }

    check_cuda(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));

    set_holoscan_inf_onnx_session_options();

    session_ = std::make_unique<Ort::Session>(*env_, model_file_path.c_str(), session_options_);
    if (!session_) {
      HOLOSCAN_LOG_ERROR("Session creation failed in Onnx inference constructor");
      throw std::runtime_error("Onnxruntime session creation failed");
    }
    populate_model_details();
  } catch (const Ort::Exception& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  }
}

OnnxInferImpl::~OnnxInferImpl() {
  if (tensor_rt_options_) {
    Ort::GetApi().ReleaseTensorRTProviderOptions(tensor_rt_options_);
  }
  if (cuda_options_) {
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options_);
  }
  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
  }
  if (cuda_event_) {
    cudaEventDestroy(cuda_event_);
  }
}

Ort::Value OnnxInferImpl::create_tensor(const std::shared_ptr<DataBuffer>& data_buffer,
                                        const std::vector<int64_t>& dims, bool cuda_buf) {
  const size_t tensor_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  const OrtMemoryInfo* info;
  void* p_data;
  if (cuda_buf) {
    if (data_buffer->device_buffer_->size() != tensor_size) {
      HOLOSCAN_LOG_ERROR("Onnx: Device buffer size mismatch, expected {}, but is {}.",
                         tensor_size,
                         data_buffer->device_buffer_->size());
      return Ort::Value(nullptr);
    }
    p_data = data_buffer->device_buffer_->data();
    info = memory_info_cuda_;
  } else {
    if (data_buffer->host_buffer_->size() != tensor_size) {
      HOLOSCAN_LOG_ERROR("Onnx: Host buffer size mismatch, expected {}, but is {}.",
                         tensor_size,
                         data_buffer->host_buffer_->size());
      return Ort::Value(nullptr);
    }
    p_data = data_buffer->host_buffer_->data();
    info = memory_info_;
  }

  Ort::Value tensor(nullptr);
  const ONNXTensorElementDataType element_data_type =
      get_onnx_datatype(data_buffer->get_datatype());
  if (cuda_buf && !use_cuda_) {
    // create a tensor in CPU memory, we copy to/from this buffer before/after inference
    tensor = Ort::Value::CreateTensor(allocator_, dims.data(), dims.size(), element_data_type);
  } else {
    // wrap the buffer
    tensor = Ort::Value::CreateTensor(info,
                                      static_cast<float*>(p_data),
                                      tensor_size * get_element_size(data_buffer->get_datatype()),
                                      dims.data(),
                                      dims.size(),
                                      element_data_type);
  }
  return tensor;
}

InferStatus OnnxInfer::do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
                                    std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                                    cudaEvent_t cuda_event_data,
                                    cudaEvent_t* cuda_event_inference) {
  return impl_->do_inference(input_buffer, output_buffer, cuda_event_data, cuda_event_inference);
}

InferStatus OnnxInferImpl::do_inference(
    const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
    std::vector<std::shared_ptr<DataBuffer>>& output_buffer, cudaEvent_t cuda_event_data,
    cudaEvent_t* cuda_event_inference) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  try {
    // synchronize the CUDA stream used for inference with the CUDA event recorded when preparing
    // the input data
    check_cuda(cudaStreamWaitEvent(cuda_stream_, cuda_event_data));

    input_tensors_.clear();
    output_tensors_.clear();

    if (input_nodes_ != input_buffer.size()) {
      status.set_message("ONNX inference core: Input buffer size not equal to input nodes.");
      return status;
    }
    if (output_nodes_ != output_buffer.size()) {
      status.set_message("ONNX inference core: Output buffer size not equal to output nodes.");
      return status;
    }

    for (size_t a = 0; a < input_buffer.size(); a++) {
      Ort::Value i_tensor = create_tensor(input_buffer[a], input_dims_[a], cuda_buf_in_);
      if (!i_tensor) {
        status.set_message("Onnxruntime: Error creating Ort tensor.");
        return status;
      }
      if (cuda_buf_in_ && !use_cuda_) {
        // Copy the the input data to the input Ort tensor if inference is on CPU and input on
        // device
        // Note: there is a bug in the C++ API, GetTensorRawData() is returning a `const void*`
        // instead of a `void *` as the C API
        check_cuda(cudaMemcpyAsync(const_cast<void*>(i_tensor.GetTensorRawData()),
                                   input_buffer[a]->device_buffer_->data(),
                                   input_buffer[a]->device_buffer_->get_bytes(),
                                   cudaMemcpyDeviceToHost,
                                   cuda_stream_));
      }
      input_tensors_.push_back(std::move(i_tensor));
    }

    for (unsigned int a = 0; a < output_buffer.size(); a++) {
      Ort::Value o_tensor = create_tensor(output_buffer[a], output_dims_[a], cuda_buf_out_);

      if (!o_tensor) {
        status.set_message("Onnxruntime: Error creating output Ort tensor.");
        return status;
      }
      output_tensors_.push_back(std::move(o_tensor));
    }

    if (!use_cuda_) {
      // synchronize CUDA with CPU if using CPU inference
      check_cuda(cudaStreamSynchronize(cuda_stream_));
    }

    session_->Run(Ort::RunOptions{nullptr},
                  input_names_.data(),
                  input_tensors_.data(),
                  input_tensors_.size(),
                  output_names_.data(),
                  output_tensors_.data(),
                  output_tensors_.size());

    if (cuda_buf_out_ && !use_cuda_) {
      for (size_t index = 0; index < output_buffer.size(); ++index) {
        // Copy the the input data to the input Ort tensor if inference is on CPU and input on
        // device
        check_cuda(cudaMemcpyAsync(output_buffer[index]->device_buffer_->data(),
                                   output_tensors_[index].GetTensorRawData(),
                                   output_buffer[index]->device_buffer_->get_bytes(),
                                   cudaMemcpyHostToDevice,
                                   cuda_stream_));
      }
    }

    if (cuda_buf_out_ || use_cuda_) {
      // record a CUDA event and pass it back to the caller
      check_cuda(cudaEventRecord(cuda_event_, cuda_stream_));
      *cuda_event_inference = cuda_event_;
    }
  } catch (const Ort::Exception& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  }
  return InferStatus();
}

std::vector<std::vector<int64_t>> OnnxInfer::get_input_dims() const {
  return impl_->get_input_dims();
}

std::vector<std::vector<int64_t>> OnnxInferImpl::get_input_dims() const {
  return input_dims_;
}

std::vector<std::vector<int64_t>> OnnxInfer::get_output_dims() const {
  return impl_->get_output_dims();
}

std::vector<std::vector<int64_t>> OnnxInferImpl::get_output_dims() const {
  return output_dims_;
}

std::vector<holoinfer_datatype> OnnxInfer::get_input_datatype() const {
  return impl_->get_input_datatype();
}

std::vector<holoinfer_datatype> OnnxInferImpl::get_input_datatype() const {
  return input_type_;
}

std::vector<holoinfer_datatype> OnnxInfer::get_output_datatype() const {
  return impl_->get_output_datatype();
}

std::vector<holoinfer_datatype> OnnxInferImpl::get_output_datatype() const {
  return output_type_;
}

void OnnxInfer::cleanup() {
  impl_->cleanup();
}

void OnnxInferImpl::cleanup() {
  session_.reset();
  env_.reset();
}

}  // namespace inference
}  // namespace holoscan
