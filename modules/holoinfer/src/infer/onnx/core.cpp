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
#include "core.hpp"

#include <onnxruntime_cxx_api.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

// Pimpl
class OnnxInferImpl {
 public:
  // Internal only
  OnnxInferImpl(const std::string& model_file_path, bool cuda_flag);

  std::string model_path_{""};
  bool use_cuda_ = true;

  Ort::SessionOptions session_options_;
  OrtCUDAProviderOptions cuda_options_{};

  std::unique_ptr<Ort::Env> env_ = nullptr;
  std::unique_ptr<Ort::Session> session_ = nullptr;

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

  std::vector<Ort::Value> input_tensors_gpu_;
  std::vector<Ort::Value> output_tensors_gpu_;

  Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                            OrtMemType::OrtMemTypeDefault);

  Ort::MemoryInfo memory_info_cuda_ =
      Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  std::unique_ptr<Ort::Allocator> memory_allocator_cuda_;

  holoinfer_datatype get_holoinfer_datatype(ONNXTensorElementDataType datatype);

  Ort::Value create_tensor(const std::shared_ptr<DataBuffer>& input_buffer,
                           const std::vector<int64_t>& dims);
  void transfer_to_output(std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                          const size_t& index);

  // Wrapped Public APIs
  InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
                           std::vector<std::shared_ptr<DataBuffer>>& output_buffer);
  void populate_model_details();
  void print_model_details();
  int set_holoscan_inf_onnx_session_options();
  std::vector<std::vector<int64_t>> get_input_dims() const;
  std::vector<std::vector<int64_t>> get_output_dims() const;
  std::vector<holoinfer_datatype> get_input_datatype() const;
  std::vector<holoinfer_datatype> get_output_datatype() const;
  void cleanup();
};

template <typename T>
Ort::Value create_tensor_core(const std::shared_ptr<DataBuffer>& input_buffer,
                              const std::vector<int64_t>& dims, Ort::MemoryInfo& memory_info_) {
  size_t input_tensor_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  return Ort::Value::CreateTensor<T>(memory_info_,
                                     static_cast<T*>(input_buffer->host_buffer.data()),
                                     input_tensor_size,
                                     dims.data(),
                                     dims.size());
}

template <typename T>
void transfer_to_host(std::shared_ptr<DataBuffer>& output_buffer, Ort::Value& output_tensor,
                      const size_t& output_tensor_size) {
  memcpy(output_buffer->host_buffer.data(),
         output_tensor.GetTensorMutableData<T>(),
         output_tensor_size * sizeof(T));
}

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

holoinfer_datatype OnnxInferImpl::get_holoinfer_datatype(ONNXTensorElementDataType data_type) {
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
    default:
      return holoinfer_datatype::h_Unsupported;
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
    if (indim[0] <= 0) { indim[0] = 1; }
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
    if (outdim[0] <= 0) { outdim[0] = 1; }
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
  if (use_cuda_) { session_options_.AppendExecutionProvider_CUDA(cuda_options_); }
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return 0;
}

extern "C" OnnxInfer* NewOnnxInfer(const std::string& model_file_path, bool cuda_flag) {
  return new OnnxInfer(model_file_path, cuda_flag);
}

OnnxInfer::OnnxInfer(const std::string& model_file_path, bool cuda_flag)
    : impl_(new OnnxInferImpl(model_file_path, cuda_flag)) {}

OnnxInfer::~OnnxInfer() {
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

OnnxInferImpl::OnnxInferImpl(const std::string& model_file_path, bool cuda_flag)
    : model_path_(model_file_path), use_cuda_(cuda_flag) {
  try {
    set_holoscan_inf_onnx_session_options();

    auto env_local = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    env_ = std::move(env_local);

    auto _session =
        std::make_unique<Ort::Session>(*env_, model_file_path.c_str(), session_options_);
    if (!_session) {
      HOLOSCAN_LOG_ERROR("Session creation failed in Onnx inference constructor");
      throw std::runtime_error("Onnxruntime session creation failed");
    }
    session_ = std::move(_session);
    populate_model_details();
  } catch (const Ort::Exception& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  }
}

Ort::Value OnnxInferImpl::create_tensor(const std::shared_ptr<DataBuffer>& input_buffer,
                                        const std::vector<int64_t>& dims) {
  auto data_type = input_buffer->get_datatype();

  switch (data_type) {
    case holoinfer_datatype::h_Float32:
      return create_tensor_core<float>(input_buffer, dims, memory_info_);
    case holoinfer_datatype::h_Int8:
      return create_tensor_core<int8_t>(input_buffer, dims, memory_info_);
    case holoinfer_datatype::h_Int32:
      return create_tensor_core<int32_t>(input_buffer, dims, memory_info_);
    case holoinfer_datatype::h_Int64:
      return create_tensor_core<int64_t>(input_buffer, dims, memory_info_);
    case holoinfer_datatype::h_UInt8:
      return create_tensor_core<uint8_t>(input_buffer, dims, memory_info_);
    default: {
      HOLOSCAN_LOG_INFO(
          "Onnxruntime backend is supported with following data types: float, int8, int32, int64, "
          "uint8");
      HOLOSCAN_LOG_ERROR("Unsupported datatype in Onnx backend tensor creation.");
      return Ort::Value(nullptr);
    }
  }
}

void OnnxInferImpl::transfer_to_output(std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                                       const size_t& index) {
  size_t output_tensor_size = accumulate(
      output_dims_[index].begin(), output_dims_[index].end(), 1, std::multiplies<size_t>());

  auto data_type = output_buffer[index]->get_datatype();

  switch (data_type) {
    case holoinfer_datatype::h_Float32:
      transfer_to_host<float>(output_buffer[index], output_tensors_[index], output_tensor_size);
      break;
    case holoinfer_datatype::h_Int8:
      transfer_to_host<int8_t>(output_buffer[index], output_tensors_[index], output_tensor_size);
      break;
    case holoinfer_datatype::h_Int32:
      transfer_to_host<int32_t>(output_buffer[index], output_tensors_[index], output_tensor_size);
      break;
    case holoinfer_datatype::h_Int64:
      transfer_to_host<int64_t>(output_buffer[index], output_tensors_[index], output_tensor_size);
      break;
    case holoinfer_datatype::h_UInt8:
      transfer_to_host<uint8_t>(output_buffer[index], output_tensors_[index], output_tensor_size);
      break;
    default:
      HOLOSCAN_LOG_INFO(
          "Onnxruntime backend is supported with following data types: float, int8, int32, int64, "
          "uint8");
      throw std::runtime_error("Unsupported datatype in output transfer with onnxrt backend.");
  }
}

InferStatus OnnxInfer::do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
                                    std::vector<std::shared_ptr<DataBuffer>>& output_buffer) {
  return impl_->do_inference(input_buffer, output_buffer);
}

InferStatus OnnxInferImpl::do_inference(
    const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
    std::vector<std::shared_ptr<DataBuffer>>& output_buffer) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  try {
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
      if (input_buffer[a]->host_buffer.size() == 0) {
        status.set_message("ONNX inference core: Input Host buffer empty.");
        return status;
      }

      Ort::Value i_tensor = create_tensor(input_buffer[a], input_dims_[a]);

      if (!i_tensor) {
        status.set_message("Onnxruntime: Error creating Ort tensor.");
        return status;
      }
      input_tensors_.push_back(std::move(i_tensor));
    }

    for (unsigned int a = 0; a < output_buffer.size(); a++) {
      if (output_buffer[a]->host_buffer.size() == 0) {
        status.set_message("ONNX inference core: Output Host buffer empty.");
        return status;
      }

      Ort::Value o_tensor = create_tensor(output_buffer[a], output_dims_[a]);

      if (!o_tensor) {
        status.set_message("Onnxruntime: Error creating output Ort tensor.");
        return status;
      }
      output_tensors_.push_back(std::move(o_tensor));
    }

    session_->Run(Ort::RunOptions{nullptr},
                  input_names_.data(),
                  input_tensors_.data(),
                  input_tensors_.size(),
                  output_names_.data(),
                  output_tensors_.data(),
                  output_tensors_.size());

    for (unsigned int a = 0; a < output_buffer.size(); a++) {
      transfer_to_output(output_buffer, a);
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
