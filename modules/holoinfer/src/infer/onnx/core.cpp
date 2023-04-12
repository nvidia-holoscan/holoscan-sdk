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
#include "core.hpp"

namespace holoscan {
namespace inference {

void OnnxInfer::print_model_details() {
  HOLOSCAN_LOG_INFO("Input node count: {}", input_nodes_);
  HOLOSCAN_LOG_INFO("Output node count: {}", output_nodes_);
  HOLOSCAN_LOG_INFO("Input names: [{}]", fmt::join(input_names_, ", "));
  HOLOSCAN_LOG_INFO("Input dims: [{}]", fmt::join(input_dims_, ", "));
  HOLOSCAN_LOG_INFO("Input Type: {}", input_type_);
  HOLOSCAN_LOG_INFO("Output names: [{}]", fmt::join(output_names_, ", "));
  HOLOSCAN_LOG_INFO("Output dims: [{}]", fmt::join(output_dims_, ", "));
  HOLOSCAN_LOG_INFO("Output Type: {}", output_type_);
}

void OnnxInfer::populate_model_details() {
  input_nodes_ = session_->GetInputCount();
  output_nodes_ = session_->GetOutputCount();

  const char* input_name = session_->GetInputName(0, allocator_);
  input_names_.push_back(input_name);

  Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

  input_type_ = input_tensor_info.GetElementType();
  input_dims_ = input_tensor_info.GetShape();

  const char* output_name = session_->GetOutputName(0, allocator_);
  output_names_.push_back(output_name);

  Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
  auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

  output_type_ = output_tensor_info.GetElementType();
  output_dims_ = output_tensor_info.GetShape();

  input_dims_[0] = 1;
  output_dims_[0] = 1;

  print_model_details();
}

int OnnxInfer::set_holoscan_inf_onnx_session_options() {
  session_options_.SetIntraOpNumThreads(1);
  if (use_cuda_) { session_options_.AppendExecutionProvider_CUDA(cuda_options_); }

  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  return 0;
}

OnnxInfer::OnnxInfer(const std::string& model_file_path, bool cuda_flag)
    : model_path_(model_file_path), use_cuda_(cuda_flag) {
  set_holoscan_inf_onnx_session_options();
  auto env_local = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
  env_ = std::move(env_local);
  auto _session = std::make_unique<Ort::Session>(*env_, model_file_path.c_str(), session_options_);
  session_ = std::move(_session);

  populate_model_details();
}

InferStatus OnnxInfer::do_inference(std::shared_ptr<DataBuffer>& input_buffer,
                                    std::shared_ptr<DataBuffer>& output_buffer) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (input_buffer->host_buffer.size() == 0) {
    status.set_message(" ONNX inference core: Input Host buffer empty.");
    return status;
  }

  if (output_buffer->host_buffer.size() == 0) {
    status.set_message(" ONNX inference core: Output Host buffer empty.");
    return status;
  }

  auto input_tensor = input_buffer->host_buffer;
  input_tensors_.clear();
  output_tensors_.clear();

  size_t input_tensor_size =
      accumulate(input_dims_.begin(), input_dims_.end(), 1, std::multiplies<size_t>());
  input_tensor_values_.assign(input_tensor.begin(), input_tensor.end());

  size_t output_tensor_size =
      accumulate(output_dims_.begin(), output_dims_.end(), 1, std::multiplies<size_t>());
  output_tensor_values_.assign(output_tensor_size, 0);

  Ort::Value i_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                        input_tensor_values_.data(),
                                                        input_tensor_size,
                                                        input_dims_.data(),
                                                        input_dims_.size());
  if (!i_tensor) {
    status.set_message(" Onnxruntime: Error creating Ort tensor.");
    return status;
  }
  input_tensors_.push_back(std::move(i_tensor));

  Ort::Value o_tensor = Ort::Value::CreateTensor<float>(memory_info_,
                                                        output_tensor_values_.data(),
                                                        output_tensor_size,
                                                        output_dims_.data(),
                                                        output_dims_.size());
  if (!o_tensor) {
    status.set_message(" Onnxruntime: Error creating output Ort tensor.");
    return status;
  }
  output_tensors_.push_back(std::move(o_tensor));

  session_->Run(Ort::RunOptions{nullptr},
                input_names_.data(),
                input_tensors_.data(),
                1,
                output_names_.data(),
                output_tensors_.data(),
                1);

  memcpy(output_buffer->host_buffer.data(),
         output_tensors_.front().GetTensorMutableData<float>(),
         output_tensor_size * sizeof(float));
  return InferStatus();
}

std::vector<int64_t> OnnxInfer::get_input_dims() const {
  return input_dims_;
}

std::vector<int64_t> OnnxInfer::get_output_dims() const {
  return output_dims_;
}

}  // namespace inference
}  // namespace holoscan
