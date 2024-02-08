/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/script.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

// Pimpl
class TorchInferImpl {
 public:
  TorchInferImpl(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
                 bool cuda_buf_out);

  std::string model_path_{""};
  size_t input_nodes_{0}, output_nodes_{0};

  std::vector<std::vector<int64_t>> input_dims_{};
  std::vector<std::vector<int64_t>> output_dims_{};

  std::vector<holoinfer_datatype> input_type_, output_type_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::vector<torch::Tensor> input_tensors_;
  std::vector<torch::Tensor> output_tensors_;

  std::vector<torch::jit::IValue> inputs_;

  torch::jit::script::Module inference_module_;

  torch::DeviceType infer_device_;
  torch::DeviceType input_device_;
  torch::DeviceType output_device_;

  c10::cuda::CUDAStream infer_stream = c10::cuda::getStreamFromPool();
  std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;

  void print_model_details();

  InferStatus populate_model_details();

  torch::Tensor create_tensor(const std::shared_ptr<DataBuffer>& input_buffer,
                              const std::vector<int64_t>& dims);

  InferStatus transfer_to_output(std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                                 torch::Tensor out_torch_tensor, const size_t& index);
};

template <typename T>
torch::Tensor create_tensor_core(const std::shared_ptr<DataBuffer>& input_buffer,
                                 const std::vector<int64_t>& dims, torch::ScalarType data_type,
                                 torch::DeviceType infer_device, torch::DeviceType input_device,
                                 cudaStream_t cstream) {
  size_t input_tensor_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  if (dims.size() < 3 || dims.size() > 4) {
    HOLOSCAN_LOG_ERROR("Input dimension must be in CHW or NCHW format");
    return torch::empty({0});
  }

  int64_t width = dims[dims.size() - 1], height = dims[dims.size() - 2],
          channels = dims[dims.size() - 3];

  // create tensor in CHW format (N=1 is supported)
  auto tensor = torch::zeros({height, width, channels},
                             torch::TensorOptions().dtype(data_type).device(infer_device));

  if (input_device == torch::kCPU) {
    if (infer_device == torch::kCPU) {
      std::memcpy(tensor.data_ptr(),
                  reinterpret_cast<void*>(input_buffer->host_buffer.data()),
                  input_tensor_size * sizeof(T));
    } else {
      auto cstatus = cudaMemcpyAsync(tensor.data_ptr(),
                                     reinterpret_cast<void*>(input_buffer->host_buffer.data()),
                                     input_tensor_size * sizeof(T),
                                     cudaMemcpyHostToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: HtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
      cstatus = cudaStreamSynchronize(cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Cuda stream synchronization failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
    }
  } else {
    if (infer_device == torch::kCPU) {
      auto cstatus = cudaMemcpyAsync(tensor.data_ptr(),
                                     reinterpret_cast<void*>(input_buffer->device_buffer->data()),
                                     input_tensor_size * sizeof(T),
                                     cudaMemcpyDeviceToHost,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoH transfer failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
      cstatus = cudaStreamSynchronize(cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Cuda stream synchronization failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
    } else {
      auto cstatus = cudaMemcpyAsync(tensor.data_ptr(),
                                     input_buffer->device_buffer->data(),
                                     input_tensor_size * sizeof(T),
                                     cudaMemcpyDeviceToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
    }
  }

  // above tensor in channel-last format, pytorch models accept channels-first
  tensor = tensor.permute({2, 0, 1});
  return tensor;
}

torch::Tensor TorchInferImpl::create_tensor(const std::shared_ptr<DataBuffer>& input_buffer,
                                            const std::vector<int64_t>& dims) {
  auto data_type = input_buffer->get_datatype();
  auto cstream = infer_stream.stream();

  switch (data_type) {
    case holoinfer_datatype::h_Float32:
      return create_tensor_core<float>(
          input_buffer, dims, torch::kF32, infer_device_, input_device_, cstream);
    case holoinfer_datatype::h_Int8:
      return create_tensor_core<int8_t>(
          input_buffer, dims, torch::kI8, infer_device_, input_device_, cstream);
    case holoinfer_datatype::h_Int32:
      return create_tensor_core<int32_t>(
          input_buffer, dims, torch::kI32, infer_device_, input_device_, cstream);
    case holoinfer_datatype::h_UInt8:
      return create_tensor_core<uint8_t>(
          input_buffer, dims, torch::kUInt8, infer_device_, input_device_, cstream);
    default: {
      HOLOSCAN_LOG_ERROR("Unsupported datatype in Torch backend tensor creation.");
      return torch::empty({0});
    }
  }
}

template <typename T>
InferStatus transfer_from_tensor(std::shared_ptr<DataBuffer>& output_buffer,
                                 torch::Tensor& output_tensor, std::vector<int64_t>& dims,
                                 torch::DeviceType infer_device, torch::DeviceType output_device,
                                 cudaStream_t cstream) {
  size_t output_tensor_size = output_tensor.numel();
  if (output_device == torch::kCUDA) {
    output_buffer->device_buffer->resize(output_tensor_size);
  } else {
    output_buffer->host_buffer.resize(output_tensor_size);
  }

  // Populate dims for data transmission
  auto tensor_size = output_tensor.dim();
  dims.clear();
  for (size_t it = 0; it < tensor_size; it++) {
    int64_t s = output_tensor.sizes()[it];
    dims.push_back(s);
  }

  if (output_device == torch::kCPU) {
    if (infer_device == torch::kCPU) {
      memcpy(output_buffer->host_buffer.data(),
             output_tensor.data_ptr(),
             output_tensor_size * sizeof(T));
    } else {
      auto cstatus = cudaMemcpyAsync(output_buffer->host_buffer.data(),
                                     output_tensor.data_ptr(),
                                     output_tensor_size * sizeof(T),
                                     cudaMemcpyDeviceToHost,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoH transfer failed: {}", cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, DtoH transfer.");
      }
      cstatus = cudaStreamSynchronize(cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: Cuda stream synchronization failed: {}",
                           cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, Stream synchronization.");
      }
    }
  } else {
    if (infer_device == torch::kCPU) {
      auto cstatus = cudaMemcpyAsync(output_buffer->device_buffer->data(),
                                     output_tensor.data_ptr(),
                                     output_tensor_size * sizeof(T),
                                     cudaMemcpyHostToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: HtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, HtoD transfer.");
      }
      cstatus = cudaStreamSynchronize(cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: Cuda stream synchronization failed: {}",
                           cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, Stream synchronization.");
      }
    } else {
      auto cstatus = cudaMemcpyAsync(output_buffer->device_buffer->data(),
                                     output_tensor.data_ptr(),
                                     output_tensor_size * sizeof(T),
                                     cudaMemcpyDeviceToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, DtoD transfer.");
      }
    }
  }
  return InferStatus();
}

InferStatus TorchInferImpl::transfer_to_output(
    std::vector<std::shared_ptr<DataBuffer>>& output_buffer, torch::Tensor out_torch_tensor,
    const size_t& index) {
  auto data_type = output_buffer[index]->get_datatype();
  out_torch_tensor = out_torch_tensor.contiguous().flatten();
  auto cstream = infer_stream.stream();

  switch (data_type) {
    case holoinfer_datatype::h_Float32:
      return transfer_from_tensor<float>(output_buffer[index],
                                         out_torch_tensor,
                                         output_dims_[index],
                                         infer_device_,
                                         output_device_,
                                         cstream);
    case holoinfer_datatype::h_Int8:
      return transfer_from_tensor<int8_t>(output_buffer[index],
                                          out_torch_tensor,
                                          output_dims_[index],
                                          infer_device_,
                                          output_device_,
                                          cstream);
    case holoinfer_datatype::h_Int32:
      return transfer_from_tensor<int32_t>(output_buffer[index],
                                           out_torch_tensor,
                                           output_dims_[index],
                                           infer_device_,
                                           output_device_,
                                           cstream);
    case holoinfer_datatype::h_Int64:
      return transfer_from_tensor<int64_t>(output_buffer[index],
                                           out_torch_tensor,
                                           output_dims_[index],
                                           infer_device_,
                                           output_device_,
                                           cstream);
    default:
      return InferStatus(holoinfer_code::H_ERROR, "Unsupported datatype for transfer.");
  }
  return InferStatus();
}

void TorchInfer::print_model_details() {
  impl_->print_model_details();
}

void TorchInferImpl::print_model_details() {
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

InferStatus TorchInfer::populate_model_details() {
  return impl_->populate_model_details();
}

InferStatus TorchInferImpl::populate_model_details() {
  std::string config_yaml_file_path;
  if (std::filesystem::exists(model_path_)) {
    config_yaml_file_path =
        std::filesystem::path(model_path_).replace_extension("").string() + ".yaml";
    if (!std::filesystem::exists(config_yaml_file_path)) {
      HOLOSCAN_LOG_ERROR("Inference config path not found: {}", config_yaml_file_path);
      return InferStatus(holoinfer_code::H_ERROR, "Torch core, invalid config path.");
    }
  } else {
    HOLOSCAN_LOG_ERROR("Inference config path not found: {}", model_path_);
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, invalid model path.");
  }

  try {
    YAML::Node config = YAML::LoadFile(config_yaml_file_path);

    if (!config["inference"]) {
      HOLOSCAN_LOG_ERROR("Torch core: inference key not present in model config file {}",
                         config_yaml_file_path);
      return InferStatus(holoinfer_code::H_ERROR, "Torch core, incorrect config file.");
    }

    // Read input nodes
    if (!config["inference"]["input_nodes"]) {
      HOLOSCAN_LOG_ERROR("Torch core: input_nodes key not present in model config file {}",
                         config_yaml_file_path);
      return InferStatus(holoinfer_code::H_ERROR, "Torch core, incorrect config file.");
    }
    auto infer_in_config = config["inference"]["input_nodes"].as<node_type>();
    input_nodes_ = infer_in_config.size();

    auto status = parse_yaml_node(infer_in_config, input_names_, input_dims_, input_type_);
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Torch core: Node parsing failed for input.");
      return status;
    }

    if (!config["inference"]["output_nodes"]) {
      HOLOSCAN_LOG_ERROR("Torch core: output_nodes key not present in model config file {}",
                         config_yaml_file_path);
      return InferStatus(holoinfer_code::H_ERROR, "Torch core, incorrect config file.");
    }
    auto infer_out_config = config["inference"]["output_nodes"].as<node_type>();
    output_nodes_ = infer_out_config.size();

    status = parse_yaml_node(infer_out_config, output_names_, output_dims_, output_type_);
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Torch core: Node parsing failed for output.");
      return status;
    }
    print_model_details();
  } catch (const YAML::ParserException& ex) {
    HOLOSCAN_LOG_ERROR("YAML error: {}", ex.what());
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, YAML error.");
  }

  return InferStatus();
}

extern "C" TorchInfer* NewTorchInfer(const std::string& model_file_path, bool cuda_flag,
                                     bool cuda_buf_in, bool cuda_buf_out) {
  return new TorchInfer(model_file_path, cuda_flag, cuda_buf_in, cuda_buf_out);
}

TorchInfer::TorchInfer(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
                       bool cuda_buf_out)
    : impl_(new TorchInferImpl(model_file_path, cuda_flag, cuda_buf_in, cuda_buf_out)) {}

TorchInfer::~TorchInfer() {
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

TorchInferImpl::TorchInferImpl(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
                               bool cuda_buf_out)
    : model_path_(model_file_path) {
  try {
    infer_stream = c10::cuda::getStreamFromPool(true);
    stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(infer_stream);

    auto status = populate_model_details();
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR(status.get_message());
      HOLOSCAN_LOG_ERROR("Torch core: Error populating model parameters");
      throw std::runtime_error("Torch core: constructor failed.");
    }

    HOLOSCAN_LOG_INFO("Loading torchscript: {}", model_path_);
    inference_module_ = torch::jit::load(model_path_);
    inference_module_.eval();
    HOLOSCAN_LOG_INFO("Torchscript loaded");
    infer_device_ = cuda_flag ? torch::kCUDA : torch::kCPU;
    input_device_ = cuda_buf_in ? torch::kCUDA : torch::kCPU;
    output_device_ = cuda_buf_out ? torch::kCUDA : torch::kCPU;

    torch::jit::getProfilingMode() = false;  // profiling mode on slows down things in start
    // May be exposed as parameter in future releases
    torch::jit::GraphOptimizerEnabledGuard guard{false};
    inference_module_.to(infer_device_);

    torch::NoGradGuard no_grad;
  } catch (const c10::Error& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  } catch (std::exception& e) {
    HOLOSCAN_LOG_ERROR(e.what());
    throw;
  } catch (...) { throw; }
}

InferStatus TorchInfer::do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
                                     std::vector<std::shared_ptr<DataBuffer>>& output_buffer) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  impl_->stream_guard->reset_stream(impl_->infer_stream);

  if (impl_->input_nodes_ != input_buffer.size()) {
    status.set_message("Torch inference core: Input buffer size not equal to input nodes.");
    return status;
  }
  if (impl_->output_nodes_ != output_buffer.size()) {
    status.set_message("Torch inference core: Output buffer size not equal to output nodes.");
    return status;
  }

  try {
    impl_->input_tensors_.clear();
    impl_->output_tensors_.clear();
    impl_->inputs_.clear();

    for (size_t a = 0; a < input_buffer.size(); a++) {
      if (input_buffer[a]->host_buffer.size() == 0) {
        status.set_message("Torch inference core: Input Host buffer empty.");
        return status;
      }

      auto i_tensor = impl_->create_tensor(input_buffer[a], impl_->input_dims_[a]);

      if (i_tensor.numel() == 0) {
        status.set_message("Torch: Error creating torch tensor.");
        return status;
      }
      impl_->input_tensors_.push_back(std::move(i_tensor));
    }

    // Input tensors are in vector form
    impl_->inputs_.push_back(impl_->input_tensors_);
    auto outputs = impl_->inference_module_.forward(impl_->inputs_);

    if (impl_->infer_device_ == torch::kCUDA) { c10::cuda::getCurrentCUDAStream().synchronize(); }

    if (outputs.isTuple()) {
      int outputs_tuple_size = outputs.toTuple()->elements().size();
      // For rcnn, output in (losses, detections), thus tuple_index is 1.
      // For models with tuple as output, tuple index 1 is takes as inferred result
      // In future releases this can be modified from config
      if (outputs_tuple_size != 2) {
        HOLOSCAN_LOG_ERROR("Output tuple size of 2 is supported. Found {}", outputs_tuple_size);
        return status;
      }
      int tuple_index = 1;

      auto tuple_output = outputs.toTuple()->elements()[tuple_index];

      if (tuple_output.isList()) {
        auto list_output = tuple_output.toList();
        for (int lindex = 0; lindex < list_output.size(); lindex++) {
          if (list_output.get(lindex).isGenericDict()) {  // tested with retinanet application
            auto dict_outputs = list_output.get(lindex).toGenericDict();
            if (dict_outputs.size() != output_buffer.size()) {
              HOLOSCAN_LOG_ERROR(
                  "Output tensor size ({}) not equal to number of entries in the output "
                  "dictionary ({}) at index {}.",
                  output_buffer.size(),
                  dict_outputs.size(),
                  lindex);
              return status;
            }

            for (unsigned int a = 0; a < output_buffer.size(); a++) {
              if (dict_outputs.find(impl_->output_names_[a]) == dict_outputs.end()) {
                HOLOSCAN_LOG_ERROR("Tensor {} not found in model output.", impl_->output_names_[a]);
                return status;
              }
              torch::Tensor current_tensor = dict_outputs.at(impl_->output_names_[a]).toTensor();
              auto status = impl_->transfer_to_output(output_buffer, current_tensor, a);
              if (status.get_code() != holoinfer_code::H_SUCCESS) {
                HOLOSCAN_LOG_ERROR("Transfer of Tensor {} failed in inferece core.",
                                   impl_->output_names_[a]);
                return status;
              }
            }
          } else if (list_output.get(lindex).isTensor()) {  // not tested yet
            torch::Tensor current_tensor = list_output.get(lindex).toTensor();
            impl_->output_tensors_.push_back(current_tensor);
          } else {
            HOLOSCAN_LOG_ERROR("Output type {} not supported.", list_output.elementType()->str());
            return status;
          }
        }
      } else {
        if (tuple_output.isTensor()) {  // not tested yet
          torch::Tensor current_tensor = tuple_output.toTensor();
          impl_->output_tensors_.push_back(current_tensor);
        } else {
          HOLOSCAN_LOG_ERROR("Output type in the Tuple is neither Tensor nor List.");
          return status;
        }
      }
    } else if (outputs.isTensor()) {  // not tested
      torch::Tensor current_tensor = outputs.toTensor();
      impl_->output_tensors_.push_back(current_tensor);
    } else {
      HOLOSCAN_LOG_ERROR("Output type from the model is not among: Tuple, Tensor.");
      return status;
    }

    if (impl_->output_tensors_.size() > 0) {  // not tested yet. This is the case when output type
                                              // is Tensor or Tuple of List of tensors
      if (impl_->output_tensors_.size() != output_buffer.size()) {
        HOLOSCAN_LOG_ERROR(
            "Output buffer size ({}) not equal to number of tensors from the model ({})",
            output_buffer.size(),
            impl_->output_tensors_.size());
        return status;
      }
      for (unsigned int a = 0; a < output_buffer.size(); a++) {
        torch::Tensor current_tensor = impl_->output_tensors_[a];
        auto status = impl_->transfer_to_output(output_buffer, current_tensor, a);
        HOLOSCAN_LOG_ERROR("Transfer of Tensor {} failed in inferece core.",
                           impl_->output_names_[a]);
        return status;
      }
    }
  } catch (const c10::Error& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  } catch (std::exception& e) {
    HOLOSCAN_LOG_ERROR(e.what());
    throw;
  }
  return InferStatus();
}

std::vector<std::vector<int64_t>> TorchInfer::get_input_dims() const {
  return impl_->input_dims_;
}

std::vector<std::vector<int64_t>> TorchInfer::get_output_dims() const {
  return impl_->output_dims_;
}

std::vector<holoinfer_datatype> TorchInfer::get_input_datatype() const {
  return impl_->input_type_;
}

std::vector<holoinfer_datatype> TorchInfer::get_output_datatype() const {
  return impl_->output_type_;
}

}  // namespace inference
}  // namespace holoscan
