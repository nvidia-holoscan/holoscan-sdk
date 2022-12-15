/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "visualizer_icardio.hpp"

namespace nvidia {
namespace holoscan {
namespace multiai {

gxf_result_t VisualizerICardio::start() {
  if (input_on_cuda_.get()) {
    return HoloInfer::report_error(module_, "CUDA based data not supported in iCardio Visualizer");
  }

  std::vector<int> logo_dim = tensor_to_shape_.at("logo");
  size_t logo_size =
      std::accumulate(logo_dim.begin(), logo_dim.end(), 1, std::multiplies<size_t>());
  logo_image_.assign(logo_size, 0);

  std::ifstream file_logo(path_to_logo_file_);

  if (!file_logo) {
    GXF_LOG_WARNING("Logo file not found, Ignored.");
  } else {
    std::istream_iterator<int> start(file_logo), end;
    std::vector<int> data_logo(start, end);

    logo_image_ = std::move(data_logo);
  }
  return GXF_SUCCESS;
}

gxf_result_t VisualizerICardio::stop() {
  return GXF_SUCCESS;
}

gxf_result_t VisualizerICardio::tick() {
  try {
    gxf_result_t stat = HoloInfer::multiai_get_data_per_model(receivers_.get(),
                                                              in_tensor_names_.get(),
                                                              data_per_tensor,
                                                              tensor_size_map_,
                                                              input_on_cuda_.get(),
                                                              module_);
    if (stat != GXF_SUCCESS) { return HoloInfer::report_error(module_, "Tick, Data extraction"); }

    if (tensor_size_map_.find(pc_tensor_name_) == tensor_size_map_.end()) {
      HoloInfer::report_error(module_, "Dimension not found for tensor " + pc_tensor_name_);
    }
    if (data_per_tensor.find(pc_tensor_name_) == data_per_tensor.end()) {
      HoloInfer::report_error(module_, "Data not found for tensor " + pc_tensor_name_);
    }
    auto coords = data_per_tensor.at(pc_tensor_name_)->host_buffer.data();
    auto datasize = tensor_size_map_[pc_tensor_name_];

    if (transmitters_.get().size() > 0) {
      for (unsigned int a = 0; a < transmitters_.get().size(); ++a) {
        auto out_message = gxf::Entity::New(context());
        if (!out_message) {
          return HoloInfer::report_error(module_, "Tick, Out message allocation");
        }
        std::string current_tensor_name{out_tensor_names_.get()[a]};
        auto out_tensor = out_message.value().add<gxf::Tensor>(current_tensor_name.c_str());
        if (!out_tensor) { return HoloInfer::report_error(module_, "Tick, Out tensor allocation"); }
        if (tensor_to_shape_.find(current_tensor_name) == tensor_to_shape_.end()) {
          return HoloInfer::report_error(
              module_, "Tick, Output Tensor shape mapping not found for " + current_tensor_name);
        }
        std::vector<int> shape_dim = tensor_to_shape_.at(current_tensor_name);
        gxf::Shape output_shape{shape_dim[0], shape_dim[1], shape_dim[2]};

        if (current_tensor_name.compare("logo") == 0) {
          out_tensor.value()->reshape<uint8_t>(
              output_shape, gxf::MemoryStorageType::kHost, allocator_);
          if (!out_tensor.value()->pointer()) {
            return HoloInfer::report_error(module_, "Tick, Out tensor buffer allocation");
          }
          gxf::Expected<uint8_t*> out_tensor_data = out_tensor.value()->data<uint8_t>();
          if (!out_tensor_data) {
            return HoloInfer::report_error(module_, "Tick, Getting out tensor data");
          }
          uint8_t* out_tensor_buffer = out_tensor_data.value();

          for (unsigned int h = 0; h < shape_dim[0] * shape_dim[1] * shape_dim[2]; ++h) {
            out_tensor_buffer[h] = uint8_t(logo_image_.data()[h]);
          }
        } else {
          out_tensor.value()->reshape<float>(
              output_shape, gxf::MemoryStorageType::kHost, allocator_);
          if (!out_tensor.value()->pointer()) {
            return HoloInfer::report_error(module_, "Tick, Out tensor buffer allocation");
          }
          gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
          if (!out_tensor_data) {
            return HoloInfer::report_error(module_, "Tick, Getting out tensor data");
          }
          float* out_tensor_buffer = out_tensor_data.value();

          int property_size = shape_dim[2];

          if (property_size <= 3) {
            for (unsigned int i = 1; i < datasize[datasize.size() - 1] / 2; ++i) {
              unsigned int index = (i - 1) * property_size;
              out_tensor_buffer[index] = coords[2 * i + 1];
              out_tensor_buffer[index + 1] = coords[2 * i];

              if (property_size == 3) {  // keypoint
                out_tensor_buffer[index + 2] = 0.01;
              }
            }
          } else {
            if (tensor_to_index_.find(current_tensor_name) == tensor_to_index_.end()) {
              return HoloInfer::report_error(module_, "Tick, tensor to index mapping failed");
            }
            int index_coord = tensor_to_index_.at(current_tensor_name);
            if (index_coord >= 1 && index_coord <= 5) {
              out_tensor_buffer[0] = coords[2 * index_coord + 1];
              out_tensor_buffer[1] = coords[2 * index_coord];
              out_tensor_buffer[2] = 0.04;
              out_tensor_buffer[3] = 0.02;
            } else {
              return HoloInfer::report_error(module_, "Tick, invalid coordinate from tensor");
            }
          }
        }
        const auto result = transmitters_.get()[a]->publish(std::move(out_message.value()));
        if (!result) { return HoloInfer::report_error(module_, "Tick, Publishing output"); }
      }
    }
  } catch (const std::runtime_error& r_) {
    return HoloInfer::report_error(module_, "Tick, Message->" + std::string(r_.what()));
  } catch (...) { return HoloInfer::report_error(module_, "Tick, unknown exception"); }
  return GXF_SUCCESS;
}

gxf_result_t VisualizerICardio::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
      in_tensor_names_, "in_tensor_names", "Input Tensors", "Input tensors", {std::string("")});
  result &= registrar->parameter(
      out_tensor_names_, "out_tensor_names", "Output Tensors", "Output tensors", {std::string("")});
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(
      input_on_cuda_, "input_on_cuda", "Input for processing on cuda", "", false);
  result &= registrar->parameter(
      receivers_, "receivers", "Receivers", "List of receivers to take input tensors");
  result &=
      registrar->parameter(transmitters_, "transmitters", "Transmitters", "List of transmitters");

  return gxf::ToResultCode(result);
}

}  // namespace multiai
}  // namespace holoscan
}  // namespace nvidia
