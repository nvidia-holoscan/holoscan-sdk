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
#include "tool_tracking_postprocessor.hpp"

#include <string>
#include <utility>

#include "gxf/std/tensor.hpp"

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).", #stmt, \
                    __LINE__, __FILE__, cudaGetErrorString(_holoscan_cuda_err),                 \
                    _holoscan_cuda_err);                                                        \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

namespace nvidia {
namespace holoscan {
namespace tool_tracking_postprocessor {

constexpr float DEFAULT_MIN_PROB = 0.5f;
// 12 qualitative classes color scheme from colorbrewer2
static const std::vector<std::vector<float>> DEFAULT_COLORS = {
    {0.12f, 0.47f, 0.71f}, {0.20f, 0.63f, 0.17f}, {0.89f, 0.10f, 0.11f}, {1.00f, 0.50f, 0.00f},
    {0.42f, 0.24f, 0.60f}, {0.69f, 0.35f, 0.16f}, {0.65f, 0.81f, 0.89f}, {0.70f, 0.87f, 0.54f},
    {0.98f, 0.60f, 0.60f}, {0.99f, 0.75f, 0.44f}, {0.79f, 0.70f, 0.84f}, {1.00f, 1.00f, 0.60f}};

gxf_result_t Postprocessor::start() {
  return GXF_SUCCESS;
}

gxf_result_t Postprocessor::tick() {
  // Process input message
  const auto in_message = in_->receive();
  if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

  // Get tensors attached to the message
  auto maybe_tensor = in_message.value().get<gxf::Tensor>("probs");
  if (!maybe_tensor) {
    GXF_LOG_ERROR("Tensor 'probs' not found in message.");
    return GXF_FAILURE;
  }
  const gxf::Handle<gxf::Tensor> probs_tensor = maybe_tensor.value();
  std::vector<float> probs(probs_tensor->size() / sizeof(float));
  CUDA_TRY(cudaMemcpy(probs.data(), probs_tensor->pointer(), probs_tensor->size(),
                      cudaMemcpyDeviceToHost));

  maybe_tensor = in_message.value().get<gxf::Tensor>("scaled_coords");
  if (!maybe_tensor) {
    GXF_LOG_ERROR("Tensor 'scaled_coords' not found in message.");
    return GXF_FAILURE;
  }
  const gxf::Handle<gxf::Tensor> scaled_coords_tensor = maybe_tensor.value();
  std::vector<float> scaled_coords(scaled_coords_tensor->size() / sizeof(float));
  CUDA_TRY(cudaMemcpy(scaled_coords.data(), scaled_coords_tensor->pointer(),
                      scaled_coords_tensor->size(), cudaMemcpyDeviceToHost));

  maybe_tensor = in_message.value().get<gxf::Tensor>("binary_masks");
  if (!maybe_tensor) {
    GXF_LOG_ERROR("Tensor 'binary_masks' not found in message.");
    return GXF_FAILURE;
  }
  const gxf::Handle<gxf::Tensor> binary_masks_tensor = maybe_tensor.value();

  auto out_message = gxf::Entity::New(context());
  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate output message");
    return GXF_FAILURE;
  }

  // filter coordinates based on probability
  std::vector<uint32_t> visible_classes;
  {
    std::vector<float> filtered_scaled_coords;
    for (size_t index = 0; index < probs.size(); ++index) {
      if (probs[index] > min_prob_) {
        filtered_scaled_coords.push_back(scaled_coords[index * 2]);
        filtered_scaled_coords.push_back(scaled_coords[index * 2 + 1]);
        visible_classes.push_back(index);
      } else {
        filtered_scaled_coords.push_back(-1.f);
        filtered_scaled_coords.push_back(-1.f);
      }
    }

    const auto out_tensor = out_message.value().add<gxf::Tensor>("scaled_coords");
    if (!out_tensor) {
      GXF_LOG_ERROR("Failed to allocate output tensor 'scaled_coords'");
      return GXF_FAILURE;
    }

    const gxf::Shape output_shape{1, int32_t(filtered_scaled_coords.size() / 2), 2};
    out_tensor.value()->reshape<float>(output_shape, gxf::MemoryStorageType::kHost,
                                       host_allocator_);
    if (!out_tensor.value()->pointer()) {
      GXF_LOG_ERROR("Failed to allocate output tensor buffer for tensor 'scaled_coords'.");
      return GXF_FAILURE;
    }
    memcpy(out_tensor.value()->data<float>().value(), filtered_scaled_coords.data(),
           filtered_scaled_coords.size() * sizeof(float));
  }

  // filter binary mask
  {
    const auto out_tensor = out_message.value().add<gxf::Tensor>("mask");
    if (!out_tensor) {
      GXF_LOG_ERROR("Failed to allocate output tensor 'mask'");
      return GXF_FAILURE;
    }

    const gxf::Shape output_shape{binary_masks_tensor->shape().dimension(2),
                                  binary_masks_tensor->shape().dimension(3), 4};
    out_tensor.value()->reshape<float>(output_shape, gxf::MemoryStorageType::kDevice,
                                       device_allocator_);
    if (!out_tensor.value()->pointer()) {
      GXF_LOG_ERROR("Failed to allocate output tensor buffer for tensor 'mask'.");
      return GXF_FAILURE;
    }

    float* const out_data = out_tensor.value()->data<float>().value();
    const size_t layer_size = output_shape.dimension(0) * output_shape.dimension(1);
    bool first = true;
    for (auto& index : visible_classes) {
      const auto& img_color =
          overlay_img_colors_.get()[std::min(index, uint32_t(overlay_img_colors_.get().size()))];
      const std::array<float, 3> color{{img_color[0], img_color[1], img_color[2]}};
      cuda_postprocess(output_shape.dimension(0), output_shape.dimension(1), color, first,
                       binary_masks_tensor->data<float>().value() + index * layer_size,
                       reinterpret_cast<float4*>(out_data));
      first = false;
    }
  }

  const auto result = out_->publish(std::move(out_message.value()));
  if (!result) {
    GXF_LOG_ERROR("Failed to publish output!");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t Postprocessor::stop() {
  return GXF_SUCCESS;
}

gxf_result_t Postprocessor::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_, "in", "Input", "Input channel.");
  result &= registrar->parameter(out_, "out", "Output", "Output channel.");

  result &= registrar->parameter(min_prob_, "min_prob", "Minimum probability",
                                 "Minimum probability.",
                                 DEFAULT_MIN_PROB);

  result &= registrar->parameter(
      overlay_img_colors_, "overlay_img_colors", "Overlay Image Layer Colors",
      "Color of the image overlays, a list of RGB values with components between 0 and 1",
      DEFAULT_COLORS);

  result &=
      registrar->parameter(host_allocator_, "host_allocator", "Allocator", "Output Allocator");
  result &=
      registrar->parameter(device_allocator_, "device_allocator", "Allocator", "Output Allocator");
  return gxf::ToResultCode(result);
}

}  // namespace tool_tracking_postprocessor
}  // namespace holoscan
}  // namespace nvidia
