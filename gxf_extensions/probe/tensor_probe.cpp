/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensor_probe.hpp"

#include <utility>
#include <vector>

#include "gxf/core/handle.hpp"
#include "gxf/std/tensor.hpp"

namespace nvidia {
namespace holoscan {
namespace probe {

gxf_result_t TensorProbe::start() {
  return GXF_SUCCESS;
}

gxf_result_t TensorProbe::stop() {
  return GXF_SUCCESS;
}

gxf_result_t TensorProbe::tick() {
  gxf::Expected<gxf::Entity> maybe_message = rx_->receive();
  if (!maybe_message) {
    GXF_LOG_ERROR("Message not available.");
    return maybe_message.error();
  }

  std::vector<nvidia::gxf::Handle<nvidia::gxf::Tensor>> tensor_metas =
      maybe_message.value().findAll<gxf::Tensor>();
  GXF_LOG_INFO("Getting tensors");

  for (nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor_meta : tensor_metas) {
    GXF_LOG_INFO("Tensor name: %s (name length %d)", tensor_meta.name(),
                 strlen(tensor_meta.name()));

    gxf::Expected<gxf::Handle<gxf::Tensor>> maybe_tensor =
        maybe_message.value().get<gxf::Tensor>(tensor_meta.name());
    if (!maybe_tensor) {
      GXF_LOG_ERROR("Tensor %s not available.", tensor_meta.name());
      return maybe_tensor.error();
    }

    const auto& tensor = maybe_tensor.value();
    auto shape = tensor->shape();
    // Prints out dimension
    {
      std::stringbuf sbuf;
      std::ostream stream(&sbuf);
      stream << "[";
      for (uint32_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) { stream << ", "; }
        stream << shape.dimension(i);
      }
      stream << "]";

      GXF_LOG_INFO("Input tensor: %s, Dimension: %s", tensor_meta.name(), sbuf.str().c_str());
    }

    // Print element type
    GXF_LOG_INFO("Input tensor: %s, Element Type: %d", tensor_meta.name(), tensor->element_type());
  }

  return GXF_SUCCESS;
}

gxf_result_t TensorProbe::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(rx_, "rx", "RX", "Receiver of tensor message.");
  return gxf::ToResultCode(result);
}

}  // namespace probe
}  // namespace holoscan
}  // namespace nvidia
