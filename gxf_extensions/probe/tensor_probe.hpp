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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TENSOR_PROBE_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TENSOR_PROBE_HPP_

#include <cinttypes>
#include <string>
#include <vector>

#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {
namespace probe {

/// @brief Introspection codlet for observing tensor shapes and printing to standard output.
///
/// This codelet can be helpful for debugging information like tensor shapes and sizes sent between
/// codelets.
class TensorProbe : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> rx_;
};

}  // namespace probe
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_TENSOR_PROBE_HPP_
