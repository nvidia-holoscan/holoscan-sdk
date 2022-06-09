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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_DUMMY_DUMMY_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_DUMMY_DUMMY_HPP_

#include "gxf/std/codelet.hpp"

namespace nvidia {
namespace holoscan {
namespace dummy {

/// @brief Test input codelet for CMake integration.
class Source : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
};

}  // namespace dummy
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_DUMMY_DUMMY_HPP_
