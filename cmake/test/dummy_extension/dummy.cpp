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
#include "dummy.hpp"

namespace nvidia {
namespace holoscan {
namespace dummy {

gxf_result_t Source::start() {
  return GXF_SUCCESS;
}

gxf_result_t Source::stop() {
  return GXF_SUCCESS;
}

gxf_result_t Source::tick() {
  return GXF_SUCCESS;
}

gxf_result_t Source::registerInterface(gxf::Registrar* registrar) {
  return GXF_SUCCESS;
}

}  // namespace dummy
}  // namespace holoscan
}  // namespace nvidia
