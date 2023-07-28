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

#include <gxf/core/gxf.h>
#include <gxf/std/clock.hpp>

#include "holoscan/core/gxf/gxf_scheduler.hpp"

namespace holoscan::gxf {
// scheduler initialization is delayed until runtime via `GXFExecutor::initialize_scheduler`

nvidia::gxf::Clock* GXFScheduler::gxf_clock() {
  if (this->clock()) {
    return static_cast<nvidia::gxf::Clock*>(this->clock()->gxf_cptr());
  } else {
    HOLOSCAN_LOG_ERROR("GXFScheduler clock is not set");
    return nullptr;
  }
}

}  // namespace holoscan::gxf
