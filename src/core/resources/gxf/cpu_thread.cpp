/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/cpu_thread.hpp"

#include <gxf/core/gxf.h>

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

CPUThread::CPUThread(const std::string& name, nvidia::gxf::CPUThread* component)
    : gxf::GXFResource(name, component) {
  bool pin_entity = false;
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterGetBool(gxf_context_, gxf_cid_, "pin_entity", &pin_entity));
  pin_entity_ = pin_entity;
}

void CPUThread::setup(ComponentSpec& spec) {
  spec.param(pin_entity_,
             "pin_entity",
             "Pin Entity",
             "Set the entity to be pinned to a worker thread or not.",
             false);
}

}  // namespace holoscan
