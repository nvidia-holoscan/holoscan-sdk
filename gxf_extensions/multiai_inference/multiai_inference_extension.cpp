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
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

#include "multiai_inference.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xa5b5234fe9004901, 0x9e098b37d539ecbd, "MultiAIExtension",
                         "MultiAI with Holoscan Inference", "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x43c2567eb5a442f7, 0x9a1781764a01822e,
                    nvidia::holoscan::multiai::MultiAIInference, nvidia::gxf::Codelet,
                    "Multi AI Codelet.");

GXF_EXT_FACTORY_END()
}
