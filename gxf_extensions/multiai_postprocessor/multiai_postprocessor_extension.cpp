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

#include "multiai_postprocessor.hpp"

#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xaa47a2a6d6b9471c, 0xbe5292af3cbb59f8, "MultiAIPostprocessorExtension",
                         "MultiAI Postprocessor", "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0xa569e128ee09448a, 0xb6e08107f0edd2ea,
                    nvidia::holoscan::multiai::MultiAIPostprocessor, nvidia::gxf::Codelet,
                    "Multi AI Postprocessor Codelet.");

GXF_EXT_FACTORY_END()
}
