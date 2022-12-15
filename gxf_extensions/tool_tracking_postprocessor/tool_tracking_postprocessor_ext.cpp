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
#include "gxf/std/extension_factory_helper.hpp"

#include "tool_tracking_postprocessor.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xfbc6d815a2b54e7c, 0x8ced71e2f635a506,
                         "ToolTrackingPostprocessorExtension",
                         "Holoscan Tool Tracking Model Postprocessing extension", "NVIDIA", "0.4.0",
                         "LICENSE");
GXF_EXT_FACTORY_ADD(0xb07a675dec0c43a4, 0x8ce6e472cd90a9a1,
                    nvidia::holoscan::tool_tracking_postprocessor::Postprocessor,
                    nvidia::gxf::Codelet, "Tool Tracking Model Postprocessor codelet.");
GXF_EXT_FACTORY_END()
