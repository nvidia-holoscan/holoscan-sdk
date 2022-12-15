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

#include "visualizer.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xd3b5b6f7291e42bc, 0x85d4a2b01327913a, "VisualizerToolTrackingExtension",
                         "Holoscan Surgical Tool Tracking Visualizer extension", "NVIDIA", "0.2.0",
                         "LICENSE");
GXF_EXT_FACTORY_ADD(0xab207890cc6b4391, 0x88e8d1dc47f3ae11,
                    nvidia::holoscan::visualizer_tool_tracking::Sink, nvidia::gxf::Codelet,
                    "Surgical Tool Tracking Viz codelet.");
GXF_EXT_FACTORY_END()
