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

#include "segmentation_visualizer.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xf52289414bec4f59, 0xb02f1b532c3f19a4, "SegmentationVisualizerExtension",
                         "Holoscan Segmentation Visualizer extension", "NVIDIA", "0.2.0",
                         "LICENSE");
GXF_EXT_FACTORY_ADD(0xe048718b636f40ba, 0x993c15d1d530b56e,
                    nvidia::holoscan::segmentation_visualizer::Visualizer, nvidia::gxf::Codelet,
                    "OpenGL Segmentation Vizualizer codelet.");
GXF_EXT_FACTORY_END()
