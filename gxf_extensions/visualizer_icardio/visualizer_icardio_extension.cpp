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

#include "visualizer_icardio.hpp"
#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"

extern "C" {

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xc09bcf6af8b2458f, 0xb9f3b0bd935a4124, "VisualizerICardio",
                         "Visualizer iCardio", "NVIDIA", "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0x70cedec6a66e4812, 0x8114c6f9bce0048c,
                    nvidia::holoscan::multiai::VisualizerICardio, nvidia::gxf::Codelet,
                    "Visualizer iCardio Codelet.");

GXF_EXT_FACTORY_END()
}
