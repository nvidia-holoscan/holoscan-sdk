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

#include "opengl_renderer.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x534a004c0cdb47d9, 0x94ad4604cc7e81d7, "OpenGL", "OpenGL Extension",
                         "NVIDIA", "1.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x47a367a1050346cc, 0xbe9a81dd23541d75, nvidia::holoscan::OpenGLRenderer,
                    nvidia::gxf::Codelet, "OpenGL Renderer Codelet");
GXF_EXT_FACTORY_END()
