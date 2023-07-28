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

#include "gxf/std/extension_factory_helper.hpp"
#include "ucx_holoscan_component_serializer.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xe549f7ce9ecf4d53, 0x8156418727c176df, "UcxHoloscanExtension",
                         "Extension for Unified Communication X framework with Holoscan", "NVIDIA",
                         "0.0.6", "LICENSE");
GXF_EXT_FACTORY_ADD(0xb8de0c9d54c64a2d, 0x88b6b642ad1bb268,
                    nvidia::gxf::UcxHoloscanComponentSerializer, nvidia::gxf::ComponentSerializer,
                    "Holoscan component serializer for UCX.");
GXF_EXT_FACTORY_END()
