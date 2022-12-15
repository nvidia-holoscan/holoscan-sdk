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

#include "v4l2_source.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xbe34ee7829e411eb, 0xc332f44c29e411eb, "V4L2Extension",
                         "V4L2 Video Extension", "NVIDIA", "1.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0xe0a3144029fc11eb, 0xe1505f5629fc11eb, nvidia::holoscan::V4L2Source,
                    nvidia::gxf::Codelet, "V4L2 Source Codelet");
GXF_EXT_FACTORY_END()
