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

#include "ping_rx.hpp"
#include "ping_tx.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x3e9f4558bcc140bc, 0xa919ab36801265eb, "Holoscan SampleExtension",
                         "Sample extension to demonstrate the use of GXF features", "NVIDIA",
                         "1.0.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0xb52ab9e5a33341a2, 0xb72574f35d77463e, nvidia::holoscan::sample::PingTx,
                    nvidia::gxf::Codelet, "Sends an entity");
GXF_EXT_FACTORY_ADD(0x2d3a1965f0534de1, 0xbd713bccd5c85c4f, nvidia::holoscan::sample::PingRx,
                    nvidia::gxf::Codelet, "Receives an entity");

GXF_EXT_FACTORY_END()
