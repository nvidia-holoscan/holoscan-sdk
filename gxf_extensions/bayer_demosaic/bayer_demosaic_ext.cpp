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

#include "bayer_demosaic.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xd14b5bdcb2c34d8a, 0x9acf91533c4146ae, "BayerDemosaic",
                         "Bayer demosacing extension", "NVIDIA", "0.1.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0xd00d673261e142f9, 0xac511a945286fcad, nvidia::holoscan::BayerDemosaic,
                    nvidia::gxf::Codelet, "Bayer demosaic component");
GXF_EXT_FACTORY_END()
