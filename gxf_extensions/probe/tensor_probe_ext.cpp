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

#include "tensor_probe.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xada0b1069d4c445a, 0xb2e045c0b816499c, "TensorProbeExtension",
                         "Probe extension", "NVIDIA", "1.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x8be4de857921ddd9, 0x182106d326154aeb, nvidia::holoscan::probe::TensorProbe,
                    nvidia::gxf::Codelet, "Tensor name probe passthrough codelet.");
GXF_EXT_FACTORY_END()
