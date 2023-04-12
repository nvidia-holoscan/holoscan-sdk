/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensor_rt_inference.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x889fdb7676144909, 0x8d610400858c14b0, "TensorRTExtension", "TensorRT",
                         "Nvidia_Gxf_TensorRT", "2.0.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x03ff707d298e4be1, 0xb9b6b1f13974b8d9, nvidia::gxf::TensorRtInference,
                    nvidia::gxf::Codelet,
                    "Codelet taking input tensors and feed them into TensorRT for inference.");
GXF_EXT_FACTORY_END()
