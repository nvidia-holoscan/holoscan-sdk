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

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/gxf/gxf_tensor.hpp"
#include "holoscan/core/message.hpp"
#include "operator_wrapper.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x12d01b4ee06f49ef, 0x93c4961834347385, "HoloscanWrapperExtension",
                         "Holoscan Wrapper extension", "NVIDIA", "0.5.1", "LICENSE");

// Register types/components that are used by Holoscan
GXF_EXT_FACTORY_ADD_0(0x61510ca06aa9493b, 0x8a777d0bf87476b7, holoscan::Message,
                      "Holoscan message type");
GXF_EXT_FACTORY_ADD(0xa02945eaf20e418c, 0x8e6992b68672ce40, holoscan::gxf::GXFTensor,
                    nvidia::gxf::Tensor, "Holoscan's GXF Tensor type");
GXF_EXT_FACTORY_ADD_0(0xa5eb0ed57d7f4aa2, 0xb5865ccca0ef955c, holoscan::Tensor,
                      "Holoscan's Tensor type");

// Register the wrapper codelet
GXF_EXT_FACTORY_ADD(0x04f99794e01b4bd1, 0xb42653a2e6d07347, holoscan::gxf::OperatorWrapper,
                    nvidia::gxf::Codelet, "Codelet for wrapping Holoscan Operator");
GXF_EXT_FACTORY_END()
