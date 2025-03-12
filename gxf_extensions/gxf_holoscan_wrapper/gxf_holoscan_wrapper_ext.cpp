/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/gxf/gxf_scheduling_term_wrapper.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/metadata.hpp"
#include "operator_wrapper.hpp"
#include "resource_wrapper.hpp"

// Helper macros to convert macro value to string
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define HOLOSCAN_BUILD_VERSION_STR TOSTRING(HOLOSCAN_BUILD_VERSION)

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x12d01b4ee06f49ef, 0x93c4961834347385, "HoloscanWrapperExtension",
                         "Holoscan Wrapper extension", "NVIDIA", HOLOSCAN_BUILD_VERSION_STR,
                         "LICENSE");

// Register types/components that are used by Holoscan
GXF_EXT_FACTORY_ADD(0xbcfb5603b060495b, 0xad0e47c3523ee88e, holoscan::gxf::GXFWrapper,
                    nvidia::gxf::Codelet, "GXF wrapper to support Holoscan SDK native operators");

GXF_EXT_FACTORY_ADD(0x3b8b521cbda54bbe, 0xa241ed132937a1b5, holoscan::gxf::GXFSchedulingTermWrapper,
                    nvidia::gxf::SchedulingTerm,
                    "GXF wrapper to support Holoscan SDK native conditions");

GXF_EXT_FACTORY_ADD_0(0x61510ca06aa9493b, 0x8a777d0bf87476b7, holoscan::Message,
                      "Holoscan message type");
GXF_EXT_FACTORY_ADD_0(0xa5eb0ed57d7f4aa2, 0xb5865ccca0ef955c, holoscan::Tensor,
                      "Holoscan's Tensor type");
GXF_EXT_FACTORY_ADD_0(0x112607eb7b23407c, 0xb93fcd10ad8b2ba7, holoscan::MetadataDictionary,
                      "Holoscan's MetaDataDictionary type");

// Register the wrapper codelet
GXF_EXT_FACTORY_ADD(0x04f99794e01b4bd1, 0xb42653a2e6d07347, holoscan::gxf::OperatorWrapper,
                    nvidia::gxf::Codelet, "Codelet for wrapping Holoscan Operator");

// Register the wrapper component
GXF_EXT_FACTORY_ADD(0x9a790deec62646a0, 0x84e4a69d1fb671b0, holoscan::gxf::ResourceWrapper,
                    nvidia::gxf::Component, "Component for wrapping Holoscan Resource");
GXF_EXT_FACTORY_END()
