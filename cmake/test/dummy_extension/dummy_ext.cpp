/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gxf/std/extension_factory_helper.hpp"

#include "dummy.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x4eec4820439b4fba, 0x982c0ebfe3170f6f, "DummyExtension",
                         "Dummy extension for CMake testing", "NVIDIA", "0.2.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x555dc30cbf054b33, 0x85d50435e3875038, nvidia::holoscan::dummy::Source,
                    nvidia::gxf::Codelet, "Dummy for CMake.");
GXF_EXT_FACTORY_END()
