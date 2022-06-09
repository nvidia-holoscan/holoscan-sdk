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

#include "video_buffer_mock.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xa494ec3e9f114704, 0xa0d181b3f1700d2e, "TestMockExtension",
                         "Holoscan Test Mock extension", "NVIDIA", "0.2.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0xa7d54d962b244b49, 0x94e4d60c0bb39903, nvidia::holoscan::test::VideoBufferMock,
                    nvidia::gxf::Codelet, "VideoBuffer Mock codelet.");
GXF_EXT_FACTORY_END()
