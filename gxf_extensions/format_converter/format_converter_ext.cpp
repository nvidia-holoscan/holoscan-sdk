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

#include "format_converter.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x6b7ee46293a142f6, 0xa92611c951965c2b, "FormatConverterExtension",
                         "Holoscan Format Converter extension", "NVIDIA", "0.2.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0xd8cdf281f6574461, 0x830d93ef6d1252e7,
                    nvidia::holoscan::formatconverter::FormatConverter, nvidia::gxf::Codelet,
                    "Format Converter codelet.");
GXF_EXT_FACTORY_END()
