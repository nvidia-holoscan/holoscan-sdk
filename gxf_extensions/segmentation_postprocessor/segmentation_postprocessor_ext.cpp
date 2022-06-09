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

#include "segmentation_postprocessor.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0x1486ad73a0f3496f, 0xa88ab90d35658f9a,
                         "SegmentationPostprocessorExtension",
                         "Holoscan Segmentation Model Postprocessing extension", "NVIDIA", "0.2.0",
                         "LICENSE");
GXF_EXT_FACTORY_ADD(0x9676cb6784e048e1, 0xaf6ed81223dda5aa,
                    nvidia::holoscan::segmentation_postprocessor::Postprocessor,
                    nvidia::gxf::Codelet, "OpenGL Segmentation Postprocessor codelet.");
GXF_EXT_FACTORY_END()
