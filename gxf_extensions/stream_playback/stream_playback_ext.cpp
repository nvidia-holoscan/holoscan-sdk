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

#include "video_stream_serializer.hpp"

GXF_EXT_FACTORY_BEGIN()
GXF_EXT_FACTORY_SET_INFO(0xe6c168715f3f428d, 0x96cd24dce2f42f46, "StreamPlaybackExtension",
                         "Holoscan StreamPlayback extension", "NVIDIA", "0.2.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0x7ee08fcc84c94245, 0xa415022b42f4ef39,
                    nvidia::holoscan::stream_playback::VideoStreamSerializer,
                    nvidia::gxf::EntitySerializer, "VideoStreamSerializer component.");
GXF_EXT_FACTORY_END()
