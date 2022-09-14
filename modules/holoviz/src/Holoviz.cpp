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

#include "holoviz/Holoviz.h"

#include "Context.h"
#include "Window.h"

#include "layers/ImageLayer.h"
#include "layers/GeometryLayer.h"

#include <iostream>

#include <imgui.h>

namespace clara::holoviz
{

void Init(GLFWwindow *window, InitFlags flags)
{
    Context::Get().Init(window, flags);
}

void Init(uint32_t width, uint32_t height, const char *title, InitFlags flags)
{
    Context::Get().Init(width, height, title, flags);
}

void Init(const char *displayName, uint32_t width, uint32_t height, uint32_t refreshRate, InitFlags flags)
{
    Context::Get().Init(displayName, width, height, refreshRate, flags);
}

void ImGuiSetCurrentContext(ImGuiContext *context)
{
    Context::Get().ImGuiSetCurrentContext(context);
}

bool WindowShouldClose()
{
    return Context::Get().GetWindow()->ShouldClose();
}

bool WindowIsMinimized()
{
    return Context::Get().GetWindow()->IsMinimized();
}

void Shutdown()
{
    Context::Shutdown();
}

void Begin()
{
    Context::Get().Begin();
}

void End()
{
    Context::Get().End();
}

void BeginImageLayer()
{
    Context::Get().BeginImageLayer();
}

void ImageCudaDevice(uint32_t w, uint32_t h, ImageFormat fmt, CUdeviceptr device_ptr)
{
    Context::Get().GetActiveImageLayer()->ImageCudaDevice(w, h, fmt, device_ptr);
}

void ImageCudaArray(ImageFormat fmt, CUarray array)
{
    Context::Get().GetActiveImageLayer()->ImageCudaArray(fmt, array);
}

void ImageHost(uint32_t w, uint32_t h, ImageFormat fmt, const void *data)
{
    Context::Get().GetActiveImageLayer()->ImageHost(w, h, fmt, data);
}

void LUT(uint32_t size, ImageFormat fmt, size_t data_size, const void *data)
{
    Context::Get().GetActiveImageLayer()->LUT(size, fmt, data_size, data);
}

void BeginImGuiLayer()
{
    Context::Get().BeginImGuiLayer();
}

void BeginGeometryLayer()
{
    Context::Get().BeginGeometryLayer();
}

void Color(float r, float g, float b, float a)
{
    Context::Get().GetActiveGeometryLayer()->Color(r, g, b, a);
}

void LineWidth(float width)
{
    Context::Get().GetActiveGeometryLayer()->LineWidth(width);
}

void PointSize(float size)
{
    Context::Get().GetActiveGeometryLayer()->PointSize(size);
}

void Text(float x, float y, float size, const char *text)
{
    Context::Get().GetActiveGeometryLayer()->Text(x, y, size, text);
}

void Primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size, const float *data)
{
    Context::Get().GetActiveGeometryLayer()->Primitive(topology, primitive_count, data_size, data);
}

void LayerOpacity(float opacity)
{
    Context::Get().GetActiveLayer()->SetOpacity(opacity);
}
void LayerPriority(int32_t priority)
{
    Context::Get().GetActiveLayer()->SetPriority(priority);
}

void EndLayer()
{
    Context::Get().EndLayer();
}

} // namespace clara::holoviz
