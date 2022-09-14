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

#include "ImGuiLayer.h"

#include "../vulkan/Vulkan.h"

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

namespace clara::holoviz
{

struct ImGuiLayer::Impl
{
    bool pushed_style_ = false;
};

ImGuiLayer::ImGuiLayer()
    : Layer(Type::ImGui)
    , impl_(new ImGuiLayer::Impl)
{
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, GetOpacity());
    impl_->pushed_style_ = true;
}

ImGuiLayer::~ImGuiLayer()
{
    if (impl_->pushed_style_ = true)
    {
        ImGui::PopStyleVar();
    }
}

void ImGuiLayer::SetOpacity(float opacity)
{
    // call the base class
    Layer::SetOpacity(opacity);

    // set the opacity
    if (impl_->pushed_style_)
    {
        ImGui::PopStyleVar();
        impl_->pushed_style_ = false;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, GetOpacity());
    impl_->pushed_style_ = true;
}

void ImGuiLayer::Render(Vulkan *vulkan)
{
    if (impl_->pushed_style_)
    {
        ImGui::PopStyleVar();
        impl_->pushed_style_ = false;
    }

    /// @todo we can't renderer multiple ImGui layers, figure out a way to handle that (store DrawData returned
    ///       by GetDrawData()?)

    // Render UI
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vulkan->GetCommandBuffer());
}

} // namespace clara::holoviz
