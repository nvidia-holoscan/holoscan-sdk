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

#include "im_gui_layer.hpp"

#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <memory>
#include <vector>

#include "../vulkan/vulkan_app.hpp"

namespace holoscan::viz {

struct ImGuiLayer::Impl {
  const ImDrawData* draw_data_ = nullptr;

  Vulkan* vulkan_ = nullptr;
  Vulkan::Buffer* vertex_buffer_ = nullptr;
  Vulkan::Buffer* index_buffer_ = nullptr;
};

ImGuiLayer::ImGuiLayer() : Layer(Type::ImGui), impl_(new ImGuiLayer::Impl) {}

ImGuiLayer::~ImGuiLayer() {
  if (impl_->vulkan_) {
    if (impl_->vertex_buffer_) { impl_->vulkan_->destroy_buffer(impl_->vertex_buffer_); }
    if (impl_->index_buffer_) { impl_->vulkan_->destroy_buffer(impl_->index_buffer_); }
  }
}

void ImGuiLayer::set_opacity(float opacity) {
  // call the base class
  Layer::set_opacity(opacity);
}

void ImGuiLayer::end(Vulkan* vulkan) {
  /// @todo we can't renderer multiple ImGui layers, figure out
  ///       a way to handle that (store DrawData returned by GetDrawData()?)

  // Render UI
  ImGui::Render();

  impl_->draw_data_ = ImGui::GetDrawData();

  // nothing to do if there are no vertices
  if (impl_->draw_data_->TotalVtxCount > 0) {
    // copy all vertex and index data to one host buffer
    std::unique_ptr<ImDrawVert> vertex_data(new ImDrawVert[impl_->draw_data_->TotalVtxCount]);
    std::unique_ptr<ImDrawIdx> index_data(new ImDrawIdx[impl_->draw_data_->TotalIdxCount]);

    ImDrawVert* vertex = vertex_data.get();
    ImDrawIdx* index = index_data.get();

    for (int draw_list_index = 0; draw_list_index < impl_->draw_data_->CmdListsCount;
         ++draw_list_index) {
      const ImDrawList* cmd_list = impl_->draw_data_->CmdLists[draw_list_index];
      memcpy(vertex, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
      memcpy(index, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
      vertex += cmd_list->VtxBuffer.Size;
      index += cmd_list->IdxBuffer.Size;
    }

    // create device buffers from vertex and index data

    /// @todo need to remember Vulkan instance for destroying buffer,
    ///       destroy should probably be handled by Vulkan class
    impl_->vulkan_ = vulkan;

    if (impl_->vertex_buffer_) { impl_->vulkan_->destroy_buffer(impl_->vertex_buffer_); }
    if (impl_->index_buffer_) { impl_->vulkan_->destroy_buffer(impl_->index_buffer_); }

    impl_->vertex_buffer_ =
        vulkan->create_buffer(impl_->draw_data_->TotalVtxCount * sizeof(ImDrawVert),
                              vertex_data.get(),
                              vk::BufferUsageFlagBits::eVertexBuffer);
    impl_->index_buffer_ =
        vulkan->create_buffer(impl_->draw_data_->TotalIdxCount * sizeof(ImDrawIdx),
                              index_data.get(),
                              vk::BufferUsageFlagBits::eIndexBuffer);
  }
}

void ImGuiLayer::render(Vulkan* vulkan) {
  // nothing to do if there are no vertices
  if (impl_->draw_data_->TotalVtxCount == 0) { return; }

  // setup the base view matrix in a way that coordinates are in the range [0...1]
  nvmath::mat4f view_matrix_base;
  view_matrix_base.identity();
  view_matrix_base.translate({-1.f, -1.f, 0.f});
  view_matrix_base.scale(
      {2.f / impl_->draw_data_->DisplaySize.x, 2.f / impl_->draw_data_->DisplaySize.y, 1.f});

  std::vector<Layer::View> views = get_views();
  if (views.empty()) { views.push_back(Layer::View()); }

  for (const View& view : views) {
    vulkan->set_viewport(view.offset_x, view.offset_y, view.width, view.height);

    nvmath::mat4f view_matrix;
    if (view.matrix.has_value()) {
      view_matrix = view.matrix.value() * view_matrix_base;
    } else {
      view_matrix = view_matrix_base;
    }

    int vertex_offset = 0;
    int index_offset = 0;
    for (int draw_list_index = 0; draw_list_index < impl_->draw_data_->CmdListsCount;
         ++draw_list_index) {
      const ImDrawList* cmd_list = impl_->draw_data_->CmdLists[draw_list_index];
      for (int draw_cmd_index = 0; draw_cmd_index < cmd_list->CmdBuffer.size(); ++draw_cmd_index) {
        const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[draw_cmd_index];
        vulkan->draw_imgui(
            vk::DescriptorSet(reinterpret_cast<VkDescriptorSet>(ImGui::GetIO().Fonts->TexID)),
            impl_->vertex_buffer_,
            impl_->index_buffer_,
            (sizeof(ImDrawIdx) == 2) ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
            pcmd->ElemCount,
            pcmd->IdxOffset + index_offset,
            pcmd->VtxOffset + vertex_offset,
            get_opacity(),
            view_matrix);
      }
      vertex_offset += cmd_list->VtxBuffer.Size;
      index_offset += cmd_list->IdxBuffer.Size;
    }
  }
}

}  // namespace holoscan::viz
