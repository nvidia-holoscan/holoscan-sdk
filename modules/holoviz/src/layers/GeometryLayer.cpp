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

#include "GeometryLayer.h"

#include "../vulkan/Vulkan.h"

#include <imgui.h>

#include <array>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include <math.h>

namespace clara::holoviz
{

/// the segment count a circle is made of
constexpr uint32_t CIRCLE_SEGMENTS = 32;

class Attributes
{
public:
    Attributes()
        : color_({1.f, 1.f, 1.f, 1.f})
        , line_width_(1.f)
        , point_size_(1.f)
    {
    }

    bool operator==(const Attributes &rhs) const
    {
        return ((color_ == rhs.color_) && (line_width_ == rhs.line_width_) && (point_size_ == rhs.point_size_));
    }

    std::array<float, 4> color_;
    float line_width_;
    float point_size_;
};

class Primitive
{
public:
    Primitive(const Attributes &attributes, PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
              const float *data, uint32_t vertex_offset, std::vector<uint32_t> &vertex_counts,
              VkPrimitiveTopology vk_topology)
        : attributes_(attributes)
        , topology_(topology)
        , primitive_count_(primitive_count)
        , vertex_offset_(vertex_offset)
        , vertex_counts_(vertex_counts)
        , vk_topology_(vk_topology)
    {
        data_.assign(data, data + data_size);
    }
    Primitive() = delete;

    bool operator==(const Primitive &rhs) const
    {
        return ((attributes_ == rhs.attributes_) && (topology_ == rhs.topology_) &&
                (primitive_count_ == rhs.primitive_count_) && (data_ == rhs.data_));
    }

    const Attributes attributes_;

    const PrimitiveTopology topology_;
    const uint32_t primitive_count_;
    std::vector<float> data_;

    // internal state
    const uint32_t vertex_offset_;
    const std::vector<uint32_t> vertex_counts_;
    const VkPrimitiveTopology vk_topology_;
};

class Text
{
public:
    Text(const Attributes &attributes, float x, float y, float size, const char *text)
        : attributes_(attributes)
        , x_(x)
        , y_(y)
        , size_(size)
        , text_(text)
    {
    }

    bool operator==(const Text &rhs) const
    {
        return ((attributes_ == rhs.attributes_) && (x_ == rhs.x_) && (y_ == rhs.y_) && (size_ == rhs.size_) &&
                (text_ == rhs.text_));
    }

    const Attributes attributes_;
    const float x_;
    const float y_;
    const float size_;
    const std::string text_;
};

class GeometryLayer::Impl
{
public:
    bool CanBeReused(Impl &other) const
    {
        return ((vertex_count_ == other.vertex_count_) && (primitives_ == other.primitives_) &&
                (texts_ == other.texts_));
    }

    Attributes attributes_;

    std::list<class Primitive> primitives_;
    std::list<class Text> texts_;

    // internal state
    Vulkan *vulkan_ = nullptr;

    size_t vertex_count_           = 0;
    Vulkan::Buffer *vertex_buffer_ = nullptr;

    std::unique_ptr<ImDrawList> text_draw_list_;
    Vulkan::Buffer *text_vertex_buffer_ = nullptr;
    Vulkan::Buffer *text_index_buffer_  = nullptr;
};

GeometryLayer::GeometryLayer()
    : Layer(Type::Geometry)
    , impl_(new GeometryLayer::Impl)
{
}

GeometryLayer::~GeometryLayer()
{
    if (impl_->vulkan_)
    {
        if (impl_->vertex_buffer_)
        {
            impl_->vulkan_->DestroyBuffer(impl_->vertex_buffer_);
        }
        if (impl_->text_vertex_buffer_)
        {
            impl_->vulkan_->DestroyBuffer(impl_->text_vertex_buffer_);
        }
        if (impl_->text_index_buffer_)
        {
            impl_->vulkan_->DestroyBuffer(impl_->text_index_buffer_);
        }
    }
}

void GeometryLayer::Color(float r, float g, float b, float a)
{
    impl_->attributes_.color_[0] = r;
    impl_->attributes_.color_[1] = g;
    impl_->attributes_.color_[2] = b;
    impl_->attributes_.color_[3] = a;
}
void GeometryLayer::LineWidth(float width)
{
    impl_->attributes_.line_width_ = width;
}

void GeometryLayer::PointSize(float size)
{
    impl_->attributes_.point_size_ = size;
}

void GeometryLayer::Primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size, const float *data)
{
    if (primitive_count == 0)
    {
        throw std::invalid_argument("primitive_count should not be zero");
    }
    if (data_size == 0)
    {
        throw std::invalid_argument("data_size should not be zero");
    }
    if (data == nullptr)
    {
        throw std::invalid_argument("data should not be nullptr");
    }

    uint32_t required_data_size;
    std::vector<uint32_t> vertex_counts;
    VkPrimitiveTopology vkTopology;
    switch (topology)
    {
    case PrimitiveTopology::POINT_LIST:
        required_data_size = primitive_count * 2;
        vertex_counts.push_back(required_data_size / 2);
        vkTopology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        break;
    case PrimitiveTopology::LINE_LIST:
        required_data_size = primitive_count * 2 * 2;
        vertex_counts.push_back(required_data_size / 2);
        vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        break;
    case PrimitiveTopology::LINE_STRIP:
        required_data_size = 2 + primitive_count * 2;
        vertex_counts.push_back(required_data_size / 2);
        vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        break;
    case PrimitiveTopology::TRIANGLE_LIST:
        required_data_size = primitive_count * 3 * 2;
        vertex_counts.push_back(required_data_size / 2);
        vkTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        break;
    case PrimitiveTopology::CROSS_LIST:
        required_data_size = primitive_count * 3;
        vertex_counts.push_back(primitive_count * 4);
        vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        break;
    case PrimitiveTopology::RECTANGLE_LIST:
        required_data_size = primitive_count * 2 * 2;
        for (uint32_t i = 0; i < primitive_count; ++i)
        {
            vertex_counts.push_back(5);
        }
        vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        break;
    case PrimitiveTopology::OVAL_LIST:
        required_data_size = primitive_count * 4;
        for (uint32_t i = 0; i < primitive_count; ++i)
        {
            vertex_counts.push_back(CIRCLE_SEGMENTS + 1);
        }
        vkTopology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        break;
    }

    if (data_size < required_data_size)
    {
        std::stringstream buf;
        buf << "Required data array size is " << required_data_size << " but only " << data_size << " where specified";
        throw std::runtime_error(buf.str().c_str());
    }

    impl_->primitives_.emplace_back(impl_->attributes_, topology, primitive_count, data_size, data,
                                    impl_->vertex_count_, vertex_counts, vkTopology);

    for (auto &&vertex_count : vertex_counts)
    {
        impl_->vertex_count_ += vertex_count;
    }
}

void GeometryLayer::Text(float x, float y, float size, const char *text)
{
    if (size == 0)
    {
        throw std::invalid_argument("size should not be zero");
    }
    if (text == nullptr)
    {
        throw std::invalid_argument("text should not be nullptr");
    }

    impl_->texts_.emplace_back(impl_->attributes_, x, y, size, text);
}

bool GeometryLayer::CanBeReused(Layer &other) const
{
    return Layer::CanBeReused(other) && impl_->CanBeReused(*static_cast<const GeometryLayer &>(other).impl_.get());
}

void GeometryLayer::Render(Vulkan *vulkan)
{
    if (!impl_->primitives_.empty())
    {
        if (!impl_->vertex_buffer_)
        {
            /// @todo need to remember Vulkan instance for destroying buffer, destroy should propably be handled by Vulkan class
            impl_->vulkan_ = vulkan;

            // setup the vertex buffer
            std::vector<float> vertices;
            vertices.reserve(impl_->vertex_count_ * 2);

            for (auto &&primitive : impl_->primitives_)
            {
                switch (primitive.topology_)
                {
                case PrimitiveTopology::POINT_LIST:
                case PrimitiveTopology::LINE_LIST:
                case PrimitiveTopology::LINE_STRIP:
                case PrimitiveTopology::TRIANGLE_LIST:
                    // just copy
                    vertices.insert(vertices.end(), primitive.data_.begin(), primitive.data_.end());
                    break;
                case PrimitiveTopology::CROSS_LIST:
                    // generate crosses
                    for (uint32_t index = 0; index < primitive.primitive_count_; ++index)
                    {
                        const float x = primitive.data_[index * 3 + 0];
                        const float y = primitive.data_[index * 3 + 1];
                        const float s = primitive.data_[index * 3 + 2] * 0.5f;
                        vertices.insert(vertices.end(), {x - s, y, x + s, y, x, y - s, x, y + s});
                    }
                    break;
                case PrimitiveTopology::RECTANGLE_LIST:
                    // generate rectangles
                    for (uint32_t index = 0; index < primitive.primitive_count_; ++index)
                    {
                        const float x0 = primitive.data_[index * 4 + 0];
                        const float y0 = primitive.data_[index * 4 + 1];
                        const float x1 = primitive.data_[index * 4 + 2];
                        const float y1 = primitive.data_[index * 4 + 3];
                        vertices.insert(vertices.end(), {x0, y0, x1, y0, x1, y1, x0, y1, x0, y0});
                    }
                    break;
                case PrimitiveTopology::OVAL_LIST:
                    for (uint32_t index = 0; index < primitive.primitive_count_; ++index)
                    {
                        const float x  = primitive.data_[index * 4 + 0];
                        const float y  = primitive.data_[index * 4 + 1];
                        const float rx = primitive.data_[index * 4 + 2] * 0.5f;
                        const float ry = primitive.data_[index * 4 + 3] * 0.5f;
                        for (uint32_t segment = 0; segment <= CIRCLE_SEGMENTS; ++segment)
                        {
                            const float rad = (2.f * M_PI) / CIRCLE_SEGMENTS * segment;
                            const float px  = x + std::cos(rad) * rx;
                            const float py  = y + std::sin(rad) * ry;
                            vertices.insert(vertices.end(), {px, py});
                        }
                    }
                    break;
                }
            }

            impl_->vertex_buffer_ = vulkan->CreateBuffer(vertices.size() * sizeof(float), vertices.data(),
                                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        }

        for (auto &&primitive : impl_->primitives_)
        {
            uint32_t vertex_offset = primitive.vertex_offset_;
            for (auto &&vertex_count : primitive.vertex_counts_)
            {
                vulkan->Draw(primitive.vk_topology_, vertex_count, vertex_offset, impl_->vertex_buffer_, GetOpacity(),
                             primitive.attributes_.color_, primitive.attributes_.point_size_,
                             primitive.attributes_.line_width_);
                vertex_offset += vertex_count;
            }
        }
    }

    if (!impl_->texts_.empty())
    {
        if (!impl_->text_draw_list_)
        {
            impl_->text_draw_list_.reset(new ImDrawList(ImGui::GetDrawListSharedData()));
            impl_->text_draw_list_->_ResetForNewFrame();

            // ImGui is using integer coordinates for the text position, we use the 0...1 range. Therefore
            // generate vertices in larger scale and scale them down afterwards.
            const float scale = 16384.f;
            const ImVec4 clip_rect(0.f, 0.f, scale, scale);

            for (auto &&text : impl_->texts_)
            {
                const ImU32 color =
                    ImGui::ColorConvertFloat4ToU32(ImVec4(text.attributes_.color_[0], text.attributes_.color_[1],
                                                          text.attributes_.color_[2], text.attributes_.color_[3]));
                ImGui::GetFont()->RenderText(impl_->text_draw_list_.get(), text.size_ * scale,
                                             ImVec2(text.x_ * scale, text.y_ * scale), color, clip_rect,
                                             text.text_.c_str(), text.text_.c_str() + text.text_.size());
            }

            // text might be completely out of clip rectangle, if this is the case no vertices had been generated
            if (impl_->text_draw_list_->VtxBuffer.size() != 0)
            {
                // scale back vertex data
                const float inv_scale  = 1.f / scale;
                ImDrawVert *vertex     = impl_->text_draw_list_->VtxBuffer.Data;
                ImDrawVert *vertex_end = vertex + impl_->text_draw_list_->VtxBuffer.size();
                while (vertex < vertex_end)
                {
                    vertex->pos.x *= inv_scale;
                    vertex->pos.y *= inv_scale;
                    ++vertex;
                }

                impl_->text_vertex_buffer_ =
                    vulkan->CreateBuffer(impl_->text_draw_list_->VtxBuffer.size() * sizeof(ImDrawVert),
                                         impl_->text_draw_list_->VtxBuffer.Data, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
                impl_->text_index_buffer_ =
                    vulkan->CreateBuffer(impl_->text_draw_list_->IdxBuffer.size() * sizeof(ImDrawIdx),
                                         impl_->text_draw_list_->IdxBuffer.Data, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            }
        }

        for (int i = 0; i < impl_->text_draw_list_->CmdBuffer.size(); ++i)
        {
            const ImDrawCmd *pcmd = &impl_->text_draw_list_->CmdBuffer[i];
            vulkan->DrawIndexed(reinterpret_cast<VkDescriptorSet>(ImGui::GetIO().Fonts->TexID),
                                impl_->text_vertex_buffer_, impl_->text_index_buffer_,
                                (sizeof(ImDrawIdx) == 2) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32, pcmd->ElemCount,
                                pcmd->IdxOffset, pcmd->VtxOffset, GetOpacity());
        }
    }
}

} // namespace clara::holoviz
