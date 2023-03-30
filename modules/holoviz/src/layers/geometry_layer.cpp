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

#include "geometry_layer.hpp"

#include <imgui.h>
#include <math.h>
#include <nvmath/nvmath.h>

#include <array>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../context.hpp"
#include "../cuda/cuda_service.hpp"
#include "../vulkan/vulkan_app.hpp"

namespace holoscan::viz {

/// the segment count a circle is made of
constexpr uint32_t CIRCLE_SEGMENTS = 32;

class Attributes {
 public:
  Attributes() : color_({1.f, 1.f, 1.f, 1.f}), line_width_(1.f), point_size_(1.f) {}

  bool operator==(const Attributes& rhs) const {
    return ((color_ == rhs.color_) && (line_width_ == rhs.line_width_) &&
            (point_size_ == rhs.point_size_));
  }

  std::array<float, 4> color_;
  float line_width_;
  float point_size_;
};

class Primitive {
 public:
  Primitive(const Attributes& attributes, PrimitiveTopology topology, uint32_t primitive_count,
            size_t data_size, const float* data, uint32_t vertex_offset,
            std::vector<uint32_t>& vertex_counts, vk::PrimitiveTopology vk_topology)
      : attributes_(attributes),
        topology_(topology),
        primitive_count_(primitive_count),
        vertex_offset_(vertex_offset),
        vertex_counts_(vertex_counts),
        vk_topology_(vk_topology) {
    data_.assign(data, data + data_size);
  }
  Primitive() = delete;

  bool operator==(const Primitive& rhs) const {
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
  const vk::PrimitiveTopology vk_topology_;
};

class Text {
 public:
  Text(const Attributes& attributes, float x, float y, float size, const char* text)
      : attributes_(attributes), x_(x), y_(y), size_(size), text_(text) {}

  bool operator==(const Text& rhs) const {
    return ((attributes_ == rhs.attributes_) && (x_ == rhs.x_) && (y_ == rhs.y_) &&
            (size_ == rhs.size_) && (text_ == rhs.text_));
  }

  const Attributes attributes_;
  const float x_;
  const float y_;
  const float size_;
  const std::string text_;
};

class DepthMap {
 public:
  DepthMap(const Attributes& attributes, DepthMapRenderMode render_mode, uint32_t width,
           uint32_t height, CUdeviceptr depth_device_ptr, CUdeviceptr color_device_ptr,
           CUstream cuda_stream)
      : attributes_(attributes),
        render_mode_(render_mode),
        width_(width),
        height_(height),
        depth_device_ptr_(depth_device_ptr),
        color_device_ptr_(color_device_ptr),
        cuda_stream_(cuda_stream) {}

  bool operator==(const DepthMap& rhs) const {
    // check attributes, render mode, width and height. Skip *_device_ptr_ since that data
    // will be uploaded each frame. The *_device_ptr_ is updated in the can_be_reused() function
    // below.
    return ((attributes_ == rhs.attributes_) && (render_mode_ == rhs.render_mode_) &&
            (width_ == rhs.width_) && (height_ == rhs.height_));
  }

  const Attributes attributes_;
  const DepthMapRenderMode render_mode_;
  const uint32_t width_;
  const uint32_t height_;
  CUdeviceptr color_device_ptr_;
  CUdeviceptr depth_device_ptr_;
  CUstream cuda_stream_;

  // internal state
  uint32_t index_count_ = 0;
  uint32_t index_offset_ = 0;
  uint32_t vertex_offset_ = 0;
};

class GeometryLayer::Impl {
 public:
  bool can_be_reused(Impl& other) const {
    if ((vertex_count_ == other.vertex_count_) && (primitives_ == other.primitives_) &&
        (texts_ == other.texts_) && (depth_maps_ == other.depth_maps_)) {
      // update the Cuda device pointers and the cuda stream.
      // Data will be uploaded when drawing regardless if the layer is reused or not
      /// @todo this should be made explicit, first check if the layer can be reused and then
      ///     update the reused layer with these properties below which don't prevent reusing
      auto it = other.depth_maps_.begin();
      for (auto&& depth_map : depth_maps_) {
        it->depth_device_ptr_ = depth_map.depth_device_ptr_;
        it->color_device_ptr_ = depth_map.color_device_ptr_;
        it->cuda_stream_ = depth_map.cuda_stream_;
      }
      return true;
    }
    return false;
  }

  Attributes attributes_;

  std::list<class Primitive> primitives_;
  std::list<class Text> texts_;
  std::list<class DepthMap> depth_maps_;

  // internal state
  Vulkan* vulkan_ = nullptr;

  float aspect_ratio_ = 1.f;

  size_t vertex_count_ = 0;
  Vulkan::Buffer* vertex_buffer_ = nullptr;

  std::unique_ptr<ImDrawList> text_draw_list_;
  Vulkan::Buffer* text_vertex_buffer_ = nullptr;
  Vulkan::Buffer* text_index_buffer_ = nullptr;

  size_t depth_map_vertex_count_ = 0;
  Vulkan::Buffer* depth_map_vertex_buffer_ = nullptr;
  Vulkan::Buffer* depth_map_index_buffer_ = nullptr;
  Vulkan::Buffer* depth_map_color_buffer_ = nullptr;
};

GeometryLayer::GeometryLayer() : Layer(Type::Geometry), impl_(new GeometryLayer::Impl) {}

GeometryLayer::~GeometryLayer() {
  if (impl_->vulkan_) {
    if (impl_->vertex_buffer_) { impl_->vulkan_->destroy_buffer(impl_->vertex_buffer_); }
    if (impl_->text_vertex_buffer_) { impl_->vulkan_->destroy_buffer(impl_->text_vertex_buffer_); }
    if (impl_->text_index_buffer_) { impl_->vulkan_->destroy_buffer(impl_->text_index_buffer_); }
    if (impl_->depth_map_vertex_buffer_) {
      impl_->vulkan_->destroy_buffer(impl_->depth_map_vertex_buffer_);
    }
    if (impl_->depth_map_index_buffer_) {
      impl_->vulkan_->destroy_buffer(impl_->depth_map_index_buffer_);
    }
    if (impl_->depth_map_color_buffer_) {
      impl_->vulkan_->destroy_buffer(impl_->depth_map_color_buffer_);
    }
  }
}

void GeometryLayer::color(float r, float g, float b, float a) {
  impl_->attributes_.color_[0] = r;
  impl_->attributes_.color_[1] = g;
  impl_->attributes_.color_[2] = b;
  impl_->attributes_.color_[3] = a;
}
void GeometryLayer::line_width(float width) {
  impl_->attributes_.line_width_ = width;
}

void GeometryLayer::point_size(float size) {
  impl_->attributes_.point_size_ = size;
}

void GeometryLayer::primitive(PrimitiveTopology topology, uint32_t primitive_count,
                              size_t data_size, const float* data) {
  if (primitive_count == 0) { throw std::invalid_argument("primitive_count should not be zero"); }
  if (data_size == 0) { throw std::invalid_argument("data_size should not be zero"); }
  if (data == nullptr) { throw std::invalid_argument("data should not be nullptr"); }

  uint32_t required_data_size;
  std::vector<uint32_t> vertex_counts;
  vk::PrimitiveTopology vkTopology;
  switch (topology) {
    case PrimitiveTopology::POINT_LIST:
      required_data_size = primitive_count * 2;
      vertex_counts.push_back(required_data_size / 2);
      vkTopology = vk::PrimitiveTopology::ePointList;
      break;
    case PrimitiveTopology::LINE_LIST:
      required_data_size = primitive_count * 2 * 2;
      vertex_counts.push_back(required_data_size / 2);
      vkTopology = vk::PrimitiveTopology::eLineList;
      break;
    case PrimitiveTopology::LINE_STRIP:
      required_data_size = 2 + primitive_count * 2;
      vertex_counts.push_back(required_data_size / 2);
      vkTopology = vk::PrimitiveTopology::eLineStrip;
      break;
    case PrimitiveTopology::TRIANGLE_LIST:
      required_data_size = primitive_count * 3 * 2;
      vertex_counts.push_back(required_data_size / 2);
      vkTopology = vk::PrimitiveTopology::eTriangleList;
      break;
    case PrimitiveTopology::CROSS_LIST:
      required_data_size = primitive_count * 3;
      vertex_counts.push_back(primitive_count * 4);
      vkTopology = vk::PrimitiveTopology::eLineList;
      break;
    case PrimitiveTopology::RECTANGLE_LIST:
      required_data_size = primitive_count * 2 * 2;
      for (uint32_t i = 0; i < primitive_count; ++i) { vertex_counts.push_back(5); }
      vkTopology = vk::PrimitiveTopology::eLineStrip;
      break;
    case PrimitiveTopology::OVAL_LIST:
      required_data_size = primitive_count * 4;
      for (uint32_t i = 0; i < primitive_count; ++i) {
        vertex_counts.push_back(CIRCLE_SEGMENTS + 1);
      }
      vkTopology = vk::PrimitiveTopology::eLineStrip;
      break;
  }

  if (data_size < required_data_size) {
    std::stringstream buf;
    buf << "Required data array size is " << required_data_size << " but only " << data_size
        << " where specified";
    throw std::runtime_error(buf.str().c_str());
  }

  impl_->primitives_.emplace_back(impl_->attributes_,
                                  topology,
                                  primitive_count,
                                  data_size,
                                  data,
                                  impl_->vertex_count_,
                                  vertex_counts,
                                  vkTopology);

  for (auto&& vertex_count : vertex_counts) { impl_->vertex_count_ += vertex_count; }
}

void GeometryLayer::text(float x, float y, float size, const char* text) {
  if (size == 0) { throw std::invalid_argument("size should not be zero"); }
  if (text == nullptr) { throw std::invalid_argument("text should not be nullptr"); }

  impl_->texts_.emplace_back(impl_->attributes_, x, y, size, text);
}

void GeometryLayer::depth_map(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
                              ImageFormat depth_fmt, CUdeviceptr depth_device_ptr,
                              ImageFormat color_fmt, CUdeviceptr color_device_ptr) {
  if ((width == 0) || (height == 0)) {
    throw std::invalid_argument("width or height should not be zero");
  }
  if (depth_fmt != ImageFormat::R8_UNORM) {
    throw std::invalid_argument("The depth format should be ImageFormat::R8_UNORM");
  }
  if (depth_device_ptr == 0) {
    throw std::invalid_argument("The depth device pointer should not be 0");
  }
  if (color_device_ptr && (color_fmt != ImageFormat::R8G8B8A8_UNORM)) {
    throw std::invalid_argument("The color format should be ImageFormat::R8G8B8A8_UNORM");
  }

  impl_->depth_maps_.emplace_back(impl_->attributes_,
                                  render_mode,
                                  width,
                                  height,
                                  depth_device_ptr,
                                  color_device_ptr,
                                  Context::get().get_cuda_stream());
}

bool GeometryLayer::can_be_reused(Layer& other) const {
  return Layer::can_be_reused(other) &&
         impl_->can_be_reused(*static_cast<const GeometryLayer&>(other).impl_.get());
}

void GeometryLayer::end(Vulkan* vulkan) {
  // if the aspect ratio changed, re-create the text and primitive buffers because the generated
  // vertex positions depend on the aspect ratio
  if (impl_->aspect_ratio_ != vulkan->get_window()->get_aspect_ratio()) {
    impl_->aspect_ratio_ = vulkan->get_window()->get_aspect_ratio();

    impl_->text_draw_list_.reset();
    if (impl_->text_vertex_buffer_) {
      impl_->vulkan_->destroy_buffer(impl_->text_vertex_buffer_);
      impl_->text_vertex_buffer_ = nullptr;
    }
    if (impl_->text_index_buffer_) {
      impl_->vulkan_->destroy_buffer(impl_->text_index_buffer_);
      impl_->text_index_buffer_ = nullptr;
    }

    // only crosses depend on the aspect ratio
    bool has_crosses = false;
    for (auto&& primitive : impl_->primitives_) {
      if (primitive.topology_ == PrimitiveTopology::CROSS_LIST) {
        has_crosses = true;
        break;
      }
    }
    if (has_crosses) {
      if (impl_->vertex_buffer_) {
        impl_->vulkan_->destroy_buffer(impl_->vertex_buffer_);
        impl_->vertex_buffer_ = nullptr;
      }
    }
  }

  if (!impl_->primitives_.empty()) {
    if (!impl_->vertex_buffer_) {
      /// @todo need to remember Vulkan instance for destroying buffer,
      ///       destroy should probably be handled by Vulkan class
      impl_->vulkan_ = vulkan;

      // setup the vertex buffer
      std::vector<float> vertices;
      vertices.reserve(impl_->vertex_count_ * 3);

      for (auto&& primitive : impl_->primitives_) {
        switch (primitive.topology_) {
          case PrimitiveTopology::POINT_LIST:
          case PrimitiveTopology::LINE_LIST:
          case PrimitiveTopology::LINE_STRIP:
          case PrimitiveTopology::TRIANGLE_LIST:
            // just copy
            for (uint32_t index = 0; index < primitive.data_.size() / 2; ++index) {
              vertices.insert(
                  vertices.end(),
                  {primitive.data_[index * 2 + 0], primitive.data_[index * 2 + 1], 0.f});
            }
            break;
          case PrimitiveTopology::CROSS_LIST:
            // generate crosses
            for (uint32_t index = 0; index < primitive.primitive_count_; ++index) {
              const float x = primitive.data_[index * 3 + 0];
              const float y = primitive.data_[index * 3 + 1];
              const float sy = primitive.data_[index * 3 + 2] * 0.5f;
              const float sx = sy / impl_->aspect_ratio_;
              vertices.insert(vertices.end(),
                              {x - sx, y, 0.f, x + sx, y, 0.f, x, y - sy, 0.f, x, y + sy, 0.f});
            }
            break;
          case PrimitiveTopology::RECTANGLE_LIST:
            // generate rectangles
            for (uint32_t index = 0; index < primitive.primitive_count_; ++index) {
              const float x0 = primitive.data_[index * 4 + 0];
              const float y0 = primitive.data_[index * 4 + 1];
              const float x1 = primitive.data_[index * 4 + 2];
              const float y1 = primitive.data_[index * 4 + 3];
              vertices.insert(vertices.end(),
                              {x0, y0, 0.f, x1, y0, 0.f, x1, y1, 0.f, x0, y1, 0.f, x0, y0, 0.f});
            }
            break;
          case PrimitiveTopology::OVAL_LIST:
            for (uint32_t index = 0; index < primitive.primitive_count_; ++index) {
              const float x = primitive.data_[index * 4 + 0];
              const float y = primitive.data_[index * 4 + 1];
              const float rx = primitive.data_[index * 4 + 2] * 0.5f;
              const float ry = primitive.data_[index * 4 + 3] * 0.5f;
              for (uint32_t segment = 0; segment <= CIRCLE_SEGMENTS; ++segment) {
                const float rad = (2.f * M_PI) / CIRCLE_SEGMENTS * segment;
                const float px = x + std::cos(rad) * rx;
                const float py = y + std::sin(rad) * ry;
                vertices.insert(vertices.end(), {px, py, 0.f});
              }
            }
            break;
        }
      }

      impl_->vertex_buffer_ = vulkan->create_buffer(
          vertices.size() * sizeof(float), vertices.data(), vk::BufferUsageFlagBits::eVertexBuffer);
    }
  }

  if (!impl_->texts_.empty()) {
    if (!impl_->text_draw_list_) {
      impl_->text_draw_list_.reset(new ImDrawList(ImGui::GetDrawListSharedData()));
      impl_->text_draw_list_->_ResetForNewFrame();

      // ImGui is using integer coordinates for the text position, we use the 0...1 range.
      // Therefore generate vertices in larger scale and scale them down afterwards.
      const float scale = 16384.f;
      const ImVec4 clip_rect(0.f, 0.f, scale * std::max(1.f, impl_->aspect_ratio_), scale);
      const float inv_scale = 1.f / scale;

      ImDrawVert *vertex_base = nullptr, *vertex = nullptr;
      for (auto&& text : impl_->texts_) {
        const ImU32 color = ImGui::ColorConvertFloat4ToU32(ImVec4(text.attributes_.color_[0],
                                                                  text.attributes_.color_[1],
                                                                  text.attributes_.color_[2],
                                                                  text.attributes_.color_[3]));
        ImGui::GetFont()->RenderText(impl_->text_draw_list_.get(),
                                     text.size_ * scale,
                                     ImVec2(text.x_ * scale, text.y_ * scale),
                                     color,
                                     clip_rect,
                                     text.text_.c_str(),
                                     text.text_.c_str() + text.text_.size());

        // scale back vertex data
        if (vertex_base != impl_->text_draw_list_->VtxBuffer.Data) {
          const size_t offset = vertex - vertex_base;
          vertex_base = impl_->text_draw_list_->VtxBuffer.Data;
          vertex = vertex_base + offset;
        }
        while (vertex < impl_->text_draw_list_->_VtxWritePtr) {
          vertex->pos.x =
              (vertex->pos.x * inv_scale - text.x_) * (1.f / impl_->aspect_ratio_) + text.x_;
          vertex->pos.y *= inv_scale;
          ++vertex;
        }
      }

      // text might be completely out of clip rectangle,
      //      if this is the case no vertices had been generated
      if (impl_->text_draw_list_->VtxBuffer.size() != 0) {
        /// @todo need to remember Vulkan instance for destroying buffer, destroy should
        //        probably be handled by Vulkan class
        impl_->vulkan_ = vulkan;

        impl_->text_vertex_buffer_ =
            vulkan->create_buffer(impl_->text_draw_list_->VtxBuffer.size() * sizeof(ImDrawVert),
                                  impl_->text_draw_list_->VtxBuffer.Data,
                                  vk::BufferUsageFlagBits::eVertexBuffer);
        impl_->text_index_buffer_ =
            vulkan->create_buffer(impl_->text_draw_list_->IdxBuffer.size() * sizeof(ImDrawIdx),
                                  impl_->text_draw_list_->IdxBuffer.Data,
                                  vk::BufferUsageFlagBits::eIndexBuffer);
      } else {
        impl_->text_draw_list_.reset();
      }
    }
  }

  if (!impl_->depth_maps_.empty()) {
    // allocate vertex buffer
    if (!impl_->depth_map_vertex_buffer_) {
      /// @todo need to remember Vulkan instance for destroying buffer, destroy should probably be
      /// handled by Vulkan class
      impl_->vulkan_ = vulkan;

      // calculate the index count needed
      size_t index_count = 0;
      bool has_color_buffer = false;
      impl_->depth_map_vertex_count_ = 0;
      for (auto&& depth_map : impl_->depth_maps_) {
        switch (depth_map.render_mode_) {
          case DepthMapRenderMode::POINTS:
            // points don't need indices
            depth_map.index_count_ = 0;
            break;
          case DepthMapRenderMode::LINES:
            // each point in the depth map needs two lines (each line has two indices) except the
            // last column and row which needs one line only
            depth_map.index_count_ = ((depth_map.width_ - 1) * (depth_map.height_ - 1) * 2 +
                                      (depth_map.width_ - 1) + (depth_map.height_ - 1)) *
                                     2;
            break;
          case DepthMapRenderMode::TRIANGLES:
            // need two triangle (each triangle has three indices) for each full quad of depth
            // points
            depth_map.index_count_ = (depth_map.width_ - 1) * (depth_map.height_ - 1) * 2 * 3;
            break;
          default:
            throw std::runtime_error("Unhandled render mode");
        }
        index_count += depth_map.index_count_;
        has_color_buffer |= (depth_map.color_device_ptr_ != 0);
        impl_->depth_map_vertex_count_ += depth_map.width_ * depth_map.height_;
      }

      if (index_count) {
        // generate index data
        std::unique_ptr<uint32_t> index_data(new uint32_t[index_count]);

        uint32_t* dst = index_data.get();
        for (auto&& depth_map : impl_->depth_maps_) {
          depth_map.index_offset_ = dst - index_data.get();

          switch (depth_map.render_mode_) {
            case DepthMapRenderMode::LINES:
              for (uint32_t y = 0; y < depth_map.height_ - 1; ++y) {
                for (uint32_t x = 0; x < depth_map.width_ - 1; ++x) {
                  const uint32_t i = (y * depth_map.width_) + x;
                  dst[0] = i;
                  dst[1] = i + 1;
                  dst[2] = i;
                  dst[3] = i + depth_map.width_;
                  dst += 4;
                }
                // last column
                const uint32_t i = (y * depth_map.width_) + depth_map.width_ - 1;
                dst[0] = i;
                dst[1] = i + depth_map.width_;
                dst += 2;
              }
              // last row
              for (uint32_t x = 0; x < depth_map.width_ - 1; ++x) {
                const uint32_t i = ((depth_map.height_ - 1) * depth_map.width_) + x;
                dst[0] = i;
                dst[1] = i + 1;
                dst += 2;
              }
              break;
            case DepthMapRenderMode::TRIANGLES:
              for (uint32_t y = 0; y < depth_map.height_ - 1; ++y) {
                for (uint32_t x = 0; x < depth_map.width_ - 1; ++x) {
                  const uint32_t i = (y * depth_map.width_) + x;
                  dst[0] = i;
                  dst[1] = i + 1;
                  dst[2] = i + depth_map.width_;
                  dst[3] = i + 1;
                  dst[4] = i + depth_map.width_ + 1;
                  dst[5] = i + depth_map.width_;
                  dst += 6;
                }
              }
              break;
            default:
              throw std::runtime_error("Unhandled render mode");
          }
        }
        if ((dst - index_data.get()) != index_count) {
          throw std::runtime_error("Index count mismatch.");
        }
        impl_->depth_map_index_buffer_ =
            vulkan->create_buffer(index_count * sizeof(uint32_t),
                                  index_data.get(),
                                  vk::BufferUsageFlagBits::eIndexBuffer);
      }

      if (has_color_buffer) {
        impl_->depth_map_color_buffer_ = vulkan->create_buffer_for_cuda_interop(
            impl_->depth_map_vertex_count_ * 4 * sizeof(uint8_t),
            vk::BufferUsageFlagBits::eVertexBuffer);
      }

      impl_->depth_map_vertex_buffer_ =
          vulkan->create_buffer(impl_->depth_map_vertex_count_ * 3 * sizeof(float),
                                nullptr,
                                vk::BufferUsageFlagBits::eVertexBuffer);
    }

    // generate and upload vertex data
    std::unique_ptr<float> vertex_data(new float[impl_->depth_map_vertex_count_ * 3]);
    float* dst = vertex_data.get();

    const CudaService::ScopedPush cuda_context = CudaService::get().PushContext();

    for (auto&& depth_map : impl_->depth_maps_) {
      depth_map.vertex_offset_ = (dst - vertex_data.get()) / 3;

      // download the depth map data and create 3D position data for each depth value
      /// @todo this is not optimal. It would be faster to use a vertex or geometry shader which
      ///       directly fetches the depth data and generates primitive (geometry shader) or emits
      ///       the 3D position vertex (vertex shader)
      std::unique_ptr<uint8_t> host_data(new uint8_t[depth_map.width_ * depth_map.height_]);
      CudaCheck(cuMemcpyDtoHAsync(reinterpret_cast<void*>(host_data.get()),
                                  depth_map.depth_device_ptr_,
                                  depth_map.width_ * depth_map.height_ * sizeof(uint8_t),
                                  depth_map.cuda_stream_));
      CudaCheck(cuStreamSynchronize(depth_map.cuda_stream_));

      const uint8_t* src = host_data.get();
      const float inv_width = 1.f / float(depth_map.width_);
      const float inv_height = 1.f / float(depth_map.height_);
      for (uint32_t y = 0; y < depth_map.height_; ++y) {
        for (uint32_t x = 0; x < depth_map.width_; ++x) {
          dst[0] = float(x) * inv_width - 0.5f;
          dst[1] = float(y) * inv_height - 0.5f;
          dst[2] = float(src[0]) / 255.f;
          src += 1;
          dst += 3;
        }
      }
    }

    if ((dst - vertex_data.get()) != impl_->depth_map_vertex_count_ * 3) {
      throw std::runtime_error("Vertex data count mismatch.");
    }

    // upload vertex data
    vulkan->upload_to_buffer(impl_->depth_map_vertex_count_ * 3 * sizeof(float),
                             vertex_data.get(),
                             impl_->depth_map_vertex_buffer_);

    // upload color data
    size_t offset = 0;
    for (auto&& depth_map : impl_->depth_maps_) {
      const size_t size = depth_map.width_ * depth_map.height_ * sizeof(uint8_t) * 4;
      if (depth_map.color_device_ptr_) {
        vulkan->upload_to_buffer(size,
                                 depth_map.color_device_ptr_,
                                 impl_->depth_map_color_buffer_,
                                 offset,
                                 depth_map.cuda_stream_);
      }
      offset += size;
    }
  }
}

void GeometryLayer::render(Vulkan* vulkan) {
  // setup the view matrix in a way that geometry coordinates are in the range [0...1]
  nvmath::mat4f view_matrix;
  view_matrix.identity();
  view_matrix.scale({2.f, 2.f, 1.f});
  view_matrix.translate({-.5f, -.5f, 0.f});

  // draw geometry primitives
  for (auto&& primitive : impl_->primitives_) {
    uint32_t vertex_offset = primitive.vertex_offset_;
    for (auto&& vertex_count : primitive.vertex_counts_) {
      vulkan->draw(primitive.vk_topology_,
                   vertex_count,
                   vertex_offset,
                   {impl_->vertex_buffer_},
                   get_opacity(),
                   primitive.attributes_.color_,
                   primitive.attributes_.point_size_,
                   primitive.attributes_.line_width_,
                   view_matrix);
      vertex_offset += vertex_count;
    }
  }

  // draw text
  if (impl_->text_draw_list_) {
    for (int i = 0; i < impl_->text_draw_list_->CmdBuffer.size(); ++i) {
      const ImDrawCmd* pcmd = &impl_->text_draw_list_->CmdBuffer[i];
      vulkan->draw_text_indexed(
          vk::DescriptorSet(reinterpret_cast<VkDescriptorSet>(ImGui::GetIO().Fonts->TexID)),
          impl_->text_vertex_buffer_,
          impl_->text_index_buffer_,
          (sizeof(ImDrawIdx) == 2) ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
          pcmd->ElemCount,
          pcmd->IdxOffset,
          pcmd->VtxOffset,
          get_opacity(),
          view_matrix);
    }
  }

  // draw depth maps
  if (!impl_->depth_maps_.empty()) {
    vulkan->get_window()->get_view_matrix(&view_matrix);

    for (auto&& depth_map : impl_->depth_maps_) {
      std::vector<Vulkan::Buffer*> vertex_buffers;
      vertex_buffers.push_back(impl_->depth_map_vertex_buffer_);
      if (depth_map.color_device_ptr_) { vertex_buffers.push_back(impl_->depth_map_color_buffer_); }

      if ((depth_map.render_mode_ == DepthMapRenderMode::LINES) ||
          (depth_map.render_mode_ == DepthMapRenderMode::TRIANGLES)) {
        vulkan->draw_indexed((depth_map.render_mode_ == DepthMapRenderMode::LINES)
                                 ? vk::PrimitiveTopology::eLineList
                                 : vk::PrimitiveTopology::eTriangleList,
                             vertex_buffers,
                             impl_->depth_map_index_buffer_,
                             vk::IndexType::eUint32,
                             depth_map.index_count_,
                             depth_map.index_offset_,
                             depth_map.vertex_offset_,
                             get_opacity(),
                             depth_map.attributes_.color_,
                             depth_map.attributes_.point_size_,
                             depth_map.attributes_.line_width_,
                             view_matrix);
      } else if (depth_map.render_mode_ == DepthMapRenderMode::POINTS) {
        vulkan->draw(vk::PrimitiveTopology::ePointList,
                     depth_map.width_ * depth_map.height_,
                     depth_map.vertex_offset_,
                     vertex_buffers,
                     get_opacity(),
                     depth_map.attributes_.color_,
                     depth_map.attributes_.point_size_,
                     depth_map.attributes_.line_width_,
                     view_matrix);
      } else {
        throw std::runtime_error("Unhandled depth render mode.");
      }
    }
  }
}

}  // namespace holoscan::viz
