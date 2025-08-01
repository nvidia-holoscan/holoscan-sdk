/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <array>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../context.hpp"
#include "../cuda/cuda_service.hpp"
#include "../cuda/gen_depth_map.hpp"
#include "../cuda/gen_primitive_vertices.hpp"
#include "../vulkan/buffer.hpp"
#include "../vulkan/vulkan_app.hpp"

namespace holoscan::viz {

class Attributes {
 public:
  Attributes() : color_({1.F, 1.F, 1.F, 1.F}), line_width_(1.F), point_size_(1.F) {}

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
            size_t data_size, const float* host_data, CUdeviceptr device_data,
            uint32_t vertex_offset, CUstream cuda_stream)
      : attributes_(attributes),
        topology_(topology),
        primitive_count_(primitive_count),
        device_data_(device_data),
        vertex_offset_(vertex_offset),
        cuda_stream_(cuda_stream) {
    size_t required_data_size;
    switch (topology) {
      case PrimitiveTopology::POINT_LIST:
        required_data_size = primitive_count * 2;
        vertex_counts_.push_back(required_data_size / 2);
        vk_topology_ = vk::PrimitiveTopology::ePointList;
        break;
      case PrimitiveTopology::LINE_LIST:
        required_data_size = primitive_count * 2 * 2;
        vertex_counts_.push_back(required_data_size / 2);
        vk_topology_ = vk::PrimitiveTopology::eLineList;
        break;
      case PrimitiveTopology::LINE_STRIP:
        required_data_size = 2 + primitive_count * 2;
        vertex_counts_.push_back(required_data_size / 2);
        vk_topology_ = vk::PrimitiveTopology::eLineStrip;
        break;
      case PrimitiveTopology::TRIANGLE_LIST:
        required_data_size = primitive_count * 3 * 2;
        vertex_counts_.push_back(required_data_size / 2);
        vk_topology_ = vk::PrimitiveTopology::eTriangleList;
        break;
      case PrimitiveTopology::CROSS_LIST:
        required_data_size = primitive_count * 3;
        vertex_counts_.push_back(primitive_count * 4);
        vk_topology_ = vk::PrimitiveTopology::eLineList;
        break;
      case PrimitiveTopology::RECTANGLE_LIST:
        required_data_size = primitive_count * 2 * 2;
        for (uint32_t i = 0; i < primitive_count; ++i) {
          vertex_counts_.push_back(5);
        }
        vk_topology_ = vk::PrimitiveTopology::eLineStrip;
        break;
      case PrimitiveTopology::OVAL_LIST:
        required_data_size = primitive_count * 4;
        for (uint32_t i = 0; i < primitive_count; ++i) {
          vertex_counts_.push_back(CIRCLE_SEGMENTS + 1);
        }
        vk_topology_ = vk::PrimitiveTopology::eLineStrip;
        break;
      case PrimitiveTopology::POINT_LIST_3D:
        required_data_size = primitive_count * 3;
        vertex_counts_.push_back(required_data_size / 3);
        vk_topology_ = vk::PrimitiveTopology::ePointList;
        break;
      case PrimitiveTopology::LINE_LIST_3D:
        required_data_size = primitive_count * 2 * 3;
        vertex_counts_.push_back(required_data_size / 3);
        vk_topology_ = vk::PrimitiveTopology::eLineList;
        break;
      case PrimitiveTopology::LINE_STRIP_3D:
        required_data_size = 3 + primitive_count * 3;
        vertex_counts_.push_back(required_data_size / 3);
        vk_topology_ = vk::PrimitiveTopology::eLineStrip;
        break;
      case PrimitiveTopology::TRIANGLE_LIST_3D:
        required_data_size = primitive_count * 3 * 3;
        vertex_counts_.push_back(required_data_size / 3);
        vk_topology_ = vk::PrimitiveTopology::eTriangleList;
        break;
    }

    if (data_size < required_data_size) {
      std::stringstream buf;
      buf << "Required data array size is " << required_data_size << " but only " << data_size
          << " where specified";
      throw std::runtime_error(buf.str().c_str());
    }

    if (host_data) {
      host_data_.assign(host_data, host_data + required_data_size);
    }
    data_size_ = required_data_size;
  }
  Primitive() = delete;

  /**
   * @return true for 3D primitives
   */
  bool three_dimensional() const {
    switch (topology_) {
      case PrimitiveTopology::POINT_LIST_3D:
      case PrimitiveTopology::LINE_LIST_3D:
      case PrimitiveTopology::LINE_STRIP_3D:
      case PrimitiveTopology::TRIANGLE_LIST_3D:
        return true;
        break;
      default:
        return false;
    }
  }

  bool operator==(const Primitive& rhs) const {
    // we can reuse if the attributes, topology and primitive count match and
    // if we did not switch from host to device memory and vice versa
    return ((attributes_ == rhs.attributes_) && (topology_ == rhs.topology_) &&
            (primitive_count_ == rhs.primitive_count_) &&
            ((host_data_.empty() && !device_data_) ||
             (((!host_data_.empty()) == (rhs.device_data_ == 0)) &&
              ((device_data_ != 0) == (rhs.host_data_.empty())))));
  }

  const Attributes attributes_;

  const PrimitiveTopology topology_;
  const uint32_t primitive_count_;
  std::vector<float> host_data_;
  CUdeviceptr device_data_;
  CUstream cuda_stream_ = 0;

  // internal state
  const uint32_t vertex_offset_;  ///< vertex start offset (in units of 3 * float)
  size_t data_size_;              ///< size of input data
  std::vector<uint32_t> vertex_counts_;
  vk::PrimitiveTopology vk_topology_;
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
           uint32_t height, ImageFormat depth_fmt, CUdeviceptr depth_device_ptr,
           ImageFormat color_fmt, CUdeviceptr color_device_ptr, CUstream cuda_stream)
      : attributes_(attributes),
        render_mode_(render_mode),
        width_(width),
        height_(height),
        depth_fmt_(depth_fmt),
        depth_device_ptr_(depth_device_ptr),
        color_fmt_(color_fmt),
        color_device_ptr_(color_device_ptr),
        cuda_stream_(cuda_stream) {}

  bool operator==(const DepthMap& rhs) const {
    // check attributes, render mode, width and height. Skip *_device_ptr_ since that data
    // will be uploaded each frame. The *_device_ptr_ is updated in the can_be_reused() function
    // below.
    return ((attributes_ == rhs.attributes_) && (render_mode_ == rhs.render_mode_) &&
            (width_ == rhs.width_) && (height_ == rhs.height_) && (depth_fmt_ == rhs.depth_fmt_) &&
            (color_fmt_ == rhs.color_fmt_));
  }

  const Attributes attributes_;
  const DepthMapRenderMode render_mode_;
  const uint32_t width_;
  const uint32_t height_;
  ImageFormat color_fmt_;
  CUdeviceptr color_device_ptr_;
  ImageFormat depth_fmt_;
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
      auto depth_map_it = other.depth_maps_.begin();
      for (auto&& depth_map : depth_maps_) {
        depth_map_it->depth_device_ptr_ = depth_map.depth_device_ptr_;
        depth_map_it->color_device_ptr_ = depth_map.color_device_ptr_;
        depth_map_it->cuda_stream_ = depth_map.cuda_stream_;
        ++depth_map_it;
      }
      auto primitive_it = other.primitives_.begin();
      for (auto&& primitive : primitives_) {
        primitive_it->host_data_ = primitive.host_data_;
        primitive_it->device_data_ = primitive.device_data_;
        primitive_it->cuda_stream_ = primitive.cuda_stream_;
        ++primitive_it;
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
  float aspect_ratio_ = 1.F;

  size_t vertex_count_ = 0;
  std::unique_ptr<Buffer> vertex_buffer_;

  std::unique_ptr<ImDrawList> text_draw_list_;
  std::unique_ptr<Buffer> text_vertex_buffer_;
  std::unique_ptr<Buffer> text_index_buffer_;

  size_t depth_map_vertex_count_ = 0;
  std::unique_ptr<Buffer> depth_map_vertex_buffer_;
  std::unique_ptr<Buffer> depth_map_index_buffer_;
  std::unique_ptr<Buffer> depth_map_color_buffer_;
};

GeometryLayer::GeometryLayer() : Layer(Type::Geometry), impl_(new GeometryLayer::Impl) {}

GeometryLayer::~GeometryLayer() {}

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
  if (primitive_count == 0) {
    throw std::invalid_argument("primitive_count should not be zero");
  }
  if (data_size == 0) {
    throw std::invalid_argument("data_size should not be zero");
  }
  if (data == nullptr) {
    throw std::invalid_argument("data should not be nullptr");
  }

  const auto& primitive = impl_->primitives_.emplace_back(impl_->attributes_,
                                                          topology,
                                                          primitive_count,
                                                          data_size,
                                                          data,
                                                          0,
                                                          impl_->vertex_count_,
                                                          Context::get().get_cuda_stream());

  for (auto&& vertex_count : primitive.vertex_counts_) {
    impl_->vertex_count_ += vertex_count;
  }
}

void GeometryLayer::primitive_cuda_device(PrimitiveTopology topology, uint32_t primitive_count,
                                          size_t data_size, CUdeviceptr data) {
  if (primitive_count == 0) {
    throw std::invalid_argument("primitive_count should not be zero");
  }
  if (data_size == 0) {
    throw std::invalid_argument("data_size should not be zero");
  }
  if (data == 0) {
    throw std::invalid_argument("data should not be 0");
  }

  const auto& primitive = impl_->primitives_.emplace_back(impl_->attributes_,
                                                          topology,
                                                          primitive_count,
                                                          data_size,
                                                          nullptr,
                                                          data,
                                                          impl_->vertex_count_,
                                                          Context::get().get_cuda_stream());

  for (auto&& vertex_count : primitive.vertex_counts_) {
    impl_->vertex_count_ += vertex_count;
  }
}

void GeometryLayer::text(float x, float y, float size, const char* text) {
  if (size == 0) {
    throw std::invalid_argument("size should not be zero");
  }
  if (text == nullptr) {
    throw std::invalid_argument("text should not be nullptr");
  }

  impl_->texts_.emplace_back(impl_->attributes_, x, y, size, text);
}

void GeometryLayer::depth_map(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
                              ImageFormat depth_fmt, CUdeviceptr depth_device_ptr,
                              ImageFormat color_fmt, CUdeviceptr color_device_ptr) {
  if ((width == 0) || (height == 0)) {
    throw std::invalid_argument("width or height should not be zero");
  }
  if (!(depth_fmt == ImageFormat::R8_UNORM || depth_fmt == ImageFormat::D32_SFLOAT)) {
    throw std::invalid_argument(
        "The depth format should be one of: ImageFormat::R8_UNORM, ImageFormat::D32_SFLOAT");
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
                                  depth_fmt,
                                  depth_device_ptr,
                                  color_fmt,
                                  color_device_ptr,
                                  Context::get().get_cuda_stream());
}

bool GeometryLayer::can_be_reused(Layer& other) const {
  return Layer::can_be_reused(other) &&
         impl_->can_be_reused(*static_cast<const GeometryLayer&>(other).impl_.get());
}

void GeometryLayer::end(Vulkan* vulkan) {
  // if the aspect ratio changed, re-create the text buffers because the generated
  // vertex positions depend on the aspect ratio
  if (impl_->aspect_ratio_ != vulkan->get_window()->get_aspect_ratio()) {
    impl_->aspect_ratio_ = vulkan->get_window()->get_aspect_ratio();

    impl_->text_draw_list_.reset();
    impl_->text_vertex_buffer_.reset();
    impl_->text_index_buffer_.reset();
  }

  if (!impl_->primitives_.empty()) {
    if (!impl_->vertex_buffer_) {
      // allocate the vertex buffer
      impl_->vertex_buffer_ = vulkan->create_buffer_for_cuda_interop(
          impl_->vertex_count_ * 3 * sizeof(float), vk::BufferUsageFlagBits::eVertexBuffer);
    }

    // generate vertex data
    CudaService* const cuda_service = vulkan->get_cuda_service();
    const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

    for (auto&& primitive : impl_->primitives_) {
      // select the stream to be used by CUDA operations
      const CUstream stream = cuda_service->select_cuda_stream(primitive.cuda_stream_);
      impl_->vertex_buffer_->begin_access_with_cuda(stream);

      UniqueAsyncCUdeviceptr tmp_src_device_ptr;
      CUdeviceptr src_device_ptr;

      // if the data is on host, allocate temporary memory and copy the data to it
      if (!primitive.host_data_.empty()) {
        const size_t size = primitive.data_size_ * sizeof(float);
        tmp_src_device_ptr.reset([size, stream] {
          CUdeviceptr device_ptr;
          CudaCheck(cuMemAllocAsync(&device_ptr, size, stream));
          return std::pair<CUdeviceptr, CUstream>(device_ptr, stream);
        }());
        src_device_ptr = tmp_src_device_ptr.get().first;
        // copy from host to device
        CudaCheck(cuMemcpyHtoDAsync(src_device_ptr,
                                    reinterpret_cast<const float*>(primitive.host_data_.data()),
                                    size,
                                    stream));
      } else {
        src_device_ptr = primitive.device_data_;
      }

      // generate vertex data
      gen_primitive_vertices(
          primitive.topology_,
          primitive.primitive_count_,
          primitive.vertex_counts_,
          impl_->aspect_ratio_,
          src_device_ptr,
          impl_->vertex_buffer_->device_ptr_.get() + primitive.vertex_offset_ * sizeof(float) * 3,
          stream);

      impl_->vertex_buffer_->end_access_with_cuda(stream);
    }
  }

  if (!impl_->texts_.empty()) {
    if (!impl_->text_draw_list_) {
      impl_->text_draw_list_.reset(new ImDrawList(ImGui::GetDrawListSharedData()));
      impl_->text_draw_list_->_ResetForNewFrame();

      // ImGui is using integer coordinates for the text position, we use the 0...1 range.
      // Therefore generate vertices in larger scale and scale them down afterwards.
      const float scale = 16384.F;
      const ImVec4 clip_rect(0.F, 0.F, scale * std::max(1.F, impl_->aspect_ratio_), scale);
      const float inv_scale = 1.F / scale;

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
              (vertex->pos.x * inv_scale - text.x_) * (1.F / impl_->aspect_ratio_) + text.x_;
          vertex->pos.y *= inv_scale;
          ++vertex;
        }
      }

      // text might be completely out of clip rectangle,
      //      if this is the case no vertices had been generated
      if (impl_->text_draw_list_->VtxBuffer.size() != 0) {
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
        impl_->depth_map_index_buffer_ = vulkan->create_buffer_for_cuda_interop(
            index_count * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer);

        CudaService* const cuda_service = vulkan->get_cuda_service();
        const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

        const CUstream stream = Context::get().get_cuda_stream();
        impl_->depth_map_index_buffer_->begin_access_with_cuda(stream);

        size_t offset = 0;
        for (auto&& depth_map : impl_->depth_maps_) {
          depth_map.index_offset_ = offset;
          const CUdeviceptr dst = impl_->depth_map_index_buffer_->device_ptr_.get() + offset;
          offset += GenDepthMapIndices(
              depth_map.render_mode_, depth_map.width_, depth_map.height_, dst, stream);
        }
        // indicate that the index buffer had been used by CUDA
        impl_->depth_map_index_buffer_->end_access_with_cuda(stream);
      }

      if (has_color_buffer) {
        impl_->depth_map_color_buffer_ = vulkan->create_buffer_for_cuda_interop(
            impl_->depth_map_vertex_count_ * 4 * sizeof(uint8_t),
            vk::BufferUsageFlagBits::eVertexBuffer);
      }

      impl_->depth_map_vertex_buffer_ =
          vulkan->create_buffer_for_cuda_interop(impl_->depth_map_vertex_count_ * 3 * sizeof(float),
                                                 vk::BufferUsageFlagBits::eVertexBuffer);
    }

    // generate the vertex data
    {
      CudaService* const cuda_service = vulkan->get_cuda_service();
      const CudaService::ScopedPush cuda_context = cuda_service->PushContext();

      size_t offset = 0;
      for (auto&& depth_map : impl_->depth_maps_) {
        // select the stream to be used by CUDA operations
        const CUstream stream = cuda_service->select_cuda_stream(depth_map.cuda_stream_);

        impl_->depth_map_vertex_buffer_->begin_access_with_cuda(stream);

        depth_map.vertex_offset_ = offset;
        const CUdeviceptr dst = impl_->depth_map_vertex_buffer_->device_ptr_.get() + offset;
        GenDepthMapCoords(depth_map.depth_fmt_,
                          depth_map.width_,
                          depth_map.height_,
                          depth_map.depth_device_ptr_,
                          dst,
                          stream);
        offset += depth_map.width_ * depth_map.height_ * sizeof(float) * 3;

        // indicate that the buffer had been used by CUDA
        impl_->depth_map_vertex_buffer_->end_access_with_cuda(stream);

        CudaService::sync_with_selected_stream(depth_map.cuda_stream_, stream);
      }
    }

    // upload color data
    size_t offset = 0;
    for (auto&& depth_map : impl_->depth_maps_) {
      const size_t size = depth_map.width_ * depth_map.height_ * sizeof(uint8_t) * 4;
      if (depth_map.color_device_ptr_) {
        vulkan->upload_to_buffer(size,
                                 depth_map.color_device_ptr_,
                                 impl_->depth_map_color_buffer_.get(),
                                 offset,
                                 depth_map.cuda_stream_);
      }
      offset += size;
    }
  }
}

void GeometryLayer::render(Vulkan* vulkan) {
  // setup the 2D view matrix in a way that geometry coordinates are in the range [0...1]
  nvmath::mat4f view_matrix_2d_base;
  view_matrix_2d_base.identity();
  view_matrix_2d_base.scale({2.F, 2.F, 1.F});
  view_matrix_2d_base.translate({-.5F, -.5F, 0.F});

  std::vector<Layer::View> views = get_views();
  if (views.empty()) {
    views.push_back(Layer::View());
  }

  for (const View& view : views) {
    vulkan->set_viewport(view.offset_x, view.offset_y, view.width, view.height);

    nvmath::mat4f view_matrix_2d, view_matrix_3d;
    if (view.matrix.has_value()) {
      view_matrix_2d = view.matrix.value();
      view_matrix_3d = view.matrix.value();
    } else {
      vulkan->get_window()->get_view_matrix(&view_matrix_3d);
      view_matrix_2d = view_matrix_2d_base;
    }

    // draw geometry primitives
    for (auto&& primitive : impl_->primitives_) {
      uint32_t vertex_offset = primitive.vertex_offset_;
      for (auto&& vertex_count : primitive.vertex_counts_) {
        vulkan->draw(primitive.vk_topology_,
                     vertex_count,
                     vertex_offset,
                     {impl_->vertex_buffer_.get()},
                     get_opacity(),
                     primitive.attributes_.color_,
                     primitive.attributes_.point_size_,
                     primitive.attributes_.line_width_,
                     primitive.three_dimensional() ? view_matrix_3d : view_matrix_2d);
        vertex_offset += vertex_count;
      }
    }

    // draw text
    if (impl_->text_draw_list_) {
      for (int i = 0; i < impl_->text_draw_list_->CmdBuffer.size(); ++i) {
        const ImDrawCmd* pcmd = &impl_->text_draw_list_->CmdBuffer[i];
        vulkan->draw_imgui(
            vk::DescriptorSet(reinterpret_cast<VkDescriptorSet>(ImGui::GetIO().Fonts->TexID)),
            impl_->text_vertex_buffer_.get(),
            impl_->text_index_buffer_.get(),
            (sizeof(ImDrawIdx) == 2) ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
            pcmd->ElemCount,
            pcmd->IdxOffset,
            pcmd->VtxOffset,
            get_opacity(),
            view_matrix_2d);
      }
    }

    // draw depth maps
    if (!impl_->depth_maps_.empty()) {
      for (auto&& depth_map : impl_->depth_maps_) {
        std::vector<Buffer*> vertex_buffers;
        vertex_buffers.push_back(impl_->depth_map_vertex_buffer_.get());
        if (depth_map.color_device_ptr_) {
          vertex_buffers.push_back(impl_->depth_map_color_buffer_.get());
        }

        if ((depth_map.render_mode_ == DepthMapRenderMode::LINES) ||
            (depth_map.render_mode_ == DepthMapRenderMode::TRIANGLES)) {
          vulkan->draw_indexed((depth_map.render_mode_ == DepthMapRenderMode::LINES)
                                   ? vk::PrimitiveTopology::eLineList
                                   : vk::PrimitiveTopology::eTriangleList,
                               vertex_buffers,
                               impl_->depth_map_index_buffer_.get(),
                               vk::IndexType::eUint32,
                               depth_map.index_count_,
                               depth_map.index_offset_,
                               depth_map.vertex_offset_,
                               get_opacity(),
                               depth_map.attributes_.color_,
                               depth_map.attributes_.point_size_,
                               depth_map.attributes_.line_width_,
                               view_matrix_3d);
        } else if (depth_map.render_mode_ == DepthMapRenderMode::POINTS) {
          vulkan->draw(vk::PrimitiveTopology::ePointList,
                       depth_map.width_ * depth_map.height_,
                       depth_map.vertex_offset_,
                       vertex_buffers,
                       get_opacity(),
                       depth_map.attributes_.color_,
                       depth_map.attributes_.point_size_,
                       depth_map.attributes_.line_width_,
                       view_matrix_3d);
        } else {
          throw std::runtime_error("Unhandled depth render mode.");
        }
      }
    }
  }
}

}  // namespace holoscan::viz
