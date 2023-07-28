/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <string>
#include <vector>

#include "./holoviz.hpp"

#include "holoscan/core/codec_registry.hpp"
#include "holoscan/core/endpoint.hpp"
#include "holoscan/core/expected.hpp"

namespace holoscan {

// Define codec for ops::HolovizOp::InputSpec::View

template <>
struct codec<ops::HolovizOp::InputSpec::View> {
  static expected<size_t, RuntimeError> serialize(const ops::HolovizOp::InputSpec::View& view,
                                                  Endpoint* endpoint) {
    size_t total_size = 0;
    auto maybe_size = serialize_trivial_type<float>(view.offset_x_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<float>(view.offset_y_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<float>(view.width_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<float>(view.height_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    bool has_matrix = view.matrix_.has_value();
    maybe_size = serialize_trivial_type<bool>(has_matrix, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    if (has_matrix) {
      ContiguousDataHeader header;
      header.size = 16;
      header.bytes_per_element = sizeof(float);
      maybe_size = endpoint->write_trivial_type<ContiguousDataHeader>(&header);
      if (!maybe_size) { return forward_error(maybe_size); }
      total_size += maybe_size.value();

      maybe_size =
          endpoint->write(view.matrix_.value().data(), header.size * header.bytes_per_element);
      if (!maybe_size) { return forward_error(maybe_size); }
      total_size += maybe_size.value();
    }
    return total_size;
  }

  static expected<ops::HolovizOp::InputSpec::View, RuntimeError> deserialize(Endpoint* endpoint) {
    ops::HolovizOp::InputSpec::View out;
    auto offset_x = deserialize_trivial_type<float>(endpoint);
    if (!offset_x) { forward_error(offset_x); }
    out.offset_x_ = offset_x.value();

    auto offset_y = deserialize_trivial_type<float>(endpoint);
    if (!offset_y) { forward_error(offset_y); }
    out.offset_y_ = offset_y.value();

    auto width = deserialize_trivial_type<float>(endpoint);
    if (!width) { forward_error(width); }
    out.width_ = width.value();

    auto height = deserialize_trivial_type<float>(endpoint);
    if (!height) { forward_error(height); }
    out.height_ = height.value();

    auto maybe_has_matrix = deserialize_trivial_type<bool>(endpoint);
    if (!maybe_has_matrix) { forward_error(maybe_has_matrix); }
    bool has_matrix = maybe_has_matrix.value();

    if (has_matrix) {
      out.matrix_ = std::array<float, 16>{};

      ContiguousDataHeader header;
      auto header_size = endpoint->read_trivial_type<ContiguousDataHeader>(&header);
      if (!header_size) { return forward_error(header_size); }
      auto result =
          endpoint->read(out.matrix_.value().data(), header.size * header.bytes_per_element);
      if (!result) { return forward_error(result); }
    }
    return out;
  }
};

// Define codec for std::vector<ops::HolovizOp::InputSpec::View>

template <>
struct codec<std::vector<ops::HolovizOp::InputSpec::View>> {
  static expected<size_t, RuntimeError> serialize(
      const std::vector<ops::HolovizOp::InputSpec::View>& views, Endpoint* endpoint) {
    size_t total_size = 0;

    // header is just the total number of views
    size_t num_views = views.size();
    auto size = endpoint->write_trivial_type<size_t>(&num_views);
    if (!size) { return forward_error(size); }
    total_size += size.value();

    // now transmit each individual view
    for (const auto& view : views) {
      size = codec<ops::HolovizOp::InputSpec::View>::serialize(view, endpoint);
      if (!size) { return forward_error(size); }
      total_size += size.value();
    }
    return total_size;
  }
  static expected<std::vector<ops::HolovizOp::InputSpec::View>, RuntimeError> deserialize(
      Endpoint* endpoint) {
    size_t num_views;
    auto size = endpoint->read_trivial_type<size_t>(&num_views);
    if (!size) { return forward_error(size); }

    std::vector<ops::HolovizOp::InputSpec::View> data;
    data.reserve(num_views);
    for (size_t i = 0; i < num_views; i++) {
      auto view = codec<ops::HolovizOp::InputSpec::View>::deserialize(endpoint);
      if (!view) { return forward_error(view); }
      data.push_back(view.value());
    }
    return data;
  }
};

// Define codec for serialization of ops::HolovizOp::InputSpec

template <>
struct codec<ops::HolovizOp::InputSpec> {
  static expected<size_t, RuntimeError> serialize(const ops::HolovizOp::InputSpec& spec,
                                                  Endpoint* endpoint) {
    size_t total_size = 0;
    auto maybe_size = codec<std::string>::serialize(spec.tensor_name_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<ops::HolovizOp::InputType>(spec.type_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<float>(spec.opacity_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<int32_t>(spec.priority_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = codec<std::vector<float>>::serialize(spec.color_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<float>(spec.line_width_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<float>(spec.point_size_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = codec<std::vector<std::string>>::serialize(spec.text_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size = serialize_trivial_type<ops::HolovizOp::DepthMapRenderMode>(
        spec.depth_map_render_mode_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    maybe_size =
        codec<std::vector<ops::HolovizOp::InputSpec::View>>::serialize(spec.views_, endpoint);
    if (!maybe_size) { forward_error(maybe_size); }
    total_size += maybe_size.value();

    return total_size;
  }
  static expected<ops::HolovizOp::InputSpec, RuntimeError> deserialize(Endpoint* endpoint) {
    ops::HolovizOp::InputSpec out;

    auto tensor_name = codec<std::string>::deserialize(endpoint);
    if (!tensor_name) { forward_error(tensor_name); }
    out.tensor_name_ = tensor_name.value();

    auto type = deserialize_trivial_type<ops::HolovizOp::InputType>(endpoint);
    if (!type) { forward_error(type); }
    out.type_ = type.value();

    auto opacity = deserialize_trivial_type<float>(endpoint);
    if (!opacity) { forward_error(opacity); }
    out.opacity_ = opacity.value();

    auto priority = deserialize_trivial_type<int32_t>(endpoint);
    if (!priority) { forward_error(priority); }
    out.priority_ = priority.value();

    auto color = codec<std::vector<float>>::deserialize(endpoint);
    if (!color) { forward_error(color); }
    out.color_ = color.value();

    auto line_width = deserialize_trivial_type<float>(endpoint);
    if (!line_width) { forward_error(line_width); }
    out.line_width_ = line_width.value();

    auto point_size = deserialize_trivial_type<float>(endpoint);
    if (!point_size) { forward_error(point_size); }
    out.point_size_ = point_size.value();

    auto text = codec<std::vector<std::string>>::deserialize(endpoint);
    if (!text) { forward_error(text); }
    out.text_ = text.value();

    auto depth_map_render_mode =
        deserialize_trivial_type<ops::HolovizOp::DepthMapRenderMode>(endpoint);
    if (!depth_map_render_mode) { forward_error(depth_map_render_mode); }
    out.depth_map_render_mode_ = depth_map_render_mode.value();

    auto views = codec<std::vector<ops::HolovizOp::InputSpec::View>>::deserialize(endpoint);
    if (!views) { forward_error(views); }
    out.views_ = views.value();

    return out;
  }
};

// Define codec for serialization of std::vector<ops::HolovizOp::InputSpec>

template <>
struct codec<std::vector<ops::HolovizOp::InputSpec>> {
  static expected<size_t, RuntimeError> serialize(
      const std::vector<ops::HolovizOp::InputSpec>& specs, Endpoint* endpoint) {
    size_t total_size = 0;

    // header is just the total number of specs
    size_t num_specs = specs.size();
    auto size = endpoint->write_trivial_type<size_t>(&num_specs);
    if (!size) { return forward_error(size); }
    total_size += size.value();

    // now transmit each individual spec
    for (const auto& spec : specs) {
      size = codec<ops::HolovizOp::InputSpec>::serialize(spec, endpoint);
      if (!size) { return forward_error(size); }
      total_size += size.value();
    }
    return total_size;
  }
  static expected<std::vector<ops::HolovizOp::InputSpec>, RuntimeError> deserialize(
      Endpoint* endpoint) {
    size_t num_specs;
    auto size = endpoint->read_trivial_type<size_t>(&num_specs);
    if (!size) { return forward_error(size); }

    std::vector<ops::HolovizOp::InputSpec> data;
    data.reserve(num_specs);
    for (size_t i = 0; i < num_specs; i++) {
      auto spec = codec<ops::HolovizOp::InputSpec>::deserialize(endpoint);
      if (!spec) { return forward_error(spec); }
      data.push_back(spec.value());
    }
    return data;
  }
};
}  // namespace holoscan
