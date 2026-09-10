// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <holoviz/color_space.hpp>
#include <holoviz/geometry_topology.hpp>
#include <holoviz/image_format.hpp>

namespace holoviz {

/// Optional per-layer 2D view transform (offset + scale), applied when compositing a layer.
struct ViewTransform {
  float offset_x = 0.f;
  float offset_y = 0.f;
  float scale_x = 1.f;
  float scale_y = 1.f;
};

/// Declaration for an image (video) layer. Declared once at construction; a \c RenderBackend
/// uses this to pre-allocate a texture pool sized to \p max_width x \p max_height before the
/// first frame.
///
/// \p max_width / \p max_height are the *declared maximum* dimensions — the pool is sized to
/// these once. Each frame binds a \p Subregion (see frame.hpp) of that maximum, so a
/// variable-resolution source (e.g. a camera that changes resolution frame to frame) never
/// triggers a runtime reallocation as long as it stays within the declared maximum.
struct ImageLayerDesc {
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
  ColorSpace color_space = ColorSpace::kPassThrough;

  /// YUV conversion parameters — required when \p format is a YUV value; must be \c nullopt
  /// for non-YUV formats. A backend must reject a YUV \p ImageLayerDesc where either field
  /// is \c nullopt; there is no implicit default conversion or range. Part of the stable
  /// declaration surface; the YUV conversion path in the backend is wired in a later release.
  std::optional<YUVModelConversion> yuv_model;
  std::optional<YUVRange> yuv_range;

  std::optional<ViewTransform> view;
};

/// Declaration for a geometry overlay layer (rects, crosses, ovals, lines, ...).
/// \p max_primitives is a hard per-frame budget: it sizes the vertex/index pool once at
/// allocate(), and a frame whose primitive count exceeds it fails record() with
/// \c kInvalidArgument (a prevalidation rejection — see RenderBackend::record()) instead
/// of growing the pool. Overflow signals a mis-declaration; declare the true maximum.
struct GeometryLayerDesc {
  std::vector<GeometryTopology> topologies;
  uint32_t max_primitives = 0;
  float default_line_width = 1.0f;
};

/// Declaration for a text overlay layer. `max_glyphs` is a hard per-frame budget, sized
/// once at allocate(); a frame exceeding it fails record() with \c kInvalidArgument (a
/// prevalidation rejection), the same rule as GeometryLayerDesc::max_primitives.
struct TextLayerDesc {
  uint32_t max_glyphs = 0;
};

/// Declaration for a color-lookup-table layer (e.g. segmentation mask colorization).
struct LUTLayerDesc {
  uint32_t lut_size = 0;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
};

/// The descriptor for one declared layer — exactly one of the four layer kinds. The layer
/// type set is fixed and closed (image, geometry, text, LUT), so a variant (rather than a
/// polymorphic base) is the right model: it encodes the "exactly one" invariant in the type,
/// a wrong-kind access is a compile error via std::visit, and adding a layer kind turns every
/// non-exhaustive visitor into a compile error rather than a silent runtime mis-dispatch.
using LayerDesc = std::variant<ImageLayerDesc, GeometryLayerDesc, TextLayerDesc, LUTLayerDesc>;

/// One declared layer: the input port it binds to, and its typed descriptor.
struct LayerDecl {
  std::string port_name;
  LayerDesc desc;
};

/// The fully declared, fixed-order set of layers a HoloViz operator instance will render,
/// captured at construction/setup time, before \c initialize() runs.
///
/// **Compositing order:** index 0 is recorded first and rendered as the backmost (background)
/// layer; each higher index composites over the layers below it (hence "stack"). The default
/// composite order is declaration order; a \c RenderBackend may accept an explicit
/// \c CompositeSpec to reorder without changing layer data.
///
/// A \c RenderBackend's \c allocate() consumes this to pre-allocate every GPU resource the
/// frame path will ever touch: the full resource set must be knowable from the declared
/// layers alone, with no runtime allocation.
class LayerStack {
 public:
  void add_image_layer(std::string port_name, ImageLayerDesc desc) {
    layers_.push_back(LayerDecl{std::move(port_name), std::move(desc)});
  }

  void add_geometry_layer(std::string port_name, GeometryLayerDesc desc) {
    layers_.push_back(LayerDecl{std::move(port_name), std::move(desc)});
  }

  void add_text_layer(std::string port_name, TextLayerDesc desc) {
    layers_.push_back(LayerDecl{std::move(port_name), std::move(desc)});
  }

  void add_lut_layer(std::string port_name, LUTLayerDesc desc) {
    layers_.push_back(LayerDecl{std::move(port_name), std::move(desc)});
  }

  /// \return the declared layers, in declaration order (also the default composite order).
  [[nodiscard]] const std::vector<LayerDecl>& layers() const { return layers_; }

 private:
  std::vector<LayerDecl> layers_;
};

}  // namespace holoviz
