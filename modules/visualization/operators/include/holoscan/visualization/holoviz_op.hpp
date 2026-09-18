// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/visualization/holoscan_operator_holoviz_export.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/visualization/image_format.hpp>
#include <holoscan/visualization/sink_desc.hpp>

namespace holoscan::visualization {

// ---- Layer declarations ------------------------------------------------------
// ImageFormat lives in <holoscan/visualization/image_format.hpp>; the sink descriptors and the
// PresentMode in <holoscan/visualization/sink_desc.hpp>.

/// YUV → RGB color model; only meaningful for YUV \ref ImageFormat values.
enum class YUVModelConversion : uint8_t { kBT601, kBT709, kBT2020 };

/// YUV sample range; only meaningful for YUV \ref ImageFormat values.
enum class YUVRange : uint8_t { kLimited, kFull };

/// How surface data is interpreted on screen.
enum class ColorSpace {
  kPassThrough,    ///< components used as-is
  kSRGBNonlinear,  ///< sRGB
  kBT709Linear,    ///< BT.709 with a linear EOTF
};

/// 2D overlay primitive shapes for geometry layers.
enum class GeometryTopology : uint8_t {
  kLineList,
  kLineStrip,
  kCrossList,      ///< center + size
  kRectangleList,  ///< two corners
  kOvalList,       ///< center + two axis sizes
};

/// Optional per-layer 2D transform (offset then scale) applied when compositing.
struct ViewTransform {
  float offset_x = 0.f;
  float offset_y = 0.f;
  float scale_x = 1.f;
  float scale_y = 1.f;
};

/// Image (video) layer. \p max_width / \p max_height are declared maxima: GPU resources are
/// sized to them once, and every frame may be any size within them.
///
/// The bound input port receives a \c holoscan::Tensor in HWC layout whose dtype and channel
/// count must agree with \p format; a mismatching sample is logged and dropped:
///
/// | format               | accepted tensors                                        |
/// |----------------------|---------------------------------------------------------|
/// | kR8Unorm             | uint8 `{H, W}` or `{H, W, 1}`                           |
/// | kR8G8B8A8Unorm/BGRA  | uint8 `{H, W, 4}`                                       |
/// | kR16Unorm            | uint16 `{H, W}` or `{H, W, 1}`, or packed uint8 `{H, W, 2}` |
/// | kR16G16B16A16Unorm   | uint16 `{H, W, 4}`, or packed uint8 `{H, W, 8}`         |
///
/// Row pitch is taken from `strides()[0]` (bytes) and may exceed the packed row size.
struct ImageLayerDesc {
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
  ColorSpace color_space = ColorSpace::kPassThrough;
  /// Required (both) for YUV formats, must be \c nullopt otherwise.
  std::optional<YUVModelConversion> yuv_model;
  std::optional<YUVRange> yuv_range;
  std::optional<ViewTransform> view;
};

/// Geometry overlay layer. \p max_primitives is a hard per-frame budget; a frame exceeding it is
/// rejected rather than grown into.
struct GeometryLayerDesc {
  std::vector<GeometryTopology> topologies;
  uint32_t max_primitives = 0;
  float default_line_width = 1.0f;
};

/// Text overlay layer. \p max_glyphs is a hard per-frame budget.
struct TextLayerDesc {
  uint32_t max_glyphs = 0;
};

/// Color-lookup-table layer (e.g. segmentation mask colorization).
struct LUTLayerDesc {
  uint32_t lut_size = 0;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
};

/// Exactly one layer kind. Closed set: adding an alternative is an API/ABI break.
using LayerDesc = std::variant<ImageLayerDesc, GeometryLayerDesc, TextLayerDesc, LUTLayerDesc>;

/// One declared layer: the input port it binds to and its descriptor.
struct LayerDecl {
  std::string port_name;
  LayerDesc desc;
};

/// The fixed, ordered set of layers one operator renders. Index 0 is the backmost layer; each
/// higher index composites over the ones below.
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

  /// Declared layers in declaration (= composite) order.
  [[nodiscard]] const std::vector<LayerDecl>& layers() const { return layers_; }

 private:
  std::vector<LayerDecl> layers_;
};

// ---- Operator ----------------------------------------------------------------

/// Maximum number of layer input ports on one HolovizOp.
inline constexpr std::size_t kMaxLayerPorts = 8;

/// Renders a \ref LayerStack into a \ref SinkDesc at a fixed display rate.
///
/// One typed Tensor input per declared layer (up to \ref kMaxLayerPorts). Rendering is driven by
/// an OnClock trigger, so frames are produced even when a layer's input is late or missing.
///
/// ```cpp
/// LayerStack stack;
/// stack.add_image_layer("video", ImageLayerDesc{...});
/// auto op = graph.op<HolovizOp>("holoviz", std::move(stack), WindowSinkDesc{.title = "Demo"});
/// graph.partition("holoviz").add(op);
/// ```
///
/// Backend errors are translated into the Holoscan Core error domain at the lifecycle boundary.
class HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT HolovizOp : public holoscan::Operator<> {
 public:
  /// @param layer_stack   Layers to render; at most \ref kMaxLayerPorts.
  /// @param sink          Where frames go.
  /// @param display_rate  Render period (default 60 Hz).
  /// @param max_input_age Freshness window for every layer input; defaults to \p display_rate.
  ///                      Widen it for producers slower than the display rate, or their layer is
  ///                      missing on most activations.
  explicit HolovizOp(
      LayerStack layer_stack, SinkDesc sink,
      std::chrono::nanoseconds display_rate = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>{1.0 / 60.0}),
      std::optional<std::chrono::nanoseconds> max_input_age = std::nullopt);

  HolovizOp(const HolovizOp&) = delete;
  HolovizOp& operator=(const HolovizOp&) = delete;
  HolovizOp(HolovizOp&&) = delete;
  HolovizOp& operator=(HolovizOp&&) = delete;
  ~HolovizOp() override;

  /// Input port for layer \p index (< \ref kMaxLayerPorts).
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& layer_input(
      std::size_t index) const noexcept {
    switch (index) {
      case 0:
        return layer_input_0_;
      case 1:
        return layer_input_1_;
      case 2:
        return layer_input_2_;
      case 3:
        return layer_input_3_;
      case 4:
        return layer_input_4_;
      case 5:
        return layer_input_5_;
      case 6:
        return layer_input_6_;
      case 7:
        return layer_input_7_;
      default:
        std::terminate();  // index >= kMaxLayerPorts
    }
  }
  [[nodiscard]] holoscan::Input<holoscan::Tensor>& layer_input(std::size_t index) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<holoscan::Input<holoscan::Tensor>&>(
        static_cast<const HolovizOp*>(this)->layer_input(index));
  }

  void setup(holoscan::OperatorSpec& spec) override;
  /// OnClock at \p display_rate; every layer selected with LatestWithin{max_input_age}. Image
  /// layers hold their last sample for the same bound (ReuseLast); other kinds are skipped.
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;
  void stop() override;

  /// Private implementation (defined in the operator's private sources); opaque to callers.
  struct Impl;

 private:
  friend struct Impl;

  LayerStack layer_stack_;
  SinkDesc sink_;
  std::chrono::nanoseconds display_rate_;
  std::chrono::nanoseconds max_input_age_;
  std::unique_ptr<Impl> impl_;  ///< renderer state, alive between start() and stop()

  // Input<T> is non-movable, hence a fixed bank rather than an array. Port i binds to
  // layer_stack_.layers()[i]; ports beyond the declared count stay unbound.
  holoscan::Input<holoscan::Tensor> layer_input_0_;
  holoscan::Input<holoscan::Tensor> layer_input_1_;
  holoscan::Input<holoscan::Tensor> layer_input_2_;
  holoscan::Input<holoscan::Tensor> layer_input_3_;
  holoscan::Input<holoscan::Tensor> layer_input_4_;
  holoscan::Input<holoscan::Tensor> layer_input_5_;
  holoscan::Input<holoscan::Tensor> layer_input_6_;
  holoscan::Input<holoscan::Tensor> layer_input_7_;
};

}  // namespace holoscan::visualization
