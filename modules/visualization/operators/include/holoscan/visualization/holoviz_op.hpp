// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
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
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
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

/// Describe the source image's transfer function, independently of the sink's encoding.
enum class ColorSpace {
  kPassThrough,    ///< Use components as-is (treated as linear by an sRGB framebuffer).
  kSRGBNonlinear,  ///< Decode sRGB RGB before filtering; currently 8-bit RGBA/BGRA only.
  kBT709Linear,    ///< Linear BT.709/sRGB primaries; no source transfer conversion.
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
/// The bound input port receives a \c holoscan::schema::ImageT whose \c encoding and
/// \c color_space are validated against \p format / \p color_space before the underlying
/// \c holoscan::Tensor (`ImageT::data`) is used; a mismatching frame is logged and dropped. The
/// tensor itself is in HWC layout and its dtype/channel count must still agree with \p format:
///
/// | format               | accepted tensors                                        |
/// |----------------------|---------------------------------------------------------|
/// | kR8G8B8Unorm         | uint8 `{H, W, 3}`; expanded to RGBA8 with alpha 255     |
/// | kNV12                | uint8 bytes; two `plane_layouts` (Y, UV); even H and W  |
/// | kYUYV422             | uint8 `{H, W, 2}`; even W; byte order Y0, U, Y1, V    |
/// | kR8Unorm             | uint8 `{H, W}` or `{H, W, 1}`                           |
/// | kR8G8B8A8Unorm/BGRA  | uint8 `{H, W, 4}`                                       |
/// | kR16Unorm            | uint16 `{H, W}` or `{H, W, 1}`, or packed uint8 `{H, W, 2}` |
/// | kR16G16B16A16Unorm   | uint16 `{H, W, 4}`, or packed uint8 `{H, W, 8}`         |
///
/// Row pitch is taken from `strides()[0]` (bytes) and may exceed the packed row size.
/// Pixels and channels within each row must be contiguous. NV12 is multi-planar: its geometry
/// comes from `ImageT::width`/`height` and each plane's offset and row stride from
/// `ImageT::plane_layouts` (required by the schema for NV12), so the backing Tensor may be a
/// flat byte buffer and the planes need not be adjacent. Maximum dimensions refer to visible
/// pixels. Host and CUDA tensors are accepted; host rows are packed before upload.
///
/// A frame's wire `color_space` is checked against \p color_space / \p yuv_model: `UNSPECIFIED`
/// (the producer makes no claim) is accepted into any layer, `SRGB` requires
/// \c ColorSpace::kSRGBNonlinear, and `REC601`/`REC709`/`REC2020` on a YUV format require the
/// matching \p yuv_model. Any other labelled combination is rejected rather than rendered with
/// the wrong transfer function or matrix.
struct ImageLayerDesc {
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
  uint32_t max_width = 0;
  uint32_t max_height = 0;
  /// Source transfer function; alpha is always linear. With \c kBT709Linear, enable the
  /// sink's \c framebuffer_srgb to encode RGB AFTER linear-light composition for an SDR
  /// display. With a UNORM framebuffer the output remains linear. \c kSRGBNonlinear uses
  /// native sRGB texture sampling; unsupported source formats fail during setup.
  ColorSpace color_space = ColorSpace::kPassThrough;
  /// Required (both) for YUV formats, must be \c nullopt otherwise.
  /// Chroma reconstruction is centered; BT.2020 uses non-constant luminance conversion.
  /// Conversion produces opaque RGB; use ColorSpace::kPassThrough (transfer-function
  /// conversion remains unsupported).
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

/// Fixed 16-byte header that starts every row of a text layer's input tensor (see
/// \ref TextLayerDesc). Standard-layout and trivially copyable so a producer can \c memcpy it
/// into bytes 0..15 of the row; the row's UTF-8 text follows at byte 16.
struct TextRunHeader {
  /// Position of the run's top-left corner, origin at the framebuffer's top-left. The unit is
  /// the layer's \ref TextLayerDesc::normalized_coordinates: framebuffer pixels by default, or
  /// fractions of the framebuffer (0 is the left/top edge, 1 the right/bottom edge) when set. Any
  /// finite value is accepted, so a run may sit partly or wholly outside the framebuffer; glyphs
  /// beyond it are clipped at draw time.
  float x = 0.0f;
  float y = 0.0f;
  float size_px = 16.0f;         ///< Glyph height in framebuffer pixels; finite and > 0.
  uint32_t color = 0xFFFFFFFFu;  ///< Packed 0xAABBGGRR (red in the lowest byte).
};
static_assert(sizeof(TextRunHeader) == 16U && std::is_standard_layout_v<TextRunHeader> &&
                  std::is_trivially_copyable_v<TextRunHeader>,
              "TextRunHeader is the on-the-wire row header of a text layer");
static_assert(offsetof(TextRunHeader, y) == 4U && offsetof(TextRunHeader, size_px) == 8U &&
              offsetof(TextRunHeader, color) == 12U);
/// Byte length of \ref TextRunHeader, i.e. where a text row's UTF-8 bytes begin.
inline constexpr std::size_t kTextRunHeaderBytes = sizeof(TextRunHeader);

/// Text overlay layer. \p max_glyphs is a hard per-frame budget: the summed UTF-8 byte length
/// of every run in one sample must not exceed it (the built-in monospace font covers ASCII
/// 32..127 and draws any other byte as `?`, so for ASCII text bytes and glyphs coincide).
///
/// The bound input port receives a host-resident \c uint8 \c holoscan::Tensor of fixed-width
/// text records, shape `{N, 16 + L}` with packed rows: one row per positioned run, \c N runs
/// of at most \c L text bytes each.
///
/// | row bytes | content                                                              |
/// |-----------|----------------------------------------------------------------------|
/// | 0..15     | \ref TextRunHeader: `float32 x, y, size_px; uint32 color`            |
/// | 16..      | `L` bytes of UTF-8, NUL-padded; the run ends at the first NUL or at L |
///
/// Rows whose text is empty are skipped, so a producer may size \c N to its maximum and blank
/// the rows it does not use; a sample with `N == 0` draws nothing.
///
/// The port declares a frozen tensor contract (host-resident, \c uint8, rank 2) that the Runtime
/// enforces before \c compute() runs: a sample violating it is a terminal \c kProtocolError that
/// closes the session rather than a dropped frame, and a producer whose declared output layout
/// disagrees is rejected at graph compile time.
///
/// Properties the Runtime does not constrain are validated per sample: a row shorter than the
/// header, unpacked rows, a non-finite position, a non-positive or non-finite \c size_px, or more
/// text bytes than \p max_glyphs is logged and dropped for that activation only (the layer is
/// left out of the frame; the operator keeps running). A position outside the framebuffer (or
/// outside [0,1] with \p normalized_coordinates) is not an error: the run is drawn where it lands
/// and clipped to the framebuffer.
struct TextLayerDesc {
  uint32_t max_glyphs = 0;
  /// Unit of \ref TextRunHeader::x and \ref TextRunHeader::y. \c false (default): framebuffer
  /// pixels, so a run keeps its pixel offset when the window is resized. \c true: fractions of
  /// the framebuffer in [0,1], so the layout scales with the framebuffer. The record bytes are
  /// identical in both modes; the flag only selects how the operator interprets them on the CPU.
  bool normalized_coordinates = false;
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

/// Maximum number of declared layers on one HolovizOp, across every layer kind.
inline constexpr std::size_t kMaxLayerPorts = 8;

/// Physical image-kind port slots (Input<schema::ImageT>, so the wire semantics travel with the
/// frame). Sized to \ref kMaxLayerPorts so any partition of a legal stack fits: eight image layers
/// and zero geometry layers is as valid as the reverse. See holoviz_op.cpp's constructor for how a
/// declared ImageLayerDesc layer is assigned one of these slots.
inline constexpr std::size_t kMaxImageLayerPorts = kMaxLayerPorts;

/// Physical tensor-kind (geometry/text/LUT) port slots (Input<Tensor>); see
/// \ref kMaxImageLayerPorts.
inline constexpr std::size_t kMaxTensorLayerPorts = kMaxLayerPorts;

/// Render one frame per sample admitted on layer \p layer.
///
/// Lowers to an \c OnEach trigger on that layer's input port, so a frame is presented as soon as
/// the driving sample is admitted rather than on the next clock tick. The other layers composite
/// their freshest sample within the operator's \c max_input_age.
struct LayerTrigger {
  std::size_t layer{0};  ///< Index into LayerStack::layers(); must name a declared layer.
};

/// Schedule a render candidate at a fixed period, subject to input admission.
///
/// Lowers to an \c OnClock trigger; every layer is selected with \c LatestWithin. Required for a
/// layer-less operator, which activates on every tick because it has no inputs to admit.
///
/// With layers, a tick is a candidate rather than a guaranteed activation: missing-input
/// admission still applies. Once an image layer's \c ReuseLast hold (\c max_input_age) is
/// exhausted and no other layer supplies a \c SafeDefault, the tick does not run compute(), and
/// neither does the window-event polling inside it. A ClockTrigger is therefore not an
/// independent event pump; a single-image window whose producer stalls stops servicing events
/// once \c max_input_age elapses, just as it would under a \ref LayerTrigger.
struct ClockTrigger {
  /// Render period; must be positive. Defaults to 60 Hz.
  std::chrono::nanoseconds period{std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>{1.0 / 60.0})};
};

/// What starts one render activation. Closed set: adding an alternative is an API/ABI break.
using FrameTrigger = std::variant<LayerTrigger, ClockTrigger>;

/// Renders a \ref LayerStack into a \ref SinkDesc.
///
/// One typed input per declared layer (up to \ref kMaxLayerPorts): image-kind layers bind to an
/// \c Input<schema::ImageT> slot, geometry/text/LUT layers bind to an \c Input<Tensor> slot. A
/// declared layer's position in the \ref LayerStack is independent of which physical port slot
/// it binds to (see the constructor). For a \ref CudaSinkDesc sink only, one
/// \c Output<schema::ImageT> (\ref output()) carries every rendered frame as CUDA device memory
/// so a second operator -- for example another HolovizOp driving a display -- can consume it
/// without a copy. By default rendering
/// is driven by the arrival of layer 0 (\ref LayerTrigger): each admitted sample on that port
/// produces one frame, and every other layer contributes its freshest sample no older than
/// \c max_input_age. Image layers hold their last sample for that same bound when nothing new
/// arrived (\c ReuseLast); other layer kinds are skipped for the frame. A \ref ClockTrigger
/// schedules a render candidate at a fixed period instead, subject to the same missing-input
/// admission (see \ref ClockTrigger).
///
/// Under a \ref LayerTrigger an activation exists only when a trigger sample arrives, so:
/// - a transient swapchain acquire failure (backpressure, out-of-date surface) drops that
///   sample's frame; the next trigger sample retries;
/// - window events and close requests are polled only on activation; a stalled producer freezes
///   event handling until it resumes;
/// - trigger samples that arrive while the sink stalls (for example during the swapchain acquire
///   wait) queue on the trigger port, and \c OnEach then plays through that backlog one frame per
///   sample, so presentation latency grows; connect the trigger edge as a depth-1 drop-oldest
///   mailbox so a stall costs at most one stale frame (as \c gray_scale_example does);
/// - a side layer produced faster than the trigger layer leaves unselected samples queued
///   (\c LatestWithin and \c ReuseLast do not retire them), so connect such edges as a depth-1
///   drop-oldest mailbox.
/// A \ref ClockTrigger offers periodic retry opportunities for the first point while input
/// admission still permits compute(). It does not guarantee event handling after every image
/// layer's hold has expired: for a single-image window, a stalled producer freezes event handling
/// under either trigger once \c max_input_age elapses.
///
/// ```cpp
/// LayerStack stack;
/// stack.add_image_layer("video", ImageLayerDesc{...});      // layer 0: drives rendering
/// stack.add_geometry_layer("overlay", GeometryLayerDesc{...});  // layer 1: freshest within 100 ms
/// auto op = graph.op<HolovizOp>("holoviz", std::move(stack), WindowSinkDesc{.title = "Demo"});
/// const holoscan::ConnectionOptions mailbox{.queue_policy = holoscan::QueuePolicy::kDropOldest,
///                                           .queue_depth = 1U};
/// // camera->frame is an Output<schema::ImageT>; detector->boxes is an Output<Tensor>.
/// graph.add_flow(camera->frame, op->image_layer_input(0), mailbox);    // trigger edge
/// graph.add_flow(detector->boxes, op->tensor_layer_input(1), mailbox);  // side edge
/// ```
///
/// Backend errors are translated into the Holoscan Core error domain at the lifecycle boundary.
class HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT HolovizOp : public holoscan::Operator<> {
 public:
  /// @param layer_stack   Layers to render; at most \ref kMaxLayerPorts.
  /// @param sink          Where frames go.
  /// @param trigger       \ref LayerTrigger (default: layer 0) renders one frame per arriving
  ///                      sample on that layer; \ref ClockTrigger schedules a render candidate
  ///                      at a fixed period. A LayerTrigger must name a declared layer; an empty
  ///                      stack requires a ClockTrigger.
  /// @param max_input_age Freshness window for every non-trigger layer, and the bound on how long
  ///                      an image layer holds its last frame (default 100 ms). Must be positive.
  ///                      Widen it for producers slower than the trigger rate, or their layer is
  ///                      missing on most activations. Under a ClockTrigger it also bounds how
  ///                      long a lone image layer keeps the operator computing (and polling
  ///                      window events) after its producer stops.
  explicit HolovizOp(LayerStack layer_stack, SinkDesc sink, FrameTrigger trigger = LayerTrigger{},
                     std::chrono::nanoseconds max_input_age = std::chrono::milliseconds{100});

  HolovizOp(const HolovizOp&) = delete;
  HolovizOp& operator=(const HolovizOp&) = delete;
  HolovizOp(HolovizOp&&) = delete;
  HolovizOp& operator=(HolovizOp&&) = delete;
  ~HolovizOp() override;

  /// Frame output port, bound only for a \ref CudaSinkDesc sink (unbound for the display
  /// sinks). Each presented frame is one \c schema::ImageT whose \c data Tensor is drawn from a
  /// Runtime pool: `width`/`height` are the sink's declared extent, `encoding`/`color_space`
  /// follow the sink's format, and the Tensor is the format's native element type in `{H, W}`
  /// (single-channel) or `{H, W, 4}` layout (see \ref CudaSinkDesc for the table). Emitted with
  /// the activation time as its capture time; the deployment must bind the port's CUDA device
  /// (CompileOptions::deployment.bind_tensor_output_device). Port name: "output"; no layer may
  /// use that name.
  [[nodiscard]] const holoscan::Output<holoscan::schema::ImageT>& output() const noexcept {
    return output_;
  }
  [[nodiscard]] holoscan::Output<holoscan::schema::ImageT>& output() noexcept { return output_; }

  /// @brief Return the immutable layer declarations captured at construction.
  [[nodiscard]] const LayerStack& layer_stack() const noexcept { return layer_stack_; }

  /// @brief Return the immutable sink declaration captured at construction.
  [[nodiscard]] const SinkDesc& sink() const noexcept { return sink_; }

  /// @brief Return the effective immutable input-freshness window.
  [[nodiscard]] std::chrono::nanoseconds max_input_age() const noexcept { return max_input_age_; }

  /// Return the trigger this operator was constructed with.
  [[nodiscard]] const FrameTrigger& trigger() const noexcept { return trigger_; }

  /// Input port for the declared image-kind layer at \p index (an index into \ref LayerStack
  /// declaration order, the same index the app passed to `add_image_layer()` -- not a physical
  /// pool slot). For graph wiring, e.g. `graph.add_flow(producer, viz->image_layer_input(0))`.
  /// Terminates if \p index is out of range or the layer at that index is not image-kind.
  [[nodiscard]] const holoscan::Input<holoscan::schema::ImageT>& image_layer_input(
      std::size_t index) const noexcept;
  [[nodiscard]] holoscan::Input<holoscan::schema::ImageT>& image_layer_input(
      std::size_t index) noexcept;

  /// Input port for the declared geometry/text/LUT layer at \p index (an index into \ref
  /// LayerStack declaration order -- not a physical pool slot). For graph wiring. Terminates if
  /// \p index is out of range or the layer at that index is image-kind.
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& tensor_layer_input(
      std::size_t index) const noexcept;
  [[nodiscard]] holoscan::Input<holoscan::Tensor>& tensor_layer_input(std::size_t index) noexcept;

  void setup(holoscan::OperatorSpec& spec) override;
  /// Return the temporal contract: \c OnEach on the trigger layer's port (LayerTrigger) or
  /// \c OnClock at the period (ClockTrigger). Every other layer is selected with
  /// LatestWithin{max_input_age}; image layers hold their last sample for the same bound
  /// (ReuseLast) and other kinds are skipped (SafeDefault). The OnEach driver carries no selector.
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
  FrameTrigger trigger_;
  std::chrono::nanoseconds max_input_age_;
  std::unique_ptr<Impl> impl_;  ///< renderer state, alive between start() and stop()

  /// Captured-frame output; declared in setup() only when sink_ holds a CudaSinkDesc.
  holoscan::Output<holoscan::schema::ImageT> output_;

  // Declared-layer-index -> pool-slot mapping, built once in the constructor by partitioning
  // layer_stack_.layers() by kind (image vs. tensor) in declared order. A declared layer's
  // position in the app's LayerStack is independent of which physical port slot it binds to;
  // every other piece of per-layer state (image_layer_cache(), record_tensors, the render
  // list's layer index) stays indexed by declared layer index, not by this slot.
  //
  // The port objects themselves (Input<schema::ImageT> for image-kind slots,
  // Input<holoscan::Tensor> for tensor-kind slots) live on Impl, not here, so setup()/contract()/
  // build_render_list() reach them through impl_. The public image_layer_input()/
  // tensor_layer_input() accessors above are declared-index-indexed wrappers, defined
  // out-of-line in holoviz_op.cpp, that translate through this map. Keeping the port members off
  // this class does not keep schema::ImageT out of this public header, though -- the accessors'
  // own return types (Input<schema::ImageT>&) require it to be a complete type here regardless,
  // since graph wiring (graph.add_flow(producer, viz->image_layer_input(0))) needs the app to
  // hold a concretely-typed reference. sensor_io is therefore a PUBLIC dependency of this
  // operator (see the ImageT port alignment design doc's Decision 2 addendum, second correction).
  std::array<std::size_t, kMaxLayerPorts> pool_slot_for_layer_{};
};

}  // namespace holoscan::visualization
