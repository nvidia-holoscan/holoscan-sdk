// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <holoscan/visualization/operator_catalog.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace holoscan::visualization {
namespace {

using authoring::OpaqueOperatorRef;
using authoring::ParameterAssignment;
using authoring::ParameterValue;
using authoring::ParameterValueKind;

constexpr std::string_view kRegistryName{"holoscan.visualization.HolovizOp"};
constexpr std::array<std::string_view, 2> kRequiredParameters{"layers", "sink"};

[[noreturn]] void parameter_error(std::string_view token, std::string_view field = {},
                                  std::string_view detail = {}) {
  std::string message{token};
  if (!field.empty()) {
    message.append(fmt::format(" field_name={}", field));
  }
  if (!detail.empty()) {
    message.append(" ");
    message.append(detail);
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

[[nodiscard]] std::string accepted_values(std::span<const std::string_view> accepted) {
  std::string result;
  for (const std::string_view name : accepted) {
    if (!result.empty()) {
      result.append(",");
    }
    result.append(name);
  }
  return result;
}

// Borrows one structural assignment map while tracking the fields consumed by its parser.
class AssignmentView {
 public:
  AssignmentView(std::span<const ParameterAssignment> assignments, std::string path)
      : assignments_(assignments), path_(std::move(path)) {
    for (std::size_t index = 0U; index < assignments_.size(); ++index) {
      const auto duplicate = std::find_if(
          assignments_.begin() + static_cast<std::ptrdiff_t>(index + 1U),
          assignments_.end(),
          [&](const ParameterAssignment& value) { return value.name == assignments_[index].name; });
      if (duplicate != assignments_.end()) {
        parameter_error("VISUALIZATION_OPERATOR_PARAMETER_DUPLICATE", field_path(duplicate->name));
      }
    }
  }

  [[nodiscard]] const ParameterValue* find(std::string_view name) const {
    if (std::find(accepted_.begin(), accepted_.end(), name) == accepted_.end()) {
      accepted_.push_back(name);
    }
    const auto found = std::find_if(assignments_.begin(),
                                    assignments_.end(),
                                    [&](const auto& item) { return item.name == name; });
    return found == assignments_.end() ? nullptr : &found->value;
  }

  [[nodiscard]] const ParameterValue& require(std::string_view name) const {
    const ParameterValue* value = find(name);
    if (value == nullptr) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_REQUIRED", field_path(name));
    }
    return *value;
  }

  void reject_unknown() const {
    for (const ParameterAssignment& assignment : assignments_) {
      if (std::find(accepted_.begin(), accepted_.end(), assignment.name) == accepted_.end()) {
        parameter_error("VISUALIZATION_OPERATOR_PARAMETER_UNKNOWN",
                        field_path(assignment.name),
                        fmt::format("accepted={}", accepted_values(accepted_)));
      }
    }
  }

  [[nodiscard]] std::string field_path(std::string_view name) const {
    if (path_.empty()) {
      return std::string{name};
    }
    return path_ + "." + std::string{name};
  }

 private:
  std::span<const ParameterAssignment> assignments_;
  std::string path_;
  mutable std::vector<std::string_view> accepted_;
};

[[nodiscard]] AssignmentView nested(const ParameterValue& value, std::string path) {
  if (value.kind != ParameterValueKind::kAssignments) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  return AssignmentView{value.nested_assignments(), std::move(path)};
}

[[nodiscard]] std::string string_value(const ParameterValue& value, std::string_view path) {
  if (value.kind != ParameterValueKind::kString) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  return std::string{value.string};
}

[[nodiscard]] bool boolean_value(const ParameterValue& value, std::string_view path) {
  bool result{};
  if (value.kind != ParameterValueKind::kScalar || !value.scalar.try_as(result)) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  return result;
}

template <typename Integer>
[[nodiscard]] Integer integer_value(const ParameterValue& value, std::string_view path) {
  const auto convert = [&]<typename Source>() -> std::optional<Integer> {
    Source source{};
    if (!value.scalar.try_as(source)) {
      return std::nullopt;
    }
    if (!std::in_range<Integer>(source)) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                      path,
                      fmt::format("value={} accepted_range=[{},{}]",
                                  source,
                                  std::numeric_limits<Integer>::lowest(),
                                  std::numeric_limits<Integer>::max()));
    }
    return static_cast<Integer>(source);
  };
  if (value.kind != ParameterValueKind::kScalar) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  if (const auto result = convert.template operator()<std::int64_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint64_t>())
    return *result;
  if (const auto result = convert.template operator()<std::int32_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint32_t>())
    return *result;
  if (const auto result = convert.template operator()<std::int16_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint16_t>())
    return *result;
  if (const auto result = convert.template operator()<std::int8_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint8_t>())
    return *result;
  parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
}

[[nodiscard]] float float_value(const ParameterValue& value, std::string_view path) {
  if (value.kind != ParameterValueKind::kScalar) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
  }
  const auto convert = [&]<typename Source>() -> std::optional<float> {
    Source source{};
    if (!value.scalar.try_as(source)) {
      return std::nullopt;
    }
    const double widened = static_cast<double>(source);
    if (!std::isfinite(widened) || widened < std::numeric_limits<float>::lowest() ||
        widened > std::numeric_limits<float>::max()) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                      path,
                      fmt::format("value={} accepted=finite-float32", source));
    }
    return static_cast<float>(source);
  };
  if (const auto result = convert.template operator()<double>())
    return *result;
  if (const auto result = convert.template operator()<float>())
    return *result;
  if (const auto result = convert.template operator()<std::int64_t>())
    return *result;
  if (const auto result = convert.template operator()<std::uint64_t>())
    return *result;
  parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", path);
}

template <typename Enum>
struct EnumChoice {
  std::string_view name;
  Enum value;
};

template <typename Enum, std::size_t Size>
[[nodiscard]] Enum enum_value(const ParameterValue& value, std::string_view path,
                              const std::array<EnumChoice<Enum>, Size>& choices) {
  if (value.kind == ParameterValueKind::kString) {
    const auto found = std::find_if(choices.begin(), choices.end(), [&](const auto& choice) {
      return choice.name == value.string;
    });
    if (found != choices.end()) {
      return found->value;
    }
  } else if (value.kind == ParameterValueKind::kScalar) {
    using Underlying = std::underlying_type_t<Enum>;
    const Underlying numeric = integer_value<Underlying>(value, path);
    const auto found = std::find_if(choices.begin(), choices.end(), [&](const auto& choice) {
      return static_cast<Underlying>(choice.value) == numeric;
    });
    if (found != choices.end()) {
      return found->value;
    }
  }
  std::string accepted;
  for (const auto& choice : choices) {
    if (!accepted.empty()) {
      accepted.append(",");
    }
    accepted.append(choice.name);
  }
  const std::string supplied = value.kind == ParameterValueKind::kString
                                   ? fmt::format("'{}'", value.string)
                                   : std::string{"<numeric>"};
  parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                  path,
                  fmt::format("value={} accepted={}", supplied, accepted));
}

constexpr std::array kImageFormats{
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "r8_unorm", ::holoscan::visualization::ImageFormat::kR8Unorm},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "rgb8_unorm", ::holoscan::visualization::ImageFormat::kR8G8B8Unorm},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "rgba8_unorm", ::holoscan::visualization::ImageFormat::kR8G8B8A8Unorm},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "nv12", ::holoscan::visualization::ImageFormat::kNV12},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "yuyv422", ::holoscan::visualization::ImageFormat::kYUYV422},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "bgra8_unorm", ::holoscan::visualization::ImageFormat::kB8G8R8A8Unorm},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "r16_unorm", ::holoscan::visualization::ImageFormat::kR16Unorm},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "rgba16_unorm", ::holoscan::visualization::ImageFormat::kR16G16B16A16Unorm},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "rgba8_srgb", ::holoscan::visualization::ImageFormat::kR8G8B8A8Srgb},
    EnumChoice<::holoscan::visualization::ImageFormat>{
        "bgra8_srgb", ::holoscan::visualization::ImageFormat::kB8G8R8A8Srgb},
};

consteval bool image_format_catalog_is_complete() {
  if (kImageFormats.size() + 1U !=
      magic_enum::enum_count<::holoscan::visualization::ImageFormat>()) {
    return false;
  }
  for (const auto format : magic_enum::enum_values<::holoscan::visualization::ImageFormat>()) {
    const auto count = std::count_if(kImageFormats.begin(),
                                     kImageFormats.end(),
                                     [&](const auto& row) { return row.value == format; });
    if ((format == ::holoscan::visualization::ImageFormat::kUnknown && count != 0) ||
        (format != ::holoscan::visualization::ImageFormat::kUnknown && count != 1)) {
      return false;
    }
  }
  return true;
}

static_assert(image_format_catalog_is_complete(),
              "kImageFormats must cover every non-unknown ImageFormat exactly once");
constexpr std::array kColorSpaces{
    EnumChoice<::holoscan::visualization::ColorSpace>{
        "pass_through", ::holoscan::visualization::ColorSpace::kPassThrough},
    EnumChoice<::holoscan::visualization::ColorSpace>{
        "srgb_nonlinear", ::holoscan::visualization::ColorSpace::kSRGBNonlinear},
    EnumChoice<::holoscan::visualization::ColorSpace>{
        "bt709_linear", ::holoscan::visualization::ColorSpace::kBT709Linear},
};
constexpr std::array kYuvModels{
    EnumChoice<::holoscan::visualization::YUVModelConversion>{
        "bt601", ::holoscan::visualization::YUVModelConversion::kBT601},
    EnumChoice<::holoscan::visualization::YUVModelConversion>{
        "bt709", ::holoscan::visualization::YUVModelConversion::kBT709},
    EnumChoice<::holoscan::visualization::YUVModelConversion>{
        "bt2020", ::holoscan::visualization::YUVModelConversion::kBT2020},
};
constexpr std::array kYuvRanges{
    EnumChoice<::holoscan::visualization::YUVRange>{"limited",
                                                    ::holoscan::visualization::YUVRange::kLimited},
    EnumChoice<::holoscan::visualization::YUVRange>{"full",
                                                    ::holoscan::visualization::YUVRange::kFull},
};
constexpr std::array kTopologies{
    EnumChoice<::holoscan::visualization::GeometryTopology>{
        "line_list", ::holoscan::visualization::GeometryTopology::kLineList},
    EnumChoice<::holoscan::visualization::GeometryTopology>{
        "line_strip", ::holoscan::visualization::GeometryTopology::kLineStrip},
    EnumChoice<::holoscan::visualization::GeometryTopology>{
        "cross_list", ::holoscan::visualization::GeometryTopology::kCrossList},
    EnumChoice<::holoscan::visualization::GeometryTopology>{
        "rectangle_list", ::holoscan::visualization::GeometryTopology::kRectangleList},
    EnumChoice<::holoscan::visualization::GeometryTopology>{
        "oval_list", ::holoscan::visualization::GeometryTopology::kOvalList},
};
constexpr std::array kPresentModes{
    EnumChoice<::holoscan::visualization::PresentMode>{
        "auto", ::holoscan::visualization::PresentMode::kAuto},
    EnumChoice<::holoscan::visualization::PresentMode>{
        "fifo", ::holoscan::visualization::PresentMode::kFIFO},
    EnumChoice<::holoscan::visualization::PresentMode>{
        "fifo_latest_ready", ::holoscan::visualization::PresentMode::kFIFOLatestReady},
    EnumChoice<::holoscan::visualization::PresentMode>{
        "mailbox", ::holoscan::visualization::PresentMode::kMailbox},
    EnumChoice<::holoscan::visualization::PresentMode>{
        "immediate", ::holoscan::visualization::PresentMode::kImmediate},
    EnumChoice<::holoscan::visualization::PresentMode>{
        "fifo_relaxed", ::holoscan::visualization::PresentMode::kFIFORelaxed},
};

template <typename Enum, std::size_t Size>
[[nodiscard]] std::optional<Enum> optional_enum(const AssignmentView& values, std::string_view name,
                                                const std::array<EnumChoice<Enum>, Size>& choices) {
  const ParameterValue* value = values.find(name);
  if (value == nullptr || value->kind == ParameterValueKind::kNone) {
    return std::nullopt;
  }
  return enum_value(*value, values.field_path(name), choices);
}

template <typename Enum, std::size_t Size>
[[nodiscard]] Enum enum_or(const AssignmentView& values, std::string_view name, Enum default_value,
                           const std::array<EnumChoice<Enum>, Size>& choices) {
  const ParameterValue* value = values.find(name);
  return value == nullptr ? default_value : enum_value(*value, values.field_path(name), choices);
}

template <typename Integer>
[[nodiscard]] Integer integer_or(const AssignmentView& values, std::string_view name,
                                 Integer default_value) {
  const ParameterValue* value = values.find(name);
  return value == nullptr ? default_value : integer_value<Integer>(*value, values.field_path(name));
}

template <typename Value>
void require_positive(Value value, std::string_view path) {
  if (value <= Value{}) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                    path,
                    fmt::format("value={} accepted=>0", value));
  }
}

[[nodiscard]] std::optional<ViewTransform> parse_view(const AssignmentView& values) {
  const ParameterValue* value = values.find("view");
  if (value == nullptr || value->kind == ParameterValueKind::kNone) {
    return std::nullopt;
  }
  AssignmentView view = nested(*value, values.field_path("view"));
  ViewTransform result;
  if (const auto* field = view.find("offset_x"))
    result.offset_x = float_value(*field, view.field_path("offset_x"));
  if (const auto* field = view.find("offset_y"))
    result.offset_y = float_value(*field, view.field_path("offset_y"));
  if (const auto* field = view.find("scale_x"))
    result.scale_x = float_value(*field, view.field_path("scale_x"));
  if (const auto* field = view.find("scale_y"))
    result.scale_y = float_value(*field, view.field_path("scale_y"));
  view.reject_unknown();
  return result;
}

void add_layer(LayerStack& stack, std::set<std::string, std::less<>>& port_names,
               const ParameterValue& value, std::size_t index) {
  const std::string path = "layers[" + std::to_string(index) + "]";
  AssignmentView layer = nested(value, path);
  const std::string port_name =
      string_value(layer.require("port_name"), layer.field_path("port_name"));
  const std::string kind = string_value(layer.require("kind"), layer.field_path("kind"));
  if (port_name.empty()) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                    layer.field_path("port_name"),
                    "value='' accepted=nonempty-string");
  }
  if (!port_names.emplace(port_name).second) {
    parameter_error("VISUALIZATION_OPERATOR_PORT_NAME_DUPLICATE",
                    layer.field_path("port_name"),
                    fmt::format("value='{}'", port_name));
  }

  if (kind == "image") {
    ImageLayerDesc desc;
    desc.format = enum_or(layer, "format", desc.format, kImageFormats);
    desc.max_width = integer_or<std::uint32_t>(layer, "max_width", desc.max_width);
    desc.max_height = integer_or<std::uint32_t>(layer, "max_height", desc.max_height);
    require_positive(desc.max_width, layer.field_path("max_width"));
    require_positive(desc.max_height, layer.field_path("max_height"));
    desc.color_space = enum_or(layer, "color_space", desc.color_space, kColorSpaces);
    desc.yuv_model = optional_enum(layer, "yuv_model", kYuvModels);
    desc.yuv_range = optional_enum(layer, "yuv_range", kYuvRanges);
    const bool is_yuv = desc.format == ImageFormat::kNV12 || desc.format == ImageFormat::kYUYV422;
    if (is_yuv && (!desc.yuv_model.has_value() || !desc.yuv_range.has_value())) {
      parameter_error("VISUALIZATION_OPERATOR_LAYER_YUV_CONVERSION_REQUIRED",
                      path,
                      "accepted=yuv_model-and-yuv_range");
    }
    if (!is_yuv && (desc.yuv_model.has_value() || desc.yuv_range.has_value())) {
      parameter_error("VISUALIZATION_OPERATOR_LAYER_YUV_CONVERSION_FORBIDDEN",
                      path,
                      "accepted=yuv-fields-omitted");
    }
    desc.view = parse_view(layer);
    layer.reject_unknown();
    stack.add_image_layer(port_name, std::move(desc));
    return;
  }
  if (kind == "geometry") {
    GeometryLayerDesc desc;
    const ParameterValue& topology_values = layer.require("topologies");
    if (topology_values.kind != ParameterValueKind::kSequence) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH",
                      layer.field_path("topologies"));
    }
    std::set<GeometryTopology> unique_topologies;
    for (std::size_t topology_index = 0U; topology_index < topology_values.sequence_size;
         ++topology_index) {
      const auto topology = enum_value(topology_values.sequence_values()[topology_index],
                                       fmt::format("{}.topologies[{}]", path, topology_index),
                                       kTopologies);
      if (!unique_topologies.emplace(topology).second) {
        parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                        fmt::format("{}.topologies[{}]", path, topology_index),
                        "accepted=unique-topology");
      }
      desc.topologies.push_back(topology);
    }
    if (desc.topologies.empty()) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                      layer.field_path("topologies"),
                      "value=[] accepted=nonempty-sequence");
    }
    desc.max_primitives = integer_or<std::uint32_t>(layer, "max_primitives", desc.max_primitives);
    require_positive(desc.max_primitives, layer.field_path("max_primitives"));
    if (const auto* field = layer.find("default_line_width")) {
      desc.default_line_width = float_value(*field, layer.field_path("default_line_width"));
    }
    require_positive(desc.default_line_width, layer.field_path("default_line_width"));
    layer.reject_unknown();
    stack.add_geometry_layer(port_name, std::move(desc));
    return;
  }
  if (kind == "text") {
    TextLayerDesc desc;
    desc.max_glyphs = integer_or<std::uint32_t>(layer, "max_glyphs", desc.max_glyphs);
    require_positive(desc.max_glyphs, layer.field_path("max_glyphs"));
    if (const auto* field = layer.find("normalized_coordinates")) {
      desc.normalized_coordinates =
          boolean_value(*field, layer.field_path("normalized_coordinates"));
    }
    layer.reject_unknown();
    stack.add_text_layer(port_name, std::move(desc));
    return;
  }
  if (kind == "lut") {
    LUTLayerDesc desc;
    desc.lut_size = integer_or<std::uint32_t>(layer, "lut_size", desc.lut_size);
    desc.max_width = integer_or<std::uint32_t>(layer, "max_width", desc.max_width);
    desc.max_height = integer_or<std::uint32_t>(layer, "max_height", desc.max_height);
    require_positive(desc.lut_size, layer.field_path("lut_size"));
    require_positive(desc.max_width, layer.field_path("max_width"));
    require_positive(desc.max_height, layer.field_path("max_height"));
    layer.reject_unknown();
    stack.add_lut_layer(port_name, std::move(desc));
    return;
  }
  parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                  layer.field_path("kind"),
                  fmt::format("value='{}' accepted=image,geometry,text,lut", kind));
}

[[nodiscard]] LayerStack parse_layers(const ParameterValue& value) {
  if (value.kind != ParameterValueKind::kSequence) {
    parameter_error("VISUALIZATION_OPERATOR_PARAMETER_TYPE_MISMATCH", "layers");
  }
  if (value.sequence_size > kMaxLayerPorts) {
    parameter_error("VISUALIZATION_OPERATOR_LAYER_CAPACITY_EXCEEDED", "layers");
  }
  LayerStack stack;
  std::set<std::string, std::less<>> port_names;
  for (std::size_t index = 0U; index < value.sequence_size; ++index) {
    add_layer(stack, port_names, value.sequence_values()[index], index);
  }
  return stack;
}

// Materialize either FrameTrigger alternative before constructing the operator. HolovizOp
// terminates for an out-of-range layer or a non-positive period, so reject both at the authoring
// boundary instead.
[[nodiscard]] FrameTrigger parse_trigger(const AssignmentView& values, const LayerStack& layers) {
  const ParameterValue* value = values.find("trigger");
  if (value == nullptr || value->kind == ParameterValueKind::kNone) {
    if (layers.layers().empty()) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_REQUIRED",
                      "trigger",
                      "accepted=clock-trigger-for-empty-layer-stack");
    }
    return LayerTrigger{};
  }
  AssignmentView trigger = nested(*value, "trigger");
  const std::string kind = string_value(trigger.require("kind"), trigger.field_path("kind"));
  if (kind == "layer") {
    LayerTrigger desc;
    desc.layer = integer_or<std::size_t>(trigger, "layer", desc.layer);
    if (desc.layer >= layers.layers().size()) {
      parameter_error("VISUALIZATION_OPERATOR_TRIGGER_LAYER_OUT_OF_RANGE",
                      trigger.field_path("layer"),
                      fmt::format("value={} accepted=<{}", desc.layer, layers.layers().size()));
    }
    trigger.reject_unknown();
    return desc;
  }
  if (kind == "clock") {
    ClockTrigger desc;
    desc.period = std::chrono::nanoseconds{
        integer_or<std::int64_t>(trigger, "period_ns", desc.period.count())};
    if (desc.period <= std::chrono::nanoseconds::zero()) {
      parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                      trigger.field_path("period_ns"),
                      fmt::format("value={} accepted=>0", desc.period.count()));
    }
    trigger.reject_unknown();
    return desc;
  }
  parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                  trigger.field_path("kind"),
                  fmt::format("value='{}' accepted=layer,clock", kind));
}

[[nodiscard]] SinkDesc parse_sink(const ParameterValue& value) {
  AssignmentView sink = nested(value, "sink");
  const std::string kind = string_value(sink.require("kind"), sink.field_path("kind"));
  if (kind == "window") {
    WindowSinkDesc desc;
    if (const auto* field = sink.find("title"))
      desc.title = string_value(*field, sink.field_path("title"));
    desc.width = integer_or<std::uint32_t>(sink, "width", desc.width);
    desc.height = integer_or<std::uint32_t>(sink, "height", desc.height);
    desc.present_mode = enum_or(sink, "present_mode", desc.present_mode, kPresentModes);
    if (const auto* field = sink.find("framebuffer_srgb"))
      desc.framebuffer_srgb = boolean_value(*field, sink.field_path("framebuffer_srgb"));
    if (const auto* field = sink.find("fullscreen"))
      desc.fullscreen = boolean_value(*field, sink.field_path("fullscreen"));
    if (const auto* field = sink.find("display_name"))
      desc.display_name = string_value(*field, sink.field_path("display_name"));
    desc.refresh_hz = integer_or<std::uint32_t>(sink, "refresh_hz", desc.refresh_hz);
    if (const auto* field = sink.find("close_on_escape"))
      desc.close_on_escape = boolean_value(*field, sink.field_path("close_on_escape"));
    if (!desc.fullscreen) {
      // Monitor and mode selectors have no effect on a decorated window. Reject them rather than
      // accepting configuration that appears to work but is silently ignored.
      if (!desc.display_name.empty() || desc.refresh_hz != 0U) {
        parameter_error("VISUALIZATION_OPERATOR_SINK_FULLSCREEN_FIELD_FORBIDDEN",
                        "sink",
                        "accepted=fullscreen-true-for-display_name-and-refresh_hz");
      }
    }
    sink.reject_unknown();
    return desc;
  }
  if (kind == "exclusive_display") {
    ExclusiveDisplaySinkDesc desc;
    if (const auto* field = sink.find("display_name"))
      desc.display_name = string_value(*field, sink.field_path("display_name"));
    // Native display-mode selection treats each zero field independently as a wildcard. Preserve
    // width-only and height-only requests rather than imposing a paired-dimension rule here.
    desc.width = integer_or<std::uint32_t>(sink, "width", desc.width);
    desc.height = integer_or<std::uint32_t>(sink, "height", desc.height);
    desc.refresh_hz = integer_or<std::uint32_t>(sink, "refresh_hz", desc.refresh_hz);
    desc.present_mode = enum_or(sink, "present_mode", desc.present_mode, kPresentModes);
    if (const auto* field = sink.find("framebuffer_srgb"))
      desc.framebuffer_srgb = boolean_value(*field, sink.field_path("framebuffer_srgb"));
    sink.reject_unknown();
    return desc;
  }
  if (kind == "cuda") {
    CudaSinkDesc desc;
    desc.width = integer_or<std::uint32_t>(sink, "width", desc.width);
    desc.height = integer_or<std::uint32_t>(sink, "height", desc.height);
    require_positive(desc.width, sink.field_path("width"));
    require_positive(desc.height, sink.field_path("height"));
    desc.format = enum_or(sink, "format", desc.format, kImageFormats);
    sink.reject_unknown();
    return desc;
  }
  parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                  sink.field_path("kind"),
                  fmt::format("value='{}' accepted=window,exclusive_display,cuda", kind));
}

// Check invariants that depend on more than one independently parsed structural descriptor before
// invoking the native constructor, whose corresponding programming preconditions terminate.
void validate_layer_sink_compatibility(const LayerStack& layers, const SinkDesc& sink) {
  if (!std::holds_alternative<CudaSinkDesc>(sink)) {
    return;
  }
  for (std::size_t index = 0U; index < layers.layers().size(); ++index) {
    if (layers.layers()[index].port_name == "output") {
      parameter_error("VISUALIZATION_OPERATOR_PORT_NAME_RESERVED",
                      fmt::format("layers[{}].port_name", index),
                      "value='output' reserved-by=cuda-sink");
    }
  }
}

// Adapts Holoviz's structural constructor to the type-erased authoring-registry callbacks.
struct HolovizRegistrationThunks final : authoring::ManualOperatorRegistrationThunks<HolovizOp> {
  static OpaqueOperatorRef add(Graph& graph, std::string_view name, LayerStack layers,
                               SinkDesc sink, FrameTrigger trigger,
                               std::optional<std::chrono::nanoseconds> max_input_age) {
    return make_operator_ref(
        graph.op<HolovizOp>(name,
                            std::move(layers),
                            std::move(sink),
                            trigger,
                            max_input_age.value_or(std::chrono::milliseconds{100})));
  }

  static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
                                        const authoring::MaterializedParameters* params) {
    if (params != nullptr) {
      parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH");
    }
    return add(graph,
               name,
               LayerStack{},
               WindowSinkDesc{},
               // A layer-less operator has no port for an OnEach driver.
               ClockTrigger{.period = std::chrono::nanoseconds{16'666'666}},
               std::nullopt);
  }

  static void set_params(const OpaqueOperatorRef&, const authoring::MaterializedParameters&) {
    throw RuntimeError{
        Error{ErrorCode::kNotSupported, "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE"}};
  }
};

// Materialize the closed layer/sink grammar before mutating the destination Graph.
expected<OpaqueOperatorRef, Error> add_from_assignments(
    Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments) {
  return HolovizRegistrationThunks::invoke_add([&] {
    if (assignments.empty()) {
      return HolovizRegistrationThunks::add_to_graph(graph, name, nullptr);
    }
    AssignmentView values{assignments, {}};
    LayerStack layers = parse_layers(values.require("layers"));
    SinkDesc sink = parse_sink(values.require("sink"));
    validate_layer_sink_compatibility(layers, sink);
    const FrameTrigger trigger = parse_trigger(values, layers);
    std::optional<std::chrono::nanoseconds> max_input_age;
    if (const auto* field = values.find("max_input_age_ns");
        field != nullptr && field->kind != ParameterValueKind::kNone) {
      max_input_age = std::chrono::nanoseconds{
          integer_value<std::int64_t>(*field, values.field_path("max_input_age_ns"))};
      if (*max_input_age <= std::chrono::nanoseconds::zero()) {
        parameter_error("VISUALIZATION_OPERATOR_PARAMETER_VALUE_INVALID",
                        "max_input_age_ns",
                        fmt::format("value={} accepted=>0", max_input_age->count()));
      }
    }
    values.reject_unknown();
    return HolovizRegistrationThunks::add(
        graph, name, std::move(layers), std::move(sink), trigger, max_input_age);
  });
}

[[nodiscard]] authoring::RegisteredOperatorType make_row() {
  using Thunks = HolovizRegistrationThunks;
  return authoring::RegisteredOperatorType{
      .name = kRegistryName,
      .param_type = &param_type_descriptor<NoParams>(),
      .param_schema = nullptr,
      .parameter_introspection = authoring::OperatorParameterIntrospection::kPassThrough,
      .required_parameters = kRequiredParameters,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &add_from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
  };
}

[[nodiscard]] const expected<void, Error>& catalog_registration_status() {
  static const expected<void, Error> status = [] {
    return authoring::authoring_registry().register_operator(make_row());
  }();
  return status;
}

[[maybe_unused]] const bool kCatalogRegisteredAtLoad = [] {
  (void)catalog_registration_status();
  return true;
}();

}  // namespace

expected<void, Error> register_holoviz_operator_catalog() {
  const expected<void, Error>& status = catalog_registration_status();
  if (!status) {
    return make_unexpected(status.error());
  }
  return {};
}

}  // namespace holoscan::visualization
