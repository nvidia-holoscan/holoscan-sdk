// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/visualization/operator_catalog.hpp>

namespace {

using holoscan::authoring::ParameterAssignment;
using holoscan::authoring::ParameterValue;

template <typename Sink, typename Check>
bool registered_descriptor_matches(std::string_view name,
                                   std::span<const ParameterAssignment> sink_fields,
                                   Check&& check_sink) {
  const std::array view_fields{
      ParameterAssignment{"offset_x", ParameterValue::from(0.25)},
      ParameterAssignment{"offset_y", ParameterValue::from(-0.5)},
      ParameterAssignment{"scale_x", ParameterValue::from(0.75)},
      ParameterAssignment{"scale_y", ParameterValue::from(0.5)},
  };
  const std::array image_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"image"})},
      ParameterAssignment{"port_name", ParameterValue::from(std::string_view{"video"})},
      ParameterAssignment{"format", ParameterValue::from(std::string_view{"nv12"})},
      ParameterAssignment{"max_width", ParameterValue::from(std::uint64_t{1920})},
      ParameterAssignment{"max_height", ParameterValue::from(std::uint64_t{1080})},
      ParameterAssignment{"color_space", ParameterValue::from(std::string_view{"bt709_linear"})},
      ParameterAssignment{"yuv_model", ParameterValue::from(std::string_view{"bt709"})},
      ParameterAssignment{"yuv_range", ParameterValue::from(std::string_view{"limited"})},
      ParameterAssignment{"view", ParameterValue::from(std::span{view_fields})},
  };
  const std::array text_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"text"})},
      ParameterAssignment{"port_name", ParameterValue::from(std::string_view{"labels"})},
      ParameterAssignment{"max_glyphs", ParameterValue::from(std::uint64_t{256})},
      ParameterAssignment{"normalized_coordinates", ParameterValue::from(true)},
  };
  const std::array default_text_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"text"})},
      ParameterAssignment{"port_name", ParameterValue::from(std::string_view{"pixel-labels"})},
      ParameterAssignment{"max_glyphs", ParameterValue::from(std::uint64_t{64})},
  };
  const std::array trigger_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"clock"})},
      ParameterAssignment{"period_ns", ParameterValue::from(std::int64_t{20'000'000})},
  };
  const std::array layer_values{ParameterValue::from(std::span{image_fields}),
                                ParameterValue::from(std::span{text_fields}),
                                ParameterValue::from(std::span{default_text_fields})};
  const std::array assignments{
      ParameterAssignment{"layers", ParameterValue::from(std::span{layer_values})},
      ParameterAssignment{"sink", ParameterValue::from(sink_fields)},
      ParameterAssignment{"trigger", ParameterValue::from(std::span{trigger_fields})},
      ParameterAssignment{"max_input_age_ns", ParameterValue::from(std::int64_t{40'000'000})},
  };

  holoscan::Graph graph{std::string{name}};
  const auto row =
      holoscan::authoring::authoring_registry().operator_type("holoscan.visualization.HolovizOp");
  if (!row) {
    std::cerr << row.error().message << '\n';
    return false;
  }
  const auto added = row->get().add_to_graph_from_assignments(graph, "visualizer", assignments);
  if (!added) {
    std::cerr << added.error().message << '\n';
    return false;
  }

  const auto handle = graph.handle<holoscan::visualization::HolovizOp>("visualizer");
  const auto& layers = handle->layer_stack().layers();
  if (layers.size() != 3U || layers[0].port_name != "video" || layers[1].port_name != "labels" ||
      layers[2].port_name != "pixel-labels") {
    return false;
  }
  const auto* image = std::get_if<holoscan::visualization::ImageLayerDesc>(&layers[0].desc);
  const auto* text = std::get_if<holoscan::visualization::TextLayerDesc>(&layers[1].desc);
  const auto* default_text = std::get_if<holoscan::visualization::TextLayerDesc>(&layers[2].desc);
  const auto* trigger = std::get_if<holoscan::visualization::ClockTrigger>(&handle->trigger());
  if (image == nullptr || image->format != holoscan::visualization::ImageFormat::kNV12 ||
      image->max_width != 1920U || image->max_height != 1080U ||
      image->color_space != holoscan::visualization::ColorSpace::kBT709Linear ||
      image->yuv_model != holoscan::visualization::YUVModelConversion::kBT709 ||
      image->yuv_range != holoscan::visualization::YUVRange::kLimited || !image->view.has_value() ||
      image->view->offset_x != 0.25F || image->view->offset_y != -0.5F ||
      image->view->scale_x != 0.75F || image->view->scale_y != 0.5F || text == nullptr ||
      text->max_glyphs != 256U || !text->normalized_coordinates || default_text == nullptr ||
      default_text->max_glyphs != 64U || default_text->normalized_coordinates ||
      trigger == nullptr || trigger->period.count() != 20'000'000 ||
      handle->max_input_age().count() != 40'000'000) {
    return false;
  }

  const auto* sink = std::get_if<Sink>(&handle->sink());
  return sink != nullptr && std::forward<Check>(check_sink)(*sink);
}

bool registered_descriptors_are_preserved() {
  const std::array window_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"window"})},
      ParameterAssignment{"title", ParameterValue::from(std::string_view{"Registry window"})},
      ParameterAssignment{"width", ParameterValue::from(std::uint64_t{1280})},
      ParameterAssignment{"height", ParameterValue::from(std::uint64_t{720})},
      ParameterAssignment{"present_mode", ParameterValue::from(std::string_view{"fifo_relaxed"})},
      ParameterAssignment{"framebuffer_srgb", ParameterValue::from(true)},
      ParameterAssignment{"fullscreen", ParameterValue::from(true)},
      ParameterAssignment{"display_name", ParameterValue::from(std::string_view{"DP-2"})},
      ParameterAssignment{"refresh_hz", ParameterValue::from(std::uint64_t{120})},
      ParameterAssignment{"close_on_escape", ParameterValue::from(true)},
  };
  const bool window_ok = registered_descriptor_matches<holoscan::visualization::WindowSinkDesc>(
      "visualization-window-descriptor", window_fields, [](const auto& sink) {
        return sink.title == "Registry window" && sink.width == 1280U && sink.height == 720U &&
               sink.present_mode == holoscan::visualization::PresentMode::kFIFORelaxed &&
               sink.framebuffer_srgb && sink.fullscreen && sink.display_name == "DP-2" &&
               sink.refresh_hz == 120U && sink.close_on_escape;
      });

  const std::array window_default_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"window"})},
  };
  const bool window_defaults_ok =
      registered_descriptor_matches<holoscan::visualization::WindowSinkDesc>(
          "visualization-window-defaults", window_default_fields, [](const auto& sink) {
            return sink.width == 0U && sink.height == 0U;
          });
  const std::array window_width_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"window"})},
      ParameterAssignment{"width", ParameterValue::from(std::uint64_t{1920})},
  };
  const bool window_width_ok =
      registered_descriptor_matches<holoscan::visualization::WindowSinkDesc>(
          "visualization-window-width", window_width_fields, [](const auto& sink) {
            return sink.width == 1920U && sink.height == 0U;
          });
  const std::array window_height_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"window"})},
      ParameterAssignment{"height", ParameterValue::from(std::uint64_t{1080})},
  };
  const bool window_height_ok =
      registered_descriptor_matches<holoscan::visualization::WindowSinkDesc>(
          "visualization-window-height", window_height_fields, [](const auto& sink) {
            return sink.width == 0U && sink.height == 1080U;
          });

  const std::array exclusive_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"exclusive_display"})},
      ParameterAssignment{"display_name", ParameterValue::from(std::string_view{"DP-2"})},
      ParameterAssignment{"width", ParameterValue::from(std::uint64_t{3840})},
      ParameterAssignment{"height", ParameterValue::from(std::uint64_t{2160})},
      ParameterAssignment{"refresh_hz", ParameterValue::from(std::uint64_t{120})},
      ParameterAssignment{"present_mode", ParameterValue::from(std::string_view{"immediate"})},
      ParameterAssignment{"framebuffer_srgb", ParameterValue::from(true)},
  };
  const bool exclusive_ok =
      registered_descriptor_matches<holoscan::visualization::ExclusiveDisplaySinkDesc>(
          "visualization-exclusive-descriptor", exclusive_fields, [](const auto& sink) {
            return sink.display_name == "DP-2" && sink.width == 3840U && sink.height == 2160U &&
                   sink.refresh_hz == 120U &&
                   sink.present_mode == holoscan::visualization::PresentMode::kImmediate &&
                   sink.framebuffer_srgb;
          });

  const std::array exclusive_default_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"exclusive_display"})},
  };
  const bool exclusive_defaults_ok =
      registered_descriptor_matches<holoscan::visualization::ExclusiveDisplaySinkDesc>(
          "visualization-exclusive-defaults", exclusive_default_fields, [](const auto& sink) {
            return sink.width == 0U && sink.height == 0U;
          });
  const std::array exclusive_width_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"exclusive_display"})},
      ParameterAssignment{"width", ParameterValue::from(std::uint64_t{1920})},
  };
  const bool exclusive_width_ok =
      registered_descriptor_matches<holoscan::visualization::ExclusiveDisplaySinkDesc>(
          "visualization-exclusive-width", exclusive_width_fields, [](const auto& sink) {
            return sink.width == 1920U && sink.height == 0U;
          });
  const std::array exclusive_height_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"exclusive_display"})},
      ParameterAssignment{"height", ParameterValue::from(std::uint64_t{1080})},
  };
  const bool exclusive_height_ok =
      registered_descriptor_matches<holoscan::visualization::ExclusiveDisplaySinkDesc>(
          "visualization-exclusive-height", exclusive_height_fields, [](const auto& sink) {
            return sink.width == 0U && sink.height == 1080U;
          });

  const std::array cuda_fields{
      ParameterAssignment{"kind", ParameterValue::from(std::string_view{"cuda"})},
      ParameterAssignment{"width", ParameterValue::from(std::uint64_t{800})},
      ParameterAssignment{"height", ParameterValue::from(std::uint64_t{600})},
      ParameterAssignment{"format", ParameterValue::from(std::string_view{"bgra8_srgb"})},
  };
  const bool cuda_ok = registered_descriptor_matches<holoscan::visualization::CudaSinkDesc>(
      "visualization-cuda-descriptor", cuda_fields, [](const auto& sink) {
        return sink.width == 800U && sink.height == 600U &&
               sink.format == holoscan::visualization::ImageFormat::kB8G8R8A8Srgb;
      });
  return window_ok && window_defaults_ok && window_width_ok && window_height_ok && exclusive_ok &&
         exclusive_defaults_ok && exclusive_width_ok && exclusive_height_ok && cuda_ok;
}

}  // namespace

int main() {
  if (!holoscan::visualization::register_holoviz_operator_catalog()) {
    std::cerr << "failed to register HoloViz authoring catalog\n";
    return 1;
  }
  if (!registered_descriptors_are_preserved()) {
    std::cerr << "registered HoloViz descriptor translation mismatch\n";
    return 1;
  }

  holoscan::visualization::LayerStack layers;
  layers.add_image_layer(
      "video",
      ::holoscan::visualization::ImageLayerDesc{
          .format = ::holoscan::visualization::ImageFormat::kR8G8B8A8Unorm,
          .max_width = 640U,
          .max_height = 480U,
          .color_space = ::holoscan::visualization::ColorSpace::kSRGBNonlinear,
          .view = ::holoscan::visualization::ViewTransform{.scale_x = 0.5F, .scale_y = 0.5F},
      });
  const ::holoscan::visualization::WindowSinkDesc sink{
      .title = "Python parity",
      .width = 640U,
      .height = 480U,
      .present_mode = ::holoscan::visualization::PresentMode::kFIFORelaxed,
      .framebuffer_srgb = true,
  };

  holoscan::Graph graph{"visualization-python-parity"};
  const auto clock = graph.add_clock<holoscan::ManualClock>("compile-clock");
  graph.set_default_clock(clock);
  constexpr std::array<std::int64_t, 3U> max_shape{480, 640, 4};
  holoscan::TopicOptions topic_options;
  topic_options.endpoint_name = "video-input";
  topic_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
  const holoscan::TensorBoundaryContract image_data_contract{
      .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 3U,
                         .strides = holoscan::TensorStridePolicy::kContiguousOnly},
      .bounds = holoscan::tensor_bounds(max_shape),
  };
  holoscan::DescriptorTensorTopicContract<holoscan::schema::ImageT> image_contract;
  image_contract.field(holoscan::TensorFieldSelector<holoscan::schema::ImageT>{4U, 0U},
                       image_data_contract);
  const auto video = graph.import_topic<holoscan::schema::ImageT>(
      "holomq://video-input", std::move(topic_options), std::move(image_contract));
  const auto visualizer = graph.op<holoscan::visualization::HolovizOp>(
      "visualizer",
      std::move(layers),
      sink,
      holoscan::visualization::ClockTrigger{.period = std::chrono::milliseconds{20}},
      std::chrono::milliseconds{40});
  graph.add_flow(video.output(),
                 visualizer->image_layer_input(0U),
                 holoscan::ConnectionOptions{.queue_depth = 8U});

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json();
    return 1;
  }
  std::cout << plan.json();
  return 0;
}
