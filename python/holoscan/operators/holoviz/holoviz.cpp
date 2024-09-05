/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include <gxf/multimedia/camera.hpp>
#include "../../core/emitter_receiver_registry.hpp"  // EmitterReceiverRegistry
#include "../../core/io_context.hpp"                 // PyOutputContext
#include "holoscan/core/codec_registry.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

// default emitter_receiver templates can handle std::vector<holoscan::ops::HolovizOp::InputSpec>
// default emitter_receiver templates can handle std::shared_ptr<std::array<float, 16>>
// default emitter_receiver templates can handle std::shared_ptr<nvidia::gxf::Pose3D>

}  // namespace holoscan

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyHolovizOp : public HolovizOp {
 public:
  /* Inherit the constructors */
  using HolovizOp::HolovizOp;

  // Define a constructor that fully initializes the object.
  PyHolovizOp(
      Fragment* fragment, const py::args& args, std::shared_ptr<::holoscan::Allocator> allocator,
      std::vector<holoscan::IOSpec*> receivers = std::vector<holoscan::IOSpec*>(),
      const std::vector<HolovizOp::InputSpec>& tensors = std::vector<HolovizOp::InputSpec>(),
      const std::vector<std::vector<float>>& color_lut = std::vector<std::vector<float>>(),
      const std::string& window_title = "Holoviz", const std::string& display_name = "",
      uint32_t width = 1920, uint32_t height = 1080, float framerate = 60.f,
      bool use_exclusive_display = false, bool fullscreen = false, bool headless = false,
      bool framebuffer_srgb = false, bool vsync = false, bool enable_render_buffer_input = false,
      bool enable_render_buffer_output = false, bool enable_camera_pose_output = false,
      const std::string& camera_pose_output_type = "projection_matrix",
      const std::array<float, 3>& camera_eye = {0.f, 0.f, 1.f},
      const std::array<float, 3>& camera_look_at = {0.f, 0.f, 0.f},
      const std::array<float, 3>& camera_up = {0.f, 1.f, 1.f}, const std::string& font_path = "",
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      const std::string& name = "holoviz_op")
      : HolovizOp(ArgList{Arg{"allocator", allocator},
                          Arg{"color_lut", color_lut},
                          Arg{"window_title", window_title},
                          Arg{"display_name", display_name},
                          Arg{"width", width},
                          Arg{"height", height},
                          Arg{"framerate", framerate},
                          Arg{"use_exclusive_display", use_exclusive_display},
                          Arg{"fullscreen", fullscreen},
                          Arg{"headless", headless},
                          Arg{"framebuffer_srgb", framebuffer_srgb},
                          Arg{"vsync", vsync},
                          Arg{"enable_render_buffer_input", enable_render_buffer_input},
                          Arg{"enable_render_buffer_output", enable_render_buffer_output},
                          Arg{"enable_camera_pose_output", enable_camera_pose_output},
                          Arg{"camera_pose_output_type", camera_pose_output_type},
                          Arg{"camera_eye", camera_eye},
                          Arg{"camera_look_at", camera_look_at},
                          Arg{"camera_up", camera_up},
                          Arg{"font_path", font_path}}) {
    // only append tensors argument if it is not empty
    //     avoids [holoscan] [error] [gxf_operator.hpp:126] Unable to handle parameter 'tensors'
    if (tensors.size() > 0) { this->add_arg(Arg{"tensors", tensors}); }
    if (receivers.size() > 0) { this->add_arg(Arg{"receivers", receivers}); }
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_holoviz, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK HolovizOp Python Bindings
        --------------------------------------
        .. currentmodule:: _holoviz
    )pbdoc";

  py::class_<HolovizOp, PyHolovizOp, Operator, std::shared_ptr<HolovizOp>> holoviz_op(
      m, "HolovizOp", doc::HolovizOp::doc_HolovizOp);
  holoviz_op.def(py::init<Fragment*,
                          const py::args&,
                          std::shared_ptr<::holoscan::Allocator>,
                          std::vector<holoscan::IOSpec*>,
                          const std::vector<HolovizOp::InputSpec>&,
                          const std::vector<std::vector<float>>&,
                          const std::string&,
                          const std::string&,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          bool,
                          bool,
                          bool,
                          bool,
                          bool,
                          bool,
                          bool,
                          bool,
                          const std::string&,
                          const std::array<float, 3>&,
                          const std::array<float, 3>&,
                          const std::array<float, 3>&,
                          const std::string&,
                          std::shared_ptr<holoscan::CudaStreamPool>,
                          const std::string&>(),
                 "fragment"_a,
                 "allocator"_a,
                 "receivers"_a = std::vector<holoscan::IOSpec*>(),
                 "tensors"_a = std::vector<HolovizOp::InputSpec>(),
                 "color_lut"_a = std::vector<std::vector<float>>(),
                 "window_title"_a = "Holoviz",
                 "display_name"_a = "",
                 "width"_a = 1920,
                 "height"_a = 1080,
                 "framerate"_a = 60,
                 "use_exclusive_display"_a = false,
                 "fullscreen"_a = false,
                 "headless"_a = false,
                 "framebuffer_srgb"_a = false,
                 "vsync"_a = false,
                 "enable_render_buffer_input"_a = false,
                 "enable_render_buffer_output"_a = false,
                 "enable_camera_pose_output"_a = false,
                 "camera_pose_output_type"_a = "projection_matrix",
                 "camera_eye"_a = std::array<float, 3>{0.f, 0.f, 1.f},
                 "camera_look_at"_a = std::array<float, 3>{0.f, 0.f, 0.f},
                 "camera_up"_a = std::array<float, 3>{0.f, 1.f, 1.f},
                 "font_path"_a = "",
                 "cuda_stream_pool"_a = py::none(),
                 "name"_a = "holoviz_op"s,
                 doc::HolovizOp::doc_HolovizOp);

  py::enum_<HolovizOp::InputType>(holoviz_op, "InputType")
      .value("UNKNOWN", HolovizOp::InputType::UNKNOWN)
      .value("COLOR", HolovizOp::InputType::COLOR)
      .value("COLOR_LUT", HolovizOp::InputType::COLOR_LUT)
      .value("POINTS", HolovizOp::InputType::POINTS)
      .value("LINES", HolovizOp::InputType::LINES)
      .value("LINE_STRIP", HolovizOp::InputType::LINE_STRIP)
      .value("TRIANGLES", HolovizOp::InputType::TRIANGLES)
      .value("CROSSES", HolovizOp::InputType::CROSSES)
      .value("RECTANGLES", HolovizOp::InputType::RECTANGLES)
      .value("OVALS", HolovizOp::InputType::OVALS)
      .value("TEXT", HolovizOp::InputType::TEXT)
      .value("DEPTH_MAP", HolovizOp::InputType::DEPTH_MAP)
      .value("DEPTH_MAP_COLOR", HolovizOp::InputType::DEPTH_MAP_COLOR)
      .value("POINTS_3D", HolovizOp::InputType::POINTS_3D)
      .value("LINES_3D", HolovizOp::InputType::LINES_3D)
      .value("LINE_STRIP_3D", HolovizOp::InputType::LINE_STRIP_3D)
      .value("TRIANGLES_3D", HolovizOp::InputType::TRIANGLES_3D);

  py::enum_<HolovizOp::ImageFormat>(holoviz_op, "ImageFormat")
      .value("AUTO_DETECT", HolovizOp::ImageFormat::AUTO_DETECT)
      .value("R8_UINT", HolovizOp::ImageFormat::R8_UINT)
      .value("R8_SINT", HolovizOp::ImageFormat::R8_SINT)
      .value("R8_UNORM", HolovizOp::ImageFormat::R8_UNORM)
      .value("R8_SNORM", HolovizOp::ImageFormat::R8_SNORM)
      .value("R8_SRGB", HolovizOp::ImageFormat::R8_SRGB)
      .value("R16_UINT", HolovizOp::ImageFormat::R16_UINT)
      .value("R16_SINT", HolovizOp::ImageFormat::R16_SINT)
      .value("R16_UNORM", HolovizOp::ImageFormat::R16_UNORM)
      .value("R16_SNORM", HolovizOp::ImageFormat::R16_SNORM)
      .value("R16_SFLOAT", HolovizOp::ImageFormat::R16_SFLOAT)
      .value("R32_UINT", HolovizOp::ImageFormat::R32_UINT)
      .value("R32_SINT", HolovizOp::ImageFormat::R32_SINT)
      .value("R32_SFLOAT", HolovizOp::ImageFormat::R32_SFLOAT)
      .value("R8G8B8_UNORM", HolovizOp::ImageFormat::R8G8B8_UNORM)
      .value("R8G8B8_SNORM", HolovizOp::ImageFormat::R8G8B8_SNORM)
      .value("R8G8B8_SRGB", HolovizOp::ImageFormat::R8G8B8_SRGB)
      .value("R8G8B8A8_UNORM", HolovizOp::ImageFormat::R8G8B8A8_UNORM)
      .value("R8G8B8A8_SNORM", HolovizOp::ImageFormat::R8G8B8A8_SNORM)
      .value("R8G8B8A8_SRGB", HolovizOp::ImageFormat::R8G8B8A8_SRGB)
      .value("R16G16B16A16_UNORM", HolovizOp::ImageFormat::R16G16B16A16_UNORM)
      .value("R16G16B16A16_SNORM", HolovizOp::ImageFormat::R16G16B16A16_SNORM)
      .value("R16G16B16A16_SFLOAT", HolovizOp::ImageFormat::R16G16B16A16_SFLOAT)
      .value("R32G32B32A32_SFLOAT", HolovizOp::ImageFormat::R32G32B32A32_SFLOAT)
      .value("D16_UNORM", HolovizOp::ImageFormat::D16_UNORM)
      .value("X8_D24_UNORM", HolovizOp::ImageFormat::X8_D24_UNORM)
      .value("D32_SFLOAT", HolovizOp::ImageFormat::D32_SFLOAT)
      .value("A2B10G10R10_UNORM_PACK32", HolovizOp::ImageFormat::A2B10G10R10_UNORM_PACK32)
      .value("A2R10G10B10_UNORM_PACK32", HolovizOp::ImageFormat::A2R10G10B10_UNORM_PACK32)
      .value("B8G8R8A8_UNORM", HolovizOp::ImageFormat::B8G8R8A8_UNORM)
      .value("B8G8R8A8_SRGB", HolovizOp::ImageFormat::B8G8R8A8_SRGB)
      .value("A8B8G8R8_UNORM_PACK32", HolovizOp::ImageFormat::A8B8G8R8_UNORM_PACK32)
      .value("A8B8G8R8_SRGB_PACK32", HolovizOp::ImageFormat::A8B8G8R8_SRGB_PACK32)
      .value("Y8U8Y8V8_422_UNORM", HolovizOp::ImageFormat::Y8U8Y8V8_422_UNORM)
      .value("U8Y8V8Y8_422_UNORM", HolovizOp::ImageFormat::U8Y8V8Y8_422_UNORM)
      .value("Y8_U8V8_2PLANE_420_UNORM", HolovizOp::ImageFormat::Y8_U8V8_2PLANE_420_UNORM)
      .value("Y8_U8V8_2PLANE_422_UNORM", HolovizOp::ImageFormat::Y8_U8V8_2PLANE_422_UNORM)
      .value("Y8_U8_V8_3PLANE_420_UNORM", HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_420_UNORM)
      .value("Y8_U8_V8_3PLANE_422_UNORM", HolovizOp::ImageFormat::Y8_U8_V8_3PLANE_422_UNORM)
      .value("Y16_U16V16_2PLANE_420_UNORM", HolovizOp::ImageFormat::Y16_U16V16_2PLANE_420_UNORM)
      .value("Y16_U16V16_2PLANE_422_UNORM", HolovizOp::ImageFormat::Y16_U16V16_2PLANE_422_UNORM)
      .value("Y16_U16_V16_3PLANE_420_UNORM", HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_420_UNORM)
      .value("Y16_U16_V16_3PLANE_422_UNORM", HolovizOp::ImageFormat::Y16_U16_V16_3PLANE_422_UNORM);

  py::enum_<HolovizOp::YuvModelConversion>(holoviz_op, "YuvModelConversion")
      .value("YUV_601", HolovizOp::YuvModelConversion::YUV_601)
      .value("YUV_709", HolovizOp::YuvModelConversion::YUV_709)
      .value("YUV_2020", HolovizOp::YuvModelConversion::YUV_2020);

  py::enum_<HolovizOp::YuvRange>(holoviz_op, "YuvRange")
      .value("ITU_FULL", HolovizOp::YuvRange::ITU_FULL)
      .value("ITU_NARROW", HolovizOp::YuvRange::ITU_NARROW);

  py::enum_<HolovizOp::ChromaLocation>(holoviz_op, "ChromaLocation")
      .value("COSITED_EVEN", HolovizOp::ChromaLocation::COSITED_EVEN)
      .value("MIDPOINT", HolovizOp::ChromaLocation::MIDPOINT);

  py::enum_<HolovizOp::DepthMapRenderMode>(holoviz_op, "DepthMapRenderMode")
      .value("POINTS", HolovizOp::DepthMapRenderMode::POINTS)
      .value("LINES", HolovizOp::DepthMapRenderMode::LINES)
      .value("TRIANGLES", HolovizOp::DepthMapRenderMode::TRIANGLES);

  py::class_<HolovizOp::InputSpec> inputspec(
      holoviz_op, "InputSpec", doc::HolovizOp::InputSpec::doc_InputSpec);
  inputspec
      .def(py::init<const std::string&, HolovizOp::InputType>(),
           doc::HolovizOp::InputSpec::doc_InputSpec)
      .def(py::init<const std::string&, const std::string&>())
      .def_readwrite("type", &HolovizOp::InputSpec::type_)
      .def_readwrite("color", &HolovizOp::InputSpec::color_)
      .def_readwrite("opacity", &HolovizOp::InputSpec::opacity_)
      .def_readwrite("priority", &HolovizOp::InputSpec::priority_)
      .def_readwrite("image_format", &HolovizOp::InputSpec::image_format_)
      .def_readwrite("line_width", &HolovizOp::InputSpec::line_width_)
      .def_readwrite("point_size", &HolovizOp::InputSpec::point_size_)
      .def_readwrite("text", &HolovizOp::InputSpec::text_)
      .def_readwrite("yuv_model_conversion", &HolovizOp::InputSpec::yuv_model_conversion_)
      .def_readwrite("yuv_range", &HolovizOp::InputSpec::yuv_range_)
      .def_readwrite("x_chroma_location", &HolovizOp::InputSpec::x_chroma_location_)
      .def_readwrite("y_chroma_location", &HolovizOp::InputSpec::y_chroma_location_)
      .def_readwrite("depth_map_render_mode", &HolovizOp::InputSpec::depth_map_render_mode_)
      .def_readwrite("views", &HolovizOp::InputSpec::views_)
      .def("description",
           &HolovizOp::InputSpec::description,
           doc::HolovizOp::InputSpec::doc_InputSpec_description);

  py::class_<HolovizOp::InputSpec::View> view(
      inputspec, "View", doc::HolovizOp::InputSpec::doc_View);
  view.def(py::init<>(), doc::HolovizOp::InputSpec::doc_View)
      .def_readwrite("offset_x", &HolovizOp::InputSpec::View::offset_x_)
      .def_readwrite("offset_y", &HolovizOp::InputSpec::View::offset_y_)
      .def_readwrite("width", &HolovizOp::InputSpec::View::width_)
      .def_readwrite("height", &HolovizOp::InputSpec::View::height_)
      .def_readwrite("matrix", &HolovizOp::InputSpec::View::matrix_);

  // need to wrap nvidia::gxf::Pose3D for use of received Pose3D object from Python
  py::class_<nvidia::gxf::Pose3D> Pose3D(m, "Pose3D", doc::HolovizOp::Pose3D::doc_Pose3D);
  Pose3D.def_readwrite("rotation", &nvidia::gxf::Pose3D::rotation)
      .def_readwrite("translation", &nvidia::gxf::Pose3D::translation)
      .def("__repr__", [](const nvidia::gxf::Pose3D& pose) {
        return fmt::format(
            "Pose3D(rotation: {}, translation: {})", pose.rotation, pose.translation);
      });

  // Register the std::vector<InputSpec> codec when the Python module is imported.
  // This is useful for, e.g. testing serialization with pytest without having to first create a
  // HolovizOp operator (which registers the type in its initialize method).
  CodecRegistry::get_instance().add_codec<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
      "std::vector<std::vector<holoscan::ops::HolovizOp::InputSpec>>", true);

  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
        "std::vector<HolovizOp::InputSpec>"s);
    // array camera pose object
    registry.add_emitter_receiver<std::shared_ptr<std::array<float, 16>>>(
        "std::shared_ptr<std::array<float, 16>>"s);
    // Pose3D camera pose object
    registry.add_emitter_receiver<std::shared_ptr<nvidia::gxf::Pose3D>>(
        "std::shared_ptr<nvidia::gxf::Pose3D>"s);
    // camera_eye_input, camera_look_at_input, camera_up_input
    registry.add_emitter_receiver<std::array<float, 3>>("std::array<float, 3>"s);
  });
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
