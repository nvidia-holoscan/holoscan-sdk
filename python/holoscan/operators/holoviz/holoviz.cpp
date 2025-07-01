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

#include <pybind11/functional.h>  // for callbacks
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// the default range for enums is 128 which is not enough for the Key enum, increase to 512
#define MAGIC_ENUM_RANGE_MAX 512  // NOLINT(cppcoreguidelines-macro-usage)
#include <magic_enum.hpp>

#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include <gxf/multimedia/camera.hpp>
#include "../../core/emitter_receiver_registry.hpp"  // EmitterReceiverRegistry
#include "../../core/io_context.hpp"                 // PyOutputContext
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/codec_registry.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

// Automatically export enum values (see https://github.com/pybind/pybind11/issues/1759)
template <typename E, typename... Extra>
py::enum_<E> export_enum(const py::handle& scope, Extra&&... extra) {
  py::enum_<E> enum_type(
      scope, magic_enum::enum_type_name<E>().data(), std::forward<Extra>(extra)...);
  for (const auto& [value, name] : magic_enum::enum_entries<E>()) {
    enum_type.value(name.data(), value);
  }
  return enum_type;
}

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
      const std::string& window_title = "Holoviz"s, const std::string& display_name = ""s,
      uint32_t width = 1920, uint32_t height = 1080, float framerate = 60.F,
      bool use_exclusive_display = false, bool fullscreen = false, bool headless = false,
      bool framebuffer_srgb = false, bool vsync = false,
      ColorSpace display_color_space = ColorSpace::AUTO, bool enable_render_buffer_input = false,
      bool enable_render_buffer_output = false, bool enable_depth_buffer_input = false,
      bool enable_depth_buffer_output = false, bool enable_camera_pose_output = false,
      const std::string& camera_pose_output_type = "projection_matrix"s,
      const std::array<float, 3>& camera_eye = {0.F, 0.F, 1.F},
      const std::array<float, 3>& camera_look_at = {0.F, 0.F, 0.F},
      const std::array<float, 3>& camera_up = {0.F, 1.F, 1.F},
      // NOLINTBEGIN(performance-unnecessary-value-param)
      KeyCallbackFunction key_callback = KeyCallbackFunction(),
      UnicodeCharCallbackFunction unicode_char_callback = UnicodeCharCallbackFunction(),
      MouseButtonCallbackFunction mouse_button_callback = MouseButtonCallbackFunction(),
      ScrollCallbackFunction scroll_callback = ScrollCallbackFunction(),
      CursorPosCallbackFunction cursor_pos_callback = CursorPosCallbackFunction(),
      FramebufferSizeCallbackFunction framebuffer_size_callback = FramebufferSizeCallbackFunction(),
      WindowSizeCallbackFunction window_size_callback = WindowSizeCallbackFunction(),
      // NOLINTEND(performance-unnecessary-value-param)
      const std::string& font_path = ""s,
      std::shared_ptr<holoscan::CudaStreamPool> cuda_stream_pool = nullptr,
      std::shared_ptr<holoscan::BooleanCondition> window_close_condition = nullptr,
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
                          Arg{"display_color_space", display_color_space},
                          Arg{"enable_render_buffer_input", enable_render_buffer_input},
                          Arg{"enable_render_buffer_output", enable_render_buffer_output},
                          Arg{"enable_depth_buffer_input", enable_depth_buffer_input},
                          Arg{"enable_depth_buffer_output", enable_depth_buffer_output},
                          Arg{"enable_camera_pose_output", enable_camera_pose_output},
                          Arg{"camera_pose_output_type", camera_pose_output_type},
                          Arg{"camera_eye", camera_eye},
                          Arg{"camera_look_at", camera_look_at},
                          Arg{"camera_up", camera_up},
                          Arg{"font_path", font_path}}) {
    // only append tensors argument if it is not empty
    //     avoids [holoscan] [error] [gxf_operator.hpp:126] Unable to handle parameter 'tensors'
    if (!tensors.empty()) { this->add_arg(Arg{"tensors", tensors}); }
    if (!receivers.empty()) { this->add_arg(Arg{"receivers", receivers}); }
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    if (window_close_condition) {
      this->add_arg(Arg{"window_close_condition", window_close_condition});
    }
    // check if callbacks are provided, for each callback take the GIL before calling the function
    if (key_callback) {
      this->add_arg(
          Arg{"key_callback",
              KeyCallbackFunction(
                  [key_callback](Key key, KeyAndButtonAction action, KeyModifiers modifiers) {
                    py::gil_scoped_acquire guard;
                    key_callback(key, action, modifiers);
                  })});
    }
    if (unicode_char_callback) {
      this->add_arg(Arg{"unicode_char_callback",
                        UnicodeCharCallbackFunction([unicode_char_callback](uint32_t code_point) {
                          py::gil_scoped_acquire guard;
                          unicode_char_callback(code_point);
                        })});
    }
    if (mouse_button_callback) {
      this->add_arg(
          Arg{"mouse_button_callback",
              MouseButtonCallbackFunction([mouse_button_callback](MouseButton button,
                                                                  KeyAndButtonAction action,
                                                                  KeyModifiers modifiers) {
                py::gil_scoped_acquire guard;
                mouse_button_callback(button, action, modifiers);
              })});
    }
    if (scroll_callback) {
      this->add_arg(Arg{"scroll_callback",
                        ScrollCallbackFunction([scroll_callback](double x_offset, double y_offset) {
                          py::gil_scoped_acquire guard;
                          scroll_callback(x_offset, y_offset);
                        })});
    }
    if (cursor_pos_callback) {
      this->add_arg(
          Arg{"cursor_pos_callback",
              CursorPosCallbackFunction([cursor_pos_callback](double x_pos, double y_pos) {
                py::gil_scoped_acquire guard;
                cursor_pos_callback(x_pos, y_pos);
              })});
    }
    if (framebuffer_size_callback) {
      this->add_arg(Arg{"framebuffer_size_callback",
                        FramebufferSizeCallbackFunction([framebuffer_size_callback](int w, int h) {
                          py::gil_scoped_acquire guard;
                          framebuffer_size_callback(w, h);
                        })});
    }
    if (window_size_callback) {
      this->add_arg(Arg{"window_size_callback",
                        WindowSizeCallbackFunction([window_size_callback](int w, int h) {
                          py::gil_scoped_acquire guard;
                          window_size_callback(w, h);
                        })});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
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

  export_enum<HolovizOp::ColorSpace>(holoviz_op, "ColorSpace");

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
                          HolovizOp::ColorSpace,
                          bool,
                          bool,
                          bool,
                          bool,
                          bool,
                          const std::string&,
                          const std::array<float, 3>&,
                          const std::array<float, 3>&,
                          const std::array<float, 3>&,
                          const ops::HolovizOp::KeyCallbackFunction,
                          ops::HolovizOp::UnicodeCharCallbackFunction,
                          ops::HolovizOp::MouseButtonCallbackFunction,
                          ops::HolovizOp::ScrollCallbackFunction,
                          ops::HolovizOp::CursorPosCallbackFunction,
                          ops::HolovizOp::FramebufferSizeCallbackFunction,
                          ops::HolovizOp::WindowSizeCallbackFunction,
                          const std::string&,
                          std::shared_ptr<holoscan::CudaStreamPool>,
                          std::shared_ptr<holoscan::BooleanCondition>,
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
                 "display_color_space"_a = HolovizOp::ColorSpace::AUTO,
                 "enable_render_buffer_input"_a = false,
                 "enable_render_buffer_output"_a = false,
                 "enable_depth_buffer_input"_a = false,
                 "enable_depth_buffer_output"_a = false,
                 "enable_camera_pose_output"_a = false,
                 "camera_pose_output_type"_a = "projection_matrix"s,
                 "camera_eye"_a = std::array<float, 3>{0.F, 0.F, 1.F},
                 "camera_look_at"_a = std::array<float, 3>{0.F, 0.F, 0.F},
                 "camera_up"_a = std::array<float, 3>{0.F, 1.F, 1.F},
                 "key_callback"_a = HolovizOp::KeyCallbackFunction(),
                 "unicode_char_callback"_a = HolovizOp::UnicodeCharCallbackFunction(),
                 "mouse_button_callback"_a = HolovizOp::MouseButtonCallbackFunction(),
                 "scroll_callback"_a = HolovizOp::ScrollCallbackFunction(),
                 "cursor_pos_callback"_a = HolovizOp::CursorPosCallbackFunction(),
                 "framebuffer_size_callback"_a = HolovizOp::FramebufferSizeCallbackFunction(),
                 "window_size_callback"_a = HolovizOp::WindowSizeCallbackFunction(),
                 "font_path"_a = ""s,
                 "cuda_stream_pool"_a = py::none(),
                 "window_close_condition"_a = py::none(),
                 "name"_a = "holoviz_op"s,
                 doc::HolovizOp::doc_HolovizOp);

  export_enum<HolovizOp::InputType>(holoviz_op, "InputType");
  export_enum<HolovizOp::ImageFormat>(holoviz_op, "ImageFormat");
  export_enum<HolovizOp::YuvModelConversion>(holoviz_op, "YuvModelConversion");
  export_enum<HolovizOp::YuvRange>(holoviz_op, "YuvRange");
  export_enum<HolovizOp::ChromaLocation>(holoviz_op, "ChromaLocation");
  export_enum<HolovizOp::DepthMapRenderMode>(holoviz_op, "DepthMapRenderMode");
  export_enum<HolovizOp::Key>(holoviz_op, "Key");
  export_enum<HolovizOp::KeyAndButtonAction>(holoviz_op, "KeyAndButtonAction");
  export_enum<HolovizOp::MouseButton>(holoviz_op, "MouseButton");

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

  py::class_<HolovizOp::KeyModifiers>(
      holoviz_op, "KeyModifiers" /*, doc::HolovizOp::InputSpec::doc_KeyModifiers*/)
      .def_property_readonly("shift",
                             [](const HolovizOp::KeyModifiers& self) -> bool { return self.shift; })
      .def_property_readonly(
          "control", [](const HolovizOp::KeyModifiers& self) -> bool { return self.control; })
      .def_property_readonly("alt",
                             [](const HolovizOp::KeyModifiers& self) -> bool { return self.alt; })
      .def_property_readonly(
          "caps_lock", [](const HolovizOp::KeyModifiers& self) -> bool { return self.caps_lock; })
      .def_property_readonly(
          "num_lock", [](const HolovizOp::KeyModifiers& self) -> bool { return self.num_lock; })
      .def("__repr__", [](const HolovizOp::KeyModifiers& modifiers) {
        return fmt::format(
            "KeyModifiers(shift: {}, control: {}, alt: {}, caps_lock {}, num_lock: {})",
            modifiers.shift,
            modifiers.control,
            modifiers.alt,
            modifiers.caps_lock,
            modifiers.num_lock);
      });

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
  gxf::CodecRegistry::get_instance().add_codec<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
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
