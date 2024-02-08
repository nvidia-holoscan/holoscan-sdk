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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "./pydoc.hpp"

#include "holoscan/core/codec_registry.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

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
      Fragment* fragment, std::shared_ptr<::holoscan::Allocator> allocator,
      std::vector<holoscan::IOSpec*> receivers = std::vector<holoscan::IOSpec*>(),
      const std::vector<HolovizOp::InputSpec>& tensors = std::vector<HolovizOp::InputSpec>(),
      const std::vector<std::vector<float>>& color_lut = std::vector<std::vector<float>>(),
      const std::string& window_title = "Holoviz", const std::string& display_name = "DP-0",
      uint32_t width = 1920, uint32_t height = 1080, float framerate = 60.f,
      bool use_exclusive_display = false, bool fullscreen = false, bool headless = false,
      bool enable_render_buffer_input = false, bool enable_render_buffer_output = false,
      bool enable_camera_pose_output = false, const std::string& font_path = "",
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
                          Arg{"enable_render_buffer_input", enable_render_buffer_input},
                          Arg{"enable_render_buffer_output", enable_render_buffer_output},
                          Arg{"enable_camera_pose_output", enable_camera_pose_output},
                          Arg{"font_path", font_path}}) {
    // only append tensors argument if it is not empty
    //     avoids [holoscan] [error] [gxf_operator.hpp:126] Unable to handle parameter 'tensors'
    if (tensors.size() > 0) { this->add_arg(Arg{"tensors", tensors}); }
    if (receivers.size() > 0) { this->add_arg(Arg{"receivers", receivers}); }
    if (cuda_stream_pool) { this->add_arg(Arg{"cuda_stream_pool", cuda_stream_pool}); }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_holoviz, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _holoviz
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<HolovizOp, PyHolovizOp, Operator, std::shared_ptr<HolovizOp>> holoviz_op(
      m, "HolovizOp", doc::HolovizOp::doc_HolovizOp);
  holoviz_op
      .def(py::init<Fragment*,
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
                    const std::string&,
                    std::shared_ptr<holoscan::CudaStreamPool>,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "receivers"_a = std::vector<holoscan::IOSpec*>(),
           "tensors"_a = std::vector<HolovizOp::InputSpec>(),
           "color_lut"_a = std::vector<std::vector<float>>(),
           "window_title"_a = "Holoviz",
           "display_name"_a = "DP-0",
           "width"_a = 1920,
           "height"_a = 1080,
           "framerate"_a = 60,
           "use_exclusive_display"_a = false,
           "fullscreen"_a = false,
           "headless"_a = false,
           "enable_render_buffer_input"_a = false,
           "enable_render_buffer_output"_a = false,
           "enable_camera_pose_output"_a = false,
           "font_path"_a = "",
           "cuda_stream_pool"_a = py::none(),
           "name"_a = "holoviz_op"s,
           doc::HolovizOp::doc_HolovizOp_python)
      .def("initialize", &HolovizOp::initialize, doc::HolovizOp::doc_initialize)
      .def("setup", &HolovizOp::setup, "spec"_a, doc::HolovizOp::doc_setup);

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
      .def_readwrite("line_width", &HolovizOp::InputSpec::line_width_)
      .def_readwrite("point_size", &HolovizOp::InputSpec::point_size_)
      .def_readwrite("text", &HolovizOp::InputSpec::text_)
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

  // Register the std::vector<InputSpec> codec when the Python module is imported.
  // This is useful for, e.g. testing serialization with pytest without having to first create a
  // HolovizOp operator (which registers the type in its initialize method).
  CodecRegistry::get_instance().add_codec<std::vector<holoscan::ops::HolovizOp::InputSpec>>(
      "std::vector<std::vector<holoscan::ops::HolovizOp::InputSpec>>", true);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
