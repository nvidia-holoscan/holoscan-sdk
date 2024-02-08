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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "./video_stream_serializer_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/video_stream_serializer.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

class PyVideoStreamSerializer : public VideoStreamSerializer {
 public:
  /* Inherit the constructors */
  using VideoStreamSerializer::VideoStreamSerializer;

  // Define a constructor that fully initializes the object.
  explicit PyVideoStreamSerializer(Fragment* fragment,
                                   const std::string& name = "video_stream_serializer")
      : VideoStreamSerializer() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
    initialize();
  }
};

void init_video_stream_serializer(py::module_& m) {
  py::class_<VideoStreamSerializer,
             PyVideoStreamSerializer,
             gxf::GXFResource,
             std::shared_ptr<VideoStreamSerializer>>(
      m, "VideoStreamSerializer", doc::VideoStreamSerializer::doc_VideoStreamSerializer)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "video_stream_serializer"s,
           doc::VideoStreamSerializer::doc_VideoStreamSerializer_python)
      .def_property_readonly("gxf_typename",
                             &VideoStreamSerializer::gxf_typename,
                             doc::VideoStreamSerializer::doc_gxf_typename)
      .def("setup", &VideoStreamSerializer::setup, "spec"_a, doc::VideoStreamSerializer::doc_setup)
      .def("initialize",
           &VideoStreamSerializer::initialize,
           doc::VideoStreamSerializer::doc_initialize);
}
}  // namespace holoscan
