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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "../../core/component_util.hpp"
#include "../operator_util.hpp"
#include "./pydoc.hpp"

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/operators/video_stream_replayer/video_stream_replayer.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

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

class PyVideoStreamReplayerOp : public VideoStreamReplayerOp {
 public:
  /* Inherit the constructors */
  using VideoStreamReplayerOp::VideoStreamReplayerOp;

  // Define a constructor that fully initializes the object.
  PyVideoStreamReplayerOp(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, const py::args& args,
      const std::string& directory, const std::string& basename, size_t batch_size = 1UL,
      bool ignore_corrupted_entities = true, float frame_rate = 0.F, bool realtime = true,
      bool repeat = false, uint64_t count = 0UL,
      std::optional<std::shared_ptr<holoscan::Allocator>> allocator = std::nullopt,
      std::optional<std::shared_ptr<holoscan::Resource>> entity_serializer = std::nullopt,
      const std::string& name = "video_stream_replayer")
      : VideoStreamReplayerOp(ArgList{Arg{"directory", directory},
                                      Arg{"basename", basename},
                                      Arg{"batch_size", batch_size},
                                      Arg{"ignore_corrupted_entities", ignore_corrupted_entities},
                                      Arg{"frame_rate", frame_rate},
                                      Arg{"realtime", realtime},
                                      Arg{"repeat", repeat},
                                      Arg{"count", count}}) {
    add_positional_condition_and_resource_args(this, args);
    if (allocator.has_value()) {
      this->add_arg(Arg{"allocator", allocator.value()});
    }
    if (entity_serializer.has_value()) {
      this->add_arg(Arg{"entity_serializer", entity_serializer.value()});
    }
    init_operator_base(this, fragment_or_subgraph, name);
  }
};

/* The python module */

PYBIND11_MODULE(_video_stream_replayer, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK VideoStreamReplayerOp Python Bindings
        --------------------------------------------------
        .. currentmodule:: _video_stream_replayer
    )pbdoc";

  py::class_<VideoStreamReplayerOp,
             PyVideoStreamReplayerOp,
             Operator,
             std::shared_ptr<VideoStreamReplayerOp>>(
      m, "VideoStreamReplayerOp", doc::VideoStreamReplayerOp::doc_VideoStreamReplayerOp)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    size_t,
                    bool,
                    float,
                    bool,
                    bool,
                    uint64_t,
                    std::optional<std::shared_ptr<holoscan::Allocator>>,
                    std::optional<std::shared_ptr<holoscan::Resource>>,
                    const std::string&>(),
           "fragment"_a,
           "directory"_a,
           "basename"_a,
           "batch_size"_a = 1UL,
           "ignore_corrupted_entities"_a = true,
           "frame_rate"_a = 0.F,
           "realtime"_a = true,
           "repeat"_a = false,
           "count"_a = 0UL,
           "allocator"_a = py::none(),
           "entity_serializer"_a = py::none(),
           "name"_a = "video_stream_replayer"s,
           doc::VideoStreamReplayerOp::doc_VideoStreamReplayerOp);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
