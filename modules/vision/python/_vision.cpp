// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <holoscan/vision/resize_op.hpp>
#include <holoscan/vision/vision_types.hpp>

namespace nb = nanobind;

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
NB_MODULE(_vision, module) {
  using holoscan::vision::InterpolationMode;
  using holoscan::vision::ResizeOpParams;

  module.doc() = "Minimal Vision value bindings and native linkage validation.";

  nb::enum_<InterpolationMode>(module, "InterpolationMode")
      .value("NEAREST", InterpolationMode::kNearest)
      .value("LINEAR", InterpolationMode::kLinear)
      .value("CUBIC", InterpolationMode::kCubic)
      .value("AREA", InterpolationMode::kArea);

  nb::class_<ResizeOpParams>(module, "ResizeOpParams")
      .def(nb::init<>())
      .def_rw("width", &ResizeOpParams::width)
      .def_rw("height", &ResizeOpParams::height)
      .def_rw("max_batch_size", &ResizeOpParams::max_batch_size)
      .def_rw("interpolation", &ResizeOpParams::interpolation);

  // Keep the real Vision DSO in the dependency closure. Value bindings alone are header-only.
  // Construction does not start a session or create the CV-CUDA resize resource.
  module.def("_check_native_linkage", [] { const holoscan::vision::ResizeOp operation; });
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
