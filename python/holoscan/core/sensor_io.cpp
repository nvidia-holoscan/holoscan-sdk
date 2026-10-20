/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sensor_io.hpp"

#include <holoscan/core/errors.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sequence_tracker.hpp>

#include "module_name.hpp"
#include "native_io.hpp"

namespace holoscan {

void init_sensor_io(nb::module_& module) {
  auto sequence_observation = python::bind_class_in_module<sensor_io::SequenceObservation>(
      module,
      "holoscan.sensor_io",
      "SequenceObservation",
      "What one observed device sequence number implies about missing samples.");
  sequence_observation.def_ro("sequence", &sensor_io::SequenceObservation::sequence)
      .def_ro("gap", &sensor_io::SequenceObservation::gap)
      .def_ro("first", &sensor_io::SequenceObservation::first)
      .def_ro("regressed", &sensor_io::SequenceObservation::regressed);

  auto sequence_tracker = python::bind_class_in_module<sensor_io::SequenceTracker>(
      module,
      "holoscan.sensor_io",
      "SequenceTracker",
      "Track a device sequence counter and report gaps and regressions.");
  sequence_tracker.def(nb::init<>())
      .def("observe", &sensor_io::SequenceTracker::observe, nb::arg("sequence"))
      .def_prop_ro("total_gap", &sensor_io::SequenceTracker::total_gap)
      .def_prop_ro("previous", &sensor_io::SequenceTracker::previous)
      .def_prop_ro("seen", &sensor_io::SequenceTracker::seen)
      .def("reset", &sensor_io::SequenceTracker::reset);

  auto capture_time_policy = python::bind_enum_in_module<sensor_io::CaptureTimePolicy>(
      module,
      "holoscan.sensor_io",
      "CaptureTimePolicy",
      "Choose which contributor capture time a combined result inherits.");
  capture_time_policy.value("EARLIEST", sensor_io::CaptureTimePolicy::kEarliest)
      .value("LATEST", sensor_io::CaptureTimePolicy::kLatest);

  auto provenance_accumulator = python::bind_class_in_module<sensor_io::ProvenanceAccumulator>(
      module,
      "holoscan.sensor_io",
      "ProvenanceAccumulator",
      "Combine metadata from one output group's input samples into options for a fresh "
      "Output.emit(). Retained instances keep every observation until reset() is called.");
  provenance_accumulator.def(nb::init<>())
      .def(
          "observe",
          [](sensor_io::ProvenanceAccumulator& accumulator, const SampleMetadata& metadata) {
            accumulator.observe(metadata);
          },
          nb::arg("metadata"))
      .def(
          "emit_options",
          [](const sensor_io::ProvenanceAccumulator& accumulator,
             sensor_io::CaptureTimePolicy policy) {
            auto result = accumulator.emit_options(policy);
            if (!result) {
              throw RuntimeError{result.error()};
            }
            return python_authoritative_emit_options_from_native(*result);
          },
          nb::arg("policy"),
          "Return authoritative provenance for a fused result published with Output.emit().\n\n"
          "This method does not reset the accumulator. Call reset() after publishing one output "
          "group and before observing contributors for the next group. Passing the returned "
          "options to Output.forward() raises InvalidArgumentError because forwarding is a "
          "one-to-one operation whose ordinary options inherit absent fields from its source.")
      .def_prop_ro("observed", &sensor_io::ProvenanceAccumulator::observed)
      .def_prop_ro("mixed_domains", &sensor_io::ProvenanceAccumulator::mixed_domains)
      .def_prop_ro("has_unstamped_contributor",
                   &sensor_io::ProvenanceAccumulator::has_unstamped_contributor)
      .def("reset",
           &sensor_io::ProvenanceAccumulator::reset,
           "Discard every contributor observed for the current output group.");
}

}  // namespace holoscan
