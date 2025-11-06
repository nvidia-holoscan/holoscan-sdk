/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include "dataflow_tracker_pydoc.hpp"
#include "holoscan/core/dataflow_tracker.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_data_flow_tracker(py::module_& m) {
  py::enum_<DataFlowMetric>(m, "DataFlowMetric", doc::DataFlowMetric::doc_DataFlowMetric)
      .value("MAX_MESSAGE_ID", DataFlowMetric::kMaxMessageID)
      .value("MIN_MESSAGE_ID", DataFlowMetric::kMinMessageID)
      .value("MAX_E2E_LATENCY", DataFlowMetric::kMaxE2ELatency)
      .value("AVG_E2E_LATENCY", DataFlowMetric::kAvgE2ELatency)
      .value("MIN_E2E_LATENCY", DataFlowMetric::kMinE2ELatency)
      .value("NUM_SRC_MESSAGES", DataFlowMetric::kNumSrcMessages)
      .value("NUM_DST_MESSAGES", DataFlowMetric::kNumDstMessages);

  py::class_<DataFlowTracker>(m, "DataFlowTracker", doc::DataFlowTracker::doc_DataFlowTracker)
      .def(py::init<>(), doc::DataFlowTracker::doc_DataFlowTracker)
      .def("enable_logging",
           &DataFlowTracker::enable_logging,
           "filename"_a = kDefaultLogfileName,
           "num_buffered_messages"_a = kDefaultNumBufferedMessages,
           doc::DataFlowTracker::doc_enable_logging)
      .def("end_logging", &DataFlowTracker::end_logging, doc::DataFlowTracker::doc_end_logging)
      // TODO(unknown): sphinx API doc build complains if more than one overloaded get_metric method
      // has a docstring specified. For now using the docstring defined for 2-argument
      // version and describe the single argument variant in the Notes section.
      .def("get_metric",
           py::overload_cast<std::string, DataFlowMetric>(&DataFlowTracker::get_metric),
           "pathstring"_a,
           "metric"_a,
           doc::DataFlowTracker::doc_get_metric_with_pathstring)
      .def("get_metric",
           py::overload_cast<DataFlowMetric>(&DataFlowTracker::get_metric),
           "metric"_a = DataFlowMetric::kNumSrcMessages)
      .def(
          "get_num_paths", &DataFlowTracker::get_num_paths, doc::DataFlowTracker::doc_get_num_paths)
      .def("get_path_strings",
           &DataFlowTracker::get_path_strings,
           doc::DataFlowTracker::doc_get_path_strings)
      .def("print", &DataFlowTracker::print, doc::DataFlowTracker::doc_print)
      .def("set_discard_last_messages",
           &DataFlowTracker::set_discard_last_messages,
           doc::DataFlowTracker::doc_set_discard_last_messages)
      .def("set_skip_latencies",
           &DataFlowTracker::set_skip_latencies,
           doc::DataFlowTracker::doc_set_skip_latencies)
      .def("set_skip_starting_messages",
           &DataFlowTracker::set_skip_starting_messages,
           doc::DataFlowTracker::doc_set_skip_starting_messages);
}

}  // namespace holoscan
