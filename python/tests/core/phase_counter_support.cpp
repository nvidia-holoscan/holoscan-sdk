/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Source-tree access to Runtime phase counters for Python tests and benchmarks, matching the C++
// benchmarks' private RuntimeTestPlanAccess seam. Nothing here ships in the Python package.

#include "phase_counter_support.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <nanobind/nanobind.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>

#include "../../../src/runtime/runtime_phase_counters.hpp"
#include "../../../src/runtime/runtime_test_access.hpp"

namespace holoscan::python_test_support {
namespace {

namespace nb = nanobind;

// Read one plan's counters after or during its run.
class PhaseCounters {
 public:
  explicit PhaseCounters(std::shared_ptr<detail::RuntimePhaseCounters> counters)
      : counters_(std::move(counters)) {}

  // Return the reported session counters by name.
  [[nodiscard]] nb::dict snapshot() const {
    const auto values = counters_->snapshot();
    nb::dict result;
    for (const auto& [name, counter] : detail::kRuntimePhaseCounterFields) {
      result[nb::str(name.data(), name.size())] =
          values[detail::RuntimePhaseCounters::index(counter)];
    }
    return result;
  }

  // Return one record per partition owner that ran: its index, owner thread id, and counters.
  [[nodiscard]] nb::list partitions() const {
    const auto values = counters_->partition_snapshot();
    const auto owners = counters_->partition_owner_threads();
    nb::list result;
    for (std::size_t partition = 0U; partition < owners.size(); ++partition) {
      if (owners[partition] == 0U) {
        continue;
      }
      nb::dict record;
      record["index"] = partition;
      record["owner_tid"] = owners[partition];
      for (const auto& [name, counter] : detail::kRuntimePartitionCounterFields) {
        record[nb::str(name.data(), name.size())] =
            values[partition][detail::RuntimePhaseCounters::index(counter)];
      }
      result.append(std::move(record));
    }
    return result;
  }

 private:
  std::shared_ptr<detail::RuntimePhaseCounters> counters_;
};

}  // namespace

void register_phase_counter_support(nb::module_& module) {
  nb::class_<PhaseCounters>(module, "_PhaseCounters")
      .def("snapshot", &PhaseCounters::snapshot)
      .def("partitions", &PhaseCounters::partitions);
  module.def("_with_phase_counters", [](const nb::object& plan) {
    auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan counted = detail::RuntimeTestPlanAccess::with_phase_counters(
        nb::cast<const ExecutionPlan&>(plan), counters);
    if (!counted.ok()) {
      throw RuntimeError(ErrorCode::kInvalidArgument, "phase counters need an accepted plan");
    }
    // A compiled plan keeps its Python callback owners in its instance dictionary; the copy needs
    // them too, or a Python Operator could outlive its callbacks.
    nb::object counted_plan = nb::cast(std::move(counted));
    counted_plan.attr("__dict__").attr("update")(plan.attr("__dict__"));
    return nb::make_tuple(std::move(counted_plan), PhaseCounters{std::move(counters)});
  });
}

}  // namespace holoscan::python_test_support
