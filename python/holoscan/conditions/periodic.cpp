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

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "./periodic_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/periodic.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

namespace {

// using constexpr constructor instead of unordered_map here to make clang-tidy happy
// (avoids warning of type: fuchsia-statically-constructed-objects)
constexpr std::array<std::pair<std::string_view, PeriodicConditionPolicy>, 3>
    PeriodicConditionPolicyMapping = {
        {{"CatchUpMissedTicks", PeriodicConditionPolicy::kCatchUpMissedTicks},
         {"MinTimeBetweenTicks", PeriodicConditionPolicy::kMinTimeBetweenTicks},
         {"NoCatchUpMissedTicks", PeriodicConditionPolicy::kNoCatchUpMissedTicks}}};

constexpr PeriodicConditionPolicy ToPeriodicConditionPolicy(std::string_view value) {
  for (const auto& [name, policy] : PeriodicConditionPolicyMapping) {
    if (name == value) {
      return policy;
    }
  }
  // Default to CatchUpMissedTicks
  return PeriodicConditionPolicy::kCatchUpMissedTicks;
}

}  // namespace

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the condition.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the condition's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_condition<ConditionT>
 */

class PyPeriodicCondition : public PeriodicCondition {
 public:
  /* Inherit the constructors */
  using PeriodicCondition::PeriodicCondition;

  // Define constructors that fully initializes the object.
  PyPeriodicCondition(Fragment* fragment, int64_t recess_period_ns,
                      const std::variant<std::string, PeriodicConditionPolicy>& policy =
                          PeriodicConditionPolicy::kCatchUpMissedTicks,
                      const std::string& name = "noname_periodic_condition")
      : PeriodicCondition(recess_period_ns) {
    // Update policy
    if (std::holds_alternative<std::string>(policy)) {
      this->add_arg(Arg("policy", ToPeriodicConditionPolicy(std::get<std::string>(policy))));
    } else {
      this->add_arg(Arg("policy", std::get<PeriodicConditionPolicy>(policy)));
    }

    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
  template <typename Rep, typename Period>
  PyPeriodicCondition(Fragment* fragment, std::chrono::duration<Rep, Period> recess_period_duration,
                      const std::variant<std::string, PeriodicConditionPolicy>& policy =
                          PeriodicConditionPolicy::kCatchUpMissedTicks,
                      const std::string& name = "noname_periodic_condition")
      : PeriodicCondition(recess_period_duration) {
    // Update policy
    if (std::holds_alternative<std::string>(policy)) {
      this->add_arg(Arg("policy", ToPeriodicConditionPolicy(std::get<std::string>(policy))));
    } else {
      this->add_arg(Arg("policy", std::get<PeriodicConditionPolicy>(policy)));
    }

    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

void init_periodic(py::module_& m) {
  py::enum_<PeriodicConditionPolicy>(m, "PeriodicConditionPolicy")
      .value("CATCH_UP_MISSED_TICKS", PeriodicConditionPolicy::kCatchUpMissedTicks)
      .value("MIN_TIME_BETWEEN_TICKS", PeriodicConditionPolicy::kMinTimeBetweenTicks)
      .value("NO_CATCH_UP_MISSED_TICKS", PeriodicConditionPolicy::kNoCatchUpMissedTicks);

  py::class_<PeriodicCondition,
             PyPeriodicCondition,
             gxf::GXFCondition,
             std::shared_ptr<PeriodicCondition>>(
      m, "PeriodicCondition", doc::PeriodicCondition::doc_PeriodicCondition)
      // TODO(unknown): sphinx API doc build complains if more than one PeriodicCondition init
      //       method has a docstring specified. For now just set the docstring for the
      //       overload using datetime.timedelta for the recess_period.
      .def(py::init<Fragment*,
                    int64_t,
                    const std::variant<std::string, PeriodicConditionPolicy>&,
                    const std::string&>(),
           "fragment"_a,
           "recess_period"_a,
           "policy"_a = PeriodicConditionPolicy::kCatchUpMissedTicks,
           "name"_a = "noname_periodic_condition"s)
      .def(py::init<Fragment*,
                    std::chrono::nanoseconds,
                    const std::variant<std::string, PeriodicConditionPolicy>&,
                    const std::string&>(),
           "fragment"_a,
           "recess_period"_a,
           "policy"_a = PeriodicConditionPolicy::kCatchUpMissedTicks,
           "name"_a = "noname_periodic_condition"s,
           doc::PeriodicCondition::doc_PeriodicCondition)
      .def("recess_period",
           static_cast<void (PeriodicCondition::*)(int64_t)>(&PeriodicCondition::recess_period),
           doc::PeriodicCondition::doc_recess_period)
      .def("recess_period",
           static_cast<void (PeriodicCondition::*)(std::chrono::nanoseconds)>(
               &PeriodicCondition::recess_period),
           doc::PeriodicCondition::doc_recess_period)
      .def_property_readonly(
          "policy", &PeriodicCondition::policy, doc::PeriodicCondition::doc_policy)
      .def("recess_period_ns",
           &PeriodicCondition::recess_period_ns,
           doc::PeriodicCondition::doc_recess_period_ns)
      .def("last_run_timestamp",
           &PeriodicCondition::last_run_timestamp,
           doc::PeriodicCondition::doc_last_run_timestamp);
}
}  // namespace holoscan
