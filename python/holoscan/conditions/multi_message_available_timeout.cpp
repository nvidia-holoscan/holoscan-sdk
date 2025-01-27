/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "./multi_message_available_timeout_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/multi_message_available_timeout.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)
namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the condition.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the condition's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_condition<ConditionT>
 */

class PyMultiMessageAvailableTimeoutCondition : public MultiMessageAvailableTimeoutCondition {
 public:
  /* Inherit the constructors */
  using MultiMessageAvailableTimeoutCondition::MultiMessageAvailableTimeoutCondition;

  // Define a constructor that fully initializes the object.
  explicit PyMultiMessageAvailableTimeoutCondition(
      Fragment* fragment, const std::string& execution_frequency,
      std::variant<MultiMessageAvailableTimeoutCondition::SamplingMode, std::string> sampling_mode =
          MultiMessageAvailableTimeoutCondition::SamplingMode::kSumOfAll,
      std::optional<size_t> min_sum = std::nullopt,
      std::optional<std::vector<size_t>> min_sizes = std::nullopt,
      const std::string& name = "multi_message_timeout_condition")
      : MultiMessageAvailableTimeoutCondition(Arg("execution_frequency", execution_frequency)) {
    name_ = name;
    fragment_ = fragment;
    if (min_sum.has_value()) { this->add_arg(Arg("min_sum", min_sum.value())); }
    if (min_sizes.has_value()) { this->add_arg(Arg("min_sizes", min_sizes.value())); }

    // need to pass mode via a YAML::Node. Can take either a string or enum from Python
    if (std::holds_alternative<std::string>(sampling_mode)) {
      this->add_arg(Arg("sampling_mode", YAML::Node(std::get<std::string>(sampling_mode))));
    } else {
      auto mode_value =
          std::get<MultiMessageAvailableTimeoutCondition::SamplingMode>(sampling_mode);
      this->add_arg(Arg("sampling_mode", mode_value));
    }
    // Note "receivers" parameter is set automatically from GXFExecutor
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

void init_multi_message_available_timeout(py::module_& m) {
  py::class_<MultiMessageAvailableTimeoutCondition,
             PyMultiMessageAvailableTimeoutCondition,
             gxf::GXFCondition,
             std::shared_ptr<MultiMessageAvailableTimeoutCondition>>(
      m,
      "MultiMessageAvailableTimeoutCondition",
      doc::MultiMessageAvailableTimeoutCondition::doc_MultiMessageAvailableTimeoutCondition)
      .def(py::init<Fragment*,
                    const std::string&,
                    std::variant<MultiMessageAvailableTimeoutCondition::SamplingMode, std::string>,
                    std::optional<size_t>,
                    std::optional<std::vector<size_t>>,
                    const std::string&>(),
           "fragment"_a,
           "execution_frequency"_a,
           "sampling_mode"_a = MultiMessageAvailableTimeoutCondition::SamplingMode::kSumOfAll,
           "min_sum"_a = py::none(),
           "min_sizes"_a = py::none(),
           "name"_a = "multi_message_timeout_condition"s,
           doc::MultiMessageAvailableTimeoutCondition::doc_MultiMessageAvailableTimeoutCondition)
      .def_property("receivers",
                    py::overload_cast<>(&MultiMessageAvailableTimeoutCondition::receivers),
                    py::overload_cast<std::vector<std::shared_ptr<Receiver>>>(
                        &MultiMessageAvailableTimeoutCondition::receivers),
                    doc::MultiMessageAvailableTimeoutCondition::doc_receivers);
}
}  // namespace holoscan
