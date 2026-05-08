/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <variant>

#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/condition.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/pubsub/runtime/conditions/publisher_available.hpp>
#include "../core/component_util.hpp"
#include "./publisher_available_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyPublisherAvailableCondition : public PublisherAvailableCondition {
 public:
  using PublisherAvailableCondition::PublisherAvailableCondition;

  explicit PyPublisherAvailableCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      uint64_t min_publisher_count = 1UL, std::optional<const std::string> receiver = std::nullopt,
      bool require_pubsub_connector = true, int64_t poll_period_ms = 100, bool latch_ready = false,
      const std::string& name = condition_default_name_v<PublisherAvailableCondition>)
      : PublisherAvailableCondition(
            ArgList{Arg{"min_publisher_count", min_publisher_count},
                    Arg{"require_pubsub_connector", require_pubsub_connector},
                    Arg{"poll_period_ms", poll_period_ms},
                    Arg{"latch_ready", latch_ready}}) {
    if (receiver.has_value()) {
      this->add_arg(Arg("receiver", receiver.value()));
    }
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_publisher_available(py::module_& m) {
  py::class_<PublisherAvailableCondition,
             PyPublisherAvailableCondition,
             Condition,
             std::shared_ptr<PublisherAvailableCondition>>(
      m,
      "PublisherAvailableCondition",
      doc::PublisherAvailableCondition::doc_PublisherAvailableCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    uint64_t,
                    std::optional<const std::string>,
                    bool,
                    int64_t,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "min_publisher_count"_a = 1UL,
           "receiver"_a = py::none(),
           "require_pubsub_connector"_a = true,
           "poll_period_ms"_a = 100,
           "latch_ready"_a = false,
           "name"_a = std::string(condition_default_name_v<PublisherAvailableCondition>),
           doc::PublisherAvailableCondition::doc_PublisherAvailableCondition);
}

}  // namespace holoscan
