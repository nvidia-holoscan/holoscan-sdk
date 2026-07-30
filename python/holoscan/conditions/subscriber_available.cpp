/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <holoscan/pubsub/runtime/conditions/subscriber_available.hpp>
#include "../core/component_util.hpp"
#include "./subscriber_available_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PySubscriberAvailableCondition : public SubscriberAvailableCondition {
 public:
  using SubscriberAvailableCondition::SubscriberAvailableCondition;

  explicit PySubscriberAvailableCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
      uint64_t min_subscriber_count = 1UL,
      std::optional<const std::string> transmitter = std::nullopt,
      bool require_pubsub_connector = true, int64_t poll_period_ms = 100,
      int64_t stabilization_ms = 0, bool latch_ready = false, bool ready_on_shutdown = false,
      const std::string& name = condition_default_name_v<SubscriberAvailableCondition>)
      : SubscriberAvailableCondition(
            ArgList{Arg{"min_subscriber_count", min_subscriber_count},
                    Arg{"require_pubsub_connector", require_pubsub_connector},
                    Arg{"poll_period_ms", poll_period_ms},
                    Arg{"stabilization_ms", stabilization_ms},
                    Arg{"latch_ready", latch_ready},
                    Arg{"ready_on_shutdown", ready_on_shutdown}}) {
    if (transmitter.has_value()) {
      this->add_arg(Arg("transmitter", transmitter.value()));
    }
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_subscriber_available(py::module_& m) {
  py::class_<SubscriberAvailableCondition,
             PySubscriberAvailableCondition,
             Condition,
             std::shared_ptr<SubscriberAvailableCondition>>(
      m,
      "SubscriberAvailableCondition",
      doc::SubscriberAvailableCondition::doc_SubscriberAvailableCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    uint64_t,
                    std::optional<const std::string>,
                    bool,
                    int64_t,
                    int64_t,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "min_subscriber_count"_a = 1UL,
           "transmitter"_a = py::none(),
           "require_pubsub_connector"_a = true,
           "poll_period_ms"_a = 100,
           "stabilization_ms"_a = 0,
           "latch_ready"_a = false,
           "ready_on_shutdown"_a = false,
           "name"_a = std::string(condition_default_name_v<SubscriberAvailableCondition>),
           doc::SubscriberAvailableCondition::doc_SubscriberAvailableCondition);
}

}  // namespace holoscan
