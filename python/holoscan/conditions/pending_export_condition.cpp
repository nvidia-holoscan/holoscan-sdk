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
#include <string>
#include <variant>

#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/condition.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/network_context.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp>
#include <holoscan/pubsub/runtime/conditions/pending_export_condition.hpp>
#include "../core/component_util.hpp"
#include "./pending_export_condition_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyPendingExportCondition : public PendingExportCondition {
 public:
  using PendingExportCondition::PendingExportCondition;

  explicit PyPendingExportCondition(
      const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, uint64_t max_pending = 1UL,
      std::shared_ptr<NetworkContext> network_context = nullptr,
      const std::string& name = condition_default_name_v<PendingExportCondition>)
      : PendingExportCondition(ArgList{Arg{"max_pending", max_pending}}) {
    if (network_context) {
      adapter_resolver([network_context]() -> std::shared_ptr<NativeBufferProtocolAdapter> {
        if (auto dds_ctx =
                std::dynamic_pointer_cast<FastDdsPubSubNetworkContext>(network_context)) {
          return dds_ctx->native_buffer_adapter();
        }
        return std::shared_ptr<NativeBufferProtocolAdapter>{};
      });
    }
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_pending_export(py::module_& m) {
  py::class_<PendingExportCondition,
             PyPendingExportCondition,
             Condition,
             std::shared_ptr<PendingExportCondition>>(
      m, "PendingExportCondition", doc::PendingExportCondition::doc_PendingExportCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    uint64_t,
                    std::shared_ptr<NetworkContext>,
                    const std::string&>(),
           "fragment"_a,
           "max_pending"_a = 1UL,
           "network_context"_a = nullptr,
           "name"_a = std::string(condition_default_name_v<PendingExportCondition>),
           doc::PendingExportCondition::doc_PendingExportCondition);
}

}  // namespace holoscan
