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

#ifndef PYHOLOSCAN_CORE_FRAGMENT_HPP
#define PYHOLOSCAN_CORE_FRAGMENT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/network_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/scheduler.hpp"
#include "holoscan/core/subgraph.hpp"
#include "kwarg_handling.hpp"

namespace py = pybind11;

namespace holoscan {

void init_fragment(py::module_&);

/**********************************************************
 * Define trampolines for classes with virtual functions. *
 **********************************************************
 *
 * see:
 *https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
 *
 */

class PyFragment : public Fragment {
 public:
  /* Inherit the constructors */
  using Fragment::Fragment;

  ~PyFragment() override;

  explicit PyFragment(const py::object& op);

  /* Trampolines (need one for each virtual function) */
  void add_operator(const std::shared_ptr<Operator>& op) override;
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op) override;
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> io_map) override;
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op,
                IOSpec::ConnectorType connector_type) override;
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> io_map,
                IOSpec::ConnectorType connector_type) override;

  // Subgraph add_flow overloads
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs = {}) override;
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> port_pairs = {}) override;
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs = {}) override;

  // Subgraph add_flow connector type overloads
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                const IOSpec::ConnectorType connector_type) override;
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs,
                const IOSpec::ConnectorType connector_type) override;
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Operator>& downstream_op,
                const IOSpec::ConnectorType connector_type) override;
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> port_pairs,
                const IOSpec::ConnectorType connector_type) override;
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                const IOSpec::ConnectorType connector_type) override;
  void add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                const std::shared_ptr<Subgraph>& downstream_subgraph,
                std::set<std::pair<std::string, std::string>> port_pairs,
                const IOSpec::ConnectorType connector_type) override;

  void compose() override;
  void run() override;

  /**
   * Register all services with the given `id` from another fragment.
   * @param fragment  Source fragment. Must not be nullptr and must out-live *this* call.
   * @throws std::invalid_argument if `fragment == nullptr`.
   */
  bool register_service_from(Fragment* fragment, std::string_view id) override;

  /// Get a Python service from the registry
  py::object get_python_service(const std::string& service_id) const;

  /// Set a Python service in the registry
  void set_python_service(const std::string& service_id, py::object service);

  /// Clear a Python service from the registry
  void clear_python_service(const std::string& service_id);

 protected:
  void reset_state() override;

 private:
  py::object py_compose_ = py::none();
  /// Map from Operator raw pointer to the Python wrapper object
  std::unordered_map<Operator*, py::object> python_operator_registry_;
  /// Map from service ID to the original Python service object
  std::unordered_map<std::string, py::object> python_service_registry_;
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_FRAGMENT_HPP */
