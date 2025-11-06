/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_COMPONENT_UTIL_HPP
#define PYHOLOSCAN_CORE_COMPONENT_UTIL_HPP

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/subgraph.hpp"

namespace holoscan {

/**
 * @brief Utility function to resolve fragment pointer and qualified name from a variant.
 *
 * This utility is used by Python binding classes (Conditions, Resources, Operators) to support
 * constructors that can accept either a Fragment* or Subgraph* as the first argument.
 *
 * When a Fragment* is provided:
 * - Returns the fragment pointer as-is
 * - Returns the name without modification
 *
 * When a Subgraph* is provided:
 * - Extracts and returns the fragment pointer from the Subgraph
 * - Returns a qualified name using the Subgraph's instance name
 *
 * @param fragment_or_subgraph Variant containing either a Fragment* or Subgraph*
 * @param name The base name for the component
 * @param component_type The type of component (e.g., "condition", "resource", "operator")
 * @return A pair containing the Fragment pointer and the (possibly qualified) name
 */
inline std::pair<Fragment*, std::string> get_fragment_ptr_name_pair(
    const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, const std::string& name,
    const std::string& component_type) {
  if (std::holds_alternative<Fragment*>(fragment_or_subgraph)) {
    // Direct Fragment path: use as-is
    auto fragment_ptr = std::get<Fragment*>(fragment_or_subgraph);
    if (fragment_ptr == nullptr) {
      throw std::runtime_error("Fragment pointer cannot be null for " + component_type);
    }
    return {fragment_ptr, name};
  } else {
    // Subgraph path: extract fragment and create qualified name
    auto subgraph_ptr = std::get<Subgraph*>(fragment_or_subgraph);
    if (subgraph_ptr == nullptr) {
      throw std::runtime_error("Subgraph pointer cannot be null for " + component_type);
    }
    auto fragment_ptr = subgraph_ptr->fragment();
    if (fragment_ptr == nullptr) {
      throw std::runtime_error("Fragment pointer from Subgraph cannot be null for " +
                               component_type);
    }
    return {fragment_ptr, subgraph_ptr->get_qualified_name(name, component_type)};
  }
}

/**
 * @brief Initialize a component (Condition or Resource) with ComponentSpec.
 *
 * This helper function encapsulates the common five-step initialization pattern used across
 * Python bindings for Conditions and Resources:
 * 1. Extract fragment pointer and qualified name
 * 2. Set fragment via public setter
 * 3. Set name via public setter
 * 4. Create and set spec via public setter (Component now has public spec() methods)
 * 5. Call setup()
 *
 * @tparam ComponentT The component type (e.g., PyCountCondition, PyBlockMemoryPool)
 * @param component Pointer to the component being initialized (typically 'this')
 * @param fragment_or_subgraph Variant containing either a Fragment* or Subgraph*
 * @param name The base name for the component
 * @param component_type The type string (e.g., "condition", "resource")
 */
template <typename ComponentT>
inline void init_component_base(ComponentT* component,
                                const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                                const std::string& name, const std::string& component_type) {
  if (component == nullptr) {
    throw std::runtime_error("component pointer cannot be null");
  }
  auto [frag_ptr, qualified_name] =
      get_fragment_ptr_name_pair(fragment_or_subgraph, name, component_type);
  component->fragment(frag_ptr);
  component->name(qualified_name);
  auto spec = std::make_shared<ComponentSpec>(frag_ptr);
  component->spec(spec);
  component->setup(*spec);
}

/**
 * @brief Initialize a component with ComponentSpec (simplified Fragment* version).
 *
 * This is a simpler variant for components that take Fragment* directly rather than
 * std::variant<Fragment*, Subgraph*>. Used by NetworkContext, DataLogger, and Scheduler
 * Python bindings.
 *
 * This helper function encapsulates the common four-step initialization pattern:
 * 1. Set fragment via public setter
 * 2. Set name via public setter
 * 3. Create and set spec via public setter
 * 4. Call setup()
 *
 * @tparam ComponentT The component type (e.g., PyUcxContext, PyAsyncConsoleLogger)
 * @param component Pointer to the component being initialized (typically 'this')
 * @param fragment Pointer to the fragment
 * @param name The name for the component
 */
template <typename ComponentT>
inline void init_component_base(ComponentT* component, Fragment* fragment,
                                const std::string& name) {
  if (component == nullptr) {
    throw std::runtime_error("component pointer cannot be null");
  }
  if (fragment == nullptr) {
    throw std::runtime_error("fragment pointer cannot be null");
  }
  component->fragment(fragment);
  component->name(name);
  auto spec = std::make_shared<ComponentSpec>(fragment);
  component->spec(spec);
  component->setup(*spec);
}

/**
 * @brief Initialize an operator with OperatorSpec.
 *
 * This helper function encapsulates the common five-step initialization pattern used across
 * Python bindings for Operators:
 * 1. Extract fragment pointer and qualified name
 * 2. Set fragment via public setter
 * 3. Set name via public setter
 * 4. Create and set spec via public setter
 * 5. Call setup()
 *
 * @tparam OperatorT The operator type (e.g., PyPingTensorRxOp)
 * @param op Pointer to the operator being initialized (typically 'this')
 * @param fragment_or_subgraph Variant containing either a Fragment* or Subgraph*
 * @param name The base name for the operator
 */
template <typename OperatorT>
inline void init_operator_base(OperatorT* op,
                               const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                               const std::string& name) {
  if (op == nullptr) {
    throw std::runtime_error("component pointer cannot be null");
  }
  auto [frag_ptr, qualified_name] =
      get_fragment_ptr_name_pair(fragment_or_subgraph, name, "operator");
  op->fragment(frag_ptr);
  op->name(qualified_name);
  auto spec = std::make_shared<OperatorSpec>(frag_ptr);
  op->spec(spec);
  op->setup(*spec);
}

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_COMPONENT_UTIL_HPP */
