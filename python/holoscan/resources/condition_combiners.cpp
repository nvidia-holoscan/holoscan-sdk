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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "../core/component_util.hpp"
#include "./condition_combiners_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/condition_combiner.hpp"
#include "holoscan/core/subgraph.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyOrConditionCombiner : public OrConditionCombiner {
 public:
  /* Inherit the constructors */
  using OrConditionCombiner::OrConditionCombiner;

  // Define a constructor that fully initializes the object.
  explicit PyOrConditionCombiner(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                                 std::vector<std::shared_ptr<holoscan::Condition>> terms = {},
                                 const std::string& name = "or_condition_combiner")
      : OrConditionCombiner(Arg{"terms", terms}) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

void init_condition_combiners(py::module_& m) {
  py::class_<OrConditionCombiner,
             PyOrConditionCombiner,
             gxf::GXFResource,
             std::shared_ptr<OrConditionCombiner>>(
      m, "OrConditionCombiner", doc::ConditionCombiners::doc_OrConditionCombiner)
      .def(py::init<std::variant<Fragment*, Subgraph*>,
                    std::vector<std::shared_ptr<holoscan::Condition>>,
                    const std::string&>(),
           "fragment"_a,
           "terms"_a,
           "name"_a = "or_condition_combiner"s,
           doc::ConditionCombiners::doc_OrConditionCombiner);
}
}  // namespace holoscan
