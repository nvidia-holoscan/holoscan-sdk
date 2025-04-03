/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>  // for unordered_map

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "entity.hpp"
#include "gxf_pydoc.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_condition.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/gxf/gxf_extension_manager.hpp"
#include "holoscan/core/gxf/gxf_extension_registrar.hpp"
#include "holoscan/core/gxf/gxf_io_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"

#include "gxf/core/gxf.h"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

// note on gxf typedefs:
//     typedef void* gxf_context_t;
//     gxf_result_t is a C-style enum where GXF_SUCCESS is 0

namespace holoscan {

void init_gxf_operator(py::module_&);
void init_gxf_network_context(py::module_&);
void init_gxf_scheduler(py::module_&);

static const gxf_tid_t default_tid = {0, 0};

PYBIND11_MODULE(_gxf, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK GXF Python Bindings
        --------------------------------
        .. currentmodule:: _gxf
    )pbdoc";

  init_entity(m);

  // TODO(unknown): `load_extensions` can be removed once Executor::extension_manager(),
  // ExtensionManager, GXFExtensionManager are exposed to Python.
  m.def(
      "load_extensions",
      [](uint64_t context,
         const std::vector<std::string>& extension_filenames,
         const std::vector<std::string>& manifest_filenames) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
        gxf::GXFExtensionManager extension_manager(reinterpret_cast<gxf_context_t>(context));
        for (const auto& extension_filename : extension_filenames) {
          extension_manager.load_extension(extension_filename);
        }
        for (const auto& manifest_filename : manifest_filenames) {
          auto node = YAML::LoadFile(manifest_filename);
          extension_manager.load_extensions_from_yaml(node);
        }
      },
      "Loads GXF extension libraries",
      "context"_a,
      "extension_filenames"_a = std::vector<std::string>{},
      "manifest_filenames"_a = std::vector<std::string>{});

  py::class_<gxf::GXFComponent, std::shared_ptr<gxf::GXFComponent>>(
      m, "GXFComponent", doc::GXFComponent::doc_GXFComponent)
      .def(py::init<>(), doc::GXFComponent::doc_GXFComponent)
      .def_property_readonly(
          "gxf_typename", &gxf::GXFComponent::gxf_typename, doc::GXFComponent::doc_gxf_typename)
      .def_property("gxf_context",
                    py::overload_cast<>(&gxf::GXFComponent::gxf_context, py::const_),
                    py::overload_cast<void*>(&gxf::GXFComponent::gxf_context),
                    doc::GXFComponent::doc_gxf_context)
      .def_property("gxf_eid",
                    py::overload_cast<>(&gxf::GXFComponent::gxf_eid, py::const_),
                    py::overload_cast<gxf_uid_t>(&gxf::GXFComponent::gxf_eid),
                    doc::GXFComponent::doc_gxf_eid)
      .def_property("gxf_cid",
                    py::overload_cast<>(&gxf::GXFComponent::gxf_cid, py::const_),
                    py::overload_cast<gxf_uid_t>(&gxf::GXFComponent::gxf_cid),
                    doc::GXFComponent::doc_gxf_cid)
      .def_property("gxf_cname",
                    py::overload_cast<>(&gxf::GXFComponent::gxf_cname),
                    py::overload_cast<const std::string&>(&gxf::GXFComponent::gxf_cname),
                    doc::GXFComponent::doc_gxf_cname)
      .def("gxf_initialize",
           &gxf::GXFComponent::gxf_initialize,
           doc::GXFComponent::doc_gxf_initialize);

  py::class_<gxf::GXFResource, Resource, gxf::GXFComponent, std::shared_ptr<gxf::GXFResource>>(
      m, "GXFResource", doc::GXFResource::doc_GXFResource)
      .def(py::init<>(), doc::GXFResource::doc_GXFResource)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto resource = obj.cast<std::shared_ptr<gxf::GXFResource>>();
            if (resource) { return resource->description(); }
            return std::string("<GXFResource: None>");
          },
          R"doc(Return repr(self).)doc");

  py::class_<gxf::GXFSystemResourceBase,
             gxf::GXFResource,
             Resource,
             gxf::GXFComponent,
             std::shared_ptr<gxf::GXFSystemResourceBase>>(
      m, "GXFSystemResourceBase", doc::GXFSystemResourceBase::doc_GXFSystemResourceBase)
      .def(py::init<>(), doc::GXFSystemResourceBase::doc_GXFSystemResourceBase)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto resource = obj.cast<std::shared_ptr<gxf::GXFSystemResourceBase>>();
            if (resource) { return resource->description(); }
            return std::string("<GXFSystemResourceBase: None>");
          },
          R"doc(Return repr(self).)doc");

  py::class_<gxf::GXFCondition, Condition, gxf::GXFComponent, std::shared_ptr<gxf::GXFCondition>>(
      m, "GXFCondition", doc::GXFCondition::doc_GXFCondition)
      .def(py::init<>(), doc::GXFCondition::doc_GXFCondition)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto condition = obj.cast<std::shared_ptr<gxf::GXFCondition>>();
            if (condition) { return condition->description(); }
            return std::string("<GXFCondition: None>");
          },
          R"doc(Return repr(self).)doc");

  init_gxf_scheduler(m);
  init_gxf_network_context(m);
  init_gxf_operator(m);

  py::class_<gxf::GXFInputContext, InputContext, std::shared_ptr<gxf::GXFInputContext>>(
      m, "GXFInputContext", R"doc(GXF input context.)doc")
      .def(py::init<ExecutionContext*, ops::GXFOperator*>(),
           "context"_a,
           "op"_a,
           doc::GXFInputContext::doc_GXFInputContext)
      .def("receive", [](const InputContext&, const std::string&) { return py::none(); }, "name"_a);

  py::class_<gxf::GXFOutputContext, OutputContext, std::shared_ptr<gxf::GXFOutputContext>>(
      m, "GXFOutputContext", R"doc(GXF output context.)doc")
      .def(py::init<ExecutionContext*, ops::GXFOperator*>(),
           "context"_a,
           "op"_a,
           doc::GXFOutputContext::doc_GXFOutputContext)
      .def(
          "emit",
          [](const OutputContext&, py::object&, const std::string&) {},
          "data"_a,
          "name"_a = "");

  py::class_<gxf::GXFExecutionContext, ExecutionContext, std::shared_ptr<gxf::GXFExecutionContext>>(
      m, "GXFExecutionContext", R"doc(GXF execution context.)doc")
      .def(py::init<gxf_context_t, ops::GXFOperator*>(),
           "context"_a,
           "op"_a,
           doc::GXFExecutionContext::doc_GXFExecutionContext);

  py::class_<gxf::GXFWrapper>(m, "GXFWrapper", doc::GXFWrapper::doc_GXFWrapper)
      .def(py::init<>(), doc::GXFWrapper::doc_GXFWrapper)
      .def("initialize", &gxf::GXFWrapper::initialize, doc::GXFWrapper::doc_initialize)
      .def("deinitialize", &gxf::GXFWrapper::deinitialize, doc::GXFWrapper::doc_deinitialize)
      .def("registerInterface",
           &gxf::GXFWrapper::registerInterface,
           doc::GXFWrapper::doc_registerInterface)
      .def("start", &gxf::GXFWrapper::start, doc::GXFWrapper::doc_start)
      .def("tick", &gxf::GXFWrapper::tick, doc::GXFWrapper::doc_tick)
      .def("stop", &gxf::GXFWrapper::stop, doc::GXFWrapper::doc_stop)
      .def("set_operator", &gxf::GXFWrapper::set_operator, doc::GXFWrapper::doc_set_operator);
}  // PYBIND11_MODULE

}  // namespace holoscan
