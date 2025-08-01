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

#include "fragment_service.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <string_view>

#include "fragment_service_pydoc.hpp"
#include "resource.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

std::shared_ptr<Resource> PyFragmentService::resource() const {
  PYBIND11_OVERRIDE_PURE(std::shared_ptr<Resource>, FragmentService, resource);
}

void PyFragmentService::resource(const std::shared_ptr<Resource>& resource) {
  PYBIND11_OVERRIDE_PURE(void, FragmentService, resource, resource);
}

std::shared_ptr<Resource> PyDefaultFragmentService::resource() const {
  // Don't use PYBIND11_OVERRIDE here because resource is exposed as a property
  // The property getter will handle Python overrides
  return DefaultFragmentService::resource();
}

void PyDefaultFragmentService::resource(const std::shared_ptr<Resource>& resource) {
  // Don't use PYBIND11_OVERRIDE here because resource is exposed as a property
  // The property setter will handle Python overrides
  DefaultFragmentService::resource(resource);
}

namespace distributed {
void PyServiceDriverEndpoint::driver_start(std::string_view driver_ip) {
  PYBIND11_OVERRIDE_PURE(void, ServiceDriverEndpoint, driver_start, driver_ip);
}

void PyServiceDriverEndpoint::driver_shutdown() {
  PYBIND11_OVERRIDE_PURE(void, ServiceDriverEndpoint, driver_shutdown);
}

void PyServiceWorkerEndpoint::worker_connect(std::string_view driver_ip) {
  PYBIND11_OVERRIDE_PURE(void, ServiceWorkerEndpoint, worker_connect, driver_ip);
}

void PyServiceWorkerEndpoint::worker_disconnect() {
  PYBIND11_OVERRIDE_PURE(void, ServiceWorkerEndpoint, worker_disconnect);
}

}  // namespace distributed

std::shared_ptr<Resource> PyDistributedAppService::resource() const {
  PYBIND11_OVERRIDE_PURE(std::shared_ptr<Resource>, DistributedAppService, resource);
}

void PyDistributedAppService::resource(const std::shared_ptr<Resource>& resource) {
  PYBIND11_OVERRIDE_PURE(void, DistributedAppService, resource, resource);
}

void PyDistributedAppService::driver_start(std::string_view driver_ip) {
  PYBIND11_OVERRIDE_PURE(void, DistributedAppService, driver_start, driver_ip);
}

void PyDistributedAppService::driver_shutdown() {
  PYBIND11_OVERRIDE_PURE(void, DistributedAppService, driver_shutdown);
}

void PyDistributedAppService::worker_connect(std::string_view driver_ip) {
  PYBIND11_OVERRIDE_PURE(void, DistributedAppService, worker_connect, driver_ip);
}

void PyDistributedAppService::worker_disconnect() {
  PYBIND11_OVERRIDE_PURE(void, DistributedAppService, worker_disconnect);
}

void init_fragment_service(py::module_& m) {
  py::class_<FragmentService, PyFragmentService, std::shared_ptr<FragmentService>>(
      m, "FragmentService", doc::FragmentService::doc_FragmentService)
      .def(py::init<>(), doc::FragmentService::doc_FragmentService_default)
      .def_property("resource",
                    py::overload_cast<>(&FragmentService::resource, py::const_),
                    py::overload_cast<const std::shared_ptr<Resource>&>(&FragmentService::resource),
                    doc::FragmentService::doc_resource);

  py::class_<DefaultFragmentService,
             PyDefaultFragmentService,
             FragmentService,
             std::shared_ptr<DefaultFragmentService>>(
      m, "DefaultFragmentService", doc::DefaultFragmentService::doc_DefaultFragmentService)
      .def(py::init<>(), doc::DefaultFragmentService::doc_DefaultFragmentService_default)
      .def(py::init<const std::shared_ptr<Resource>&>(),
           "resource"_a,
           doc::DefaultFragmentService::doc_DefaultFragmentService_resource);

  py::class_<distributed::ServiceDriverEndpoint,
             distributed::PyServiceDriverEndpoint,
             std::shared_ptr<distributed::ServiceDriverEndpoint>>(
      m, "ServiceDriverEndpoint", doc::FragmentService::doc_ServiceDriverEndpoint)
      .def(py::init<>(), doc::FragmentService::doc_ServiceDriverEndpoint_default)
      .def("driver_start",
           &distributed::ServiceDriverEndpoint::driver_start,
           "driver_ip"_a,
           doc::FragmentService::doc_driver_start)
      .def("driver_shutdown",
           &distributed::ServiceDriverEndpoint::driver_shutdown,
           doc::FragmentService::doc_driver_shutdown);

  py::class_<distributed::ServiceWorkerEndpoint,
             distributed::PyServiceWorkerEndpoint,
             std::shared_ptr<distributed::ServiceWorkerEndpoint>>(
      m, "ServiceWorkerEndpoint", doc::FragmentService::doc_ServiceWorkerEndpoint)
      .def(py::init<>(), doc::FragmentService::doc_ServiceWorkerEndpoint_default)
      .def("worker_connect",
           &distributed::ServiceWorkerEndpoint::worker_connect,
           "driver_ip"_a,
           doc::FragmentService::doc_worker_connect)
      .def("worker_disconnect",
           &distributed::ServiceWorkerEndpoint::worker_disconnect,
           doc::FragmentService::doc_worker_disconnect);

  py::class_<DistributedAppService,
             PyDistributedAppService,
             FragmentService,
             distributed::ServiceDriverEndpoint,
             distributed::ServiceWorkerEndpoint,
             std::shared_ptr<DistributedAppService>>(
      m, "DistributedAppService", doc::DistributedAppService::doc_DistributedAppService)
      .def(py::init<>(), doc::DistributedAppService::doc_DistributedAppService_default)
      .def("driver_start",
           &DistributedAppService::driver_start,
           "driver_ip"_a,
           doc::DistributedAppService::doc_driver_start)
      .def("driver_shutdown",
           &DistributedAppService::driver_shutdown,
           doc::DistributedAppService::doc_driver_shutdown)
      .def("worker_connect",
           &DistributedAppService::worker_connect,
           "driver_ip"_a,
           doc::DistributedAppService::doc_worker_connect)
      .def("worker_disconnect",
           &DistributedAppService::worker_disconnect,
           doc::DistributedAppService::doc_worker_disconnect);
}

}  // namespace holoscan
