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

#ifndef PYHOLOSCAN_CORE_FRAGMENT_SERVICE_PYDOC_HPP
#define PYHOLOSCAN_CORE_FRAGMENT_SERVICE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace FragmentService {

PYDOC(FragmentService, R"doc(
Base interface for services that enable sharing of resources and functionality between
operators.
)doc")

PYDOC(FragmentService_default, R"doc(
Construct a new FragmentService object.
)doc")

PYDOC(resource, R"doc(
The resource associated with the service.
)doc")

PYDOC(ServiceDriverEndpoint, R"doc(
Interface for a distributed service endpoint on the driver side.

This class defines the methods that are called on the driver (main) process
when managing a distributed service.
)doc")

PYDOC(ServiceDriverEndpoint_default, R"doc(
Construct a new ServiceDriverEndpoint object.
)doc")

PYDOC(driver_start, R"doc(
Start the driver-side service.

Parameters
----------
driver_ip : str
    The IP address of the driver.
)doc")

PYDOC(driver_shutdown, R"doc(
Shut down the driver-side service.
)doc")

PYDOC(ServiceWorkerEndpoint, R"doc(
Interface for a distributed service endpoint on the worker side.

This class defines the methods that are called on a worker process when
managing a distributed service.
)doc")

PYDOC(ServiceWorkerEndpoint_default, R"doc(
Construct a new ServiceWorkerEndpoint object.
)doc")

PYDOC(worker_connect, R"doc(
Connect the worker-side service to the driver.

Parameters
----------
driver_ip : str
    The IP address of the driver to connect to.
)doc")

PYDOC(worker_disconnect, R"doc(
Disconnect the worker-side service from the driver.
)doc")

}  // namespace FragmentService

namespace DefaultFragmentService {

PYDOC(DefaultFragmentService, R"doc(
Base class for services to be registered in the fragment service registry.

Fragment services provide a way to share resources and functionality across
operators within a fragment or application.
)doc")

PYDOC(DefaultFragmentService_default, R"doc(
Construct a new DefaultFragmentService object.
)doc")

PYDOC(DefaultFragmentService_resource, R"doc(
Construct a new DefaultFragmentService object.

Parameters
----------
resource : holoscan.core.Resource
    The resource to associate with the service.
)doc")

}  // namespace DefaultFragmentService

namespace DistributedAppService {

PYDOC(DistributedAppService, R"doc(
Composite service class that implements FragmentService and distributed service endpoints.

This class combines the functionality of FragmentService, ServiceDriverEndpoint,
and ServiceWorkerEndpoint to provide a single service that can be used as both
a driver and worker endpoint in distributed environments.
)doc")

PYDOC(DistributedAppService_default, R"doc(
Construct a new DistributedAppService object.
)doc")

PYDOC(driver_start, R"doc(
Start the driver-side service.
)doc")

PYDOC(driver_shutdown, R"doc(
Shut down the driver-side service.
)doc")

PYDOC(worker_connect, R"doc(
Connect the worker-side service to the driver.
)doc")

PYDOC(worker_disconnect, R"doc(
Disconnect the worker-side service from the driver.
)doc")

}  // namespace DistributedAppService

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_FRAGMENT_SERVICE_PYDOC_HPP */
