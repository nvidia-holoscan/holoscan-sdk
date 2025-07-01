"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import weakref

from holoscan.conditions import CountCondition
from holoscan.core import (
    Application,
    DefaultFragmentService,
    FragmentService,
    Operator,
    OperatorSpec,
    Resource,
)


class MyServiceResource(Resource):
    """A resource that can be used as a fragment service."""

    def __init__(self, fragment, *args, int_value=0, **kwargs):
        self.int_value = int_value

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        # In Python, we don't use spec.param for simple attributes
        # Instead, we just store the value directly
        pass

    def value(self):
        """Get the integer value stored in the resource."""
        return self.int_value


class MyResourceManagerSelfRef(Resource, FragmentService):
    """A resource that also implements FragmentService interface with self-reference."""

    def __init__(self, fragment, *args, int_value=0, **kwargs):
        self.int_value = int_value

        Resource.__init__(self, fragment, *args, **kwargs)
        FragmentService.__init__(self)
        self._resource_ref = weakref.ref(self)

    def resource(self, new_resource=None):
        """Overrides the C++ resource property."""
        if new_resource is not None:
            # This is the setter path, called from C++
            self._resource_ref = weakref.ref(new_resource)
            # Also call the C++ base class's resource setter to keep it in sync
            super().resource(new_resource)
        # The getter path
        return self._resource_ref()

    def value(self):
        """Get the integer value stored in the resource."""
        return self.int_value


class MyResourceManagerEnabledShared(Resource, FragmentService):
    """A resource that also implements FragmentService interface."""

    def __init__(self, fragment, *args, int_value=0, **kwargs):
        self.int_value = int_value
        Resource.__init__(self, fragment, *args, **kwargs)
        FragmentService.__init__(self)

    def resource(self, _new_resource=None):
        """Overrides the C++ resource property. The setter is a no-op."""
        # The setter is called from C++ during registration, but we don't need to
        # store the new_resource since this object is its own resource.
        return self

    def value(self):
        """Get the integer value stored in the resource."""
        return self.int_value


class MyOp(Operator):
    """An operator that retrieves and uses fragment services."""

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("MyOp.compute() executed")

        # 1,2) Fragment service resource can be retrieved via the base DefaultFragmentService type
        my_service = self.service(DefaultFragmentService, "my_service")
        if my_service:
            resource = my_service.resource
            if hasattr(resource, "value"):
                print(f"MyService value (via FragmentService): {resource.value()}")
            else:
                print("ERROR: Resource doesn't have value() method")
        else:
            print("ERROR: Could not retrieve service via FragmentService")

        # 1,2) Fragment service resource can also be retrieved directly via the resource type
        # Note: When retrieving a resource by its specific type, use the resource name as ID
        my_service_resource = self.service(MyServiceResource, "my_service")
        if my_service_resource:
            print(f"MyService value (via MyServiceResource): {my_service_resource.value()}")
        else:
            print("ERROR: Could not retrieve service via MyServiceResource")

        # 3) Fragment service resource can be retrieved via the resource type inheriting both
        #    Resource and FragmentService interfaces (self-reference)
        my_service_resource_selfref = self.service(
            MyResourceManagerSelfRef, "my_service_resource_selfref"
        )
        if my_service_resource_selfref:
            print(
                f"MyResourceManager value (via MyResourceManagerSelfRef): "
                f"{my_service_resource_selfref.value()}"
            )
        else:
            print("ERROR: Could not retrieve service via MyResourceManagerSelfRef")

        # 4) Fragment service resource can be retrieved via the resource type inheriting both
        #    Resource and FragmentService interfaces
        my_service_resource_enabled_shared = self.service(
            MyResourceManagerEnabledShared, "my_resource_manager_enabled_shared"
        )
        if my_service_resource_enabled_shared:
            print(
                f"MyResourceManager value (via MyResourceManagerEnabledShared): "
                f"{my_service_resource_enabled_shared.value()}"
            )
        else:
            print("ERROR: Could not retrieve service via MyResourceManagerEnabledShared")


class FragmentServiceApp(Application):
    """Application demonstrating fragment service usage with resources."""

    def compose(self):
        # WARNING: The Fragment Service API is currently experimental in Holoscan SDK.
        # Future SDK releases may introduce breaking changes to this API.

        # 1,2) Create a service resource with an integer value of 20
        my_service_resource = MyServiceResource(self, name="my_service", int_value=20)

        # # 1) Fragment service can be registered with a FragmentService instance (created with
        # #    a Resource object)
        # my_service = DefaultFragmentService(my_service_resource)
        # self.register_service(my_service)  # ID automatically handled for resources

        # 2) Fragment service can also be registered directly with a Resource object
        # Note: For resources, the ID parameter is ignored and the resource's name is used
        self.register_service(my_service_resource)  # id parameter defaults to "" and can be omitted

        # 3) Create and register a fragment service that inherits from both Resource and
        #    FragmentService.
        my_service_resource_selfref = MyResourceManagerSelfRef(
            self, name="my_service_resource_selfref", int_value=30
        )
        self.register_service(my_service_resource_selfref)

        # 4) Create and register a fragment service that inherits from both Resource and
        #    FragmentService.
        my_resource_manager_enabled_shared = MyResourceManagerEnabledShared(
            self, name="my_resource_manager_enabled_shared", int_value=40
        )
        self.register_service(my_resource_manager_enabled_shared)

        # Create an operator with a CountCondition to run once
        my_op = MyOp(self, CountCondition(self, 1), name="my_op")

        # Add the operator to the application
        self.add_operator(my_op)


def main():
    app = FragmentServiceApp()
    app.run()


if __name__ == "__main__":
    main()
