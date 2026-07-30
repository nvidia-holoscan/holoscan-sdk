"""
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, ComponentSpec, Operator, Resource
from holoscan.resources import CudaStreamPool


class NativeResource(Resource):
    def __init__(self, fragment, msg="test", *args, **kwargs):
        self.msg = msg
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: ComponentSpec):
        print("** native resource setup method called **")

    def get_message(self):
        return self.msg


class MinimalOp(Operator):
    def __init__(self, *args, expected_message="test", **kwargs):
        self.expected_message = expected_message
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def compute(self, op_input, op_output, context):
        print("** MinimalOp compute method called **")
        resource = self.resource("msg_resource")
        assert isinstance(resource, NativeResource)

        # can call a custom method implemented for the resource
        msg = resource.get_message()
        assert msg == self.expected_message

        # test case when no resource with the specified name exists
        nonexistent_resource = self.resource("nonexistent")
        assert nonexistent_resource is None

        # test retrieving all resources
        resources = self.resources
        assert isinstance(resources, dict)
        # Two resources are expected: the native resource and the default CudaStreamPool
        assert len(resources) == 2
        assert isinstance(resources["msg_resource"], NativeResource)
        assert isinstance(resources[f"{self.name}_stream_pool"], CudaStreamPool)


class MinimalNativeResourceApp(Application):
    def compose(self):
        msg = "native resource message"
        native_resource = NativeResource(self, msg=msg, name="msg_resource")
        mx = MinimalOp(
            self,
            CountCondition(self, 1),
            native_resource,
            expected_message=msg,
            name="mx",
        )
        self.add_operator(mx)


def main():
    app = MinimalNativeResourceApp()
    app.run()


if __name__ == "__main__":
    main()
