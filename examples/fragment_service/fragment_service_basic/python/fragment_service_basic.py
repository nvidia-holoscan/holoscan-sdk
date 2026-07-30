"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, DefaultFragmentService, Operator, OperatorSpec


class MyService(DefaultFragmentService):
    """A simple fragment service that holds an integer value."""

    def __init__(self, value):
        super().__init__()
        self._value = value

    def value(self):
        """Get the value stored in the service."""
        return self._value


class MyOp(Operator):
    """An operator that retrieves and uses a fragment service."""

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("MyOp.compute() executed")

        # Retrieve the service from the fragment
        my_service = self.service(MyService)
        print(f"MyService value: {my_service.value()}")


class FragmentServiceApp(Application):
    """Application demonstrating fragment service usage."""

    def compose(self):
        # Create a service instance with a value of 10
        my_service = MyService(10)

        # Register the service with the fragment
        self.register_service(my_service)

        # Create an operator with a CountCondition to run once
        my_op = MyOp(self, CountCondition(self, 1), name="my_op")

        # Add the operator to the application
        self.add_operator(my_op)


def main():
    app = FragmentServiceApp()
    app.run()


if __name__ == "__main__":
    main()
