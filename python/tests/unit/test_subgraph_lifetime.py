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
"""  # noqa: E501

import gc
import weakref

from holoscan.core import Application, Operator, Subgraph


class LifetimeTestOp(Operator):
    """Simple test operator."""

    def setup(self, spec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        pass


class LifetimeTestSubgraph(Subgraph):
    """Test subgraph that adds an operator."""

    def compose(self):
        # Create an operator within the subgraph
        op = LifetimeTestOp(self, name="test_op")
        self.add_operator(op)

        # Create a weak reference to verify lifetime
        self.op_weakref = weakref.ref(op)


class LifetimeTestApp(Application):
    """Test application."""

    def compose(self):
        sg = LifetimeTestSubgraph(self, "test_sg")

        # Store the weakref from the subgraph
        self.op_weakref = sg.op_weakref


def test_operator_lifetime():
    """Test that operators added via subgraph.add_operator maintain Python references.

    This is expected due to virtual dispatch to PyFragment::add_operator() which stores a
    reference to the operator in the python_operator_registry_.
    """
    app = LifetimeTestApp()
    app.compose_graph()

    op_weakref = app.op_weakref
    assert op_weakref() is not None, "Operator should exist before gc"

    # Force garbage collection
    gc.collect()

    # The operator should STILL exist because it's in the registry
    # If PyFragment::add_operator was called via virtual dispatch,
    # the operator will be in python_operator_registry_ and won't be GC'd
    op_obj = op_weakref()
    assert op_obj is not None, (
        "Operator was garbage collected! "
        "This means PyFragment::add_operator was NOT called "
        "and the python_operator_registry_ does not have a reference"
    )

    # Check that the name was qualified by the subgraph
    assert op_obj.name == "test_sg_test_op", f"Expected qualified name, got {op_obj.name}"
