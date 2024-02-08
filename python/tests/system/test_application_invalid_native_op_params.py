"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec


class InvalidOp(Operator):
    def __init__(self, *args, protected_property="fragment", **kwargs):
        self.count = 1
        self.param_value = None

        # Set a property that matches a base class property or method.
        # This will lead to an AttributeError for read-only properties
        # or a TypeError for other properties.
        # (Note this will segfault if the value assigned is None)
        if protected_property:
            setattr(self, protected_property, 5)

        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("param_value", 500)

    def compute(self, op_input, op_output, context):
        self.count += 1


@pytest.mark.parametrize(
    "protected_property, exception_type",
    [
        # __setattr__ override will raise AttributeError on read-only properties
        ("fragment", AttributeError),
        ("conditions", AttributeError),
        ("resources", AttributeError),
        ("operator_type", AttributeError),
        ("description", AttributeError),
        # other properties/methods called during init will raise TypeError
        ("name", TypeError),
        ("spec", TypeError),
        ("setup", TypeError),
    ],
)
def test_invalid_operator_property_assignment(fragment, protected_property, exception_type):
    with pytest.raises(exception_type):
        InvalidOp(fragment, protected_property=protected_property)


class InvalidMinimalApp(Application):
    def __init__(self, *args, protected_property="fragment", **kwargs):
        self.protected_property = protected_property
        super().__init__(*args, **kwargs)

    def compose(self):
        mx = InvalidOp(
            self, CountCondition(self, 1), protected_property=self.protected_property, name="mx"
        )
        self.add_operator(mx)


@pytest.mark.parametrize(
    "protected_property, exception_type",
    [
        ("fragment", AttributeError),
        ("conditions", AttributeError),
        ("resources", AttributeError),
        ("operator_type", AttributeError),
        ("description", AttributeError),
        ("name", TypeError),
        ("spec", TypeError),
        ("setup", TypeError),
        # Setting any of the following methods would still cause a segfault
        # ("initialize", TypeError),
        # ("start", TypeError),
        # ("compute", TypeError),
        # ("stop", TypeError),
    ],
)
def test_invalid_operator_property_assignment_app(protected_property, exception_type):
    app = InvalidMinimalApp(protected_property=protected_property)
    with pytest.raises(exception_type):
        app.run()
