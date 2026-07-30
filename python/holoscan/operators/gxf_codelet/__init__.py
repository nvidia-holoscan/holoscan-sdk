"""
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import holoscan.core  # noqa: F401
import holoscan.gxf  # noqa: F401
from holoscan.core import OperatorSpec, _Fragment

from ._gxf_codelet import GXFCodeletOp as _GXFCodeletOp

__all__ = ["GXFCodeletOp"]


class GXFCodeletOp(_GXFCodeletOp):
    def __setattr__(self, name, value):
        readonly_attributes = [
            "fragment",
            "gxf_typename",
            "conditions",
            "resources",
            "operator_type",
            "description",
        ]
        if name in readonly_attributes:
            raise AttributeError(f'cannot override read-only property "{name}"')
        super().__setattr__(name, value)

    def __init__(self, fragment, *args, **kwargs):
        if not isinstance(fragment, _Fragment):
            raise ValueError(
                "The first argument to an Operator's constructor must be the Fragment "
                "(Application) to which it belongs."
            )
        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _GXFCodeletOp.__init__(self, self, fragment, *args, **kwargs)
        # Create a PyOperatorSpec object and pass it to the C++ API
        spec = OperatorSpec(fragment=self.fragment, op=self)
        self.spec = spec
        # Call setup method in the derived class
        self.setup(spec)

    def setup(self, spec: OperatorSpec):
        # This method is invoked by the derived class to set up the operator.
        super().setup(spec)

    def initialize(self):
        # Place holder for initialize method
        pass


# copy docstrings defined in pydoc.hpp
GXFCodeletOp.__doc__ = _GXFCodeletOp.__doc__
GXFCodeletOp.__init__.__doc__ = _GXFCodeletOp.__init__.__doc__
