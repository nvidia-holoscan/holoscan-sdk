# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module provides a Python API for the core C++ API classes.

The `Application` class is the primary class that should be derived from to
create a custom application.

.. autosummary::

    holoscan.core.Application
    holoscan.core.Arg
    holoscan.core.ArgContainerType
    holoscan.core.ArgElementType
    holoscan.core.ArgList
    holoscan.core.ArgType
    holoscan.core.CLIOptions
    holoscan.core.Component
    holoscan.core.ComponentSpec
    holoscan.core.ConditionType
    holoscan.core.Condition
    holoscan.core.Config
    holoscan.core.DataFlowMetric
    holoscan.core.DataFlowTracker
    holoscan.core.DLDevice
    holoscan.core.DLDeviceType
    holoscan.core.ExecutionContext
    holoscan.core.Executor
    holoscan.core.Fragment
    holoscan.core.Graph
    holoscan.core.InputContext
    holoscan.core.IOSpec
    holoscan.core.Message
    holoscan.core.NetworkContext
    holoscan.core.Operator
    holoscan.core.OperatorSpec
    holoscan.core.OutputContext
    holoscan.core.ParameterFlag
    holoscan.core.Resource
    holoscan.core.Tensor
    holoscan.core.Tracker
    holoscan.core.arg_to_py_object
    holoscan.core.arglist_to_kwargs
    holoscan.core.kwargs_to_arglist
    holoscan.core.py_object_to_arg
"""

# Note: Python 3.7+ expects the threading module to be initialized (imported) before additional
# threads are created (by C++ modules using pybind11).
# Otherwise you will get an assert tlock.locked() error on exit.
# (CLARAHOLOS-765)
import threading as _threading  # noqa: F401

from ..graphs._graphs import FragmentGraph, OperatorGraph
from ._core import Application as _Application
from ._core import (
    Arg,
    ArgContainerType,
    ArgElementType,
    ArgList,
    ArgType,
    CLIOptions,
    Component,
    ComponentSpec,
    Condition,
    ConditionType,
    Config,
    DataFlowMetric,
    DataFlowTracker,
    DLDevice,
    DLDeviceType,
    ExecutionContext,
    Executor,
)
from ._core import Fragment as _Fragment
from ._core import InputContext, IOSpec, Message, NetworkContext
from ._core import Operator as _Operator
from ._core import OutputContext, ParameterFlag
from ._core import PyOperatorSpec as OperatorSpec
from ._core import PyTensor as Tensor
from ._core import (
    Resource,
    Scheduler,
    arg_to_py_object,
    arglist_to_kwargs,
    kwargs_to_arglist,
    py_object_to_arg,
)

Graph = OperatorGraph  # define alias for backward compatibility

__all__ = [
    "Application",
    "Arg",
    "ArgContainerType",
    "ArgElementType",
    "ArgList",
    "ArgType",
    "CLIOptions",
    "Component",
    "ComponentSpec",
    "ConditionType",
    "Condition",
    "Config",
    "DataFlowMetric",
    "DataFlowTracker",
    "DLDevice",
    "DLDeviceType",
    "ExecutionContext",
    "Executor",
    "Fragment",
    "FragmentGraph",
    "Graph",
    "InputContext",
    "IOSpec",
    "Message",
    "NetworkContext",
    "Operator",
    "OperatorSpec",
    "OperatorGraph",
    "OutputContext",
    "ParameterFlag",
    "Resource",
    "Scheduler",
    "Tensor",
    "Tracker",
    "arg_to_py_object",
    "arglist_to_kwargs",
    "kwargs_to_arglist",
    "py_object_to_arg",
]


class Application(_Application):
    def __init__(self, argv=[], *args, **kwargs):
        # If no arguments are provided, instead of letting the C++ API initialize the application
        # from the command line (through '/proc/self/cmdline'), we initialize the application
        # with the command line arguments retrieved from the Python interpreter.
        # This is because the C++ API will not be able to discard arguments that are not meant for
        # the Python application.
        # For example, if the user runs the application with the following
        # command line arguments:
        #   /usr/bin/python3 -m pytest -v -k test_init /workspace/holoscan-sdk/public/python/tests
        # then the C++ API will get the following arguments:
        #   ['/usr/bin/python3', '-m', 'pytest', '-v', '-k', 'test_init',
        #    '/workspace/holoscan-sdk/public/python/tests']
        # whereas the Python interpreter (sys.argv) will get the following arguments:
        #   ['/usr/lib/python3/dist-packages/pytest.py', '-v', '-k', 'test_init',
        #    '/workspace/holoscan-sdk/public/python/tests']
        # For the reason above, we initialize the application with the arguments:
        #   [sys.executable, *sys.argv]
        # which will be equivalent to the following command line arguments:
        #   ['/usr/bin/python3', '/usr/lib/python3/dist-packages/pytest.py', '-v', '-k',
        #    'test_init', '/workspace/holoscan-sdk/public/python/tests']
        # and ``Application().argv`` will return the same arguments as ``sys.argv``.

        if not argv:
            import sys

            argv = [sys.executable, *sys.argv]

        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _Application.__init__(self, argv, *args, **kwargs)

    def run_async(self):
        """Run the application asynchronously.

        This method is a convenience method that creates a thread pool with
        one thread and runs the application in that thread. The thread pool
        is created using `concurrent.futures.ThreadPoolExecutor`.

        Returns
        -------
        future : ``concurrent.futures.Future`` object
        """
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self.run)

    # If we created a context via `gxf.context_create` then we would need to
    # call `gxf.context_destroy` in a destructor. However, in the __init__
    # here, the C++ API creates the GXFExecutor and its context and the
    # C++ object will also take care of the context deletion.

    # def __del__(self):
    #    context_destroy(self._context)


# copy docstrings defined in core_pydoc.hpp
Application.__doc__ = _Application.__doc__
Application.__init__.__doc__ = _Application.__init__.__doc__


class Fragment(_Fragment):
    def __init__(self, app=None, name="", *args, **kwargs):
        if app is not None and not isinstance(app, _Application):
            raise ValueError(
                "The first argument to a Fragment's constructor must be the Application "
                "to which it belongs."
            )
        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _Fragment.__init__(self, self, *args, **kwargs)

        self.name = name
        self.application = app
        # Set the fragment config to the application config.
        if app:
            self.config(app.config())

    def compose(self):
        pass

    def run_async(self):
        """Run the fragment asynchronously.

        This method is a convenience method that creates a thread pool with
        one thread and runs the fragment in that thread. The thread pool
        is created using `concurrent.futures.ThreadPoolExecutor`.

        Returns
        -------
        future : ``concurrent.futures.Future`` object
        """
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1)
        return executor.submit(self.run)


# copy docstrings defined in core_pydoc.hpp
Fragment.__doc__ = _Fragment.__doc__
Fragment.__init__.__doc__ = _Fragment.__init__.__doc__


class Operator(_Operator):
    def __init__(self, fragment, *args, **kwargs):
        if not isinstance(fragment, _Fragment):
            raise ValueError(
                "The first argument to an Operator's constructor must be the Fragment "
                "(Application) to which it belongs."
            )
        # It is recommended to not use super()
        # (https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python)
        _Operator.__init__(self, self, fragment, *args, **kwargs)
        # Create a PyOperatorSpec object and pass it to the C++ API
        spec = OperatorSpec(fragment=self.fragment, op=self)
        self.spec = spec
        # Call setup method in PyOperator class
        self.setup(spec)

    def setup(self, spec: OperatorSpec):
        """Default implementation of setup method."""
        pass


# copy docstrings defined in core_pydoc.hpp
Operator.__doc__ = _Operator.__doc__
Operator.__init__.__doc__ = _Operator.__init__.__doc__
# TODO: remove from core_pydoc.hpp and just define docstrings in this file?


class Tracker:
    """Context manager to add data flow tracking to an application."""

    def __init__(
        self,
        app,
        *,
        filename=None,
        num_buffered_messages=100,
        num_start_messages_to_skip=10,
        num_last_messages_to_discard=10,
        latency_threshold=0,
    ):
        """
        Parameters
        ----------
        app : holoscan.core.Application
            on which flow tracking should be applied.
        filename : str or None, optional
            If none, logging to file will be disabled. Otherwise, logging will
            write to the specified file.
        num_buffered_messages : int, optional
            Controls the number of messages buffered between file writing when
            `filename` is not ``None``.
        num_start_messages_to_skip : int, optional
            The number of messages to skip at the beginning of the execution. This does not affect
            the log file or the number of source messages metric.
        num_last_messages_to_discard : int, optional
            The number of messages to discard at the end of the execution. This does not affect
            the log file or the number of source messages metric.
        latency_threshold : int, optional
            The minimum end-to-end latency in milliseconds to account for in the end-to-end
            latency metric calculations.
        """
        self.app = app
        self.enable_logging = filename is not None
        if self.enable_logging:
            self.logging_kwargs = dict(
                filename=filename,
                num_buffered_messages=num_buffered_messages,
            )
        self.tracker_kwargs = dict(
            num_start_messages_to_skip=num_start_messages_to_skip,
            num_last_messages_to_discard=num_last_messages_to_discard,
            latency_threshold=latency_threshold,
        )

    def __enter__(self):
        self.tracker = self.app.track(**self.tracker_kwargs)
        if self.enable_logging:
            self.tracker.enable_logging(**self.logging_kwargs)
        return self.tracker

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.enable_logging:
            self.tracker.end_logging()
