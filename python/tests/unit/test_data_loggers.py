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

from holoscan.core import AsyncDataLoggerResource, AsyncQueuePolicy, DataLogger, DataLoggerResource
from holoscan.core import _Resource as ResourceBase
from holoscan.core._core import ComponentSpec as ComponentSpecBase
from holoscan.data_loggers import (
    AsyncConsoleLogger,
    BasicConsoleLogger,
    GXFConsoleLogger,
    SimpleTextSerializer,
)


class TestSimpleTextSerializer:
    def check_resource(self, resource, name):
        assert isinstance(resource, ResourceBase)
        assert resource.id == -1  # native Resource type (not GXFResource)

        assert isinstance(resource.spec, ComponentSpecBase)

        if name is not None:
            assert f"name: {name}" in repr(resource)

    def test_kwarg_based_initialization(self, app, capfd):
        name = "console-logger"
        resource = SimpleTextSerializer(
            fragment=app,
            name=name,
            max_elements=10,
            max_metadata_items=10,
            log_python_object_contents=True,
        )
        self.check_resource(resource, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_init_from_config(self, app, data_loggers_config_file, capfd):
        app.config(data_loggers_config_file)
        name = "text-serializer"
        resource = SimpleTextSerializer(
            fragment=app,
            name=name,
            **app.kwargs("simple_text_serializer"),
        )
        self.check_resource(resource, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app, capfd):
        resource = SimpleTextSerializer(app)
        self.check_resource(resource, None)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err


class TestBasicConsoleLogger:
    def check_data_logger(self, data_logger, name):
        assert isinstance(data_logger, DataLogger)
        assert isinstance(data_logger, DataLoggerResource)
        assert isinstance(data_logger, ResourceBase)
        assert data_logger.id == -1  # native Resource type (not GXFResource)

        assert isinstance(data_logger.spec, ComponentSpecBase)

        if name is not None:
            assert f"name: {name}" in repr(data_logger)

    def test_kwarg_based_initialization(self, app, capfd):
        name = "console-logger"
        data_logger = BasicConsoleLogger(
            fragment=app,
            name=name,
            log_inputs=True,
            log_outputs=True,
            log_metadata=True,
            log_tensor_data_content=True,
            serializer=SimpleTextSerializer(app, name="text-serializer"),
        )
        self.check_data_logger(data_logger, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_init_from_config(self, app, data_loggers_config_file, capfd):
        app.config(data_loggers_config_file)
        name = "console-logger"
        data_logger = BasicConsoleLogger(
            fragment=app,
            name=name,
            **app.kwargs("basic_console_logger"),
        )
        self.check_data_logger(data_logger, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app, capfd):
        data_logger = BasicConsoleLogger(app)
        self.check_data_logger(data_logger, None)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err


class TestGXFConsoleLogger:
    def check_data_logger(self, data_logger, name):
        assert isinstance(data_logger, DataLogger)
        assert isinstance(data_logger, DataLoggerResource)
        assert isinstance(data_logger, ResourceBase)
        assert data_logger.id == -1  # native Resource type (not GXFResource)

        assert isinstance(data_logger.spec, ComponentSpecBase)

        if name is not None:
            assert f"name: {name}" in repr(data_logger)

    def test_kwarg_based_initialization(self, app, capfd):
        name = "gxf-console-logger"
        data_logger = GXFConsoleLogger(
            fragment=app,
            name=name,
            log_inputs=True,
            log_outputs=True,
            log_metadata=True,
            log_tensor_data_content=True,
            serializer=SimpleTextSerializer(app, name="text-serializer"),
        )
        self.check_data_logger(data_logger, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_init_from_config(self, app, data_loggers_config_file, capfd):
        app.config(data_loggers_config_file)
        name = "gxf-console-logger"
        data_logger = GXFConsoleLogger(
            fragment=app,
            name=name,
            **app.kwargs("basic_console_logger"),
        )
        self.check_data_logger(data_logger, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app, capfd):
        data_logger = GXFConsoleLogger(app)
        self.check_data_logger(data_logger, None)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err


class TestAsyncConsoleLogger:
    def check_data_logger(self, data_logger, name):
        assert isinstance(data_logger, DataLogger)
        assert isinstance(data_logger, ResourceBase)
        assert isinstance(data_logger, DataLoggerResource)
        assert isinstance(data_logger, AsyncDataLoggerResource)
        assert data_logger.id == -1  # native Resource type (not GXFResource)

        assert isinstance(data_logger.spec, ComponentSpecBase)

        if name is not None:
            assert f"name: {name}" in repr(data_logger)

    def test_kwarg_based_initialization(self, app, capfd):
        name = "async-console-logger"
        data_logger = AsyncConsoleLogger(
            fragment=app,
            name=name,
            log_inputs=True,
            log_outputs=True,
            log_metadata=True,
            log_tensor_data_content=True,
            max_queue_size=5000,
            worker_sleep_time=50000,
            queue_policy=AsyncQueuePolicy.REJECT,
            large_data_max_queue_size=1000,
            large_data_worker_sleep_time=200000,
            large_data_queue_policy=AsyncQueuePolicy.REJECT,
            enable_large_data_queue=True,
            serializer=SimpleTextSerializer(app, name="text-serializer"),
        )
        self.check_data_logger(data_logger, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_init_from_config(self, app, data_loggers_config_file, capfd):
        app.config(data_loggers_config_file)
        name = "async-console-logger"
        data_logger = AsyncConsoleLogger(
            fragment=app,
            name=name,
            **app.kwargs("async_console_logger"),
        )
        self.check_data_logger(data_logger, name)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app, capfd):
        data_logger = AsyncConsoleLogger(app)
        self.check_data_logger(data_logger, None)

        # assert no errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
