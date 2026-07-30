"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import os
import re
import subprocess
from xml.etree import ElementTree

import pytest

from holoscan.core import Application, Fragment
from holoscan.logger import LogLevel, set_log_level
from holoscan.resources import CudaGreenContextPool

# set log level to INFO during testing
set_log_level(LogLevel.INFO)


@pytest.fixture
def app():
    return Application()


@pytest.fixture
def fragment():
    return Fragment()


@pytest.fixture
def operators_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "operator_parameters.yaml")
    return config_file


@pytest.fixture
def data_loggers_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "data_logger_parameters.yaml")
    return config_file


@pytest.fixture
def subgraphs_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "subgraph_parameters.yaml")
    return config_file


@pytest.fixture
def ping_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "app_config_ping.yaml")
    return config_file


@pytest.fixture
def deprecated_extension_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "deprecated_stream_playback.yaml")
    return config_file


# Cached result for Green Context availability.
_green_context_available = None
_cached_nvidia_smi_output = None
_cached_nvidia_smi_loaded = False
_green_context_device_properties = None
_green_context_device_properties_loaded = False
# Holoscan SDK FAQ prerequisite documents that CUDA Green Context features
# require CUDA Driver API version >= 12.4.
# Ref: https://docs.nvidia.com/holoscan/sdk-user-guide/hsdk_faq.html
MIN_GREEN_CONTEXT_CUDA_DRIVER_API = 12040


def _nvidia_smi_output():
    global _cached_nvidia_smi_loaded, _cached_nvidia_smi_output
    if _cached_nvidia_smi_loaded:
        return _cached_nvidia_smi_output
    _cached_nvidia_smi_loaded = True
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        _cached_nvidia_smi_output = None
        return None
    _cached_nvidia_smi_output = proc.stdout
    return _cached_nvidia_smi_output


def _cuda_driver_api_version():
    output = _nvidia_smi_output()
    if not output:
        return None
    try:
        root = ElementTree.fromstring(output)
    except ElementTree.ParseError:
        return None
    cuda_version = root.findtext("cuda_version")
    if not cuda_version:
        return None
    match = re.search(r"(\d+)\.(\d+)", cuda_version)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    return major * 1000 + minor * 10


def green_context_available():
    """Return True if Green Context is supported in this environment.

    Tests that require it should skip when this returns False.
    """
    global _green_context_available
    if _green_context_available is not None:
        return _green_context_available
    version = _cuda_driver_api_version()
    _green_context_available = version is not None and version >= MIN_GREEN_CONTEXT_CUDA_DRIVER_API
    return _green_context_available


def green_context_min_sm_size_for_device_major(device_major):
    """Return a typical minimum SM block size for Green Context pools by architecture.

    Tests use this to choose `min_sm_size` / `sms_per_partition` together so
    partition sizes stay compatible with the driver's SM grouping for the GPU.

    `device_major` is CUDA compute capability major (`cudaDeviceProp.major`):
      - 7: Volta/Turing generation (SM 7.x)
      - 8: Ampere generation (SM 8.x)
      - 9: Hopper generation (SM 9.x)
      - 10+: Blackwell generation and newer (SM 10.x+)

    For reference on some aarch64 boards when sizing tests:
      - Jetson Orin AGX has 16 SMs
      - Jetson Orin Nano has 8 SMs
      - Jetson AGX Thor has 20 SMs (Blackwell sm_110)
    """
    if device_major == 7:
        return 2
    if device_major in (8, 9):
        return 4
    if device_major >= 10:
        return 8
    return 2


def green_context_device_properties():
    """Return cached CUDA device properties needed by Green Context tests.

    Returns:
        dict | None: {"major": int, "sm_count": int, "min_sm_size": int} on success,
        otherwise None when CuPy/CUDA is unavailable or properties cannot be queried.
    """
    global _green_context_device_properties, _green_context_device_properties_loaded
    if _green_context_device_properties_loaded:
        return _green_context_device_properties
    _green_context_device_properties_loaded = True
    try:
        import cupy as cp  # noqa: PLC0415
    except ImportError:
        _green_context_device_properties = None
        return None
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        major = int(props["major"])
        sm_count = int(props["multiProcessorCount"])
        _green_context_device_properties = {
            "major": major,
            "sm_count": sm_count,
            "min_sm_size": green_context_min_sm_size_for_device_major(major),
        }
    except (RuntimeError, cp.cuda.runtime.CUDARuntimeError):
        _green_context_device_properties = None
    return _green_context_device_properties


def green_context_partitions_supported(partitions):
    """Return True if `partitions` fit this GPU for typical Green Context tests.

    After basic arithmetic checks, calls
    :meth:`CudaGreenContextPool.is_partitioning_supported` to verify that
    the CUDA driver accepts the SM split and per-partition resource
    generation.  This catches driver-level rejections that arithmetic
    checks alone would miss.

    Returns False if device properties cannot be read.
    """
    props = green_context_device_properties()
    if not props:
        return False
    min_sm = props["min_sm_size"]
    sm_count = props["sm_count"]
    total = sum(partitions)
    if sm_count < total:
        return False
    for p in partitions:
        if p < min_sm or p % min_sm != 0:
            return False
    remainder = sm_count - total
    if remainder != 0 and remainder % min_sm != 0:
        return False

    return CudaGreenContextPool.is_partitioning_supported(0, min_sm, partitions)


def pytest_configure(config):  # noqa: ARG001
    os.environ["HOLOSCAN_DISABLE_BACKTRACE"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        "--run-slow",
        action="store_true",
        help="include tests marked slow (--runslow and --run-slow are equivalent)",
    )
    parser.addoption(
        "--run-realtime",
        action="store_true",
        default=False,
        help="run tests marked as requiring real-time kernel config",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow (or --runslow) option to run")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)

    if not config.getoption("--run-realtime"):
        skip_realtime = pytest.mark.skip(reason="need --run-realtime option to run")
        for item in items:
            if item.get_closest_marker("realtime"):
                item.add_marker(skip_realtime)


# Note: see [tool.pytest.ini_options] in pyproject.toml for marker definitions
