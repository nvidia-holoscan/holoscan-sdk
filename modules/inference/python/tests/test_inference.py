# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Inference module authoring tests."""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from holoscan_inference.operators import InferenceOp

import holoscan as hs
import holoscan.operators as core_operators


def one_model_configuration() -> dict[str, object]:
    """Return the representative direct configuration shared by authoring tests."""

    return {
        "model_path_map": {"model": "/tmp/model.onnx"},
        "input_map": {"model": ["input"]},
        "output_map": {"model": ["result"]},
        "input_on_cuda": False,
        "output_on_cuda": False,
        "transmit_on_cuda": False,
        "enable_cuda_graphs": False,
        "parallel_inference": False,
    }


# REQ: PY-API-016
def test_fresh_import_initializes_only_the_module_owned_catalog():
    # Key setup: Inspect the registry before and after importing the Inference provider.
    script = """
import holoscan
from holoscan._core import _registered_operator_names

assert 'holoscan.inference.InferenceOp' not in _registered_operator_names()
from holoscan_inference.operators import InferenceOp
assert InferenceOp.name == 'holoscan.inference.InferenceOp'
assert 'holoscan.inference.InferenceOp' in _registered_operator_names()
assert set(_registered_operator_names()) == {'holoscan.inference.InferenceOp'}
assert 'holoscan.visualization.HolovizOp' not in _registered_operator_names()
assert not any(name.startswith('holoscan.vision.') for name in _registered_operator_names())
"""

    # Key assertion: Core stays provider-neutral and the provider installs only its own row.
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_direct_configuration_declares_only_external_tensor_ports():
    # Key setup: Author two chained models with one external input and one external output.
    #
    #   source --Tensor--> inference[first] --hidden--> inference[second] --Tensor--> sink
    graph = hs.Graph("inference-ports")
    handle = InferenceOp(
        graph,
        "inference",
        model_path_map={"first": "/tmp/first.onnx", "second": "/tmp/second.onnx"},
        input_map={"first": ["input"], "second": ["hidden"]},
        output_map={"first": ["hidden"], "second": ["output"]},
        trt_opt_profile={"first": ["1,4,8"]},
        dla_core_map={"second": "-1"},
        input_on_cuda=False,
        output_on_cuda=False,
        transmit_on_cuda=False,
    )

    # Key assertion: The internal handoff is hidden and only boundary ports are public.
    assert set(handle.ports) == {"input", "output"}
    assert handle.input.direction is hs.PortDirection.INPUT
    assert handle.output.direction is hs.PortDirection.OUTPUT
    with pytest.raises(AttributeError, match="hidden"):
        _ = handle.hidden


def test_resource_descriptions_are_accepted_without_runtime_initialization():
    # Key setup: Supply both module resource descriptions during graph authoring.
    graph = hs.Graph("inference-resources")
    handle = InferenceOp(
        graph,
        "inference",
        **one_model_configuration(),
        cuda_stream_pool={
            "device_id": 0,
            "reserved_size": 1,
            "max_size": 2,
            "nvtx_identifier": "python-inference-streams",
        },
        cuda_green_context_pool={
            "device_id": 0,
            "num_partitions": 0,
            "default_context": None,
            "nvtx_identifier": "python-inference-green-contexts",
        },
    )

    # Key assertion: Cold authoring constructs resources but does not initialize CUDA state.
    assert handle.graph_valid
    assert set(handle.ports) == {"input", "result"}


def test_yaml_configuration_uses_the_existing_cpp_loader(tmp_path: Path):
    # Key setup: Author through config_path using the existing InferenceOpParams YAML grammar.
    config = tmp_path / "inference.yaml"
    config.write_text(
        """
model_path_map: {model: model.onnx}
input_map: {model: [input]}
output_map: {model: [result]}
trt_opt_profile: {model: ["1,4,8"]}
dla_core_map: {model: "-1"}
input_on_cuda: false
output_on_cuda: false
transmit_on_cuda: false
enable_cuda_graphs: false
parallel_inference: false
""".lstrip(),
        encoding="utf-8",
    )
    graph = hs.Graph("inference-yaml")
    handle = InferenceOp(graph, "inference", config_path=str(config))

    # Key assertion: The YAML path declares the same dynamic ports without starting inference.
    assert set(handle.ports) == {"input", "result"}


# REQ: PY-API-003
def test_structural_errors_precede_graph_mutation(tmp_path: Path):
    # Key setup: Exercise facade errors and the fully qualified pass-through registry row.
    graph = hs.Graph("inference-invalid")
    registered = hs.registry.resolve("holoscan.inference.InferenceOp", kind="operator")

    # Key assertion: Every catalog guard rejects before adding an operator.
    with pytest.raises(TypeError, match="requires: input_map, output_map"):
        InferenceOp(graph, "missing", model_path_map={"model": "/tmp/model.onnx"})
    with pytest.raises(TypeError, match="cannot be combined"):
        InferenceOp(graph, "conflict", config_path="config.yaml", **one_model_configuration())
    with pytest.raises(hs.errors.InvalidArgumentError, match="TYPE_MISMATCH"):
        InferenceOp(
            graph,
            "malformed",
            model_path_map={"model": "/tmp/model.onnx"},
            input_map={"model": "input"},
            output_map={"model": ["result"]},
        )
    with pytest.raises(TypeError, match="unexpected"):
        InferenceOp(graph, "unknown", **one_model_configuration(), unexpected=True)
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"PARAMETER_REQUIRED.*input_map"):
        registered(graph, "catalog-missing", model_path_map={"model": "/tmp/model.onnx"})
    with pytest.raises(hs.errors.InvalidArgumentError, match="CONFIG_PATH_CONFLICT"):
        registered(graph, "catalog-conflict", config_path="config.yaml", backend="trt")
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"CONFIG_PATH_INVALID.*missing.yaml"):
        registered(graph, "missing-config", config_path="missing.yaml")
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"PARAMETER_UNKNOWN.*unexpected"):
        registered(graph, "catalog-unknown", **one_model_configuration(), unexpected=True)
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"PARAMETER_UNKNOWN.*max_sizee"):
        registered(
            graph,
            "nested-unknown",
            **one_model_configuration(),
            cuda_stream_pool={"max_sizee": 2},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"model_path_map.*nonempty-map"):
        registered(graph, "empty", model_path_map={}, input_map={}, output_map={})
    with pytest.raises(
        hs.errors.InvalidArgumentError, match=r"model_path_map.model.*nonempty-path"
    ):
        registered(
            graph,
            "empty-path",
            model_path_map={"model": ""},
            input_map={"model": ["input"]},
            output_map={"model": ["result"]},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"output_map.*same-model-keys"):
        registered(
            graph,
            "key-mismatch",
            model_path_map={"model": "/tmp/model.onnx"},
            input_map={"model": ["input"]},
            output_map={"typo": ["result"]},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"input_map.model.*nonempty-sequence"):
        registered(
            graph,
            "empty-inputs",
            model_path_map={"model": "/tmp/model.onnx"},
            input_map={"model": []},
            output_map={"model": ["result"]},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"input_map.model\[1\].*nonempty"):
        registered(
            graph,
            "empty-tensor",
            model_path_map={"model": "/tmp/model.onnx"},
            input_map={"model": ["input", ""]},
            output_map={"model": ["result"]},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"output_map.second\[0\].*unique"):
        registered(
            graph,
            "duplicate-output",
            model_path_map={"first": "/tmp/first.onnx", "second": "/tmp/second.onnx"},
            input_map={"first": ["first_input"], "second": ["second_input"]},
            output_map={"first": ["result"], "second": ["result"]},
        )
    with pytest.raises(hs.errors.InvalidArgumentError, match=r"max_batch_size.*accepted_range"):
        registered(graph, "overflow", **one_model_configuration(), max_batch_size=2**31)
    for field, value in (("trt_opt_profile", ["1,4,8"]), ("dla_core_map", "-1")):
        params = one_model_configuration()
        params[field] = {"typo": value}
        config = tmp_path / f"{field}.yaml"
        config.write_text(json.dumps(params), encoding="utf-8")
        for constructor in (InferenceOp, registered):
            for arguments in (params, {"config_path": str(config)}):
                with pytest.raises(
                    hs.errors.InvalidArgumentError,
                    match=rf"{field}\.typo.*model-key-present-in-model_path_map",
                ):
                    constructor(graph, f"unknown-model-{field}", **arguments)
    assert "operators=0" in repr(graph)


# REQ: PY-API-003
def test_python_and_cpp_authoring_produce_identical_plan_bytes():
    # Key setup: Author and compile the same bounded Inference graph from Python and C++.
    #
    #   holomq://inference-input --Tensor--> inference --Tensor--> (unconnected result output)
    graph = hs.Graph("inference-python-parity")
    clock = graph.add_clock(hs.ManualClock, "compile-clock")
    graph.set_default_clock(clock)
    source = graph.import_topic(
        "holomq://inference-input",
        hs.payload.TensorType,
        endpoint_name="inference-input",
        tensor=hs.tensor.TensorBoundaryContract(
            representation=hs.tensor.TensorRepresentation(
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1),
                rank=1,
                strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
            ),
            bounds=hs.tensor.TensorBounds(max_shape=(1,)),
        ),
    )
    handle = InferenceOp(graph, "inference", **one_model_configuration())
    graph.add_flow(source.output, handle.input)
    assert handle.input.direction is hs.PortDirection.INPUT
    assert handle.result.direction is hs.PortDirection.OUTPUT
    plan = hs.compile(graph)
    assert plan, plan.json()

    helper = os.environ["HOLOSCAN_INFERENCE_PLAN_HELPER"]
    cpp_plan = subprocess.run([helper], check=True, capture_output=True, text=True).stdout

    # Key assertion: The module facade and typed C++ constructor produce identical plan bytes.
    assert plan.json() == cpp_plan


# REQ: PY-API-016
def test_module_facade_is_canonical_and_documents_pass_through_discovery():
    # Key setup: Inspect the module facade and the generic Core operator namespace.

    # Key assertion: Alternative configuration modes stay on the typed facade without a weak alias.
    assert InferenceOp.required_parameters == frozenset()
    assert InferenceOp.parameter_introspection == "pass_through"
    assert not hasattr(core_operators, "InferenceOp")


# REQ: PY-API-018
def test_runtime_and_committed_stub_call_signatures_match():
    # Key setup: Read the facade's committed stub and locate its callable declaration.
    stub_path = (
        Path(__file__).resolve().parents[1] / "holoscan_inference" / "operators" / "__init__.pyi"
    )
    tree = ast.parse(stub_path.read_text(encoding="utf-8"))
    facade = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_InferenceOpType"
    )
    call = next(
        node
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    stub_names = [argument.arg for argument in call.args.args[1:]] + [
        argument.arg for argument in call.args.kwonlyargs
    ]
    runtime = inspect.signature(InferenceOp)

    # Key assertion: Manual facade and manual stub expose the same parameter names.
    assert list(runtime.parameters) == stub_names
