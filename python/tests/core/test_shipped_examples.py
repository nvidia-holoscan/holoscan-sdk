# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-language plan parity for the shipped Python examples."""

import json
import os
import subprocess

import pytest
from example_sources import load_shipped_example

import holoscan as hs

EXAMPLE_PROCESS_TIMEOUT_SECONDS = 30.0


def semantic_plan_json(value: str) -> dict:
    """Canonicalize process-local identities before cross-language plan comparison."""
    document = json.loads(value)
    document.pop("graph_generation", None)

    plan_generations = set()

    def collect_plan_generations(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "plan_generation":
                    plan_generations.add(child)
                collect_plan_generations(child)
        elif isinstance(item, list):
            for child in item:
                collect_plan_generations(child)

    collect_plan_generations(document)
    assert len(plan_generations) == 1
    plan_generation = str(next(iter(plan_generations)))

    def canonicalize(item):
        if isinstance(item, dict):
            return {
                key: "<plan-generation>" if key == "plan_generation" else canonicalize(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [canonicalize(child) for child in item]
        if isinstance(item, str):
            return item.replace(plan_generation, "<plan-generation>")
        return item

    return canonicalize(document)


def semantic_hosted_plan_json(value: str) -> dict:
    """Exclude Python-host implementation evidence from otherwise equivalent plans."""
    document = semantic_plan_json(value)
    # Hosted parameter schemas intentionally change graph_hash despite equivalent graph semantics.
    for key in (
        "graph_hash",
        "lifecycle_checkpoints",
        "lifecycle_restart_closures",
        "operator_hosts",
        "operator_parameters",
        "parameter_fields",
    ):
        document.pop(key, None)
    document["diagnostics"] = [
        diagnostic
        for diagnostic in document["diagnostics"]
        if diagnostic["token"] != "OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY"
    ]
    return document


def tensor_structure(document: dict) -> dict:
    """Return language-neutral Tensor/rate evidence from an already canonical plan."""
    input_contracts = [
        {
            "id": port["id"],
            "owner": port["owner"],
            "name": port["local_name"],
            "contract": port["tensor_contract"],
        }
        for port in document["ports"]
        if port["direction"] == 0 and port["tensor_contract"] is not None
    ]
    byte_spans = [
        {
            "output": evidence["output"],
            "resolved_max_byte_span": evidence["resolved_max_byte_span"],
            "resolved_max_staging_byte_span": evidence["resolved_max_staging_byte_span"],
            "update_disposition": evidence["update_disposition"],
        }
        for evidence in document.get("tensor_byte_span_derivations", [])
    ]
    memory_kinds = [
        {
            "output": evidence["output"],
            "form": evidence["form"],
            "mapping": [
                (case["source_raw_bits"], case["memory_kind"]) for case in evidence["mapping"]
            ],
            "resolved_memory_kind": evidence["resolved_memory_kind"],
            "update_disposition": evidence["update_disposition"],
        }
        for evidence in document.get("tensor_memory_kind_derivations", [])
    ]
    return {
        "input_contracts": input_contracts,
        "rates": document["port_rates"],
        "allocations": document.get("tensor_allocations", []),
        "byte_spans": byte_spans,
        "memory_kinds": memory_kinds,
    }


# REQ: PY-API-004
# REQ: PY-API-019
@pytest.mark.parametrize(
    ("name", "executable_environment"),
    [
        ("slow_fast_fusion", "HOLOSCAN_CPP_SLOW_FAST_FUSION"),
        ("graph_composition", "HOLOSCAN_CPP_GRAPH_COMPOSITION"),
    ],
)
def test_shipped_example_plan_matches_cpp_authoring(name, executable_environment):
    # Key setup: Build each shipped Python graph and invoke the matching shipped C++ program.
    #
    #   shipped Python graph --> Plan <-- matching C++ graph
    module = load_shipped_example(name)
    plan = module.build_plan()

    # Key assertion: Every shipped graph is admitted and semantically matches its C++ plan.
    assert plan.ok

    cpp_json = subprocess.run(
        [os.environ[executable_environment], "--plan-json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=EXAMPLE_PROCESS_TIMEOUT_SECONDS,
    ).stdout
    assert semantic_hosted_plan_json(plan.json()) == semantic_hosted_plan_json(cpp_json)


# REQ: PY-API-019
def test_camera_imu_tensor_structure_matches_cpp_authoring():
    # Key setup: Compile the independent native-Python and C++ camera/IMU graphs.
    #
    #   camera --> window fusion --> control topic
    module = load_shipped_example("camera_imu_control")
    plan = module.build_plan()
    cpp_json = subprocess.run(
        [os.environ["HOLOSCAN_CPP_CAMERA_IMU_CONTROL"], "--plan-json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=EXAMPLE_PROCESS_TIMEOUT_SECONDS,
    ).stdout

    # Key assertion: Tensor contracts, allocation evidence, and rates have C++ parity.
    assert plan.ok
    python_structure = tensor_structure(semantic_plan_json(plan.json()))
    cpp_structure = tensor_structure(semantic_plan_json(cpp_json))
    assert python_structure["allocations"]
    assert python_structure == cpp_structure


# REQ: PY-API-022
# REQ: PY-API-019
@pytest.mark.parametrize(
    ("shape", "output_storage"),
    [
        ((1080, 1920, 4), "host"),
        ((720, 1280, 4), "host"),
        ((720, 1280, 4), "pinned"),
    ],
)
def test_derived_tensor_pool_capacity_plan_matches_cpp_authoring(shape, output_storage):
    # Key setup: Compile independent native-Python and C++ graphs with equivalent structure.
    #
    #   camera --> resize --+--> packed grayscale
    #                       +--> conservative grayscale
    module = load_shipped_example("derived_tensor_pool_capacity")
    plan = module.build_plan(shape, output_storage)

    # Key assertion: Named public evidence and the full semantic plans agree for every variant.
    assert plan.ok
    for operator_name in (
        "resize",
        "packed_grayscale",
        "conservative_grayscale",
        "projected_batch",
    ):
        evidence = plan.find_tensor_byte_span_derivation(operator_name, "output")
        assert evidence is not None
        assert evidence.field_id is None
        assert evidence.tensor_index is None
        assert evidence.resolved_max_byte_span > 0
        assert evidence.resolved_max_staging_byte_span is None
        assert (
            evidence.update_disposition
            is hs.execution.TensorByteSpanDerivationUpdateDisposition.RECOMPILE_REQUIRED
        )
    memory_evidence = plan.find_tensor_memory_kind_derivation("resize", "output")
    assert memory_evidence is not None
    assert memory_evidence.parameter_path == "pinned_output"
    assert memory_evidence.resolved_memory_kind is (
        hs.tensor.MemoryKind.PINNED_HOST
        if output_storage == "pinned"
        else hs.tensor.MemoryKind.HOST
    )

    cpp_json = subprocess.run(
        [
            os.environ["HOLOSCAN_CPP_DERIVED_TENSOR_POOL_CAPACITY"],
            "--plan-json",
            *(str(extent) for extent in shape),
            output_storage,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=EXAMPLE_PROCESS_TIMEOUT_SECONDS,
    ).stdout
    python_structure = tensor_structure(semantic_plan_json(plan.json()))
    cpp_structure = tensor_structure(semantic_plan_json(cpp_json))
    assert python_structure["allocations"]
    assert python_structure["byte_spans"]
    assert python_structure["memory_kinds"]
    assert python_structure == cpp_structure


# REQ: PY-API-019
def test_derived_tensor_pool_capacity_example_rejects_unbounded_input():
    # Key setup: Compile the shipped failure graph with no finite producer byte bound.
    #
    #   unbounded source --> conservative grayscale
    module = load_shipped_example("derived_tensor_pool_capacity")
    plan = module.build_unbounded_plan()

    # Key assertion: Input-derived sizing fails closed with the stable diagnostic.
    assert not plan.ok
    assert "PHYSICAL_TENSOR_BYTE_SPAN_INPUT_UNBOUNDED" in plan.json()
