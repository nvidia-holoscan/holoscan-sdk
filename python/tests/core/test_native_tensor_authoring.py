# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural Tensor and rate authoring through native Python Operators."""

import json

import pytest

import holoscan as hs


class StructuralTensorSource(hs.Operator):
    """Declare the producer half of test_structural_tensor_and_rate_authoring_lowers."""

    out = hs.Output("out", hs.payload.Tensor)

    def setup(self, spec):
        spec.output(self.out).nominal_rate(60).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
                strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
                bounds=hs.tensor.TensorBounds(max_shape=(16, 16, 4), max_byte_span=1024),
            )
        )

    def compute(self, context):
        pass


class StructuralTensorTransform(hs.Operator):
    """Exercise the P2-7c structural surface in its representative plan test."""

    inp = hs.Input("in", hs.payload.Tensor)
    out = hs.Output("out", hs.payload.Tensor)
    shape = hs.Param([8, 8, 4], max_elements=3, recompile_required=True)
    pinned = hs.Param(False, recompile_required=True)

    def setup(self, spec):
        spec.input(self.inp).expected_rate(60).expects_tensor(
            hs.tensor.InputSpec(
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
                strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
                bounds=hs.tensor.TensorBounds(max_shape=(16, 16, 4), max_byte_span=1024),
            )
        )
        memory_kind = hs.tensor.parameter_memory_kind(
            type(self).pinned,
            {
                False: hs.tensor.MemoryKind.HOST,
                True: hs.tensor.MemoryKind.PINNED_HOST,
            },
        )
        spec.output(self.out).follows(self.inp, numerator=1, denominator=4).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
            ),
            hs.tensor.parameter_contiguous_byte_span(type(self).shape),
            memory_kind=memory_kind,
        )

    def compute(self, context):
        pass


# REQ: TENSOR-PY-017
def test_structural_tensor_and_rate_authoring_lowers():
    # Key setup: compile a native Tensor producer and transform with every P2-7c declaration.
    #
    #   source[60 Hz, host Tensor] --Tensor--> transform[expected 60 Hz, follows at 1/4]
    graph = hs.Graph("python-structural-tensor-authoring", default_clock=hs.RealtimeClock)
    source = graph.op(StructuralTensorSource, "source")
    transform = graph.op(StructuralTensorTransform, "transform", pinned=True)
    graph.add_flow(source.out, transform["in"])
    plan = hs.compile(graph)
    document = json.loads(plan.json())

    input_port = next(
        port for port in document["ports"] if port["local_name"] == "in" and port["direction"] == 0
    )
    fixed_rate = next(rate for rate in document["port_rates"] if rate["kind"] == 1)
    expected_rate = next(rate for rate in document["port_rates"] if rate["kind"] == 2)
    followed_rate = next(rate for rate in document["port_rates"] if rate["kind"] == 3)

    # Key assertion: exact contracts, rational rates, and frozen mapped memory evidence survive.
    assert plan.ok
    assert input_port["tensor_contract"]["representation"] == {
        "dtype": {"bits": 8, "code": 1, "lanes": 1},
        "memory_kind": 1,
        "rank": 3,
        "strides": 0,
    }
    assert input_port["tensor_contract"]["bounds"]["max_byte_span"] == 1024
    assert fixed_rate["declared_nominal_rate"] == {
        "denominator_hz": 1,
        "numerator_hz": 60,
    }
    assert expected_rate["declared_expected_rate"] == {
        "denominator_hz": 1,
        "numerator_hz": 60,
    }
    assert followed_rate["follows_ratio"] == {"denominator": 4, "numerator": 1}
    memory_derivation = plan.find_tensor_memory_kind_derivation("transform", "out")
    assert memory_derivation is not None
    assert memory_derivation.resolved_memory_kind is hs.tensor.MemoryKind.PINNED_HOST
    assert memory_derivation.parameter_path == "pinned"
    assert sorted(
        allocation["bounds"]["max_byte_span"] for allocation in document["tensor_allocations"]
    ) == [256, 1024]


# REQ: TENSOR-PY-017
def test_scalar_rate_authoring_uses_the_registered_non_tensor_path():
    # Key setup: Declare fixed and follows rates on ordinary scalar ports.
    #
    #   source[30 Hz] --uint32--> transform[expected 30 Hz, follows at 1/2]
    class ScalarRateSource(hs.Operator):
        """Declare a scalar nominal rate for this non-Tensor dispatch test."""

        out = hs.Output("out", hs.payload.uint32)

        def setup(self, spec):
            spec.output(self.out).nominal_rate(30)

        def compute(self, context):
            pass

    class ScalarRateTransform(hs.Operator):
        """Declare scalar expected/follows rates for this dispatch test."""

        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def setup(self, spec):
            spec.input(self.inp).expected_rate(30)
            spec.output(self.out).follows(self.inp, hs.execution.RateRatio(1, 2))

        def compute(self, context):
            pass

    graph = hs.Graph("python-scalar-rate-authoring", default_clock=hs.RealtimeClock)
    source = graph.op(ScalarRateSource, "source")
    transform = graph.op(ScalarRateTransform, "transform")
    graph.add_flow(source.out, transform["in"])
    document = json.loads(hs.compile(graph).json())

    # Key assertion: The erased scalar builder records both fixed and rational rate forms.
    assert any(rate["kind"] == 1 for rate in document["port_rates"])
    assert any(rate["kind"] == 2 for rate in document["port_rates"])
    assert any(
        rate["kind"] == 3 and rate["follows_ratio"] == {"denominator": 2, "numerator": 1}
        for rate in document["port_rates"]
    )


# REQ: TENSOR-PY-017
@pytest.mark.parametrize(
    ("case", "token"),
    [
        ("expected-output", "GRAPH_PORT_DIRECTION_MISMATCH"),
        ("nominal-input", "GRAPH_PORT_DIRECTION_MISMATCH"),
        ("zero-rate", "PORT_RATE_INVALID"),
        ("zero-ratio", "PORT_RATE_INVALID"),
        ("tensor-on-scalar", "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"),
    ],
)
def test_structural_port_authoring_rejects_invalid_direction_and_rate(case, token):
    # Key setup: Route one malformed structural declaration through a native Python setup callback.
    class InvalidStructuralPort(hs.Operator):
        """Select one invalid port declaration for this parameterized test."""

        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def setup(self, spec):
            if case == "expected-output":
                spec.output(self.out).expected_rate(1)
            elif case == "nominal-input":
                spec.input(self.inp).nominal_rate(1)
            elif case == "zero-rate":
                spec.input(self.inp).expected_rate(1, 0)
            elif case == "zero-ratio":
                spec.input(self.inp)
                spec.output(self.out).follows(self.inp, numerator=1, denominator=0)
            else:
                spec.input(self.inp).expects_tensor(hs.tensor.InputSpec())

        def compute(self, context):
            pass

    # Key assertion: Direction, rational-domain, and payload errors retain their diagnostic.
    with pytest.raises(hs.HoloscanError) as raised:
        hs.Graph(f"python-invalid-structural-port-{case}").op(InvalidStructuralPort, "invalid")
    assert token in str(raised.value)


# REQ: TENSOR-PY-017
@pytest.mark.parametrize(
    "mapping",
    [
        {0: hs.tensor.MemoryKind.HOST, 1: hs.tensor.MemoryKind.PINNED_HOST},
        {False: hs.tensor.MemoryKind.UNKNOWN, True: hs.tensor.MemoryKind.PINNED_HOST},
    ],
    ids=["integer-keys", "unknown-target"],
)
def test_parameter_memory_kind_rejects_invalid_mappings(mapping):
    # Key setup: Use integer lookalikes for the required Boolean mapping domain.
    class InvalidMappedMemorySource(hs.Operator):
        """Exercise mapping-key validation in this representative rejection test."""

        out = hs.Output("out", hs.payload.Tensor)
        pinned = hs.Param(False, recompile_required=True)

        def setup(self, spec):
            memory_kind = hs.tensor.parameter_memory_kind(
                type(self).pinned,
                mapping,
            )
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                    dtype="uint8",
                    rank=1,
                    bounds=hs.tensor.TensorBounds(max_byte_span=64),
                ),
                memory_kind=memory_kind,
            )

        def compute(self, context):
            pass

    graph = hs.Graph("python-invalid-memory-kind-mapping")

    # Key assertion: Python equality between 0/1 and False/True cannot bypass exact key typing.
    with pytest.raises(hs.errors.InvalidArgumentError) as raised:
        graph.op(InvalidMappedMemorySource, "invalid")
    assert "PYTHON_TENSOR_MEMORY_KIND_MAPPING_INVALID" in str(raised.value)


# REQ: TENSOR-PY-017
@pytest.mark.parametrize("foreign", [False, True], ids=["non-boolean", "foreign-owner"])
def test_parameter_memory_kind_rejects_invalid_parameter_dependency(foreign):
    # Key setup: Use either a non-Boolean field or another Operator class's descriptor.
    class ForeignOwner(hs.Operator):
        """Own the descriptor used by the foreign-dependency test case."""

        pinned = hs.Param(False, recompile_required=True)

        def compute(self, context):
            pass

    class InvalidDependency(hs.Operator):
        """Declare an invalid memory-kind dependency for this test."""

        out = hs.Output("out", hs.payload.Tensor)
        extent = hs.Param(64, recompile_required=True)

        def setup(self, spec):
            descriptor = ForeignOwner.pinned if foreign else type(self).extent
            memory_kind = hs.tensor.parameter_memory_kind(
                descriptor,
                {
                    False: hs.tensor.MemoryKind.HOST,
                    True: hs.tensor.MemoryKind.PINNED_HOST,
                },
            )
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                    dtype="uint8",
                    rank=1,
                    bounds=hs.tensor.TensorBounds(max_byte_span=64),
                ),
                memory_kind=memory_kind,
            )

        def compute(self, context):
            pass

    # Key assertion: Only an exact Boolean descriptor owned by this Operator can drive placement.
    with pytest.raises(hs.errors.InvalidArgumentError) as raised:
        hs.Graph(f"python-invalid-memory-dependency-{foreign}").op(InvalidDependency, "invalid")
    assert "GRAPH_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID" in str(raised.value)


# REQ: TENSOR-PY-017
def test_tensor_structural_authoring_rejects_invalid_python_arguments():
    # Key setup: Exercise eager validation that does not require a live Graph setup callback.
    literal = hs.tensor.OutputSpec(
        storage="runtime_pool",
        memory_kind=hs.tensor.MemoryKind.HOST,
        dtype="uint8",
        rank=1,
        bounds=hs.tensor.TensorBounds(max_byte_span=64),
    )

    # Key assertion: Convenience spellings round-trip and invalid alternatives explain the domain.
    assert literal.storage is hs.tensor.OutputStorage.RUNTIME_POOL
    with pytest.raises(hs.errors.InvalidArgumentError) as storage_error:
        hs.tensor.OutputSpec(storage="runtime-pool")
    assert "accepted=pass_through,runtime_pool" in str(storage_error.value)

    class ValueInsteadOfDescriptor(hs.Operator):
        """Pass a parameter value where its descriptor is required in this test."""

        out = hs.Output("out", hs.payload.Tensor)
        pinned = hs.Param(False, recompile_required=True)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                literal,
                memory_kind=hs.tensor.parameter_memory_kind(
                    self.pinned,
                    {
                        False: hs.tensor.MemoryKind.HOST,
                        True: hs.tensor.MemoryKind.PINNED_HOST,
                    },
                ),
            )

        def compute(self, context):
            pass

    with pytest.raises(hs.errors.InvalidArgumentError) as mapping_error:
        hs.Graph("python-tensor-descriptor-hint").op(ValueInsteadOfDescriptor, "invalid")
    assert "hint=use_type(self).parameter" in str(mapping_error.value)
