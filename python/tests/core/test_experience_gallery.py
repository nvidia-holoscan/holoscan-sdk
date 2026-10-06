# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executable Python authoring experience and failure gallery."""

import gc

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

ConfigDetector = hs.registry.resolve("holoscan.tests.ConfigDetector", kind="operator")
CounterSource = hs.registry.resolve("holoscan.tests.CounterSource", kind="operator")
DoubleIt = hs.registry.resolve("holoscan.tests.DoubleIt", kind="operator")
ParameterProbe = hs.registry.resolve("holoscan.tests.ParameterProbe", kind="operator")
Sink = hs.registry.resolve("holoscan.tests.Sink", kind="operator")
TopicCameraSource = hs.registry.resolve("holoscan.tests.TopicCameraSource", kind="operator")
TopicWindowFusion = hs.registry.resolve("holoscan.tests.TopicWindowFusion", kind="operator")


# REQ: PY-API-016
def test_sixty_second_tour_composes_compiles_and_runs():
    # Key setup: Compose and run the concise registry-proxy pipeline.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph = hs.Graph("experience-tour")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    source = graph.op(CounterSource, "source", limit=1_000_000)
    doubler = graph.op(DoubleIt, "doubler")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, doubler["in"], queue_depth=4)
    graph.add_flow(doubler.out, sink["in"], queue_depth=4)

    plan = hs.compile(graph)

    # Key assertion: Authoring values remain concise and the session stops cooperatively.
    assert plan, plan.summary()
    assert repr(graph) == "<Graph 'experience-tour' operators=3 flows=2>"
    assert repr(source.out) == "<Output 'out' of source (CounterSource): uint32>"
    assert repr(hs.payload.uint32) == "<PayloadType holoscan.uint32 (epoch 1)>"

    with hs.run_async(plan) as session:
        assert not session.wait(timeout=0.0)
        session.request_stop()
        assert session.wait(timeout=5.0)

    assert session.termination is hs.RunTermination.STOPPED
    assert repr(session) == "<holoscan.RunSession STOPPED>"


# REQ: PY-API-015
# REQ: PY-API-016
def test_configuration_gallery_uses_the_same_sparse_validation_path(tmp_path):
    # Key setup: Apply a sparse document to a private source/detector test graph.
    #
    #   source --uint32--> detector
    graph = hs.Graph("configuration-gallery", default_clock=hs.RealtimeClock)
    source = graph.op(CounterSource, "source")
    detector = graph.op(ConfigDetector, "detector", threshold=6, gain=1.0, window=8)
    graph.add_flow(source.out, detector["in"], queue_depth=8)
    config_path = tmp_path / "app.yaml"
    config_path.write_text("detector:\n  threshold: 9\n  gain: 2.5\n", encoding="utf-8")

    document = hs.config.load(config_path)
    provenance = hs.params.set_params(graph, document, [hs.params.bind(detector, "/detector")])
    plan = hs.compile(graph)

    # Key assertion: Sparse validation reaches the normal accepted compile path.
    assert plan.ok
    assert provenance.config_field_count == 2


# REQ: PY-API-001
# REQ: PY-API-016
def test_failure_gallery_preserves_tokens_message_shape_and_suggestions():
    # Key setup: Connect a Tensor output to the scalar input beside a compatible Tensor input.
    #
    #   camera.frame --Tensor--> fusion.imu (uint32)
    mismatch_graph = hs.Graph("mismatch-gallery")
    camera = mismatch_graph.op(TopicCameraSource, "camera")
    fusion = mismatch_graph.op(TopicWindowFusion, "fusion")

    with pytest.raises(hs.errors.IncompatibleError) as mismatch:
        mismatch_graph.add_flow(camera.frame, fusion.imu)

    # Key assertion: The failure preserves endpoint detail and suggests the compatible input.
    mismatch_message = str(mismatch.value)
    assert mismatch.value.token == "GRAPH_PORT_TYPE_MISMATCH"
    assert "source: camera.frame (Output, holoscan.Tensor, epoch 1)" in mismatch_message
    assert "target: fusion.imu (Input, holoscan.uint32, epoch 1)" in mismatch_message
    assert "Typed edges require identical payload identity." in mismatch_message
    assert "Did you mean fusion.camera?" in mismatch_message
    assert repr(mismatch_graph) == "<Graph 'mismatch-gallery' operators=2 flows=0>"

    rejected_graph = hs.Graph("rejected-gallery")
    rejected_source = rejected_graph.op(CounterSource, "source")
    rejected_sink = rejected_graph.op(Sink, "sink")
    rejected_graph.add_flow(rejected_source.out, rejected_sink["in"])
    rejected_plan = hs.compile(rejected_graph)

    assert not rejected_plan
    assert repr(rejected_plan).startswith("<holoscan.Plan 'rejected-gallery': REJECTED,")
    assert "TEMPORAL_CLOCK_UNDECLARED" in {error.token for error in rejected_plan.errors}

    stale_graph = hs.Graph("stale-gallery")
    stale_source = stale_graph.op(CounterSource, "source")
    stale_port = stale_source.out
    del stale_source
    del stale_graph
    gc.collect()
    replacement = hs.Graph("replacement")
    replacement_sink = replacement.op(Sink, "sink")
    with pytest.raises(hs.errors.InvalidArgumentError) as stale:
        replacement.add_flow(stale_port, replacement_sink["in"])
    assert stale.value.token == "GRAPH_HANDLE_INVALID"


# REQ: PY-API-003
# REQ: PY-API-016
def test_experience_repr_never_exposes_sensitive_parameter_values():
    # Key setup: Author an operator with a non-default sensitive value.
    graph = hs.Graph("redaction-gallery")
    operator = graph.op(ParameterProbe, "probe", secret="private-value")

    # Key assertion: Public representations identify redaction without revealing the value.
    assert "secret=<redacted>" in repr(operator)
    assert "private-value" not in repr(operator)


@pytest.mark.parametrize(
    "block",
    [
        "python_operator_camera_imu_fusion",
        "python_operator_gpu_compute",
        "runtime_python_failure_diagnostic",
        "function_form_operator",
    ],
)
@pytest.mark.skip(reason="experience block requires the hosted Python operator surface")
def test_hosted_python_operator_experience_blocks_are_explicitly_deferred(block):
    assert block
