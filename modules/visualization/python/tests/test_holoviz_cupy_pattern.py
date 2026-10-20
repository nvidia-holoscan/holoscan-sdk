# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate CPU decisions, received CUDA Images, and rendered CUDA Images."""

import json
import subprocess
import sys
import threading
import weakref
from contextlib import contextmanager, nullcontext
from io import StringIO
from types import SimpleNamespace

import holoviz_cupy_pattern_driver as driver
import numpy as np
import pytest
from holoscan_visualization.operators import HolovizOp

example = driver.example
NATIVE_NO_WINDOW_ROUTE = (
    "create_window failed (HoloViz kNotFound): [holoviz] No window system route: "
    "DISPLAY, WAYLAND_DISPLAY, WAYLAND_SOCKET and XDG_RUNTIME_DIR are all unset."
)


@pytest.mark.parametrize("option", ["seconds", "startup-timeout", "fps"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
# REQ: PY-API-019
def test_invalid_float_before_prerequisites(option, value, monkeypatch):
    """Reject invalid timing values before loading optional CuPy prerequisites."""
    monkeypatch.setattr(example, "load_cupy", lambda: pytest.fail("CuPy loaded"))
    with pytest.raises(SystemExit) as error:
        (driver.main if option == "startup-timeout" else example.main)([f"--{option}={value}"])
    assert error.value.code == 2


@pytest.mark.parametrize("option", ["width", "height"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "8193"])
# REQ: PY-API-019
def test_invalid_size_before_prerequisites(option, value, monkeypatch):
    """Reject invalid frame dimensions before probing optional dependencies."""
    monkeypatch.setattr(example, "load_cupy", lambda: pytest.fail("CuPy loaded"))
    with pytest.raises(SystemExit) as error:
        example.main([f"--{option}={value}"])
    assert error.value.code == 2


def test_single_visible_device_is_explicit():
    """Accept one visible CUDA device, skip none, and reject ambiguous device selection."""
    for count in [0, 1, 2]:
        cupy = SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda count=count: count))
        )
        if count == 0:
            with pytest.raises(example.UnavailableError, match="no CUDA device"):
                example.require_single_device(cupy)
        elif count == 1:
            assert example.require_single_device(cupy) == 0
        else:
            with pytest.raises(ValueError, match="one visible CUDA device"):
                example.require_single_device(cupy)


@pytest.mark.parametrize(
    "failure,expected",
    [
        (example.UnavailableError("CuPy dependency absent"), 77),
        (RuntimeError("broken CUDA installation"), None),
    ],
)
def test_optional_prerequisite_only_skip(failure, expected, monkeypatch):
    """Translate absent prerequisites to a skip while preserving broken-installation errors."""

    def unavailable():
        """Raise the parameterized prerequisite failure without importing CuPy."""
        raise failure

    monkeypatch.setattr(example, "load_cupy", unavailable)
    monkeypatch.setattr(example.hs, "compile", lambda *_: pytest.fail("graph compiled"))
    if expected is None:
        with pytest.raises(RuntimeError, match="broken CUDA"):
            example.main([])
    else:
        assert example.main([]) == expected


def test_excessive_rate_rejected_before_prerequisites(monkeypatch):
    """Reject frame rates above the limit before loading CuPy."""
    monkeypatch.setattr(example, "load_cupy", lambda: pytest.fail("CuPy loaded"))
    with pytest.raises(SystemExit) as error:
        example.main(["--fps=1001"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "fault", ["shape", "extent", "dtype", "device", "readonly", "strides", "null"]
)
# REQ: PY-API-019
def test_borrowed_view_rejects_invalid_loan(fault):
    """Reject incompatible storage before exposing a raw pointer to CuPy."""
    tensor = SimpleNamespace(
        shape=(48, 64, 4),
        nbytes=48 * 64 * 4,
        data=4096,
        dtype=SimpleNamespace(code=1, bits=8, lanes=1),
        is_read_only=False,
        is_contiguous=lambda: True,
        __dlpack_device__=lambda: (2, 0),
    )
    if fault == "shape":
        tensor.shape = (48, 63, 4)
    elif fault == "extent":
        tensor.nbytes -= 1
    elif fault == "dtype":
        tensor.dtype.bits = 16
    elif fault == "device":
        tensor.__dlpack_device__ = lambda: (2, 1)
    elif fault == "readonly":
        tensor.is_read_only = True
    elif fault == "strides":
        tensor.is_contiguous = lambda: False
    else:
        tensor.data = 0
    with pytest.raises(ValueError, match="writable contiguous RGBA8"):
        example.fill_pattern(None, tensor, (48, 64, 4), None, 0)


@pytest.mark.parametrize("failure_point", ["fill", "commit"])
# REQ: PY-API-019
def test_pool_pressure_defers_only_acquisition(failure_point, monkeypatch):
    """A full loan provider preserves the frame; later fill and commit errors still propagate."""
    config = example.parse_args(["--width=4", "--height=3"])
    cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()))
    source_type = example.pattern_source(config, cupy)
    state = SimpleNamespace(frame=0, coordinates=(None, None))
    filled = []
    point = None

    class Loan:
        tensors = {"data": object()}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            if point == "commit":
                raise example.hs.errors.ResourceExhaustedError("commit failure")

    def allocate(*_):
        if point == "acquire":
            raise example.hs.errors.ResourceExhaustedError("pool held by consumer")
        return Loan()

    def fill(_cupy, _tensor, _shape, _coordinates, frame):
        if point == "fill":
            raise example.hs.errors.ResourceExhaustedError("fill failure")
        filled.append(frame)

    state.out = SimpleNamespace(allocate_tensors=allocate)
    context = SimpleNamespace(stream=lambda: nullcontext())
    monkeypatch.setattr(example, "fill_pattern", fill)
    point = "acquire"
    source_type.compute(state, context)
    assert state.frame == 0
    assert not filled
    point = None
    source_type.compute(state, context)
    assert state.frame == 1
    assert filled == [0]
    point = failure_point
    with pytest.raises(example.hs.errors.ResourceExhaustedError, match=failure_point):
        source_type.compute(state, context)
    assert state.frame == 1


class Timeline:
    def __init__(self, signal=0.1, overshoot=0.0):
        """Configure a virtual clock, signal time, and simulated wait overshoot."""
        self.now = 0.0
        self.signal = signal
        self.overshoot = overshoot

    def is_set(self):
        """Report whether the virtual clock has reached the configured signal time."""
        return self.signal is not None and self.now >= self.signal

    def wait(self, timeout):
        """Advance past the wait budget by the configured overshoot and inspect the signal."""
        self.now += timeout + self.overshoot
        return self.is_set()


class Session:
    def __init__(self, timeline, terminal=None, diagnostics=(), termination=None):
        """Configure terminal timing and final results for a simulated runtime session."""
        self.timeline = timeline
        self.terminal = terminal
        self.diagnostics = diagnostics
        self.termination = termination or example.hs.RunTermination.STOPPED
        self.stopped = False
        self.duration_starts = []

    def wait(self, timeout=0):
        """Advance timed waits and report cooperative stop or scheduled termination."""
        if self.stopped:
            return True
        if timeout:
            self.duration_starts.append(self.timeline.now)
            self.timeline.now += timeout
        return self.terminal is not None and self.timeline.now >= self.terminal

    def request_stop(self):
        """Mark the simulated session as stopped for subsequent waits."""
        self.stopped = True


def diagnostic(message, token="RUNTIME_LIFECYCLE_CALLBACK_FAILED", *, code=None):
    """Build a minimal START-stage diagnostic with the requested token and message."""
    return SimpleNamespace(
        token=token,
        detail=SimpleNamespace(
            stage=example.hs.lifecycle.LifecycleStage.START,
            operator_path="visualizer",
            error=SimpleNamespace(
                message=message,
                code=code if code is not None else example.hs.errors.ErrorCode.FAILURE,
            ),
        ),
    )


@pytest.mark.parametrize(
    "signal,overshoot,terminal,expected",
    [
        (0.1, 0, None, 0),
        (None, 0, None, 1),
        (0.21, 0.2, None, 1),
        (0.1, 0, 0.1, 1),
        (0.1, 0, 0.5, 1),
    ],
)
# REQ: PY-API-019
def test_lifecycle_deadline_and_early_termination(signal, overshoot, terminal, expected):
    """Check startup deadlines, early termination, and duration measured after first emission."""
    timeline = Timeline(signal, overshoot)
    session = Session(timeline, terminal)
    assert (
        driver.run_lifecycle(session, timeline, 1, 0.2, monotonic=lambda: timeline.now) == expected
    )
    assert session.stopped
    if expected == 0:
        assert session.duration_starts == [0.1]


@pytest.mark.parametrize("signal,overshoot", [(0.19, 0.1), (0.21, 0.2)])
# REQ: PY-API-019
def test_lifecycle_rejects_late_wait_wakeup(signal, overshoot):
    """Reject signals first observed after the deadline, even when the wait returns true."""
    timeline = Timeline(signal=signal, overshoot=overshoot)
    session = Session(timeline)

    assert driver.run_lifecycle(session, timeline, 1, 0.2, monotonic=lambda: timeline.now) == 1
    assert not session.duration_starts
    assert session.stopped


# REQ: PY-API-019
def test_lifecycle_rejects_event_observed_after_deadline():
    """Sample the admission timestamp after observing the event to prevent stale-time success."""
    timeline = Timeline(signal=0.25)

    def delayed_observation():
        """Simulate event observation crossing the deadline before the clock is sampled."""
        timeline.now = 0.25
        return True

    timeline.is_set = delayed_observation
    session = Session(timeline)

    assert driver.run_lifecycle(session, timeline, 1, 0.2, monotonic=lambda: timeline.now) == 1
    assert not session.duration_starts
    assert session.stopped


@pytest.mark.parametrize(
    "diagnostics,termination,expected",
    [
        (
            [
                diagnostic(
                    NATIVE_NO_WINDOW_ROUTE,
                    code=example.hs.errors.ErrorCode.NOT_FOUND,
                )
            ],
            None,
            77,
        ),
        (
            [
                diagnostic(
                    "[holoviz] No window system route: unavailable",
                    code=example.hs.errors.ErrorCode.NOT_FOUND,
                )
            ],
            None,
            77,
        ),
        ([diagnostic("create_window failed: Failed to create GLFW window")], None, 1),
        ([diagnostic("create_window failed: Failed to initialize GLFW")], None, 1),
        ([diagnostic("create_window failed")], None, 1),
        (
            [diagnostic(NATIVE_NO_WINDOW_ROUTE)],
            None,
            1,
        ),
        (
            [
                diagnostic(
                    NATIVE_NO_WINDOW_ROUTE,
                    code=example.hs.errors.ErrorCode.NOT_FOUND,
                ),
                diagnostic("device mismatch"),
            ],
            None,
            1,
        ),
        ([], example.hs.RunTermination.FAILED, 1),
    ],
)
# REQ: PY-API-019
def test_final_result_precedence(diagnostics, termination, expected):
    """Give final diagnostics and termination precedence over apparent startup progress."""
    timeline = Timeline(0)
    session = Session(timeline, diagnostics=diagnostics, termination=termination)
    assert (
        driver.run_lifecycle(session, timeline, 1, 0.2, monotonic=lambda: timeline.now) == expected
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("operator_path", "source"),
        ("stage", example.hs.lifecycle.LifecycleStage.STOP),
    ],
)
# REQ: PY-API-019
def test_no_display_requires_visualizer_start(field, value):
    """Do not classify the same text outside the visualizer START context as display absence."""
    failure = diagnostic(NATIVE_NO_WINDOW_ROUTE, code=example.hs.errors.ErrorCode.NOT_FOUND)
    setattr(failure.detail, field, value)
    assert not example.display_unavailable(SimpleNamespace(diagnostics=[failure]))


# REQ: PY-API-019
def test_multiple_devices_main_is_concise_failure(monkeypatch, capsys):
    """Reject unsupported device selection before compilation without an uncaught traceback."""
    cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 2)))
    monkeypatch.setattr(example, "load_cupy", lambda: cupy)
    monkeypatch.setattr(example.hs, "compile", lambda *_: pytest.fail("graph compiled"))
    assert example.main([]) == 1
    assert "exactly one visible CUDA device" in capsys.readouterr().err


@pytest.mark.parametrize(
    "early,diagnostics,expected",
    [
        (False, (), 0),
        (True, (), 1),
        (False, (diagnostic("compute failed"),), 1),
        (
            True,
            (diagnostic(NATIVE_NO_WINDOW_ROUTE, code=example.hs.errors.ErrorCode.NOT_FOUND),),
            77,
        ),
    ],
)
# REQ: PY-API-019
def test_main_session_duration_and_final_result(early, diagnostics, expected, monkeypatch):
    """The shipped CLI measures a session interval and final errors override elapsed time."""
    timeline = Timeline(signal=None)
    session = Session(timeline, terminal=0 if early else None, diagnostics=diagnostics)
    monkeypatch.setattr(example, "load_cupy", lambda: object())
    monkeypatch.setattr(example, "require_single_device", lambda _: 0)
    monkeypatch.setattr(example, "build_graph", lambda *_: object())
    monkeypatch.setattr(example.hs, "compile", lambda *_: object())
    monkeypatch.setattr(example.hs, "run_async", lambda _: nullcontext(session))
    assert example.main(["--seconds=0.25"]) == expected
    assert session.duration_starts == [0.0]
    assert timeline.now == 0.25
    assert session.stopped


# REQ: PY-API-019
def test_main_shutdown_timeout_reports_before_cleanup(monkeypatch, capsys):
    """Bound the stop wait and flush its failure before entering context cleanup."""
    events = []

    class Stderr(StringIO):
        def flush(self):
            events.append(("flush", self.getvalue()))

    def wait(timeout=None):
        events.append(("wait", timeout))
        return False

    session = SimpleNamespace(
        wait=wait,
        request_stop=lambda: events.append("stop"),
        diagnostics=(),
        termination=example.hs.RunTermination.STOPPED,
    )

    @contextmanager
    def run_session(_):
        try:
            yield session
        finally:
            events.append("cleanup")

    monkeypatch.setattr(example, "load_cupy", lambda: object())
    monkeypatch.setattr(example, "require_single_device", lambda _: 0)
    monkeypatch.setattr(example, "build_graph", lambda *_: object())
    monkeypatch.setattr(example.hs, "compile", lambda *_: object())
    monkeypatch.setattr(example.hs, "run_async", run_session)
    monkeypatch.setattr(example.sys, "stderr", Stderr())

    assert example.main(["--seconds=0.25"]) == 1
    assert events == [
        ("wait", 0.25),
        "stop",
        ("wait", 10.0),
        ("flush", "shutdown timeout after 10s; cleanup may still block\n"),
        "cleanup",
    ]
    assert not capsys.readouterr().out


class AllSkipped:
    """Translate only a successful, nonempty, entirely skipped pytest selection."""

    def __init__(self):
        """Collect skipped and passed node identities across pytest report phases."""
        self.skipped = set()
        self.passed = set()

    def pytest_runtest_logreport(self, report):
        """Record skipped nodes and successful call phases without counting setup as a pass."""
        if report.skipped:
            self.skipped.add(report.nodeid)
        if report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        """Map only a successful, nonempty all-skipped selection to exit code 77."""
        selected = {item.nodeid for item in session.items}
        if exitstatus == 0 and selected and not self.passed and self.skipped == selected:
            session.exitstatus = 77


@pytest.mark.parametrize(
    "body,expected",
    [
        ('def test_one(): pytest.skip("no device")', 77),
        ('@pytest.fixture\ndef setup(): pytest.skip("no device")\ndef test_one(setup): pass', 77),
        ('def test_one(): pytest.skip("no device")\ndef test_two(): pass', 0),
        ('def test_one(): pytest.skip("no device")\ndef test_two(): assert False', 1),
        (
            '@pytest.fixture\ndef setup():\n yield\n raise RuntimeError("teardown failed")\n'
            'def test_one(setup): pytest.skip("no device")',
            1,
        ),
        ('raise RuntimeError("collection failed")', 2),
        ("value = 1", 5),
    ],
)
# REQ: PY-API-019
def test_pytest_result_translation(tmp_path, body, expected):
    """Preserve failures and collection errors while translating exclusively skipped selections."""
    target = tmp_path / "test_selection.py"
    target.write_text("import pytest\n" + body + "\n")
    result = subprocess.run([sys.executable, __file__, "-q", str(target)], timeout=30)
    assert result.returncode == expected


@pytest.fixture
def cupy():
    """Provide CuPy on one usable CUDA device, skipping only genuinely absent prerequisites."""
    try:
        module = example.load_cupy()
        example.require_single_device(module)
    except example.UnavailableError as error:
        pytest.skip(str(error))
    # Allocation errors after device discovery are failures, not availability skips.
    module.zeros(1, dtype=module.uint8)
    return module


def assert_pattern_rgb(actual, frame):
    """Compare all received channels to an independent NumPy float64 reference."""
    height, width, channels = actual.shape
    assert channels == 4
    assert actual.dtype == np.uint8
    columns, rows = np.meshgrid(np.arange(width) * 0.05, np.arange(height) * 0.05)
    phase = frame * 0.2
    expected = np.stack(
        [
            127.5 * (1 + np.sin(columns + phase)),
            127.5 * (1 + np.sin(rows - phase)),
            127.5 * (1 + np.sin(columns + rows + phase)),
        ],
        axis=-1,
    ).astype(np.uint8)
    # CPU float64/GPU float32 sin and uint8 truncation can straddle one code value.
    np.testing.assert_allclose(actual[:, :, :3], expected, rtol=0, atol=1)
    np.testing.assert_array_equal(actual[:, :, 3], 255)


@pytest.mark.parametrize("fault", ["swap", "formula"])
# REQ: PY-API-019
def test_rgb_oracle_rejects_wrong_pattern(fault):
    """Detect a channel swap and a phase error even when alpha and frame changes look right."""
    x, y = np.meshgrid(np.arange(64) * 0.05, np.arange(48) * 0.05)
    image = np.empty((48, 64, 4), dtype=np.uint8)
    image[:, :, 0] = 127.5 * (1 + np.sin(x + 0.2))
    image[:, :, 1] = 127.5 * (1 + np.sin(y - 0.2))
    image[:, :, 2] = 127.5 * (1 + np.sin(x + y + 0.2))
    image[:, :, 3] = 255
    assert_pattern_rgb(image, 1)
    if fault == "swap":
        image[:, :, [0, 1]] = image[:, :, [1, 0]]
    else:
        image[:, :, 2] = 127.5 * (1 + np.sin(x + y - 0.2))
    with pytest.raises(AssertionError):
        assert_pattern_rgb(image, 1)


class IndependentTensorCopy:
    # The unversioned fast C exchange cannot convey read-only pool ownership.
    # Request an independent copy through the public Python DLPack protocol.
    def __init__(self, tensor):
        """Retain a pool-backed Tensor for export through the Python DLPack protocol."""
        self.tensor = tensor

    def __dlpack_device__(self):
        """Expose the wrapped Tensor's device identity to the importing array library."""
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, stream=None):
        """Request an independent copy with readiness ordered on the consumer stream."""
        return self.tensor.__dlpack__(stream=stream, copy=True)


@pytest.mark.parametrize("rendered", [False, True], ids=["received", "rendered"])
# REQ: SCHEMA-PY-004
# REQ: PY-API-019
def test_cuda_images_values_stream_device_and_lifetime(cupy, rendered, monkeypatch):
    """Inspect received or rendered frames across CUDA streams and later source allocations."""
    hs = example.hs
    pending_writes = []
    borrowed_views = []
    delay = cupy.RawKernel(
        'extern "C" __global__ void delay(unsigned long long cycles) {'
        " const unsigned long long start = clock64();"
        " while (clock64() - start < cycles) {} }",
        "delay",
    )
    fill = example.fill_pattern

    def delayed_fill(*args):
        """Leave writes queued on the runtime stream when the loan guard commits."""
        assert cupy.cuda.get_current_stream().ptr != 0
        delay((1,), (1,), (np.uint64(20_000_000),))
        tensor = args[1]

        def unowned_memory(pointer, extent, *, owner, device_id):
            assert pointer == tensor.data
            assert extent == tensor.nbytes
            assert owner is tensor
            assert device_id == 0
            return cupy.cuda.UnownedMemory(pointer, extent, owner, device_id)

        def ndarray(*shape_args, **kwargs):
            view = cupy.ndarray(*shape_args, **kwargs)
            assert view.data.ptr == tensor.data
            assert view.data.mem.size == tensor.nbytes
            assert view.device.id == 0
            borrowed_views.append(weakref.ref(view))
            return view

        checked_cupy = SimpleNamespace(
            cuda=SimpleNamespace(
                UnownedMemory=unowned_memory, MemoryPointer=cupy.cuda.MemoryPointer
            ),
            ndarray=ndarray,
            sin=cupy.sin,
            uint8=cupy.uint8,
        )
        fill(checked_cupy, *args[1:])
        assert borrowed_views[-1]() is None, "borrowed CuPy view escaped its fill scope"
        ready = cupy.cuda.Event()
        ready.record()
        pending_writes.append(not ready.done)

    monkeypatch.setattr(example, "fill_pattern", delayed_fill)

    config = example.parse_args(["--width=64", "--height=48", "--fps=100"])
    started = threading.Event()
    received = []
    complete = threading.Event()
    reused = threading.Event()
    pointers = []

    class Capture(hs.Operator):
        inp = hs.Input("in", example.Image)

        def setup(self, spec):
            """Register a lossless input queue for capturing independently inspected Images."""
            spec.input(self.inp, queue_depth=8)

        def contract(self):
            """Consume one Image per input-triggered invocation."""
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            """Retain three Image descriptors and signal that capture is complete."""
            sample = self.inp.receive()
            pointers.append(sample.data.data.data)
            if len(received) < 3:
                received.append(sample)
            if len(received) == 3:
                complete.set()
            if len(pointers) >= 64:
                reused.set()

    graph = hs.Graph("pattern-capture", default_clock=hs.RealtimeClock)
    source = graph.op(
        driver.observe_emissions(example.pattern_source(config, cupy), started), "source"
    )
    sink = graph.op(Capture, "capture")
    output = source.out
    if rendered:
        visualizer = HolovizOp(
            graph,
            "visualizer",
            layers=[
                dict(
                    kind="image",
                    port_name="video",
                    format="rgba8_unorm",
                    max_width=config.width,
                    max_height=config.height,
                )
            ],
            sink=dict(kind="cuda", width=config.width, height=config.height, format="rgba8_unorm"),
            trigger=dict(kind="layer", layer=0),
        )
        graph.add_flow(
            output, visualizer.video, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
        )
        output = visualizer["output"]
        graph.partition("render").add(visualizer)
    graph.add_flow(output, sink["in"], queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=8)
    if rendered:
        graph.auto_partition_remaining()
    options = example.compile_options()
    if rendered:
        options.deployment.bind_tensor_output_device(
            operator_path="visualizer", output_port="output", device=hs.execution.DeviceId(0)
        )
    plan = hs.compile(graph, options)
    assert plan, plan.summary()

    with hs.run_async(plan) as session:
        try:
            assert complete.wait(15), session.diagnostics
            assert started.is_set()
            assert len(received) == 3
            values = []
            # Read retained descriptors on a separate stream while their session still owns
            # the readiness streams. Producer-local arrays have already left compute scope.
            with cupy.cuda.Stream(non_blocking=True) as stream:
                for sample in received:
                    image = sample.data
                    assert (image.height, image.width) == (48, 64)
                    assert image.encoding is example.ImageEncoding.RGBA8
                    assert image.color_space is example.ColorSpace.UNSPECIFIED
                    assert image.data.shape == (48, 64, 4)
                    assert image.data.__dlpack_device__() == (2, 0)
                    array = cupy.from_dlpack(IndependentTensorCopy(image.data))
                    assert array.dtype == cupy.uint8
                    values.append(array)
                stream.synchronize()
            for value in values:
                assert bool(cupy.all(value[:, :, 3] == 255))
            assert not bool(cupy.array_equal(values[0], values[1]))
            assert not bool(cupy.array_equal(values[1], values[2]))
            snapshots = [cupy.asnumpy(value) for value in values]
            if not rendered:
                sequences = [sample.metadata.sequence for sample in received]
                assert sequences == [1, 2, 3]
                for sequence, snapshot in zip(sequences, snapshots, strict=True):
                    assert_pattern_rgb(snapshot, sequence - 1)
            assert reused.wait(15), session.diagnostics
            assert any(pending_writes), "producer writes never remained pending at guard commit"
            assert len(set(pointers[3:])) < len(pointers[3:]), "pool reuse not exercised"
            assert not set(pointers[:3]).intersection(pointers[3:])
            assert not session.wait(timeout=0.1)  # More source allocations and publications.
            with cupy.cuda.Stream(non_blocking=True) as stream:
                for sample, snapshot in zip(received, snapshots, strict=True):
                    image = sample.data
                    actual = cupy.from_dlpack(IndependentTensorCopy(image.data))
                    stream.synchronize()
                    cupy.testing.assert_array_equal(actual, snapshot)
        finally:
            session.request_stop()
            stopped = session.wait(timeout=10)
    assert stopped
    assert session.termination is hs.RunTermination.STOPPED
    assert not session.diagnostics


def full_pool_capture_count(slot_count, published, held):
    """Freeze only a fully published and captured inventory of distinct provider slots."""
    if published < slot_count or len(held) < slot_count:
        return None
    assert published == len(held) == slot_count, (slot_count, published, len(held))
    assert [sample.metadata.sequence for sample in held] == list(range(1, slot_count + 1))
    assert len({sample.data.data.data for sample in held}) == slot_count
    return slot_count


# REQ: PY-API-019
def test_full_pool_capture_waits_for_delayed_delivery():
    """Do not freeze two captures when the source has published all three pool slots."""

    def sample(sequence):
        return SimpleNamespace(
            metadata=SimpleNamespace(sequence=sequence),
            data=SimpleNamespace(data=SimpleNamespace(data=sequence * 4096)),
        )

    held = [sample(1), sample(2)]
    assert full_pool_capture_count(3, 3, held) is None
    held.append(sample(3))  # Deliver the already-published sample after the parent observation.
    assert full_pool_capture_count(3, 3, held) == 3


def image_pool_slot_count(plan, config):
    """Resolve this source's singular Image.data pool through public plan evidence."""
    document = json.loads(plan.json())

    def one(rows, description):
        assert len(rows) == 1, f"expected one {description}, got {rows}"
        return rows[0]

    owner = one(
        [row for row in document["operator_parameters"] if row["operator_path"] == "source"],
        "source operator",
    )["owner"]["id"]
    port = one(
        [
            row
            for row in document["ports"]
            if row["owner"] == owner and row["local_name"] == "out" and row["direction"] == 1
        ],
        "source.out port",
    )
    assert port["payload_identity"]["schema"]["canonical_name"] == "holoscan.schema.Image"
    # Image.data is schema field 4 and the singular descriptor Tensor at index 0.
    derivation = one(
        [row for row in document["tensor_byte_span_derivations"] if row["output"] == port["id"]],
        "Image.data byte-span derivation",
    )
    assert (derivation["field_id"], derivation["tensor_index"]) == (4, 0), derivation
    assert derivation["resolved_max_byte_span"] == config.height * config.width * 4
    assert not [
        row for row in document.get("tensor_field_allocations", []) if row["output"] == port["id"]
    ], "expected singular Image Tensor allocation, not a multi-field pool"
    allocation = one(
        [row for row in document["tensor_allocations"] if row["output"] == port["id"]],
        "source Image Tensor allocation",
    )
    assert allocation["bounds"]["max_byte_span"] == derivation["resolved_max_byte_span"]
    assert allocation["device"] == 0, allocation
    assert allocation["slot_count"] > 1, allocation
    return allocation["slot_count"]


# REQ: PY-API-019
def test_cuda_full_pool_recovers_without_advancing_frame(cupy):
    """Acknowledge every planned slot before one pressure probe and post-release recovery."""
    hs = example.hs
    config = example.parse_args(["--width=16", "--height=12", "--fps=100"])
    changed = threading.Condition()
    state = SimpleNamespace(phase="fill", published=0, held=[], attempts=[], recovered=False)
    source_type = example.pattern_source(config, cupy)

    class ObservedSource(source_type):
        def compute(self, context):
            with changed:
                phase = state.phase
                if phase not in {"fill", "probe", "recover"}:
                    return
                if phase == "probe":
                    state.phase = "probing"  # Exactly one pre-release attempt is admitted.
            before = self.frame
            super().compute(context)
            with changed:
                state.published = self.frame
                state.attempts.append((phase, before, self.frame))
                if phase == "fill" and self.frame == slot_count:
                    state.phase = "full"
                elif phase == "probe":
                    state.phase = "probed"
                elif phase == "recover" and self.frame == slot_count + 1:
                    state.phase = "recovered"
                changed.notify_all()

    class Capture(hs.Operator):
        inp = hs.Input("in", example.Image)

        def setup(self, spec):
            spec.input(self.inp, queue_depth=1)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            sample = self.inp.receive()
            with changed:
                state.held.append(sample)
                if sample.metadata.sequence == slot_count + 1:
                    assert state.phase in {"recover", "recovered"}, state.phase
                    state.recovered = True
                changed.notify_all()

    def snapshot(sample):
        with cupy.cuda.Stream(non_blocking=True) as stream:
            value = cupy.from_dlpack(IndependentTensorCopy(sample.data.data))
            stream.synchronize()
        return cupy.asnumpy(value)

    graph = hs.Graph("pattern-pool-pressure", default_clock=hs.RealtimeClock)
    source = graph.op(ObservedSource, "source")
    capture = graph.op(Capture, "capture")
    graph.add_flow(source.out, capture["in"], queue_policy=hs.QueuePolicy.BLOCKING, queue_depth=1)
    plan = hs.compile(graph, example.compile_options())
    assert plan, plan.summary()
    slot_count = image_pool_slot_count(plan, config)
    with hs.run_async(plan) as session:

        def wait_for(predicate, boundary):
            with changed:
                assert changed.wait_for(predicate, timeout=10), (
                    f"{boundary}: planned slots={slot_count}, phase={state.phase}, "
                    f"published={state.published}, captured="
                    f"{[sample.metadata.sequence for sample in state.held]}, "
                    f"attempts={state.attempts}; diagnostics={session.diagnostics}"
                )

        try:
            wait_for(
                lambda: state.phase == "full"
                and full_pool_capture_count(slot_count, state.published, state.held) == slot_count,
                "pool never reached its complete captured geometry",
            )
            with changed:
                state.phase = "probe"
            wait_for(
                lambda: state.phase == "probed", "pre-release acquisition probe did not finish"
            )
            with changed:
                probes = [attempt for attempt in state.attempts if attempt[0] == "probe"]
                assert probes == [("probe", slot_count, slot_count)], probes
                retained = list(state.held[1:])
                state.phase = "releasing"
            snapshots = [(sample, snapshot(sample)) for sample in retained]
            for sample, pixels in snapshots:
                assert_pattern_rgb(pixels, sample.metadata.sequence - 1)
            with changed:
                released = state.held.pop(0)
            del released  # Destroy the only retained first-Image owner outside the condition lock.
            with changed:
                state.phase = "recover"
            wait_for(
                lambda: state.phase == "recovered" and state.recovered,
                "post-release publication/capture did not recover",
            )
            with changed:
                recovered = state.held[-1]
                recovery = [attempt for attempt in state.attempts if attempt[0] == "recover"]
                assert recovery[-1] == ("recover", slot_count, slot_count + 1), recovery
                assert all(before == after == slot_count for _, before, after in recovery[:-1])
            assert recovered.metadata.sequence == slot_count + 1
            assert_pattern_rgb(snapshot(recovered), slot_count)
            for sample, pixels in snapshots:
                np.testing.assert_array_equal(snapshot(sample), pixels)
        finally:
            session.request_stop()
            stopped = session.wait(timeout=10)
    assert stopped
    assert session.termination is hs.RunTermination.STOPPED
    assert not session.diagnostics


if __name__ == "__main__":
    raise SystemExit(pytest.main(sys.argv[1:], plugins=[AllSkipped()]))
