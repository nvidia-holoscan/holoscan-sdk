# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native constructor handles passed from Python through registered Operator rows."""

import ast
import ctypes
import gc
import importlib.util
import os
import subprocess
import sys
import weakref
from types import SimpleNamespace

import holoscan_constructor_handle_provider as provider
import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

provider.register_rows()
HandleReceiver = hs.registry.resolve("holoscan.test.py.HandleReceiver", kind="operator")
PassThroughReceiver = hs.registry.resolve(
    "holoscan.test.py.PassThroughHandleReceiver", kind="operator"
)
Sink = hs.registry.resolve("holoscan.test.Sink", kind="operator")


def _compile(graph, receiver):
    sink = Sink(graph, "sink")
    graph.add_flow(receiver.output, sink.in_)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return hs.compile(graph)


# REQ: PY-API-024
def test_discovery_publishes_handle_inputs_without_a_device():
    # Key setup: Resolve the row; description must not construct a handle-bearing instance.
    constructions = provider.receiver_constructions()
    handles = HandleReceiver.constructor_handles

    # Key assertion: Typed handle metadata and ports are visible, and help() lists the handles.
    assert provider.receiver_constructions() == constructions
    assert list(handles) == ["data_channel", "camera_sensor"]
    assert handles["data_channel"].type_name == "holoscan.test.py.TestChannel"
    assert handles["data_channel"].required
    assert not handles["camera_sensor"].required
    assert "output" in HandleReceiver.ports
    assert "Constructor handles:" in HandleReceiver.__doc__
    assert "data_channel: holoscan.test.py.TestChannel, required" in HandleReceiver.__doc__


# REQ: PY-API-024
@pytest.mark.parametrize("spelling", ["proxy", "graph-op"])
def test_both_spellings_pass_native_handles_and_values(spelling):
    # Key setup: Pass both handles in reverse declaration order with one declared value.
    #
    #   receiver --> sink
    graph = hs.Graph(f"handles-{spelling}")
    channel = provider.TestChannel(1)
    sensor = provider.TestSensor()
    kwargs = {"camera_sensor": sensor, "frame_size": 128, "data_channel": channel}
    if spelling == "proxy":
        receiver = HandleReceiver(graph, "receiver", **kwargs)
    else:
        receiver = graph.op(HandleReceiver, "receiver", **kwargs)

    # Key assertion: The declaration compiles, callbacks were not invoked, and repr shows an
    # inert summary instead of the provider objects or any address.
    assert _compile(graph, receiver).ok
    assert sensor.starts == 0
    rendered = repr(receiver)
    assert "frame_size=128" in rendered
    assert "data_channel=<holoscan.test.py.TestChannel: process-local>" in rendered
    assert "camera_sensor=<holoscan.test.py.TestSensor: process-local>" in rendered
    assert "0x" not in rendered


# REQ: PY-API-024
def test_native_owners_outlive_python_wrappers_graph_and_last_plan():
    # Key setup: Author with temporary Python objects, then drop every Python reference.
    #
    #   receiver --> sink
    gc.collect()
    channels = provider.live_channels()
    sensors = provider.live_sensors()
    graph = hs.Graph("handle-lifetime")
    channel = provider.TestChannel(2)
    wrapper = weakref.ref(channel)
    receiver = HandleReceiver(
        graph, "receiver", data_channel=channel, camera_sensor=provider.TestSensor()
    )
    del channel
    gc.collect()

    # Key assertion: The Python wrapper is released while the native owner stays alive through
    # compilation; the plan retains it after the Graph is gone and releases it last.
    assert wrapper() is None
    assert provider.live_channels() == channels + 1
    plan = _compile(graph, receiver)
    assert plan.ok
    del receiver, graph
    gc.collect()
    assert provider.live_channels() == channels + 1
    assert provider.live_sensors() == sensors + 1
    del plan
    gc.collect()
    assert provider.live_channels() == channels
    assert provider.live_sensors() == sensors


class _NoProtocol:
    pass


class _RaisingExporter:
    def __holoscan_constructor_handle__(self, type_name):
        raise RuntimeError("device secret 1234")


class _NoneExporter:
    def __holoscan_constructor_handle__(self, type_name):
        return None


class _ForeignCapsuleExporter:
    """Return a well-formed capsule that did not come from this Holoscan Core instance."""

    def __init__(self, name):
        self._storage = ctypes.c_int(0)
        self._name = name

    def __holoscan_constructor_handle__(self, type_name):
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
        return capsule_new(ctypes.addressof(self._storage), self._name, None)


# REQ: PY-API-024
@pytest.mark.parametrize(
    ("kwargs", "error", "token"),
    [
        ({"data_channel": _NoProtocol()}, hs.errors.IncompatibleError, "PROTOCOL_MISSING"),
        (
            {"data_channel": provider.TestSensor()},
            hs.errors.IncompatibleError,
            "TYPE_MISMATCH",
        ),
        ({"data_channel": _RaisingExporter()}, hs.errors.FailureError, "EXPORT_FAILED"),
        ({"data_channel": _NoneExporter()}, hs.errors.InvalidArgumentError, "PROTOCOL_INVALID"),
        (
            {"data_channel": _ForeignCapsuleExporter(b"holoscan.constructor_handle.v1")},
            hs.errors.InvalidArgumentError,
            "PROTOCOL_INVALID",
        ),
        (
            {"data_channel": _ForeignCapsuleExporter(b"other.capsule")},
            hs.errors.InvalidArgumentError,
            "PROTOCOL_INVALID",
        ),
        ({"camera_sensor": None}, hs.errors.InvalidArgumentError, "REQUIRED"),
    ],
)
def test_invalid_handles_fail_before_mutation_and_the_name_can_be_retried(kwargs, error, token):
    # Key setup: Reject a plain object, a wrong provider type, a raising exporter, an exporter that
    # returns None, foreign or misnamed capsules, and a missing required handle.
    graph = hs.Graph("invalid-handles")
    gc.collect()
    channels = provider.live_channels()
    constructions = provider.receiver_constructions()
    with pytest.raises(error) as caught:
        HandleReceiver(graph, "receiver", **kwargs)

    # Key assertion: The field-qualified token never repeats provider text, nothing was
    # constructed or retained, and the same name then succeeds.
    assert caught.value.token == f"AUTHORING_CONSTRUCTOR_HANDLE_{token}"
    assert "field=" in str(caught.value)
    assert "secret" not in str(caught.value)
    assert provider.receiver_constructions() == constructions
    gc.collect()
    assert provider.live_channels() == channels
    assert HandleReceiver(graph, "receiver", data_channel=provider.TestChannel(3)).graph_valid


# REQ: PY-API-024
def test_pass_through_rows_route_handles_before_values():
    # Key setup: Use the pass-through row with an omitted optional handle and one value.
    graph = hs.Graph("pass-through-handles")
    receiver = PassThroughReceiver(
        graph, "receiver", data_channel=provider.TestChannel(4), camera_sensor=None, frame_size=32
    )

    # Key assertion: Handles are routed by declared name; unknown values still fail normally.
    assert receiver.graph_valid
    assert "camera_sensor=None" in repr(receiver)
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        PassThroughReceiver(graph, "other", data_channel=provider.TestChannel(5), frames=1)
    assert caught.value.token == "AUTHORING_PARAMETER_UNKNOWN"


# REQ: PY-API-024
def test_undeclared_objects_remain_rejected_as_values():
    # Key setup: Pass a provider object to a row that declares no constructor handle.
    graph = hs.Graph("undeclared-object")
    with pytest.raises(hs.errors.InvalidArgumentError) as caught:
        graph.op(
            "holoscan.test.ManualRowSourceWithoutDescriptor",
            "source",
            limit=provider.TestChannel(6),
        )

    # Key assertion: Ordinary value conversion still refuses the object.
    assert caught.value.token == "AUTHORING_PARAMETER_TYPE_MISMATCH"


# REQ: PY-API-024
def test_python_anchored_exporter_keeps_its_wrapper_alive():
    # Key setup: Export a Python-owned object through nanobind's shared_ptr caster, the recipe that
    # the provider helper documents as unsuitable.
    graph = hs.Graph("anchored-handle")
    channel = provider.AnchoredChannel(7)
    wrapper = weakref.ref(channel)
    HandleReceiver(graph, "receiver", data_channel=channel)
    del channel
    gc.collect()

    # Key assertion: The retained native owner pins the Python wrapper, unlike the native recipe.
    assert wrapper() is not None
    del graph
    gc.collect()
    assert wrapper() is None


def _load_stub_generator():
    path = os.environ.get("HOLOSCAN_OPERATOR_STUB_GENERATOR")
    if path is None:
        pytest.skip("operator stub generator path is unavailable")
    spec = importlib.util.spec_from_file_location("_generate_operator_stubs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _class_node(rendered, name):
    tree = ast.parse(rendered)
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _typed_dict_keys(rendered, name):
    """Return the keys exactly as Python would see them in the rendered TypedDict ``name``."""

    compile(rendered, "<generated stub>", "exec")
    for node in ast.parse(rendered).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            # Class syntax: compile the body so normalization and private-name mangling apply.
            namespace = {}
            body = "\n".join(
                f"    {statement.target.id}: object"
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
            )
            exec(f"class {name}:\n{body or '    pass'}\n", namespace)  # noqa: S102
            return list(namespace[name].__annotations__)
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id == name:
            return [key.value for key in node.value.args[1].keys]
    raise AssertionError(f"{name} not rendered")


# REQ: PY-API-024
def test_generated_stubs_type_handle_keywords():
    # Key setup: Render stubs for the declared and pass-through handle rows.
    generator = _load_stub_generator()
    aliases = {"HandleReceiver": HandleReceiver, "PassThroughReceiver": PassThroughReceiver}
    annotations = {
        "holoscan.test.py.TestChannel": "holoscan_constructor_handle_provider.TestChannel",
        "holoscan.test.py.TestSensor": "holoscan_constructor_handle_provider.TestSensor",
    }

    rendered = generator.render_operator_proxies(aliases, annotations)

    # Key assertion: Handles become typed keywords with their required status; a handle type
    # without an annotation is an error rather than Any.
    assert "import holoscan_constructor_handle_provider" in rendered
    assert "data_channel: Required[holoscan_constructor_handle_provider.TestChannel]" in rendered
    assert "camera_sensor: holoscan_constructor_handle_provider.TestSensor | None" in rendered
    assert (
        "*, data_channel: holoscan_constructor_handle_provider.TestChannel, "
        "camera_sensor: holoscan_constructor_handle_provider.TestSensor | None = None, "
        "**params: Any" in rendered
    )
    call = next(
        node
        for node in _class_node(rendered, "_PassThroughReceiverType").body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert [arg.arg for arg in call.args.kwonlyargs] == ["data_channel", "camera_sensor"]
    with pytest.raises(ValueError, match="holoscan.test.py.TestSensor"):
        generator.render_operator_proxies(
            aliases, {"holoscan.test.py.TestChannel": annotations["holoscan.test.py.TestChannel"]}
        )


# Handle names that Python rejects or rewrites as class-body or parameter names: keywords,
# non-identifiers, `__debug__`, NFKC-changing spellings (fullwidth "self" and the Kelvin sign), and
# class-private names that are mangled.
_REWRITTEN_HANDLE_NAMES = [
    "in",
    "class",
    "__debug__",
    "data-channel",
    "\uff53\uff45\uff4c\uff46",
    "\u212a",
    "__channel",
]


# REQ: PY-API-024
@pytest.mark.parametrize("name", [*_REWRITTEN_HANDLE_NAMES, "params", "graph", "instance_name"])
def test_stub_generation_rejects_unrepresentable_pass_through_handle_names(monkeypatch, name):
    # Key setup: Give the pass-through row a handle input whose name Python would reject or rewrite
    # as a parameter, or that collides with the generated signature's own parameters.
    generator = _load_stub_generator()
    handle = hs.registry.OperatorConstructorHandleInfo(
        name=name, type_name="holoscan.test.py.TestChannel", description="", required=True
    )
    monkeypatch.setattr(PassThroughReceiver, "constructor_handles", {name: handle})

    # Key assertion: Generation fails with an actionable error instead of emitting invalid Python
    # or silently renaming the keyword.
    with pytest.raises(ValueError, match="cannot be a keyword parameter"):
        generator.render_operator_proxies(
            {"PassThroughReceiver": PassThroughReceiver},
            {"holoscan.test.py.TestChannel": "holoscan_constructor_handle_provider.TestChannel"},
        )


# REQ: PY-API-024
@pytest.mark.parametrize("name", _REWRITTEN_HANDLE_NAMES)
def test_declared_row_stubs_keep_the_literal_handle_key(monkeypatch, name):
    # Key setup: Give the declared-schema row a handle whose name class syntax would reject or
    # rewrite, next to an ordinary handle.
    generator = _load_stub_generator()
    handles = {
        key: hs.registry.OperatorConstructorHandleInfo(
            name=key, type_name="holoscan.test.py.TestChannel", description="", required=False
        )
        for key in ("data_channel", name)
    }
    monkeypatch.setattr(HandleReceiver, "constructor_handles", handles)

    rendered = generator.render_operator_proxies(
        {"HandleReceiver": HandleReceiver},
        {"holoscan.test.py.TestChannel": "holoscan_constructor_handle_provider.TestChannel"},
    )

    # Key assertion: The stub compiles and its TypedDict carries both registry keys verbatim.
    keys = _typed_dict_keys(rendered, "_HandleReceiverParams")
    assert "data_channel" in keys
    assert name in keys


def _run_packaged_generator(tmp_path, *extra):
    # A provider module registers its rows on import, as an installed provider package does.
    (tmp_path / "register_handle_rows.py").write_text(
        "import holoscan_constructor_handle_provider\n"
        "holoscan_constructor_handle_provider.register_rows()\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(tmp_path), environment.get("PYTHONPATH")))
    )
    output = tmp_path / "operators.pyi"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "holoscan._tools.generate_operator_stubs",
            "--output",
            str(output),
            "--provider-module",
            "register_handle_rows",
            "--registry-prefix",
            "holoscan.test.py.",
            "--handle-annotation",
            "holoscan.test.py.TestChannel=holoscan_constructor_handle_provider.TestChannel",
            "--handle-annotation",
            "holoscan.test.py.TestSensor=holoscan_constructor_handle_provider.TestSensor",
            *extra,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    return completed, output


# REQ: PY-API-032
def test_packaged_generator_writes_a_provider_stub_with_its_own_header(tmp_path):
    # Key setup: Run the generator shipped in holoscan._tools, as a downstream build does, with a
    # provider-owned license header.
    header = tmp_path / "header.txt"
    header.write_text("# Copyright (c) Example Corp.\n# SPDX-License-Identifier: MIT\n")
    completed, output = _run_packaged_generator(tmp_path, "--header-file", str(header))

    # Key assertion: The stub carries the provider's header, keeps the formatter guard, compiles,
    # and types the provider's handle rows.
    assert completed.returncode == 0, completed.stderr
    rendered = output.read_text()
    assert rendered.startswith(
        "# Copyright (c) Example Corp.\n# SPDX-License-Identifier: MIT\n# fmt: off\n"
    )
    assert "NVIDIA" not in rendered.split('"""', 1)[0]
    compile(rendered, str(output), "exec")
    assert "class _HandleReceiverType(" in rendered


# REQ: PY-API-034
def test_packaged_generator_imports_each_name_once_and_only_when_used(tmp_path):
    # Key setup: Generate a provider stub whose rows declare no Duration, Required, or port-info
    # members that would need every optional import.
    completed, output = _run_packaged_generator(tmp_path)
    assert completed.returncode == 0, completed.stderr
    tree = ast.parse(output.read_text())

    # Key assertion: Every import binds a distinct name that the stub's declarations reference,
    # so a committed copy needs no unused-import or redefinition suppressions.
    bindings = [
        alias.asname or alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert len(bindings) == len(set(bindings)), bindings
    referenced = {
        node.id
        for statement in tree.body
        if not isinstance(statement, (ast.Import, ast.ImportFrom))
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert set(bindings) <= referenced, set(bindings) - referenced
    assert "holoscan" in bindings
    assert "holoscan_constructor_handle_provider" in bindings


# REQ: PY-API-034
@pytest.mark.parametrize("annotation", ["int", "holoscan.time.Duration | datetime.timedelta"])
def test_generated_stub_imports_datetime_only_for_annotations(monkeypatch, annotation):
    # Key setup: A declared parameter named datetime is an assignment target; its annotation
    # independently determines whether the datetime module is needed.
    monkeypatch.setattr(
        HandleReceiver,
        "parameters",
        {
            "datetime": SimpleNamespace(
                name="datetime",
                python_type=annotation,
                nested_parameters=None,
                optional=False,
                required=False,
                alias=None,
            )
        },
    )
    generator = _load_stub_generator()
    rendered = generator.render_operator_proxies(
        {"HandleReceiver": HandleReceiver},
        {
            "holoscan.test.py.TestChannel": "holoscan_constructor_handle_provider.TestChannel",
            "holoscan.test.py.TestSensor": "holoscan_constructor_handle_provider.TestSensor",
        },
    )

    # Key assertion: The scalar field needs no datetime import, while a timedelta annotation
    # retains both the module import and the SDK Duration alias.
    compile(rendered, "<generated stub>", "exec")
    assert ("import datetime\n" in rendered) == (annotation != "int")
    assert ("from holoscan.time import Duration as _time_Duration" in rendered) == (
        annotation != "int"
    )
    expected = annotation.replace("holoscan.time.Duration", "_time_Duration")
    assert f"    datetime: {expected}\n" in rendered


# REQ: PY-API-032
def test_packaged_generator_rejects_a_header_with_code(tmp_path):
    # Key setup: Supply a header file whose second line is not a comment.
    header = tmp_path / "header.txt"
    header.write_text("# Copyright (c) Example Corp.\nimport os\n")
    completed, output = _run_packaged_generator(tmp_path, "--header-file", str(header))

    # Key assertion: Generation fails instead of emitting code into every provider stub.
    assert completed.returncode != 0
    assert "--header-file must contain only comment lines" in completed.stderr
    assert not output.exists()


# REQ: PY-API-032
@pytest.mark.parametrize("annotation", ["TestChannel", "provider.Test Channel", ".TestChannel"])
def test_packaged_generator_rejects_a_handle_annotation_that_is_not_a_dotted_path(
    tmp_path, annotation
):
    # Key setup: Map a handle type to a class name the stub could not import.
    completed, output = _run_packaged_generator(
        tmp_path, "--handle-annotation", f"holoscan.test.py.TestChannel={annotation}"
    )

    # Key assertion: Generation fails instead of emitting an unresolvable annotation.
    assert completed.returncode != 0
    assert "--handle-annotation expects TYPE_NAME=MODULE.CLASS" in completed.stderr
    assert not output.exists()


# REQ: PY-API-024
def test_capsule_helper_reports_a_null_owner_as_a_python_exception():
    # Key setup: Ask the provider helper to export an empty owner for the requested type.
    with pytest.raises(ValueError, match="AUTHORING_CONSTRUCTOR_HANDLE_NULL"):
        provider.export_null_channel("holoscan.test.py.TestChannel")

    # Key assertion: A different requested type still yields NotImplemented, not an error.
    assert provider.export_null_channel("holoscan.other.Type") is NotImplemented
