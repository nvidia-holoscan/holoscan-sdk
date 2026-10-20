# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public namespace contract tests for the private native extension."""

import builtins
import os
import subprocess
import sys
import textwrap
from importlib import import_module
from pathlib import Path

import holoscan as hs
from holoscan import _module_discovery
from holoscan._public_api import DYNAMIC_PUBLIC_MODULES, PUBLIC_API


# REQ: PY-API-013
def test_public_api_module_keys_are_sorted():
    # Key setup: Read the module order from the manually maintained public API manifest.
    module_names = list(PUBLIC_API)

    # Key assertion: Keep concept modules easy to scan and deterministic for every consumer.
    assert module_names == sorted(module_names)


# REQ: PY-API-013
def test_root_allowlist_is_manifest_driven():
    # Key setup: Derive the root symbols and concept modules from the checked-in API manifest.
    expected_exports = [name for name in PUBLIC_API["holoscan"] if not hasattr(builtins, name)]
    assert hs.__all__ == ["__version__", *expected_exports]

    concept_modules = {name.rsplit(".", 1)[-1] for name in PUBLIC_API if name.count(".") == 1}
    provider_modules = _module_discovery.registered_names()
    visible = {name for name in dir(hs) if not name.startswith("_")}

    # Key assertion: The root exposes only allowlisted objects, concept modules, and providers.
    assert visible == set(PUBLIC_API["holoscan"]) | concept_modules | provider_modules
    assert concept_modules.isdisjoint(hs.__all__)
    assert provider_modules.isdisjoint(hs.__all__)
    assert set(hs.__all__).isdisjoint(dir(builtins))


# REQ: PY-API-013
def test_native_extension_surface_is_fully_manifested():
    # Key setup: Compare native bindings with manifested names that the extension actually owns.
    manifested = {
        name for names in PUBLIC_API.values() for name in names if hasattr(hs._core, name)
    }
    native = {name for name in dir(hs._core) if not name.startswith("_")}

    # Key assertion: No public-looking native binding can bypass module and export checks.
    assert native == manifested


# REQ: PY-API-013
def test_dynamic_concept_modules_are_explicitly_manifested():
    # Key setup: Inspect the package modules whose public names depend on loaded providers.

    # Key assertion: Dynamic modules remain explicit and retain an empty fixed-name manifest.
    assert {"holoscan.graphs"} == DYNAMIC_PUBLIC_MODULES
    assert all(module_name in PUBLIC_API for module_name in DYNAMIC_PUBLIC_MODULES)
    assert PUBLIC_API["holoscan.graphs"] == ()


# REQ: PY-API-013
def test_root_does_not_restore_provisional_spellings():
    # Key setup: Enumerate private and retired provisional root spellings.
    assert hs._core.__name__ == "holoscan._core"

    # Key assertion: None of the retired names leak back into the public root namespace.
    for name in (
        "core",
        "DLDevice",
        "ErrorCode",
        "InvalidArgumentError",
        "OperatorHandle",
        "PayloadType",
        "ms",
        "payload_type",
        "uint32",
    ):
        assert not hasattr(hs, name)


# REQ: PY-OP-005
def test_python_predicate_trigger_is_intentionally_absent():
    # Key setup: Inspect the root surface used to author hosted Python temporal contracts.

    # Key assertion: Python callbacks cannot enter the scheduler's readiness-evaluation path.
    assert not hasattr(hs, "OnPredicate")


# REQ: PY-API-013
def test_intentional_aliases_preserve_identity():
    # Key setup: Resolve the two public aliases intentionally retained at the package root.

    # Key assertion: The root aliases are the same native objects as their concept-module names.
    assert hs.Tensor is hs.payload.Tensor
    assert hs.HoloscanError is hs.errors.HoloscanError


# REQ: PY-API-013
def test_public_native_objects_report_canonical_modules():
    # Key setup: Map the intentional aliases whose canonical module differs from their access path.
    canonical_alias_modules = {
        ("holoscan", "HoloscanError"): "holoscan.errors",
        ("holoscan", "ManualClock"): "holoscan.clocks",
        ("holoscan", "pyobject"): "holoscan.payload",
        ("holoscan", "RealtimeClock"): "holoscan.clocks",
        ("holoscan", "SyntheticClock"): "holoscan.clocks",
        ("holoscan.channel", "QueuePolicy"): "holoscan",
        ("holoscan.channel", "Sample"): "holoscan",
        ("holoscan.channel", "SampleMetadata"): "holoscan",
        ("holoscan.payload", "Tensor"): "holoscan",
        ("holoscan.payload.schema", "Header"): "holoscan.payload",
        ("holoscan.payload.schema", "String"): "holoscan.payload",
        ("holoscan.payload.schema", "TensorPacket"): "holoscan.payload",
        ("holoscan.payload.schema", "Time"): "holoscan.payload",
        ("holoscan.schema", "Header"): "holoscan.schema.header",
        ("holoscan.schema", "Image"): "holoscan.schema.image",
        ("holoscan.schema", "Imu"): "holoscan.schema.imu",
        ("holoscan.schema", "String"): "holoscan.schema.string",
        ("holoscan.schema", "TensorPacket"): "holoscan.schema.tensor_packet",
        ("holoscan.schema", "Time"): "holoscan.schema.time",
        ("holoscan.execution", "ProbeFailurePolicy"): "holoscan.topics",
    }
    for module_name, names in PUBLIC_API.items():
        module = hs if module_name == "holoscan" else import_module(module_name)
        for name in names:
            value = getattr(module, name)
            if not hasattr(value, "__module__"):
                continue
            expected = canonical_alias_modules.get((module_name, name), module_name)

            # Key assertion: Every public native object reports its canonical documentation module.
            assert value.__module__ == expected, f"{module_name}.{name}"


# REQ: PY-API-018
def test_root_import_does_not_require_array_libraries(tmp_path):
    """The annotation surface must not turn array libraries into import dependencies."""

    # Key setup: Shadow optional array libraries with modules that fail if imported.
    blocker = tmp_path / "optional_dependency_blocker"
    blocker.mkdir()
    for name in ("cupy", "numpy", "torch"):
        (blocker / f"{name}.py").write_text(
            f'raise AssertionError("holoscan eagerly imported {name}")\n', encoding="utf-8"
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(blocker), str(Path(hs.__file__).resolve().parent.parent))
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import holoscan"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    # Key assertion: Importing Holoscan succeeds without importing any optional array library.
    assert completed.returncode == 0, completed.stderr


# REQ: SCHEMA-PY-004
# REQ: SCHEMA-PY-006
def test_root_import_publishes_core_schema_values_in_fresh_interpreter():
    # Key setup: Use only token-declared ports after importing the root package.
    script = textwrap.dedent(
        """
        import holoscan as hs

        facade_value = hs.schema.Time(time_ns=13, acqtime_ns=11)
        assert (facade_value.time_ns, facade_value.acqtime_ns) == (13, 11)

        class Source(hs.Operator):
            out = hs.Output("out", hs.payload.schema.Time)

            def setup(self, spec):
                spec.output(self.out)

            def contract(self):
                return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

            def compute(self, context):
                self.out.emit({"time_ns": 13, "acqtime_ns": 11})

        class Sink(hs.Operator):
            inp = hs.Input("in", hs.payload.schema.Time)
            received = None

            def setup(self, spec):
                spec.input(self.inp)

            def contract(self):
                return hs.Contract().trigger(hs.OnEach(self.inp))

            def compute(self, context):
                type(self).received = self.inp.receive_data()

        graph = hs.Graph("fresh-token-schema", default_clock=hs.RealtimeClock)
        source = graph.op(Source, "source")
        sink = graph.op(Sink, "sink")
        graph.add_flow(source.out, sink["in"])
        plan = hs.compile(graph)
        assert plan.ok, [error.token for error in plan.errors]
        hs.run(plan)
        assert Sink.received == {"time_ns": 13, "acqtime_ns": 11}
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    # Key assertion: The root import installs the facade before token-only emit and receive.
    assert completed.returncode == 0, completed.stderr


# REQ: PY-API-013
# REQ: PY-API-024
def test_registry_exports_every_public_definition():
    # Key setup: Collect the public classes and functions that holoscan.registry itself defines.
    registry = import_module("holoscan.registry")
    defined = {
        name
        for name, value in vars(registry).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == registry.__name__
    }

    # Key assertion: The manifest-driven __all__ names every one of them, including the
    # constructor-handle metadata class.
    assert "OperatorConstructorHandleInfo" in defined
    assert defined <= set(registry.__all__)
