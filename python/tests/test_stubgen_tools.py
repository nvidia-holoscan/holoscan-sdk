# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests: no Holoscan import, CUDA toolkit, or GPU is required."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_tool(name):
    path = Path(__file__).parents[1] / f"{name}.py"
    if name == "run_stubgen":
        path = Path(__file__).parents[2] / "cmake/modules/stubgen/run_stubgen.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_tool("run_stubgen")


def write_elf(path, machine=62):
    header = bytearray(20)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = machine.to_bytes(2, "little")
    path.write_bytes(header)
    return path


@pytest.fixture
def environment(tmp_path, monkeypatch):
    python = write_elf(tmp_path / "python")
    extension = write_elf(tmp_path / "extension.so")
    stub = write_elf(tmp_path / "libcuda.so")
    monkeypatch.setattr(runner.sys, "executable", str(python))
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "missing_libraries", lambda *_args: {"libcuda.so.1"})
    return dict(
        mode="auto", extensions=[extension], stub_library=stub, scratch_dir=tmp_path / "scratch"
    )


def test_fallback_is_scoped_and_cleaned(environment, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    original = os.environ.copy()
    directories = []

    def child(command, *, env, check):
        assert check is False
        directory, previous = env["LD_LIBRARY_PATH"].split(":")
        directories.append(Path(directory))
        assert previous == "/existing/lib"
        assert (Path(directory) / "libcuda.so.1").resolve() == environment["stub_library"]
        assert command == ["generator", "argument with spaces"]
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", child)
    assert runner.run_generator(["generator", "argument with spaces"], **environment) == 7
    assert os.environ == original
    assert not directories[0].exists()


def test_cleanup_on_launch_error(environment, monkeypatch):
    def child(*_args, **_kwargs):
        raise OSError("cannot launch")

    monkeypatch.setattr(runner.subprocess, "run", child)
    with pytest.raises(OSError, match="cannot launch"):
        runner.run_generator(["generator"], **environment)
    assert not list(environment["scratch_dir"].iterdir())


@pytest.mark.parametrize("mode", ["native", "auto"])
def test_native_environment_needs_no_toolkit(environment, monkeypatch, mode):
    environment.update(mode=mode, stub_library=None)
    monkeypatch.setattr(runner, "missing_libraries", lambda *_args: set())

    def child(*_args, env, **_kwargs):
        assert env == dict(os.environ)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    assert runner.run_generator(["generator"], **environment) == 0
    assert not environment["scratch_dir"].exists()


def test_explicit_stub_mode_even_when_driver_resolves(environment, monkeypatch):
    environment["mode"] = "stub"
    monkeypatch.setattr(runner, "missing_libraries", lambda *_args: set())

    def child(*_args, env, **_kwargs):
        assert Path(env["LD_LIBRARY_PATH"].split(":")[0], "libcuda.so.1").is_symlink()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    assert runner.run_generator(["generator"], **environment) == 0


@pytest.mark.parametrize("key", ["stub_library", "extension"])
def test_rejects_wrong_architecture(environment, key):
    path = environment["stub_library"] if key == "stub_library" else environment["extensions"][0]
    write_elf(path, machine=183)
    with pytest.raises(ValueError, match="architecture"):
        runner.run_generator(["must-not-run"], **environment)


def test_rejects_unrelated_missing_library(environment, monkeypatch):
    monkeypatch.setattr(runner, "missing_libraries", lambda *_args: {"libcuda.so.1", "libother.so"})
    with pytest.raises(ValueError, match="non-driver libraries: libother.so"):
        runner.run_generator(["must-not-run"], **environment)


def test_missing_toolkit_fails(environment):
    environment["stub_library"] = None
    with pytest.raises(ValueError, match="stub-library path"):
        runner.run_generator(["must-not-run"], **environment)


def test_parallel_generators_get_private_directories(environment, monkeypatch):
    directories = []

    def child(*_args, env, **_kwargs):
        directory = Path(env["LD_LIBRARY_PATH"].split(":")[0])
        directories.append(directory)
        assert (directory / "libcuda.so.1").exists()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: runner.run_generator(["generator"], **environment), range(8))
        )
    assert results == [0] * 8
    assert len(set(directories)) == 8
    assert all(not directory.exists() for directory in directories)


def test_ldd_parser(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="libcuda.so.1 => not found\nlibc.so.6 => /lib/libc.so.6 (0x1234)\n"
        ),
    )
    assert runner.missing_libraries(tmp_path / "extension.so", {}) == {"libcuda.so.1"}


def test_cli_preserves_child_exit_status(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            runner.__file__,
            "--mode",
            "native",
            "--scratch-dir",
            str(tmp_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(19)",
        ],
        check=False,
    )
    assert result.returncode == 19


@pytest.fixture
def installed_stubgen(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("CMake is unavailable")
    public = Path(__file__).parents[2]
    source = tmp_path / "tooling producer"
    source.mkdir()
    # Exercise the SDK's actual install rules without configuring its native dependencies.
    rules = re.findall(
        r'install\(FILES "\$\{holoscan_SOURCE_DIR\}/cmake/modules/'
        r'(?:HoloscanStubGeneration\.cmake|HoloscanPythonWheel\.cmake|stubgen/run_stubgen\.py)".*?\n\)',
        (public / "CMakeLists.txt").read_text(),
        re.DOTALL,
    )
    assert len(rules) == 3
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(tooling LANGUAGES NONE)\n"
        f'set(holoscan_SOURCE_DIR "{public}")\nset(HOLOSCAN_INSTALL_LIB_DIR lib)\n'
        + "\n".join(rules)
    )
    build = tmp_path / "tooling build"
    prefix = tmp_path / "original prefix"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    subprocess.run(
        ["cmake", "--install", str(build), "--prefix", str(prefix), "--component", "holoscan-core"],
        check=True,
    )
    relocated = tmp_path / "relocated SDK"
    prefix.rename(relocated)
    module = relocated / "lib/cmake/holoscan/HoloscanStubGeneration.cmake"
    assert module.is_file()
    assert (module.parent / "stubgen/run_stubgen.py").is_file()
    assert (module.parent / "HoloscanPythonWheel.cmake").is_file()
    # A narrow config fixture uses the exact include exported by the full SDK config.
    config_text = (public / "cmake/templates/holoscan-config.cmake.in").read_text()
    includes = [
        f'include("${{CMAKE_CURRENT_LIST_DIR}}/{name}.cmake")'
        for name in ["HoloscanStubGeneration", "HoloscanPythonWheel"]
    ]
    assert all(include in config_text for include in includes)
    (module.parent / "holoscan-config.cmake").write_text("\n".join(includes) + "\n")
    return relocated


def make_consumer(tmp_path, installed_stubgen):
    source = tmp_path / "source with spaces"
    build = tmp_path / "build with spaces"
    source.mkdir()
    # A fake generator checks the command boundary without importing a native SDK.
    (source / "stubgen.py").write_text(
        "import os, pathlib, sys\n"
        "assert 'python/lib' in os.environ['PYTHONPATH']\n"
        "assert sys.argv[1:4] == ['-q', '-m', 'fixture']\n"
        "pathlib.Path(sys.argv[5]).write_text('generated')\n"
    )
    (source / "extension.so").touch()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(fixture LANGUAGES NONE)\n"
        "get_cmake_property(before_include VARIABLES)\n"
        f'find_package(holoscan PATHS "{installed_stubgen}" NO_DEFAULT_PATH REQUIRED)\n'
        # Include alone must not discover tools, define loader policy, or create state.
        "get_cmake_property(after_include VARIABLES)\n"
        "list(REMOVE_ITEM after_include ${before_include})\n"
        "foreach(variable IN LISTS after_include)\n"
        '  if(variable MATCHES "^(Python|CUDAToolkit|HOLOSCAN_PYTHON_STUBGEN)")\n'
        '    message(FATAL_ERROR "Including the module had side effects: ${variable}")\n'
        "  endif()\nendforeach()\n"
        "add_library(camera_python SHARED IMPORTED)\n"
        "set_target_properties(camera_python PROPERTIES\n"
        '  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/extension.so")\n'
        "if(NOT DEFINED TEST_MODE)\nset(TEST_MODE native)\nendif()\n"
        "holoscan_add_nanobind_stubs(TARGET generate-fixture MODULE fixture ALL\n"
        f'  PYTHON_EXECUTABLE "{sys.executable}"\n'
        '  CUDA_MODE "${TEST_MODE}"\n'
        '  GENERATOR_SCRIPT "${CMAKE_SOURCE_DIR}/stubgen.py"\n'
        '  PYTHON_PATH "${CMAKE_BINARY_DIR}/python/lib"\n'
        '  OUTPUT "${CMAKE_BINARY_DIR}/fixture.pyi" EXTENSION_TARGETS camera_python)\n'
        'install(FILES "${CMAKE_BINARY_DIR}/fixture.pyi" DESTINATION fixture)\n'
    )
    return source, build


@pytest.mark.parametrize("generator", ["Unix Makefiles", "Ninja"])
def test_cmake_adapter_build_and_incremental_rebuild(tmp_path, installed_stubgen, generator):
    if generator == "Ninja" and shutil.which("ninja") is None:
        pytest.skip("Ninja is unavailable")
    source, build = make_consumer(tmp_path, installed_stubgen)
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        generator,
    ]
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    output = build / "fixture.pyi"
    assert output.read_text() == "generated"
    timestamp = output.stat().st_mtime_ns
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    assert output.stat().st_mtime_ns == timestamp
    # A changed loader policy is a file dependency, including with Make generators.
    time.sleep(1.05)  # Apple Make compares file timestamps at second granularity.
    subprocess.run([*configure, "-DHOLOSCAN_PYTHON_STUBGEN_CUDA_STUB_LIBRARY=/unused"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    assert output.stat().st_mtime_ns != timestamp
    timestamp = output.stat().st_mtime_ns
    time.sleep(1.05)
    (source / "extension.so").touch()
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    assert output.stat().st_mtime_ns != timestamp


def test_scikit_build_consumer_wheel(tmp_path, installed_stubgen, monkeypatch):
    pytest.importorskip("scikit_build_core")
    source, _ = make_consumer(tmp_path, installed_stubgen)
    # Both helpers come from the relocated SDK install. The inert extension and
    # patchelf recorder exercise packaging rules without requiring native Linux binaries.
    config = source / "CMakeLists.txt"
    config.write_text(
        config.read_text().replace(
            'install(FILES "${CMAKE_BINARY_DIR}/fixture.pyi" DESTINATION fixture)',
            "holoscan_add_python_wheel_component(MODULE fixture PACKAGE fixture\n"
            '  PYTHON_DIRECTORY "${CMAKE_BINARY_DIR}" PYTHON_FILES fixture.pyi\n'
            "  EXTENSION_TARGET camera_python)",
        )
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    patchelf = tools / "patchelf"
    log = tmp_path / "patchelf.json"
    patchelf.write_text(
        f"#!{sys.executable}\nimport json, pathlib, sys\n"
        f"pathlib.Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    patchelf.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["scikit-build-core==1.0.3"]\n'
        'build-backend = "scikit_build_core.build"\n'
        '[project]\nname = "stubgen-fixture"\nversion = "0.0.1"\n'
        "[tool.scikit-build]\nwheel.packages = []\n"
        'install.components = ["holoscan-wheel-fixture"]\n'
    )
    destination = tmp_path / "wheels"
    destination.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from scikit_build_core.build import build_wheel; import sys; build_wheel(sys.argv[1])",
            str(destination),
        ],
        cwd=source,
        check=True,
    )
    wheels = list(destination.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert wheel.read("fixture/fixture.pyi") == b"generated"
        assert "fixture/extension.so" in wheel.namelist()
        assert not any(
            "libcuda" in name or "run_stubgen" in name or name.endswith(".cmake")
            for name in wheel.namelist()
        )
    assert json.loads(log.read_text())[:2] == ["--set-rpath", "$ORIGIN/lib"]


def test_nanobind_discovery_uses_selected_python(tmp_path, installed_stubgen):
    pytest.importorskip("nanobind")
    source, build = make_consumer(tmp_path, installed_stubgen)
    config = source / "CMakeLists.txt"
    config.write_text(
        config.read_text().replace('  GENERATOR_SCRIPT "${CMAKE_SOURCE_DIR}/stubgen.py"\n', "")
    )
    package = build / "python/lib"
    package.mkdir(parents=True)
    (package / "fixture.py").write_text("def camera_id() -> int:\n    return 7\n")
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    assert "def camera_id() -> int:" in (build / "fixture.pyi").read_text()


def test_generic_multiple_consumers(tmp_path, installed_stubgen):
    source = tmp_path / "generic source"
    build = tmp_path / "generic build"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(generic LANGUAGES NONE)\n"
        f'find_package(holoscan PATHS "{installed_stubgen}" NO_DEFAULT_PATH REQUIRED)\n'
        "add_subdirectory(first)\nadd_subdirectory(second)\n"
    )
    for name in ("first", "second"):
        directory = source / name
        directory.mkdir()
        (directory / "extension.so").touch()
        (directory / "generate.py").write_text(
            "import os, pathlib, sys\n"
            "assert pathlib.Path.cwd().name == 'working directory'\n"
            f"assert os.environ['STUBGEN_FIXTURE'] == '{name}'\n"
            "pathlib.Path(sys.argv[1]).write_text(os.environ['STUBGEN_FIXTURE'])\n"
        )
        (directory / "CMakeLists.txt").write_text(
            f"add_library({name}_python MODULE IMPORTED)\n"
            f"set_target_properties({name}_python PROPERTIES\n"
            '  IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/extension.so")\n'
            f"holoscan_add_stub_generation(TARGET {name}_stubs OUTPUTS result.pyi\n"
            f'  PYTHON_EXECUTABLE "{sys.executable}"\n'
            f"  EXTENSION_TARGETS {name}_python CUDA_MODE native\n"
            '  WORKING_DIRECTORY "working directory"\n'
            f"  ENVIRONMENT STUBGEN_FIXTURE={name}\n"
            '  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/generate.py"\n'
            f'  COMMAND "{sys.executable}" "${{CMAKE_CURRENT_SOURCE_DIR}}/generate.py"\n'
            '    "${CMAKE_CURRENT_BINARY_DIR}/result.pyi")\n'
        )
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    assert not (build / "first/result.pyi").exists()  # ALL is deliberately opt-in.
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            "4",
            "--target",
            "first_stubs",
            "second_stubs",
        ],
        check=True,
    )
    for name in ("first", "second"):
        assert (build / name / "result.pyi").read_text() == name
        assert (build / name / "holoscan-stubgen" / f"{name}_stubs/policy.txt").is_file()


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [
        (
            ("EXTENSION_TARGETS camera_python", "EXTENSION_TARGETS missing"),
            "Unknown stub generation",
        ),
        (("set(TEST_MODE native)", "set(TEST_MODE invalid)"), "Invalid stub generation CUDA_MODE"),
    ],
)
def test_consumer_argument_validation(tmp_path, installed_stubgen, replacement, expected):
    source, build = make_consumer(tmp_path, installed_stubgen)
    config = source / "CMakeLists.txt"
    config.write_text(config.read_text().replace(*replacement))
    result = subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build)], check=False, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert expected in result.stderr


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("cc") is None,
    reason="ELF loader integration requires Linux and a C compiler",
)
@pytest.mark.parametrize("mode", ["auto", "stub"])
def test_real_elf_loader_with_synthetic_driver(tmp_path, monkeypatch, mode):
    # This is a loader fixture, not an implementation of the CUDA API.
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    source = tmp_path / "driver.c"
    source.write_text("int stubgen_fixture(void) { return 34; }\n")
    stub = tmp_path / "libcuda.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), "-Wl,-soname,libcuda.so.1", "-o", str(stub)],
        check=True,
    )
    source = tmp_path / "extension.c"
    source.write_text(
        "extern int stubgen_fixture(void);\nint probe(void) { return stubgen_fixture(); }\n"
    )
    extension = tmp_path / "extension.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), f"-L{tmp_path}", "-lcuda", "-o", str(extension)],
        check=True,
    )
    if mode == "auto" and "libcuda.so.1" not in runner.missing_libraries(
        extension, dict(os.environ)
    ):
        pytest.skip("auto fallback requires a driverless host")
    status = runner.run_generator(
        [
            sys.executable,
            "-c",
            "import ctypes, sys; assert ctypes.CDLL(sys.argv[1]).probe() == 34",
            str(extension),
        ],
        mode=mode,
        extensions=[extension],
        stub_library=stub,
        scratch_dir=tmp_path / "scratch",
    )
    assert status == 0
    assert not list((tmp_path / "scratch").iterdir())


def make_operator_stub_consumer(tmp_path, installed_stubgen, *, call_extra=""):
    """Configure a consumer of holoscan_add_operator_stubs with inert tools and extension."""

    source = tmp_path / "operator source"
    build = tmp_path / "operator build"
    source.mkdir()
    (source / "extension.so").touch()
    (source / "header.txt").write_text("# Copyright (c) Example Corp.\n")
    (source / "committed.pyi").write_text("committed\n")
    package = source / "python root/holoscan"
    (package / "_tools").mkdir(parents=True)
    # Locating the tools must never import holoscan itself.
    (package / "__init__.py").write_text("raise ImportError('holoscan must not be imported')\n")
    # The generator records its arguments and the holoscan package its import path selects.
    (package / "_tools/generate_operator_stubs.py").write_text(
        "import importlib.util, json, pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "output = pathlib.Path(arguments[arguments.index('--output') + 1])\n"
        "holoscan = getattr(importlib.util.find_spec('holoscan'), 'origin', None)\n"
        "output.write_text(json.dumps({'arguments': arguments, 'holoscan': holoscan}))\n"
    )
    (package / "_tools/check_stub_coverage.py").write_text(
        "import pathlib, sys\n"
        "arguments = sys.argv[1:]\n"
        "assert '--match-declarations' in arguments\n"
        "committed = pathlib.Path(arguments[arguments.index('--committed') + 1])\n"
        "raise SystemExit(0 if committed.read_text() == 'committed\\n' else 3)\n"
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(provider LANGUAGES NONE)\n"
        f'find_package(holoscan PATHS "{installed_stubgen}" NO_DEFAULT_PATH REQUIRED)\n'
        "add_library(provider_python MODULE IMPORTED)\n"
        "set_target_properties(provider_python PROPERTIES\n"
        '  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/extension.so")\n'
        "holoscan_add_operator_stubs(TARGET provider-stubs\n"
        '  OUTPUT "${CMAKE_BINARY_DIR}/generated/operators.pyi"\n'
        "  PROVIDER_MODULES provider_camera provider_extras\n"
        "  REGISTRY_PREFIX holoscan.camera.\n"
        '  MODULE_DOC "Registered Camera Operator proxies."\n'
        '  HEADER_FILE "header.txt"\n'
        "  HANDLE_ANNOTATIONS holoscan.camera.Service=provider_camera.Service\n"
        f'  PYTHON_EXECUTABLE "{sys.executable}"\n'
        '  PYTHON_PATH "${CMAKE_SOURCE_DIR}/python root"\n'
        "  EXTENSION_TARGETS provider_python CUDA_MODE native\n"
        '  COMMITTED_STUB "committed.pyi" CHECK_TARGET check-provider-stubs\n'
        f"{call_extra})\n"
    )
    return source, build


# REQ: PY-API-032
def test_operator_stubs_use_the_tools_beside_the_selected_holoscan(tmp_path, installed_stubgen):
    source, build = make_operator_stub_consumer(tmp_path, installed_stubgen)
    subprocess.run(["cmake", "-S", str(source), "-B", str(build)], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "check-provider-stubs"], check=True)

    # Key assertion: The generator receives every provider setting in one stable command.
    recorded = json.loads((build / "generated/operators.pyi").read_text())
    assert recorded["holoscan"] == str(source / "python root/holoscan/__init__.py")
    assert recorded["arguments"] == [
        "--output",
        str(build / "generated/operators.pyi"),
        "--registry-prefix",
        "holoscan.camera.",
        "--provider-module",
        "provider_camera",
        "--provider-module",
        "provider_extras",
        "--handle-annotation",
        "holoscan.camera.Service=provider_camera.Service",
        "--module-doc",
        "Registered Camera Operator proxies.",
        "--header-file",
        str(source / "header.txt"),
    ]

    # Key assertion: The check target fails when the committed stub drifts.
    (source / "committed.pyi").write_text("drifted\n")
    result = subprocess.run(
        ["cmake", "--build", str(build), "--target", "check-provider-stubs"], check=False
    )
    assert result.returncode != 0


# REQ: PY-API-032
@pytest.mark.parametrize("source_of_holoscan", ["installed SDK", "inherited PYTHONPATH"])
def test_operator_stubs_generate_with_the_path_that_located_the_tools(
    tmp_path, installed_stubgen, source_of_holoscan
):
    # Key setup: Omit the optional PYTHON_PATH. A second SDK on the inherited PYTHONPATH stands in
    # for another holoscan installed in the selected interpreter.
    source, build = make_operator_stub_consumer(tmp_path, installed_stubgen)
    config = source / "CMakeLists.txt"
    config.write_text(
        config.read_text().replace('  PYTHON_PATH "${CMAKE_SOURCE_DIR}/python root"\n', "")
    )
    selected = source / "python root"
    other = tmp_path / "other SDK"
    (other / "holoscan").mkdir(parents=True)
    (other / "holoscan/__init__.py").write_text("raise ImportError('other SDK')\n")
    configure = ["cmake", "-S", str(source), "-B", str(build)]
    environment = os.environ.copy()
    if source_of_holoscan == "installed SDK":
        configure.append(f"-Dholoscan_PYTHON_PACKAGE_DIR={selected}")
        environment["PYTHONPATH"] = str(other)
    else:
        environment["PYTHONPATH"] = str(selected)
    subprocess.run(configure, check=True, env=environment)
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "provider-stubs"], check=True, env=environment
    )

    # Key assertion: The generator imports the holoscan package beside the tools it runs.
    recorded = json.loads((build / "generated/operators.pyi").read_text())
    assert recorded["holoscan"] == str(selected / "holoscan/__init__.py")


# REQ: PY-API-032
def test_operator_stubs_prefer_an_explicit_tools_directory(tmp_path, installed_stubgen):
    source, build = make_operator_stub_consumer(tmp_path, installed_stubgen)
    explicit = tmp_path / "explicit tools"
    shutil.copytree(source / "python root/holoscan/_tools", explicit)
    shutil.rmtree(source / "python root")
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), f"-DHOLOSCAN_PYTHON_TOOLS_DIR={explicit}"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--target", "provider-stubs"], check=True)
    assert (build / "generated/operators.pyi").is_file()


# REQ: PY-API-032
@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"tools_dir": "absent tools"}, "does not contain the holoscan typing tools"),
        ({"call_extra": "  HANDLE_ANNOTATIONS missing-separator\n"}, "TYPE_NAME=MODULE.CLASS"),
        ({"call_extra": "  HANDLE_ANNOTATIONS Foo=bar\n"}, "TYPE_NAME=MODULE.CLASS"),
        ({"remove": "  REGISTRY_PREFIX holoscan.camera.\n"}, "requires REGISTRY_PREFIX"),
        ({"remove": " CHECK_TARGET check-provider-stubs"}, "COMMITTED_STUB and CHECK_TARGET"),
    ],
)
def test_operator_stub_argument_validation(tmp_path, installed_stubgen, change, expected):
    source, build = make_operator_stub_consumer(
        tmp_path,
        installed_stubgen,
        call_extra=change.get("call_extra", ""),
    )
    configure = ["cmake", "-S", str(source), "-B", str(build)]
    if "tools_dir" in change:
        configure.append(f"-DHOLOSCAN_PYTHON_TOOLS_DIR={tmp_path / change['tools_dir']}")
    if "remove" in change:
        config = source / "CMakeLists.txt"
        config.write_text(config.read_text().replace(change["remove"], ""))
    result = subprocess.run(configure, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr


GENERATED_OPERATOR_STUB = '''\
from typing import Final, Required, TypedDict, Unpack

__all__: list[str] = ["ResizeOp"]

class _ResizeOpParams(TypedDict, total=False):
    """Parameters."""

    max_batch_size: Required[int]
    interpolation: str
    __all__: int

class _ResizeOpType:
    def __call__(self, graph: object, **params: Unpack[_ResizeOpParams]) -> object: ...

ResizeOp: Final[_ResizeOpType]
'''


# REQ: PY-API-032
# REQ: PY-API-034
@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (("", "\ndef helper() -> None: ...\n"), None),
        (("    max_batch_size: Required[int]\n", ""), "_ResizeOpParams.max_batch_size"),
        (
            ("max_batch_size: Required[int]", "max_batch_size: int"),
            "_ResizeOpParams.max_batch_size",
        ),
        (("    __all__: int\n", ""), "_ResizeOpParams.__all__"),
        (("__all__: int", "__all__: str"), "_ResizeOpParams.__all__"),
        (("graph: object,", "graph: object, extra: int,"), "_ResizeOpType.__call__"),
        (('"""Parameters."""', '"""Edited."""'), "_ResizeOpParams"),
        (("Final[_ResizeOpType]", "_ResizeOpType"), "ResizeOp"),
        # A committed stub may export more Operators than one build configuration registers.
        (('["ResizeOp"]', '["ResizeOp", "WarpOp"]'), None),
        (('["ResizeOp"]', "[]"), "__all__"),
    ],
)
def test_operator_stub_check_compares_private_declarations(tmp_path, change, expected):
    # Key setup: Edit a committed copy of a generated stub; only public names stay unchanged.
    generated = tmp_path / "generated.pyi"
    committed = tmp_path / "committed.pyi"
    generated.write_text(GENERATED_OPERATOR_STUB)
    old, new = change
    committed.write_text(
        GENERATED_OPERATOR_STUB.replace(old, new) if old else GENERATED_OPERATOR_STUB + new
    )
    checker = Path(__file__).parents[1] / "holoscan/_tools/check_stub_coverage.py"

    def check(*options):
        return subprocess.run(
            [
                sys.executable,
                str(checker),
                "--generated",
                str(generated),
                "--committed",
                str(committed),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    # Key assertion: The default refined-stub check sees only public names and members, while
    # declaration matching rejects any change to a generated declaration and allows additions.
    assert check().returncode == 0, check().stderr
    result = check("--match-declarations")
    if expected is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert f"missing or changed declarations: {expected}" in result.stderr


@pytest.mark.parametrize("generator", ["Unix Makefiles", "Ninja"])
@pytest.mark.parametrize("adapter", ["nanobind", "operator"])
def test_sibling_provider_waits_for_core_schema_imports(
    tmp_path, installed_stubgen, generator, adapter
):
    if generator == "Ninja" and shutil.which("ninja") is None:
        pytest.skip("Ninja is unavailable")
    public = Path(__file__).parents[2]
    sdk_python = (public / "python/holoscan/CMakeLists.txt").read_text()
    # Reuse the production collection (including schema codec targets) and registration.
    collect_start = sdk_python.index("get_property(HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES")
    collect_end = sdk_python.index("unset(_holoscan_schema_import_target)", collect_start)
    collection = sdk_python[collect_start:collect_end]
    registration = re.search(
        r"set_property\(TARGET holoscan_core_python PROPERTY\s+"
        r"HOLOSCAN_PYTHON_IMPORT_DEPENDENCIES.*?\)",
        sdk_python,
        re.DOTALL,
    )
    assert registration is not None
    source = tmp_path / "source"
    sdk = source / "sdk"
    provider = source / "provider"
    sdk.mkdir(parents=True)
    provider.mkdir()
    (sdk / "core.so").touch()
    (provider / "provider.so").touch()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(sibling LANGUAGES NONE)\n"
        f'find_package(holoscan PATHS "{installed_stubgen}" NO_DEFAULT_PATH REQUIRED)\n'
        "add_subdirectory(sdk)\nadd_subdirectory(provider)\n"
    )
    prerequisites = {
        "holoscan-python-init": "init.py",
        "schema-python-install": "schema.py",
        "schema-codec": "codec.so",
    }
    rules = [
        "add_library(holoscan_core_python MODULE IMPORTED GLOBAL)\n",
        "set_target_properties(holoscan_core_python PROPERTIES IMPORTED_LOCATION "
        '"${CMAKE_CURRENT_SOURCE_DIR}/core.so")\n',
        "add_custom_target(holoscan-python)\n",
    ]
    for target, filename in prerequisites.items():
        rules.append(
            f'add_custom_command(OUTPUT "${{CMAKE_CURRENT_BINARY_DIR}}/{filename}"\n'
            f'  COMMAND "${{CMAKE_COMMAND}}" -E touch "${{CMAKE_CURRENT_BINARY_DIR}}/{filename}")\n'
            f'add_custom_target({target} DEPENDS "${{CMAKE_CURRENT_BINARY_DIR}}/{filename}")\n'
        )
    rules.extend(
        [
            "set_property(GLOBAL PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS "
            "schema-python-install)\n",
            "set_property(TARGET schema-python-install PROPERTY "
            "HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET schema-codec)\n",
            collection,
            registration.group(0),
            # Match the SDK's cleanup before a later sibling module is configured.
            "\nunset(HOLOSCAN_SDK_STUBGEN_DEPENDS)\n",
            "unset(HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES)\n",
        ]
    )
    (sdk / "CMakeLists.txt").write_text("".join(rules))
    tools = provider / "tools"
    tools.mkdir()
    (tools / "generate_operator_stubs.py").write_text(
        "import pathlib, sys\n"
        "flag = '-o' if '-o' in sys.argv else '--output'\n"
        "output = pathlib.Path(sys.argv[sys.argv.index(flag) + 1])\n"
        f"files = {list(prerequisites.values())!r}\n"
        "for name in files:\n"
        "    assert (output.parent.parent / 'sdk' / name).is_file(), name\n"
        "output.write_text('generated after schema imports')\n"
    )
    arguments = (
        "MODULE provider GENERATOR_SCRIPT "
        '"${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_operator_stubs.py"'
        if adapter == "nanobind"
        else "PROVIDER_MODULES provider REGISTRY_PREFIX holoscan.provider."
    )
    command = (
        "holoscan_add_nanobind_stubs" if adapter == "nanobind" else "holoscan_add_operator_stubs"
    )
    (provider / "CMakeLists.txt").write_text(
        "if(DEFINED HOLOSCAN_SDK_STUBGEN_DEPENDS OR "
        "DEFINED HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES)\n"
        '  message(FATAL_ERROR "SDK directory-local variables leaked into the consumer")\nendif()\n'
        "add_library(provider_python MODULE IMPORTED)\n"
        "set_target_properties(provider_python PROPERTIES IMPORTED_LOCATION "
        '"${CMAKE_CURRENT_SOURCE_DIR}/provider.so")\n'
        'set(HOLOSCAN_PYTHON_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tools")\n'
        f"{command}(TARGET provider-stubs {arguments}\n"
        '  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/provider.pyi"\n'
        f'  PYTHON_EXECUTABLE "{sys.executable}" CUDA_MODE native\n'
        "  EXTENSION_TARGETS provider_python holoscan_core_python)\n"
    )
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", generator], check=True)
    # Build the consumer alone: the SDK aggregate's ALL dependencies cannot mask missing edges.
    build_command = [
        "cmake",
        "--build",
        str(build),
        "--parallel",
        "8",
        "--target",
        "provider-stubs",
    ]
    subprocess.run(build_command, check=True)
    output = build / "provider/provider.pyi"
    assert output.read_text() == "generated after schema imports"
    timestamp = output.stat().st_mtime_ns
    subprocess.run(build_command, check=True)
    assert output.stat().st_mtime_ns == timestamp
