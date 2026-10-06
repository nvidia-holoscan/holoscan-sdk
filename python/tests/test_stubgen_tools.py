# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests: no Holoscan import, CUDA toolkit, or GPU is required."""

import importlib.util
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
        r'(?:HoloscanStubGeneration\.cmake|stubgen/run_stubgen\.py)".*?\n\)',
        (public / "CMakeLists.txt").read_text(),
        re.DOTALL,
    )
    assert len(rules) == 2
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
    # A narrow config fixture uses the exact include exported by the full SDK config.
    config_text = (public / "cmake/templates/holoscan-config.cmake.in").read_text()
    include = 'include("${CMAKE_CURRENT_LIST_DIR}/HoloscanStubGeneration.cmake")'
    assert include in config_text
    (module.parent / "holoscan-config.cmake").write_text(include + "\n")
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
        f'find_package(holoscan REQUIRED CONFIG PATHS "{installed_stubgen}" NO_DEFAULT_PATH)\n'
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
    subprocess.run(["cmake", "--build", str(build)], check=True)
    output = build / "fixture.pyi"
    assert output.read_text() == "generated"
    timestamp = output.stat().st_mtime_ns
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert output.stat().st_mtime_ns == timestamp
    # A changed loader policy is a file dependency, including with Make generators.
    time.sleep(1.05)  # Apple Make compares file timestamps at second granularity.
    subprocess.run([*configure, "-DHOLOSCAN_PYTHON_STUBGEN_CUDA_STUB_LIBRARY=/unused"], check=True)
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert output.stat().st_mtime_ns != timestamp
    timestamp = output.stat().st_mtime_ns
    time.sleep(1.05)
    (source / "extension.so").touch()
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert output.stat().st_mtime_ns != timestamp


def test_scikit_build_consumer_wheel(tmp_path, installed_stubgen):
    pytest.importorskip("scikit_build_core")
    source, _ = make_consumer(tmp_path, installed_stubgen)
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["scikit-build-core==1.0.3"]\n'
        'build-backend = "scikit_build_core.build"\n'
        '[project]\nname = "stubgen-fixture"\nversion = "0.0.1"\n'
        "[tool.scikit-build]\nwheel.packages = []\n"
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
        assert not any("libcuda" in name or "run_stubgen" in name for name in wheel.namelist())


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
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert "def camera_id() -> int:" in (build / "fixture.pyi").read_text()


def test_generic_multiple_consumers(tmp_path, installed_stubgen):
    source = tmp_path / "generic source"
    build = tmp_path / "generic build"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(generic LANGUAGES NONE)\n"
        f'find_package(holoscan REQUIRED CONFIG PATHS "{installed_stubgen}" NO_DEFAULT_PATH)\n'
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
    subprocess.run(["cmake", "--build", str(build)], check=True)
    assert not (build / "first/result.pyi").exists()  # ALL is deliberately opt-in.
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "first_stubs", "second_stubs"], check=True
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
