#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the run script."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PUBLIC_ROOT / "run"
VERSION_FILE = PUBLIC_ROOT / "VERSION"
DEV_DOCKERFILE = PUBLIC_ROOT / "Dockerfile.dev"


def source_run_script(run_script: Path, command: str) -> subprocess.CompletedProcess[str]:
    """Source run_script and execute one of its functions."""
    return subprocess.run(
        ["bash", "-c", f'source "$1"\n{command}', str(run_script), str(run_script)],
        check=False,
        capture_output=True,
        text=True,
    )


def copy_source_archive(destination: Path) -> Path:
    """Create the minimal source-archive layout needed to exercise run."""
    destination.mkdir()
    shutil.copy2(RUN_SCRIPT, destination / "run")
    shutil.copy2(VERSION_FILE, destination / "VERSION")
    return destination / "run"


def run_dryrun_command(command: str, *extra_arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a CLI command without Docker or host-device dependencies."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        fake_bin = Path(temporary_directory)
        fake_commands = {
            "cat": """#!/bin/sh
if [ "${1-}" = "/etc/group" ]; then
    printf '%s' "${FAKE_ETC_GROUP-}"
    exit 0
fi
exec /bin/cat "$@"
""",
            "docker": """#!/bin/sh
if [ "${1-}" = "inspect" ]; then
    exit 0
fi
exit 99
""",
            "custom-docker": """#!/bin/sh
if [ "${1-}" = "inspect" ]; then
    printf '%s\n' custom-docker-inspect >&2
    exit 0
fi
exit 99
""",
            "find": "#!/bin/sh\nexit 0\n",
            "realpath": "#!/bin/sh\nprintf .\n",
        }
        for name, contents in fake_commands.items():
            executable = fake_bin / name
            executable.write_text(contents)
            executable.chmod(0o755)

        environment = os.environ.copy()
        environment.update(
            {
                "CMAKE_BUILD_PARALLEL_LEVEL": "2",
                "HOLOSCAN_BUILD_ARCH": "x86_64",
                "HOLOSCAN_USE_SCCACHE": "OFF",
                "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            }
        )
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)

        return subprocess.run(
            [
                shutil.which("bash") or "bash",
                str(RUN_SCRIPT),
                command,
                "--dryrun",
                "--standalone",
                *extra_arguments,
            ],
            cwd=PUBLIC_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )


class RunScriptIntegrationTestCase(unittest.TestCase):
    def dryrun_output(self, command: str, *extra_arguments: str) -> str:
        result = run_dryrun_command(command, *extra_arguments)
        output = result.stdout + result.stderr

        assert result.returncode == 0, output
        assert "[dryrun]" in output
        return output

    def assert_dryrun_command(self, command: str, *expected_fragments: str) -> None:
        output = self.dryrun_output(command)

        for fragment in expected_fragments:
            assert fragment in output

    def assert_archive_fallback(self, run_script: Path) -> None:
        version = VERSION_FILE.read_text().strip()

        result = source_run_script(
            run_script,
            "if is_git_repo; then printf true; else printf false; fi",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "false"

        result = source_run_script(run_script, "get_git_sha")
        assert result.returncode != 0

        result = source_run_script(run_script, "get_build_img_tag")
        assert result.returncode == 0, result.stderr
        assert result.stdout == version

        result = source_run_script(run_script, "get_image_tag_flags holoscan-test")
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"-t holoscan-test:latest -t holoscan-test:{version} "

    def test_git_checkout_uses_commit_sha(self) -> None:
        expected_sha = subprocess.run(
            ["git", "-C", str(PUBLIC_ROOT), "rev-parse", "--short=9", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        result = source_run_script(RUN_SCRIPT, "get_build_img_tag")

        assert result.returncode == 0, result.stderr
        assert result.stdout == expected_sha

    def test_image_tag_flags_deduplicate_repeated_values(self) -> None:
        result = source_run_script(
            RUN_SCRIPT,
            """
is_git_repo() { return 0; }
get_version() { printf duplicate; }
get_git_branch() { printf duplicate; }
get_git_tag() { printf duplicate; }
get_git_sha() { printf duplicate; }
get_image_tag_flags holoscan-test
""",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == "-t holoscan-test:latest -t holoscan-test:duplicate "

    def test_build_image_dryrun(self) -> None:
        self.assert_dryrun_command(
            "build_image",
            "docker build",
            "--platform linux/amd64",
            "holoscan-sdk-build-x86_64:latest",
        )

    def test_build_dev_image_dryrun(self) -> None:
        output = self.dryrun_output("build_dev_image")

        assert "--target pytorch-downloader" in output
        assert "-f " + str(DEV_DOCKERFILE) in output
        assert "holoscan-sdk-dev-x86_64:latest" in output
        assert "HOLOSCAN_RELEASE_STAGE" not in output

    def test_build_dev_image_builds_sdk_prerequisite_by_default(self) -> None:
        result = source_run_script(
            RUN_SCRIPT,
            """
ARCH=x86_64
DO_DRY_RUN=true
build() { printf 'sdk-prerequisite args=%s\\n' "$#"; }
build_dev_image --label=development-image
""",
        )
        output = result.stdout + result.stderr

        assert result.returncode == 0, output
        assert "sdk-prerequisite args=0" in output
        assert output.index("sdk-prerequisite") < output.index("--target pytorch-downloader")
        assert output.count("--label=development-image") == 2

    def test_build_dev_image_standalone_skips_sdk_prerequisite(self) -> None:
        result = source_run_script(
            RUN_SCRIPT,
            """
ARCH=x86_64
DO_DRY_RUN=true
DO_STANDALONE=true
build() { printf 'unexpected-sdk-prerequisite\\n'; return 1; }
build_dev_image
""",
        )
        output = result.stdout + result.stderr

        assert result.returncode == 0, output
        assert "unexpected-sdk-prerequisite" not in output
        assert "--target pytorch-downloader" in output

    def test_build_dev_image_standalone_requires_install_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            result = source_run_script(
                run_script,
                "ARCH=x86_64; DO_STANDALONE=true; build_dev_image",
            )
        output = result.stdout + result.stderr

        assert result.returncode != 0
        assert "SDK installation" in output
        assert "Run './public/run build' first" in output

    def test_source_archive_uses_the_version_tag_for_development_images(self) -> None:
        version = VERSION_FILE.read_text().strip()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            result = source_run_script(
                run_script,
                "ARCH=x86_64; DO_DRY_RUN=true; DO_STANDALONE=true; build_dev_image",
            )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert f"holoscan-sdk-build-x86_64:{version}-torch-wheels" in output

    def test_source_archive_launches_the_matching_development_image_tag(self) -> None:
        version = VERSION_FILE.read_text().strip()
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            result = source_run_script(
                run_script,
                "ARCH=x86_64; DO_DRY_RUN=true; DO_STANDALONE=true; launch_dev --run-cmd true",
            )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert f"holoscan-sdk-dev-x86_64:{version}" in output

    def test_launch_dev_builds_artifact_by_default(self) -> None:
        result = source_run_script(
            RUN_SCRIPT,
            """
ARCH=x86_64
DO_DRY_RUN=true
build_dev_image() { printf 'development-image-prerequisite\\n'; }
launch_dev --run-cmd true
""",
        )
        output = result.stdout + result.stderr

        assert result.returncode == 0, output
        assert output.count("development-image-prerequisite") == 1
        assert output.index("development-image-prerequisite") < output.index("docker run")

    def test_dev_dockerfile_uses_usr_contract_without_hololink(self) -> None:
        dockerfile = DEV_DOCKERFILE.read_text()

        assert "ARG INSTALL_PATH=/usr\n" in dockerfile
        assert "ARG SAMPLE_DATA_PATH=/usr/share/holoscan/data\n" in dockerfile
        assert "WORKDIR ${WORKSPACE_PATH}" in dockerfile
        assert "FROM release-with-holoscan-sdk AS release\n" in dockerfile
        assert "release-with-hololink" not in dockerfile
        assert "HOLOLINK_INSTALL_PATH" not in dockerfile

    def test_build_dev_image_rejects_removed_features(self) -> None:
        result = run_dryrun_command("build_dev_image", "--feature", "hololink")
        output = result.stdout + result.stderr

        assert result.returncode != 0
        assert "Development-container features are not supported" in output

    def test_launch_dev_rejects_removed_features(self) -> None:
        result = run_dryrun_command("launch_dev", "--feature", "hololink")
        output = result.stdout + result.stderr

        assert result.returncode != 0
        assert "Development-container features are not supported" in output

    def test_build_dryrun(self) -> None:
        self.assert_dryrun_command(
            "build",
            "docker run",
            "cmake --build build-x86_64 -j 2",
            "holoscan-sdk-build-x86_64:",
        )

    def test_build_selects_inference_python_install_component(self) -> None:
        enabled = self.dryrun_output("build")
        assert "if [ \"ON\" != 'OFF' ]; then" in enabled
        assert "install_component_if_present holoscan-inference-python_libs" in enabled

        disabled = self.dryrun_output("build", "--build-python", "false")
        assert "if [ \"OFF\" != 'OFF' ]; then" in disabled
        assert "install_component_if_present holoscan-inference-python_libs" in disabled

    def test_test_dryrun(self) -> None:
        self.assert_dryrun_command(
            "test",
            "docker run",
            "ctest .",
            "--timeout 480",
        )

    def test_launch_dryrun(self) -> None:
        self.assert_dryrun_command(
            "launch",
            "docker run",
            "-w /workspace/holoscan-sdk/./build-x86_64",
            "--runtime=nvidia",
            "            bash",
        )

    def test_launch_selects_runc_without_a_gpu_runtime_flag(self) -> None:
        for runtime_args in (("--runtime=runc",), ("--runtime", "runc")):
            with self.subTest(runtime_args=runtime_args):
                output = self.dryrun_output("launch", "..", *runtime_args)
                docker_command = next(
                    line for line in output.splitlines() if "[dryrun] $ docker run" in line
                )

                assert docker_command.count("--runtime=") == 1
                assert "--runtime=runc" in docker_command
                assert "--runtime=nvidia" not in docker_command
                assert "--gpus" not in docker_command

    def test_launch_rejects_an_empty_runtime(self) -> None:
        for runtime_args in (("--runtime=",), ("--runtime",)):
            with self.subTest(runtime_args=runtime_args):
                result = run_dryrun_command("launch", "..", *runtime_args)

                assert result.returncode != 0
                assert "--runtime requires a value" in result.stdout + result.stderr

    def test_launch_keeps_a_leading_flag_out_of_the_working_directory(self) -> None:
        output = self.dryrun_output("launch", "--cpus=8")

        assert "-w /workspace/holoscan-sdk/./build-x86_64" in output
        assert "--cpus=8" in output, "the flag must still reach docker run"
        assert "/./--cpus=8" not in output

    def test_launch_uses_a_leading_positional_working_directory(self) -> None:
        output = self.dryrun_output("launch", "install-x86_64")

        assert "-w /workspace/holoscan-sdk/./install-x86_64" in output

    def test_launch_preserves_whitespace_in_path_arguments(self) -> None:
        output = self.dryrun_output(
            "launch", "build  directory", "--mount-point", "/tmp/source  directory"
        )

        assert "-v /tmp/source  directory:/workspace/holoscan-sdk" in output
        assert "-w /workspace/holoscan-sdk/./build  directory" in output
        assert "HOLOSCAN_LIB_PATH=/workspace/holoscan-sdk/./build  directory/lib" in output
        assert 'export PATH="$PATH:/workspace/holoscan-sdk/./build  directory/bin";' in output

    def test_launch_omits_missing_optional_groups(self) -> None:
        output = self.dryrun_output("launch")

        assert "--group-add" not in output

    def test_launch_uses_configured_docker_executable_for_inspection(self) -> None:
        with mock.patch.dict(os.environ, {"HOLOSCAN_DOCKER_EXE": "custom-docker"}):
            output = self.dryrun_output("launch")

        assert "custom-docker-inspect" in output
        assert "custom-docker run" in output

    def test_pass_through_arguments_keep_their_quoting(self) -> None:
        # run_command echoes "$*", so the doubled space is the signal here: it survives only when
        # the argument reaches docker as one element. If quoting is lost the argument is split and
        # rejoined, collapsing the two spaces into one.
        for command in ("build_image", "build", "test", "launch", "bench"):
            with self.subTest(command=command):
                output = self.dryrun_output(command, "--label=a  b")

                assert "--label=a  b" in output

    def test_source_archive_outside_git_uses_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            self.assert_archive_fallback(run_script)

    def test_untracked_source_archive_inside_git_uses_version(self) -> None:
        with tempfile.TemporaryDirectory(dir=PUBLIC_ROOT) as temporary_directory:
            run_script = copy_source_archive(Path(temporary_directory) / "archive")
            self.assert_archive_fallback(run_script)


if __name__ == "__main__":
    unittest.main()
