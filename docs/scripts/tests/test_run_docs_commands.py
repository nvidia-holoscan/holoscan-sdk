#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

REPO_ROOT = Path(__file__).resolve().parents[4]
INTERNAL_RUN = REPO_ROOT / "public" / "run"
DOCS_HELPER = "docs/scripts/build_holoscan_docs.py"


class RunDocsCommandsTestCase(TestCase):
    def _runner_command(self, run_path: Path, *args: str) -> list[str]:
        return [shutil.which("bash") or "bash", str(run_path), *args]

    def _fake_python(self, directory: Path) -> tuple[Path, Path]:
        executable = directory / "fake-python"
        record = directory / "argv.json"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['FAKE_RECORD']).write_text("
            "json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            "raise SystemExit(int(os.environ.get('FAKE_EXIT', '0')))\n",
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable, record

    def _run(
        self,
        run_path: Path,
        fake_python: Path,
        record: Path,
        *args: str,
        exit_code: int = 0,
        cwd: Path | None = None,
        fern_token: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("FERN_TOKEN", None)
        environment.update(
            {
                "FAKE_RECORD": str(record),
                "FAKE_EXIT": str(exit_code),
                "HOLOSCAN_PY_EXE": str(fake_python),
            }
        )
        if fern_token is not None:
            environment["FERN_TOKEN"] = fern_token
        return subprocess.run(
            self._runner_command(run_path, *args),
            cwd=cwd or REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def _recorded_argv(self, record: Path) -> list[str]:
        return json.loads(record.read_text(encoding="utf-8"))

    def test_commands_are_listed_and_have_command_help(self) -> None:
        usage = subprocess.run(
            self._runner_command(INTERNAL_RUN),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert usage.returncode == 0, usage.stderr
        assert "Documentation" in usage.stdout + usage.stderr
        assert "build_docs" in usage.stdout + usage.stderr
        assert "live_docs" in usage.stdout + usage.stderr

        for command in ("build_docs", "live_docs"):
            help_result = subprocess.run(
                self._runner_command(INTERNAL_RUN, command, "--help"),
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert help_result.returncode == 0, help_result.stderr
            assert DOCS_HELPER in help_result.stdout + help_result.stderr

    def test_internal_layout_forwards_build_arguments_and_status(self) -> None:
        with TemporaryDirectory() as tmpdir:
            fake_python, record = self._fake_python(Path(tmpdir))
            result = self._run(
                INTERNAL_RUN,
                fake_python,
                record,
                "build_docs",
                "--container-name",
                "docs with spaces",
                "--skip-library-mdx",
                exit_code=37,
            )

            assert result.returncode == 37, result.stderr
            assert self._recorded_argv(record) == [
                str(REPO_ROOT / "public" / DOCS_HELPER),
                "--container-name",
                "docs with spaces",
                "--skip-library-mdx",
            ]

    def test_internal_tokenless_build_defaults_to_skip_library_mdx(self) -> None:
        with TemporaryDirectory() as tmpdir:
            fake_python, record = self._fake_python(Path(tmpdir))
            result = self._run(INTERNAL_RUN, fake_python, record, "build_docs")

            assert result.returncode == 0, result.stderr
            assert self._recorded_argv(record) == [
                str(REPO_ROOT / "public" / DOCS_HELPER),
                "--skip-library-mdx",
            ]

    def test_internal_tokenless_live_preview_adds_preview_and_skip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            fake_python, record = self._fake_python(Path(tmpdir))
            result = self._run(
                INTERNAL_RUN,
                fake_python,
                record,
                "live_docs",
                exit_code=23,
            )

            assert result.returncode == 23, result.stderr
            assert self._recorded_argv(record) == [
                str(REPO_ROOT / "public" / DOCS_HELPER),
                "--preview",
                "--skip-library-mdx",
            ]

    def test_token_preserves_original_zero_argument_mapping(self) -> None:
        with TemporaryDirectory() as tmpdir:
            fake_python, record = self._fake_python(Path(tmpdir))

            build_result = self._run(
                INTERNAL_RUN,
                fake_python,
                record,
                "build_docs",
                fern_token="organization-token",
            )
            assert build_result.returncode == 0, build_result.stderr
            assert self._recorded_argv(record) == [str(REPO_ROOT / "public" / DOCS_HELPER)]

            live_result = self._run(
                INTERNAL_RUN,
                fake_python,
                record,
                "live_docs",
                fern_token="organization-token",
            )
            assert live_result.returncode == 0, live_result.stderr
            assert self._recorded_argv(record) == [
                str(REPO_ROOT / "public" / DOCS_HELPER),
                "--preview",
            ]

    def test_public_layout_resolves_helper_relative_to_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            mirror = Path(tmpdir) / "public mirror with spaces"
            (mirror / "docs" / "scripts").mkdir(parents=True)
            mirror_run = mirror / "run"
            shutil.copy2(INTERNAL_RUN, mirror_run)
            mirror_run.chmod(mirror_run.stat().st_mode | stat.S_IXUSR)
            helper = mirror / DOCS_HELPER
            helper.touch()

            fake_python, record = self._fake_python(Path(tmpdir))
            result = self._run(
                mirror_run,
                fake_python,
                record,
                "live_docs",
                "--preview-id",
                "branch with spaces",
                cwd=mirror,
            )

            assert result.returncode == 0, result.stderr
            recorded_argv = self._recorded_argv(record)
            assert recorded_argv == [
                str(helper.resolve()),
                "--preview",
                "--preview-id",
                "branch with spaces",
            ], recorded_argv

            tokenless_result = self._run(
                mirror_run,
                fake_python,
                record,
                "build_docs",
                cwd=mirror,
                fern_token="",
            )
            assert tokenless_result.returncode == 0, tokenless_result.stderr
            assert self._recorded_argv(record) == [
                str(helper.resolve()),
                "--skip-library-mdx",
            ]


if __name__ == "__main__":
    main()
