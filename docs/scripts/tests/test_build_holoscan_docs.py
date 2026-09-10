#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import ANY, patch

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import build_holoscan_docs as docs_builder  # noqa: E402
from build_holoscan_docs import (  # noqa: E402
    _external_git_metadata_mounts,
    _needs_tokenless_fern_project,
    _tokenless_fern_project,
)


class ExternalGitMetadataMountsTestCase(TestCase):
    def test_skips_metadata_already_mounted_with_checkout(self) -> None:
        repo_root = Path("/workspace/holoscan-sdk")
        git_dir = repo_root / ".git"

        with patch(
            "build_holoscan_docs._git_metadata_paths",
            return_value=(git_dir, git_dir),
        ):
            assert _external_git_metadata_mounts(repo_root) == []

    def test_mounts_linked_worktree_common_directory_once_read_only(self) -> None:
        repo_root = Path("/tmp/holoscan-sdk-worktree")
        common_dir = Path("/src/holoscan-sdk/.git")
        git_dir = common_dir / "worktrees" / "holoscan-sdk-worktree"

        with patch(
            "build_holoscan_docs._git_metadata_paths",
            return_value=(git_dir, common_dir),
        ):
            assert _external_git_metadata_mounts(repo_root) == [f"{common_dir}:{common_dir}:ro"]

    def test_mounts_distinct_external_metadata_roots_deterministically(self) -> None:
        repo_root = Path("/workspace/holoscan-sdk")
        git_dir = Path("/git-admin/worktrees/holoscan-sdk")
        common_dir = Path("/git-common")

        with patch(
            "build_holoscan_docs._git_metadata_paths",
            return_value=(git_dir, common_dir),
        ):
            assert _external_git_metadata_mounts(repo_root) == [
                f"{common_dir}:{common_dir}:ro",
                f"{git_dir}:{git_dir}:ro",
            ]


class DockerHomeTestCase(TestCase):
    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            fern_dir=docs_builder.DOCS_ROOT / "fern",
            preview_url_file=None,
            no_clean=False,
            with_library_mdx=False,
            skip_library_mdx=True,
            skip_fern_check=False,
            skip_link_check=False,
            preview=True,
            publish=False,
            publish_preview=False,
            preview_id=None,
            delete_preview_id=None,
            force=False,
            container_name="docs-test",
            verbose=False,
            build_role="current",
            source_ref="test-ref",
            allow_rendered_source_links=False,
            allowed_source_browser_url=[],
            allowed_source_ref=[],
        )

    def _run_and_capture_home(
        self,
        *,
        interrupt: bool,
        args: SimpleNamespace | None = None,
        stdin_isatty: bool = False,
    ) -> tuple[Path, list[str]]:
        observed: dict[str, object] = {}

        def capture_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
            observed["command"] = cmd
            observed["home"] = Path(
                next(part.removeprefix("-eHOME=") for part in cmd if part.startswith("-eHOME="))
            )
            if interrupt:
                raise KeyboardInterrupt
            return SimpleNamespace(returncode=0)

        with (
            patch.object(docs_builder, "_run"),
            patch.object(docs_builder, "_external_git_metadata_mounts", return_value=[]),
            patch.object(docs_builder, "_mount_host_fern_auth") as mount_auth,
            patch.object(docs_builder.subprocess, "run", side_effect=capture_run),
            patch.object(
                docs_builder.sys,
                "stdin",
                SimpleNamespace(isatty=lambda: stdin_isatty),
            ),
            patch.dict(docs_builder.os.environ, {"FERN_TOKEN": ""}),
        ):
            if interrupt:
                # Fern CI runs this suite with stdlib unittest and no optional test packages.
                with self.assertRaises(KeyboardInterrupt):  # noqa: PT027
                    docs_builder._run_in_docker(args or self._args())
            else:
                assert docs_builder._run_in_docker(args or self._args()) == 0

        home = observed["home"]
        command = observed["command"]
        assert isinstance(home, Path)
        assert isinstance(command, list)
        mount_auth.assert_called_once_with(docker_home=home, docker_mounts=ANY)
        return home, command

    def test_mounts_ephemeral_docker_home_outside_checkout_and_cleans_it(self) -> None:
        home, command = self._run_and_capture_home(interrupt=False)

        assert not docs_builder._is_under(home, docs_builder.REPO)
        assert f"{home}:{home}" in command
        assert not home.exists()

    def test_cleans_ephemeral_docker_home_after_interrupt(self) -> None:
        home, command = self._run_and_capture_home(interrupt=True)

        assert not docs_builder._is_under(home, docs_builder.REPO)
        assert f"{home}:{home}" in command
        assert not home.exists()

    def test_publish_preview_allocates_tty_for_interactive_fern_prompt(self) -> None:
        args = self._args()
        args.preview = False
        args.publish_preview = True

        _, command = self._run_and_capture_home(
            interrupt=False,
            args=args,
            stdin_isatty=True,
        )

        assert "-ti" in command


class TokenlessFernProjectTestCase(TestCase):
    def _write_project(self, parent: Path) -> tuple[Path, str, str]:
        fern_dir = parent / "fern"
        fern_dir.mkdir()
        docs_yml = """title: Holoscan SDK Documentation
# holoscan-tokenless-library:begin
libraries:
  holoscan-cpp:
    output:
      path: ./generated/api-reference/cpp
# holoscan-tokenless-library:end
instances:
  - url: example.docs.buildwithfern.com/holoscan/sdk-user-guide
"""
        index_yml = """navigation:
  - section: Introduction
    contents:
      - page: Overview
        path: ../overview.mdx
# holoscan-tokenless-api-navigation:begin
  - section: API reference
    contents:
      - folder: ./generated/api-reference/cpp/holoscan
# holoscan-tokenless-api-navigation:end
  - section: Resources
    contents: []
"""
        (fern_dir / "docs.yml").write_text(docs_yml, encoding="utf-8")
        (fern_dir / "index.yml").write_text(index_yml, encoding="utf-8")
        (fern_dir / "fern.config.json").write_text('{"organization": "nvidia"}\n')
        (fern_dir / "assets").mkdir()
        (fern_dir / "assets" / "logo.svg").write_text("<svg />\n", encoding="utf-8")
        (parent / "overview.mdx").write_text("Overview\n", encoding="utf-8")
        generated = fern_dir / "generated" / "api-reference" / "cpp"
        generated.mkdir(parents=True)
        (generated / "incomplete.txt").write_text("not MDX\n", encoding="utf-8")
        return fern_dir, docs_yml, index_yml

    def test_filters_and_rebases_authored_paths_in_temporary_copy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            fern_dir, docs_yml, index_yml = self._write_project(parent)

            with _tokenless_fern_project(fern_dir) as effective_dir:
                assert effective_dir != fern_dir
                assert effective_dir.name == "fern"
                assert effective_dir.parent.name.startswith(".fern-tokenless-")
                assert effective_dir.parent.parent == fern_dir.parent
                assert "libraries:" not in (effective_dir / "docs.yml").read_text()
                effective_index = (effective_dir / "index.yml").read_text()
                assert "API reference" not in effective_index
                assert "path: ../../overview.mdx" in effective_index
                assert (effective_dir / "assets" / "logo.svg").is_file()
                assert not (effective_dir.parent / "overview.mdx").exists()
                effective_overview = effective_dir / "../../overview.mdx"
                assert effective_overview.resolve() == (parent / "overview.mdx").resolve()
                assert not effective_overview.is_symlink()
                (fern_dir.parent / "overview.mdx").write_text("Updated overview\n")
                assert effective_overview.read_text(encoding="utf-8") == "Updated overview\n"
                assert not (effective_dir / "generated").exists()
                temporary_dir = effective_dir

            assert not temporary_dir.exists()
            assert (fern_dir / "docs.yml").read_text(encoding="utf-8") == docs_yml
            assert (fern_dir / "index.yml").read_text(encoding="utf-8") == index_yml

    def test_cleans_up_after_an_interrupted_action(self) -> None:
        with TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            fern_dir, _, _ = self._write_project(parent)

            try:
                with _tokenless_fern_project(fern_dir) as effective_dir:
                    temporary_dir = effective_dir
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                pass

            assert not temporary_dir.exists()

    def test_fails_closed_when_filter_markers_are_invalid(self) -> None:
        begin_marker = "# holoscan-tokenless-library:begin\n"
        end_marker = "# holoscan-tokenless-library:end\n"
        for name in ("missing", "duplicate"):
            with self.subTest(name=name), TemporaryDirectory() as tmpdir:
                parent = Path(tmpdir)
                fern_dir, _, _ = self._write_project(parent)
                docs_path = fern_dir / "docs.yml"
                docs_yml = docs_path.read_text(encoding="utf-8")
                invalid_docs_yml = (
                    docs_yml.replace(end_marker, "")
                    if name == "missing"
                    else docs_yml.replace(begin_marker, begin_marker * 2)
                )
                docs_path.write_text(invalid_docs_yml, encoding="utf-8")

                with (
                    # Fern CI runs this suite with stdlib unittest and no optional test packages.
                    self.assertRaisesRegex(ValueError, r"tokenless.*marker"),  # noqa: PT027
                    _tokenless_fern_project(fern_dir),
                ):
                    pass
                assert not list(parent.glob(".fern-tokenless-*"))

    def test_selects_fallback_only_for_local_actions_without_real_api_mdx(self) -> None:
        assert _needs_tokenless_fern_project(
            local_fern_action=True,
            with_library_mdx=False,
            cpp_api_present=False,
        )
        for case in (
            {
                "local_fern_action": False,
                "with_library_mdx": False,
                "cpp_api_present": False,
            },
            {
                "local_fern_action": True,
                "with_library_mdx": True,
                "cpp_api_present": False,
            },
            {
                "local_fern_action": True,
                "with_library_mdx": False,
                "cpp_api_present": True,
            },
        ):
            with self.subTest(case=case):
                assert not _needs_tokenless_fern_project(**case)

    def test_validates_links_with_canonical_config_before_local_filtering(self) -> None:
        with TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            fern_dir, _, _ = self._write_project(parent)
            source = parent / "docs"
            source.mkdir()
            link_checker = parent / "check_doc_links.py"
            link_checker.touch()
            args = SimpleNamespace(
                fern_dir=fern_dir,
                preview_url_file=None,
                no_clean=False,
                with_library_mdx=False,
                skip_library_mdx=True,
                skip_fern_check=False,
                skip_link_check=False,
                preview=False,
                publish=False,
                publish_preview=False,
                preview_id=None,
                delete_preview_id=None,
                force=False,
                verbose=False,
                build_role="current",
                source_ref="test-ref",
                allow_rendered_source_links=False,
                allowed_source_browser_url=[],
                allowed_source_ref=[],
            )

            with (
                patch.object(docs_builder, "SOURCE", source),
                patch.object(docs_builder, "CHECK_DOC_LINKS", link_checker),
                patch.object(docs_builder, "_fern_exe", return_value="fern"),
                patch.object(docs_builder, "_fern_assets_ok", return_value=True),
                patch.object(docs_builder, "_run") as run,
            ):
                run.return_value.returncode = 0
                assert docs_builder.run_pipeline(args) == 0

            link_call = next(
                call for call in run.call_args_list if str(link_checker) in call.args[0]
            )
            link_command = link_call.args[0]
            assert Path(link_command[-1]).resolve() == fern_dir.resolve()
            fern_call = next(
                call for call in run.call_args_list if call.args[0][:2] == ["fern", "check"]
            )
            fern_cwd = fern_call.kwargs["cwd"]
            assert fern_cwd != fern_dir
            assert fern_cwd.name == "fern"
            assert fern_cwd.parent.name.startswith(".fern-tokenless-")
            assert not fern_cwd.parent.exists()


class BuildHoloscanDocsTestCase(TestCase):
    def test_derives_documentation_version_from_checkout(self) -> None:
        with TemporaryDirectory() as tmpdir:
            docs_root = Path(tmpdir) / "public" / "docs"
            docs_root.mkdir(parents=True)
            (docs_root.parent / "VERSION").write_text("4.5.0\n", encoding="utf-8")

            with patch.object(docs_builder, "DOCS_ROOT", docs_root):
                assert docs_builder._documentation_version() == "v4.5"

    def test_derives_documentation_version_for_staged_docs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "release"
            (repo / "public").mkdir(parents=True)
            (repo / "public" / "VERSION").write_text("4.5.0\n", encoding="utf-8")
            docs_root = root / "rendered" / "public" / "docs"
            docs_root.mkdir(parents=True)

            with (
                patch.object(docs_builder, "DOCS_ROOT", docs_root),
                patch.object(docs_builder, "REPO", repo),
            ):
                assert docs_builder._documentation_version() == "v4.5"

    def test_prefers_exact_tag_as_checkout_source_ref(self) -> None:
        with TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q", repo], check=True)
            subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
            subprocess.run(
                ["git", "-C", repo, "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    repo,
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "Initial",
                ],
                check=True,
            )
            subprocess.run(["git", "-C", repo, "tag", "v4.4.0.2"], check=True)

            with patch.object(docs_builder, "REPO", repo):
                assert docs_builder._checkout_source_ref() == "v4.4.0.2"

    def test_local_library_generation_requires_docker(self) -> None:
        with patch.object(docs_builder.shutil, "which", return_value=None):
            assert "requires Docker" in docs_builder._local_library_runtime_error()

    def test_local_library_generation_requires_running_daemon(self) -> None:
        completed_process = subprocess.CompletedProcess(
            ["docker", "info"],
            1,
            stdout="",
            stderr="Cannot connect to the Docker daemon\n",
        )

        with (
            patch.object(docs_builder.shutil, "which", return_value="/usr/bin/docker"),
            patch.object(docs_builder.subprocess, "run", return_value=completed_process),
        ):
            assert "daemon is unavailable" in docs_builder._local_library_runtime_error()

    def test_container_generation_forwards_active_docker_socket(self) -> None:
        completed_process = subprocess.CompletedProcess(
            ["docker", "run"],
            0,
            stdout="991\n",
            stderr="",
        )
        with (
            patch.dict(
                docs_builder.os.environ,
                {"DOCKER_HOST": "unix:///tmp/colima/docker.sock"},
            ),
            patch.object(
                docs_builder.subprocess,
                "run",
                return_value=completed_process,
            ) as run,
        ):
            options = docs_builder._container_docker_options(
                image="holoscan-docs:test",
                with_library_mdx=True,
            )

        run.assert_called_once()
        assert "-eDOCKER_HOST=unix:///var/run/docker.sock" in options
        assert "991" in options
        assert "/tmp/colima/docker.sock:/var/run/docker.sock" in options

    def test_container_generation_uses_active_docker_context_socket(self) -> None:
        context_process = subprocess.CompletedProcess(
            ["docker", "context", "inspect"],
            0,
            stdout="unix:///run/user/1000/docker.sock\n",
            stderr="",
        )
        stat_process = subprocess.CompletedProcess(
            ["docker", "run"],
            0,
            stdout="991\n",
            stderr="",
        )
        with (
            patch.dict(docs_builder.os.environ, {}, clear=True),
            patch.object(
                docs_builder.subprocess,
                "run",
                side_effect=[context_process, stat_process],
            ) as run,
        ):
            options = docs_builder._container_docker_options(
                image="holoscan-docs:test",
                with_library_mdx=True,
            )

        assert run.call_count == 2
        assert "/run/user/1000/docker.sock:/var/run/docker.sock" in options

    def test_container_reuse_does_not_forward_docker_socket(self) -> None:
        with patch.object(docs_builder.subprocess, "run") as run:
            options = docs_builder._container_docker_options(
                image="holoscan-docs:test",
                with_library_mdx=False,
            )

        run.assert_not_called()
        assert options == []

    def test_rendered_fern_directory_keeps_docs_root_path_in_container(self) -> None:
        root = Path("/workspace")
        repo_root = root / "release"
        script_repo = root / "scripts"
        docs_root = root / "work" / "rendered" / "public" / "docs"
        fern_dir = docs_root / "fern"

        mounts, container_fern_dir = docs_builder._container_mounts(
            repo_root=repo_root,
            script_repo=script_repo,
            docs_root=docs_root,
            fern_dir=fern_dir,
        )

        assert container_fern_dir == fern_dir
        assert f"{docs_root}:{docs_root}" in mounts
        assert not any(str(docs_builder.EXTERNAL_FERN_DIR_MOUNT) in mount for mount in mounts)

    def test_independent_fern_directory_uses_container_mount_alias(self) -> None:
        root = Path("/workspace")
        repo_root = root / "release"
        script_repo = root / "scripts"
        docs_root = root / "work" / "rendered" / "public" / "docs"
        fern_dir = root / "independent-fern"

        mounts, container_fern_dir = docs_builder._container_mounts(
            repo_root=repo_root,
            script_repo=script_repo,
            docs_root=docs_root,
            fern_dir=fern_dir,
        )

        assert container_fern_dir == docs_builder.EXTERNAL_FERN_DIR_MOUNT
        assert f"{fern_dir}:{docs_builder.EXTERNAL_FERN_DIR_MOUNT}" in mounts


if __name__ == "__main__":
    main()
