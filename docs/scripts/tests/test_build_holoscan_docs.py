#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import ANY, patch

SCRIPT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "scripts"))

import build_holoscan_docs as docs_builder  # noqa: E402
from _git_environment import git_environment  # noqa: E402
from build_holoscan_docs import (  # noqa: E402
    _expand_unpack_params_for_fern,
    _external_git_metadata_mounts,
    _needs_tokenless_fern_project,
    _public_python_api_input,
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
    def setUp(self) -> None:
        runtime = TemporaryDirectory()
        self.addCleanup(runtime.cleanup)
        self.fern_state_dir = Path(runtime.name).resolve()
        environment = patch.dict(
            docs_builder.os.environ,
            {"HOLOSCAN_DOCS_FERN_STATE_DIR": str(self.fern_state_dir)},
        )
        environment.start()
        self.addCleanup(environment.stop)

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

    def test_preview_requires_generated_folders_before_publication(self) -> None:
        with TemporaryDirectory() as tmpdir:
            args = self._args()
            args.fern_dir = Path(tmpdir) / "fern"
            args.fern_dir.mkdir()
            args.preview = False
            args.publish_preview = True
            args.preview_id = "test-preview"

            def check_command(command: list[str], **_kwargs: object) -> SimpleNamespace:
                if str(docs_builder.CHECK_DOC_LINKS) in command:
                    assert "--require-generated" in command
                    return SimpleNamespace(returncode=2)
                return SimpleNamespace(returncode=0)

            with (
                patch.object(docs_builder, "_run", side_effect=check_command),
                patch.object(docs_builder, "_require_fern_auth") as authenticate,
                patch.object(docs_builder, "_publish_remote_preview") as publish,
            ):
                assert docs_builder.run_pipeline(args) == 2
                authenticate.assert_not_called()
                publish.assert_not_called()

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

    def test_reuses_docker_home_and_cleans_unique_parser_scratch(self) -> None:
        home, command = self._run_and_capture_home(interrupt=False)
        scratch = Path(
            next(part.removeprefix("-eTMPDIR=") for part in command if part.startswith("-eTMPDIR="))
        )
        assert not docs_builder._is_under(home, docs_builder.REPO)
        assert f"{home}:{home}" in command
        assert "git config --global --replace-all safe.directory" in command[-1]
        assert " && python3" in command[-1]
        assert "2>/dev/null || true" not in command[-1]
        assert home == self.fern_state_dir
        assert scratch.parent == home / "tmp"
        assert not scratch.exists()
        cache = home / ".fern" / "preview"
        cache.mkdir(parents=True)
        marker = cache / "bundle.marker"
        marker.write_text("cached bundle\n")

        next_home, next_command = self._run_and_capture_home(interrupt=False)
        next_scratch = Path(
            next(
                part.removeprefix("-eTMPDIR=")
                for part in next_command
                if part.startswith("-eTMPDIR=")
            )
        )
        assert next_home == home
        assert next_scratch != scratch
        assert not next_scratch.exists()
        assert marker.read_text() == "cached bundle\n"

    def test_container_home_defaults_under_home_not_host_tmpdir(self) -> None:
        with (
            patch.object(Path, "home", return_value=self.fern_state_dir),
            patch.dict(docs_builder.os.environ, {"TMPDIR": "/unshared/host/tmp"}),
        ):
            docs_builder.os.environ.pop("HOLOSCAN_DOCS_FERN_STATE_DIR")
            home, _ = self._run_and_capture_home(interrupt=False)
            assert home == self.fern_state_dir / ".cache/holoscan-docs"

    def test_container_home_uses_fern_state_directory_and_preserves_siblings(self) -> None:
        marker = self.fern_state_dir / "keep.txt"
        marker.write_text("unrelated state\n")
        home, _ = self._run_and_capture_home(interrupt=False)
        assert home == self.fern_state_dir
        assert home.is_dir()
        assert marker.read_text() == "unrelated state\n"

    def test_container_home_rejects_fern_state_inside_watched_tree(self) -> None:
        with (
            patch.dict(
                docs_builder.os.environ,
                {"HOLOSCAN_DOCS_FERN_STATE_DIR": str(self.fern_state_dir / "rendered/fern-state")},
            ),
            self.assertRaisesRegex(ValueError, "overlaps watched content"),  # noqa: PT027
            docs_builder._external_docker_home(
                watched_roots=(self.fern_state_dir / "rendered",), docker_mounts=[]
            ),
        ):
            raise AssertionError("watched Fern state directory was accepted")
        assert not (self.fern_state_dir / "rendered").exists()

    def test_retains_home_but_cleans_scratch_and_releases_lock_after_interrupt(self) -> None:
        home, command = self._run_and_capture_home(interrupt=True)

        assert not docs_builder._is_under(home, docs_builder.REPO)
        assert f"{home}:{home}" in command
        assert home.is_dir()
        assert list((home / "tmp").iterdir()) == []
        next_home, _ = self._run_and_capture_home(interrupt=False)
        assert next_home == home

    def test_rejects_concurrent_use_of_same_home(self) -> None:
        with docs_builder._external_docker_home(watched_roots=(), docker_mounts=[], reuse=True):
            with (
                self.assertRaisesRegex(ValueError, "already in use"),  # noqa: PT027
                docs_builder._external_docker_home(watched_roots=(), docker_mounts=[], reuse=True),
            ):
                raise AssertionError("concurrent use was accepted")
            # A failed contender must not remove the lock file or unlock the owner.
            with (
                self.assertRaisesRegex(ValueError, "already in use"),  # noqa: PT027
                docs_builder._external_docker_home(watched_roots=(), docker_mounts=[], reuse=True),
            ):
                raise AssertionError("failed contender released the lock")
        with docs_builder._external_docker_home(
            watched_roots=(), docker_mounts=[], reuse=True
        ) as home:
            assert home == self.fern_state_dir

    def test_non_preview_modes_keep_per_run_home_and_cleanup(self) -> None:
        for mode in ("check", "publish", "publish_preview", "delete_preview_id"):
            with self.subTest(mode=mode):
                args = self._args()
                args.preview = False
                if mode != "check":
                    setattr(args, mode, "old-preview" if mode == "delete_preview_id" else True)
                first_home, _ = self._run_and_capture_home(interrupt=False, args=args)
                second_home, _ = self._run_and_capture_home(interrupt=False, args=args)
                assert first_home.parent == self.fern_state_dir
                assert second_home.parent == self.fern_state_dir
                assert first_home != second_home
                assert not first_home.exists()
                assert not second_home.exists()
                interrupted_home, _ = self._run_and_capture_home(interrupt=True, args=args)
                assert not interrupted_home.exists()

    def test_node_options_default_and_explicit_overrides(self) -> None:
        for mode, value, expected in (
            ("preview", None, ["-eNODE_OPTIONS=--max-old-space-size=4096"]),
            ("check", None, []),
            ("publish_preview", None, []),
            ("publish", None, []),
            ("preview", "--max-old-space-size=2048 --trace-warnings", ["-eNODE_OPTIONS"]),
            ("preview", "", ["-eNODE_OPTIONS"]),
            ("check", "--max-old-space-size=2048", ["-eNODE_OPTIONS"]),
        ):
            with self.subTest(mode=mode, node_options=value), patch.dict(docs_builder.os.environ):
                args = self._args()
                args.preview = mode == "preview"
                args.publish_preview = mode == "publish_preview"
                args.publish = mode == "publish"
                if value is None:
                    docs_builder.os.environ.pop("NODE_OPTIONS", None)
                else:
                    docs_builder.os.environ["NODE_OPTIONS"] = value
                _, command = self._run_and_capture_home(interrupt=False, args=args)
                assert [part for part in command if part.startswith("-eNODE_OPTIONS")] == expected
                assert docs_builder.os.environ.get("NODE_OPTIONS") == value

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


class PublicPythonApiInputTestCase(TestCase):
    def test_expands_unpack_typed_dict_into_keyword_parameters(self) -> None:
        source = """\
from typing import Literal, TypedDict, Unpack

class _Params(TypedDict, total=False):
    limit: int
    mode: Literal["safe", "fast"]

class Proxy:
    def __call__(self, graph: object, instance_name: str, **params: Unpack[_Params]) -> object: ...
"""

        expanded = _expand_unpack_params_for_fern(source)

        assert "**params: Unpack[_Params]" not in expanded
        assert '*, limit: int = ..., mode: Literal["safe", "fast"] = ...' in expanded

    def test_expands_functional_typed_dict_fields(self) -> None:
        source = """\
from typing import TypedDict, Unpack

_Params = TypedDict("_Params", {"mode": str}, total=False)

class Proxy:
    def __call__(self, graph: object, instance_name: str, **params: Unpack[_Params]) -> object: ...
"""

        expanded = _expand_unpack_params_for_fern(source)

        assert "**params: Unpack[_Params]" not in expanded
        assert "*, mode: str = ..." in expanded

    def test_rejects_unparseable_or_colliding_proxy_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "parse"):  # noqa: PT027
            _expand_unpack_params_for_fern("class\n, **params: Unpack[_Params]")

        source = """\
from typing import TypedDict, Unpack

class _Params(TypedDict, total=False):
    graph: str

class Proxy:
    def __call__(self, graph: object, instance_name: str, **params: Unpack[_Params]) -> object: ...
"""
        with self.assertRaisesRegex(ValueError, "collide"):  # noqa: PT027
            _expand_unpack_params_for_fern(source)

    def test_preserves_stubs_without_unpack_signatures(self) -> None:
        source = "class GraphType[HandleT]: ...\n"

        assert _expand_unpack_params_for_fern(source) == source

    def test_projects_only_public_facade_stubs_as_parser_sources(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo = root / "repo"
            docs_root = repo / "public" / "docs"
            stub_root = repo / "public" / "python" / "holoscan"
            (stub_root / "payload").mkdir(parents=True)
            docs_root.mkdir(parents=True)
            (stub_root / "__init__.pyi").write_text("class Graph: ...\n", encoding="utf-8")
            (stub_root / "errors.pyi").write_text(
                "class HoloscanError(Exception): ...\n", encoding="utf-8"
            )
            (stub_root / "payload" / "__init__.pyi").write_text(
                "class PayloadType: ...\n", encoding="utf-8"
            )
            (stub_root / "_core.pyi").write_text("class Private: ...\n", encoding="utf-8")
            (stub_root / "registry.py").write_text("runtime_only = True\n", encoding="utf-8")

            with (
                patch.object(docs_builder, "REPO", repo),
                patch.object(docs_builder, "DOCS_ROOT", docs_root),
            ):
                package_root = _public_python_api_input()
                assert (package_root / "__init__.py").read_text() == "class Graph: ...\n"
                assert (package_root / "errors.py").is_file()
                assert (package_root / "payload" / "__init__.py").is_file()
                assert not (package_root / "_core.py").exists()
                assert not (package_root / "registry.py").exists()
                assert (package_root.parent / "pyproject.toml").is_file()
                input_root = package_root.parent

            assert input_root.exists()


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
# holoscan-tokenless-python-api-navigation:begin
      - folder: ./generated/api-reference/python/holoscan
        title: Python API
# holoscan-tokenless-python-api-navigation:end
# holoscan-tokenless-api-navigation:begin
  - section: API reference
    contents:
      - folder: ./generated/api-reference/cpp/holoscan
      - link: Python API
        href: /holoscan/sdk-user-guide/python/api-reference/holoscan
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
                assert "Python API" not in effective_index
                assert "generated/api-reference" not in effective_index
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
            library_api_present=False,
        )
        for case in (
            {
                "local_fern_action": False,
                "with_library_mdx": False,
                "library_api_present": False,
            },
            {
                "local_fern_action": True,
                "with_library_mdx": True,
                "library_api_present": False,
            },
            {
                "local_fern_action": True,
                "with_library_mdx": False,
                "library_api_present": True,
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
        # _checkout_source_ref also invokes Git, so isolate the full fixture scope.
        with patch.dict(os.environ, git_environment(), clear=True), TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            subprocess.run(["git", "init", "-q", repo], check=True)
            top_level = subprocess.check_output(["git", "-C", repo, "rev-parse", "--show-toplevel"])
            assert Path(os.fsdecode(top_level).strip()).resolve() == repo.resolve()
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
                {"DOCKER_HOST": "unix:///run/custom/docker.sock"},
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
        assert "/run/custom/docker.sock:/var/run/docker.sock" in options

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

    def test_container_generation_falls_back_from_host_proxy_on_either_os(self) -> None:
        for platform, proxy in (
            ("linux", "/home/developer/.colima/default/docker.sock"),
            ("darwin", "/Users/developer/.colima/default/docker.sock"),
        ):
            with (
                self.subTest(platform=platform),
                patch.object(docs_builder.sys, "platform", platform),
                patch.object(docs_builder, "_active_docker_socket", return_value=proxy),
                patch.object(
                    docs_builder.subprocess,
                    "run",
                    side_effect=[
                        SimpleNamespace(returncode=1, stdout="", stderr="Cannot bind proxy"),
                        SimpleNamespace(returncode=0, stdout="991\n", stderr=""),
                    ],
                ) as run,
            ):
                options = docs_builder._container_docker_options(
                    image="holoscan-docs:test", with_library_mdx=True
                )

                assert run.call_count == 2
                assert "/var/run/docker.sock:/var/run/docker.sock" in options
                assert (
                    f"type=bind,source={proxy},target=/var/run/docker.sock"
                    in (run.call_args_list[0].args[0])
                )
                assert (
                    "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock"
                    in run.call_args.args[0]
                )

    def test_container_generation_reports_unusable_socket_paths(self) -> None:
        for socket_path, attempts in (
            ("/run/custom/docker.sock", 2),
            ("/var/run/docker.sock", 1),
        ):
            with (
                self.subTest(socket_path=socket_path),
                patch.object(docs_builder, "_active_docker_socket", return_value=socket_path),
                patch.object(
                    docs_builder.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=1, stdout="", stderr="No socket"),
                ) as run,
            ):
                with self.assertRaisesRegex(  # noqa: PT027 - Keep this test stdlib-only.
                    RuntimeError, "could not inspect.*No socket"
                ):
                    docs_builder._container_docker_options(
                        image="holoscan-docs:test", with_library_mdx=True
                    )
                assert run.call_count == attempts

    def test_container_generation_rejects_invalid_socket_gid(self) -> None:
        with (
            patch.object(
                docs_builder, "_active_docker_socket", return_value="/var/run/docker.sock"
            ),
            patch.object(
                docs_builder.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="not-a-gid\n", stderr=""),
            ),
            self.assertRaisesRegex(  # noqa: PT027 - Keep this test stdlib-only.
                RuntimeError, "could not inspect"
            ),
        ):
            docs_builder._container_docker_options(
                image="holoscan-docs:test", with_library_mdx=True
            )

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
