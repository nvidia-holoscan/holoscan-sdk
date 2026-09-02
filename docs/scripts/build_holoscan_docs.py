#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Build and validate Holoscan SDK Fern documentation.

User guide pages are committed as ``*.mdx`` under ``docs/`` in the public repository
and ``public/docs/`` in the internal source repository. Fern config
(``fern.config.json``, ``docs.yml``, ``index.yml``, ``assets/``, ``dist/``) lives under
the detected documentation root. C++ API pages are generated under
``fern/generated/`` there (gitignored).

1. Optionally remove prior C++ API MDX under ``generated/``.
2. With ``--with-library-mdx``: ``fern docs md generate --local`` +
   ``fix_generated_library_mdx.py``.
3. ``check_doc_links.py`` (internal link validation, which ``fern check`` does not cover).
4. ``fern check --local --warnings`` (local validation), ``fern docs dev`` (``--preview``),
   ``fern generate --docs --preview`` (``--publish-preview`` for CI), or
   ``fern generate --docs`` (``--publish`` for a production release).

Usage (from holoscan-sdk repo root):

  python3 docs/scripts/build_holoscan_docs.py
  python3 docs/scripts/build_holoscan_docs.py --preview
  python3 docs/scripts/build_holoscan_docs.py --no-docker
  python3 docs/scripts/build_holoscan_docs.py --with-library-mdx
  python3 docs/scripts/build_holoscan_docs.py --skip-fern-check
  python3 docs/scripts/build_holoscan_docs.py --skip-link-check
  python3 docs/scripts/build_holoscan_docs.py --publish --skip-library-mdx
  python3 docs/scripts/build_holoscan_docs.py --publish-preview --preview-id my-branch --force
  python3 docs/scripts/build_holoscan_docs.py --publish-preview \
    --preview-id my-branch --preview-url-file /tmp/fern-preview-url --force
  python3 docs/scripts/build_holoscan_docs.py --delete-preview-id my-branch \
    --publish-preview --preview-id main --force
  python3 /path/to/build_holoscan_docs.py \
    --repository-checkout /path/to/release-checkout \
    --no-docker --publish --skip-library-mdx

``--skip-library-mdx`` reuses an existing local generated C++ API MDX tree; it
does not exclude that tree from validation, preview, or publication. Production
reuse requires a non-empty C++ API tree and verifies that every generated MDX
file is already post-processed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPTS = Path(__file__).resolve().parent
SCRIPT_DOCS_ROOT = SCRIPTS.parent


def _initial_repository_root(docs_root: Path) -> Path:
    """Find the checkout root for internal public/docs or mirrored docs layouts."""
    for candidate in (docs_root.parent, docs_root.parent.parent):
        if (candidate / ".git").exists():
            return candidate
    if docs_root.parent.name == "public":
        return docs_root.parent.parent
    return docs_root.parent


def _find_docs_root(repo: Path) -> Path:
    """Return the internal or public documentation root in a checkout."""
    candidates = (repo / "public" / "docs", repo / "docs")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    expected = " or ".join(str(path) for path in candidates)
    raise ValueError(f"documentation root not found; expected {expected}")


SCRIPT_REPO = _initial_repository_root(SCRIPT_DOCS_ROOT)
DOCS_ROOT = SCRIPT_DOCS_ROOT
REPO = SCRIPT_REPO
SOURCE = DOCS_ROOT
FIX_LIBRARY_MDX = SCRIPTS / "fix_generated_library_mdx.py"
CHECK_DOC_LINKS = SCRIPTS / "check_doc_links.py"
CHECK_SOURCE_LINKS = SCRIPTS / "render_source_links.py"
DOCKERFILE = DOCS_ROOT / "Dockerfile"
BUILD_CONTEXT_PREFIX = ""

FERN_API_VERSION = "5.89.3"
EXTERNAL_FERN_DIR_MOUNT = Path("/mnt/fern_dir")
EXTERNAL_PREVIEW_URL_FILE_MOUNT = Path("/mnt/fern_preview_url")
PREVIEW_URL_FILE = ".fern-preview-url"
_PREVIEW_URL_RE = re.compile(
    r"https://[^\s\"'<>)\]]+\.docs\.buildwithfern\.com[^\s\"'<>)\]]*",
    re.IGNORECASE,
)
_TOKENLESS_LIBRARY_MARKERS = (
    "# holoscan-tokenless-library:begin",
    "# holoscan-tokenless-library:end",
)
_TOKENLESS_API_NAVIGATION_MARKERS = (
    "# holoscan-tokenless-api-navigation:begin",
    "# holoscan-tokenless-api-navigation:end",
)
_TOKENLESS_AUTHORED_PATH_RE = re.compile(
    r"^(?P<prefix>[ \t]*path:[ \t]+)"
    r"(?P<path>\.\./[^ \t\r\n#]+)"
    r"(?P<suffix>[ \t]*(?:#.*)?)(?P<newline>\r?\n?)$",
    re.MULTILINE,
)

USAGE_EXAMPLES = r"""pipeline phases:
  Execution environment:

    build_holoscan_docs.py (default)
      -> build the repository docs image
      -> run the pipeline with its pinned tools in Docker

    build_holoscan_docs.py --no-docker
      -> run the same pipeline with tools installed directly on the host

  C++ API output selection:

    neither flag          automatic: generate when Fern auth is detected,
                          otherwise reuse an existing local output tree
    --with-library-mdx    generate fresh output from the selected checkout's
                          local library input
    --skip-library-mdx    reuse the existing local output tree

  docs.yml configures ../../include/holoscan as the C++ API source. Fern parses
  the selected checkout with --local and writes the generated MDX into its
  fern/generated directory. This keeps user-guide and API content on the same
  source revision, including revisions that are not published on GitHub.

  Common pipeline and selected final action:

    generate or reuse local C++ API MDX
      |
      v
    post-process local MDX
      |
      v
    check internal links (skip with --skip-link-check)
      |
      +-- no action flag ...... fern check --local --warnings
      +-- --preview ........... fern docs dev
      +-- --publish-preview ... fern generate --docs --preview
      `-- --publish ........... fern generate --docs

  `fern check` does not resolve internal link targets, so `check_doc_links.py`
  runs first and fails the pipeline on links, anchors, or assets that cannot
  resolve.

  "Local validation" means `fern check --local --warnings`. This script stops
  after Fern reports publication success. The separate HSDK GA release
  workflow checks its plan-listed public routes before the GitHub Release.

examples:
  Generate the C++ API output and run local Fern validation:
    python3 docs/scripts/build_holoscan_docs.py --with-library-mdx

  Publish using an existing generated C++ API output tree:
    python3 docs/scripts/build_holoscan_docs.py --no-docker \
      --skip-library-mdx \
      --publish
"""


def _configure_repository_checkout(checkout: Path) -> None:
    """Point repository-derived inputs at a selected HSDK checkout."""
    global REPO, DOCS_ROOT, SOURCE, DOCKERFILE

    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or f"not a Git checkout: {checkout}"
        raise ValueError(detail)
    REPO = Path(result.stdout.strip()).resolve()
    DOCS_ROOT = _find_docs_root(REPO)
    SOURCE = DOCS_ROOT
    DOCKERFILE = DOCS_ROOT / "Dockerfile"


def _configure_docs_root(docs_root: Path) -> None:
    """Use an explicit documentation tree, such as a rendered staging copy."""
    global DOCS_ROOT, SOURCE, DOCKERFILE
    docs_root = docs_root.expanduser().resolve()
    if not docs_root.is_dir():
        raise ValueError(f"documentation root not found: {docs_root}")
    if not (docs_root / "fern").is_dir():
        raise ValueError(f"Fern directory not found: {docs_root / 'fern'}")
    DOCS_ROOT = docs_root
    SOURCE = docs_root
    DOCKERFILE = docs_root / "Dockerfile"


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _container_mounts(
    *, repo_root: Path, script_repo: Path, docs_root: Path, fern_dir: Path
) -> tuple[list[str], Path]:
    """Return Docker mounts and the container-visible Fern directory."""
    docker_mounts = [f"{repo_root}:{repo_root}"]
    mounted_roots = [repo_root]

    if script_repo != repo_root:
        docker_mounts.append(f"{script_repo}:{script_repo}")
        mounted_roots.append(script_repo)
    if not any(_is_under(docs_root, root) for root in mounted_roots):
        docker_mounts.append(f"{docs_root}:{docs_root}")
        mounted_roots.append(docs_root)

    if any(_is_under(fern_dir, root) for root in mounted_roots):
        container_fern_dir = fern_dir
    else:
        container_fern_dir = EXTERNAL_FERN_DIR_MOUNT
        docker_mounts.append(f"{fern_dir}:{container_fern_dir}")

    return docker_mounts, container_fern_dir


def _git_metadata_paths(repo_root: Path) -> tuple[Path, Path]:
    """Return the checkout's absolute per-worktree and common Git directories."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "--path-format=absolute",
            "--absolute-git-dir",
            "--git-common-dir",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    paths = [Path(line) for line in result.stdout.splitlines() if line]
    if result.returncode or len(paths) != 2:
        detail = result.stderr.strip() or f"unexpected Git metadata for {repo_root}"
        raise ValueError(detail)
    return paths[0], paths[1]


def _external_git_metadata_mounts(repo_root: Path) -> list[str]:
    """Return read-only Docker mounts needed to use an externally linked worktree."""
    external_roots: list[Path] = []
    # Keep the shallowest roots because a linked worktree's per-worktree Git
    # directory is normally contained by its common directory. The container
    # needs the same absolute paths referenced by the checkout's .git file.
    metadata_paths = sorted(
        set(_git_metadata_paths(repo_root)),
        key=lambda path: (len(path.parts), str(path)),
    )
    for path in metadata_paths:
        if _is_under(path, repo_root):
            continue
        if any(_is_under(path, mounted_root) for mounted_root in external_roots):
            continue
        external_roots.append(path)
    return [f"{path}:{path}:ro" for path in external_roots]


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
    indent_output: bool = False,
    show_command: bool = True,
) -> subprocess.CompletedProcess:
    if show_command:
        prefix = "  + " if indent_output else "+ "
        print(prefix + shlex.join(cmd), flush=True)
    if not indent_output:
        return subprocess.run(cmd, cwd=cwd or REPO, env=env, check=check)

    process = subprocess.Popen(
        cmd,
        cwd=cwd or REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"  {line}", end="", flush=True)
    returncode = process.wait()
    result = subprocess.CompletedProcess(cmd, returncode)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, cmd)
    return result


def _print_section(title: str) -> None:
    print(f"\n{BUILD_CONTEXT_PREFIX}{title}", flush=True)


def _print_detail(message: str) -> None:
    print(f"  {message}", flush=True)


def _documentation_version() -> str:
    version = ""
    for version_path in (
        DOCS_ROOT.parent / "VERSION",
        REPO / "public" / "VERSION",
        REPO / "VERSION",
    ):
        try:
            version = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if version:
            break
    if not version:
        return "unknown"
    match = re.match(r"^v?(\d+)\.(\d+)", version)
    return f"v{match.group(1)}.{match.group(2)}" if match else version


def _checkout_source_ref() -> str:
    commands = (
        ["git", "-C", str(REPO), "describe", "--tags", "--exact-match", "HEAD"],
        ["git", "-C", str(REPO), "branch", "--show-current"],
        ["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"],
    )
    for command in commands:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value
    return "unknown"


def _build_purpose(args: argparse.Namespace) -> str:
    if args.build_role == "archive":
        return "backfill archive snapshot"
    if args.preview:
        return "local preview"
    if args.publish_preview:
        return "remote preview"
    if args.publish:
        return "production publication"
    return "current documentation validation"


def _configured_fern_version(fern_dir: Path) -> str:
    config = fern_dir / "fern.config.json"
    try:
        version = json.loads(config.read_text(encoding="utf-8")).get("version")
    except (OSError, json.JSONDecodeError):
        version = None
    return version or "unknown"


def _fern_exe() -> str | None:
    return shutil.which("fern")


def _local_library_runtime_error() -> str | None:
    """Return an actionable error when Fern cannot run its local parser."""
    docker_exe = shutil.which("docker")
    if docker_exe is None:
        return (
            "local C++ API generation requires Docker; install and start Docker "
            "before running with --with-library-mdx"
        )

    result = subprocess.run(
        [docker_exe, "info"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        return f"Docker is installed but its daemon is unavailable{suffix}"

    if sys.platform == "darwin":
        context = subprocess.run(
            [docker_exe, "context", "show"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        if context == "colima" and not _is_under(REPO, Path.home().resolve()):
            return (
                "Colima only shares the macOS home directory by default, but the "
                f"release checkout is {REPO}. Choose --work-dir under {Path.home()} "
                "or configure Colima to mount this path"
            )
    return None


def _require_fern_auth(*, action: str) -> int | None:
    if _fern_auth_token():
        return None
    print(
        f"error: {action} requires Fern auth (run 'fern login' or set FERN_TOKEN).",
        file=sys.stderr,
    )
    return 1


def _extract_preview_url(output: str) -> str | None:
    urls = [match.group(0).rstrip(".,;)") for match in _PREVIEW_URL_RE.finditer(output)]
    if not urls:
        return None
    for url in urls:
        if "-preview-" in url.lower():
            return url
    return urls[0]


def _docs_site_path(fern_dir: Path) -> str:
    docs_yml = fern_dir / "docs.yml"
    if docs_yml.is_file():
        for line in docs_yml.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- url:") or stripped.startswith("url:"):
                value = stripped.split(":", 1)[1].strip()
                if value.startswith("nvidia-holoscan.docs.buildwithfern.com"):
                    suffix = value.split(".docs.buildwithfern.com", 1)[-1]
                    return suffix if suffix.startswith("/") else f"/{suffix}"
    return "/holoscan/sdk-user-guide"


def _fallback_preview_url(*, fern_dir: Path, preview_id: str) -> str | None:
    config_path = fern_dir / "fern.config.json"
    if not config_path.is_file():
        return None
    try:
        org = json.loads(config_path.read_text(encoding="utf-8")).get("organization")
    except json.JSONDecodeError:
        return None
    if not org:
        return None
    slug = preview_id.strip().lower().replace("/", "-").replace("_", "-")
    return f"https://{org}-preview-{slug}.docs.buildwithfern.com{_docs_site_path(fern_dir)}"


def _write_preview_url(*, preview_url_file: Path, url: str) -> None:
    preview_url_file.parent.mkdir(parents=True, exist_ok=True)
    preview_url_file.write_text(f"{url}\n", encoding="utf-8")
    print(f"Fern docs preview URL: {url}", flush=True)


def _delete_remote_preview(
    *, fern_exe: str, fern_dir: Path, preview_id: str, verbose: bool = False
) -> None:
    print(f"Deleting Fern preview: {preview_id}", flush=True)
    result = _run(
        [fern_exe, "docs", "preview", "delete", "--id", preview_id],
        cwd=fern_dir,
        check=False,
        show_command=verbose,
    )
    if result.returncode != 0:
        print(
            f"warning: fern docs preview delete --id {preview_id!r} exited "
            f"{result.returncode} (preview may already be gone).",
            file=sys.stderr,
        )


def _fern_generate_docs_command(
    *,
    fern_exe: str,
    preview_id: str | None = None,
    force: bool = False,
) -> list[str]:
    """Build a Fern documentation publication command."""
    cmd = [fern_exe, "generate", "--docs"]
    if preview_id is not None:
        cmd.extend(["--preview", "--id", preview_id])
    if force:
        cmd.append("--force")
    return cmd


def _publish_remote_preview(
    *,
    fern_exe: str,
    fern_dir: Path,
    preview_id: str,
    force: bool,
    preview_url_file: Path | None = None,
    verbose: bool = False,
) -> str | None:
    cmd = _fern_generate_docs_command(
        fern_exe=fern_exe,
        preview_id=preview_id,
        force=force,
    )
    if verbose:
        print("+ " + shlex.join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=fern_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_chunks.append(line)
    rc = proc.wait()
    output = "".join(output_chunks)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=output)

    url = _extract_preview_url(output) or _fallback_preview_url(
        fern_dir=fern_dir,
        preview_id=preview_id,
    )
    if url:
        _write_preview_url(
            preview_url_file=preview_url_file or fern_dir / PREVIEW_URL_FILE,
            url=url,
        )
    else:
        print("warning: could not determine Fern docs preview URL", file=sys.stderr)
    return url


def _publish_production(*, fern_exe: str, fern_dir: Path, verbose: bool = False) -> None:
    """Publish the selected checkout's HSDK User Guide to production."""
    _run(
        _fern_generate_docs_command(fern_exe=fern_exe),
        cwd=fern_dir,
        show_command=verbose,
    )


def _repo_relative(path: Path) -> Path | str:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def _clean_generated_api_docs(*, generated_docs: Path, preserve: bool = False) -> None:
    if not generated_docs.is_dir():
        _print_detail("Existing output: none")
        return
    if preserve:
        _print_detail(
            f"Existing output: {_repo_relative(generated_docs)} (reused; pass "
            "--with-library-mdx to regenerate)"
        )
        return
    shutil.rmtree(generated_docs)
    _print_detail(f"Removed: {_repo_relative(generated_docs)}")


def _fern_auth_token() -> str | None:
    token = os.environ.get("FERN_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".fern" / "token"
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip()
    return None


def _resolve_with_library_mdx(explicit: str | None) -> bool:
    if explicit == "1":
        return True
    if explicit == "0":
        return False
    return _fern_auth_token() is not None


def _fern_assets_ok(fern_dir: Path) -> bool:
    dist = fern_dir / "dist"
    return (dist / "output.css").is_file() and (dist / "output.js").is_file()


def _should_check_doc_links(args: argparse.Namespace) -> bool:
    """Check links except for a standalone remote-preview deletion."""
    delete_only = args.delete_preview_id and not args.publish_preview
    return not args.skip_link_check and not delete_only


def _strip_marked_block(
    text: str,
    *,
    begin_marker: str,
    end_marker: str,
    source: Path,
) -> str:
    """Remove one complete marked block or fail closed on marker drift."""
    lines = text.splitlines(keepends=True)
    begin_indexes = [index for index, line in enumerate(lines) if line.strip() == begin_marker]
    end_indexes = [index for index, line in enumerate(lines) if line.strip() == end_marker]
    if len(begin_indexes) != 1 or len(end_indexes) != 1:
        raise ValueError(
            "tokenless Fern filter marker must occur exactly once: "
            f"{begin_marker!r} / {end_marker!r} in {source}"
        )
    begin_index = begin_indexes[0]
    end_index = end_indexes[0]
    if begin_index >= end_index:
        raise ValueError(
            "tokenless Fern filter marker order is invalid: "
            f"{begin_marker!r} / {end_marker!r} in {source}"
        )
    return "".join(lines[:begin_index] + lines[end_index + 1 :])


def _rebase_tokenless_authored_paths(
    text: str,
    *,
    canonical_fern_dir: Path,
    effective_fern_dir: Path,
) -> str:
    """Point copied navigation directly at canonical authored source files."""

    def replace(match: re.Match[str]) -> str:
        canonical_path = canonical_fern_dir / match.group("path")
        rebased_path = Path(
            os.path.relpath(canonical_path, start=effective_fern_dir)
        ).as_posix()
        return (
            f"{match.group('prefix')}{rebased_path}"
            f"{match.group('suffix')}{match.group('newline')}"
        )

    return _TOKENLESS_AUTHORED_PATH_RE.sub(replace, text)


@contextmanager
def _tokenless_fern_project(fern_dir: Path) -> Iterator[Path]:
    """Yield a temporary docs-root/fern workspace without generated API config."""
    docs_root = fern_dir.parent
    with TemporaryDirectory(prefix=".fern-tokenless-", dir=docs_root) as tmpdir:
        workspace = Path(tmpdir)
        effective_dir = workspace / "fern"
        shutil.copytree(
            fern_dir,
            effective_dir,
            ignore=shutil.ignore_patterns(
                "generated",
                ".fern",
                ".preview",
                "node_modules",
                ".next",
                "out",
                "*.log",
            ),
        )
        filters = (
            (effective_dir / "docs.yml", _TOKENLESS_LIBRARY_MARKERS),
            (effective_dir / "index.yml", _TOKENLESS_API_NAVIGATION_MARKERS),
        )
        for config_path, markers in filters:
            if not config_path.is_file():
                raise ValueError(f"tokenless Fern project is missing {config_path}")
            filtered_text = _strip_marked_block(
                config_path.read_text(encoding="utf-8"),
                begin_marker=markers[0],
                end_marker=markers[1],
                source=config_path,
            )
            if config_path.name == "index.yml":
                filtered_text = _rebase_tokenless_authored_paths(
                    filtered_text,
                    canonical_fern_dir=fern_dir,
                    effective_fern_dir=effective_dir,
                )
            config_path.write_text(filtered_text, encoding="utf-8")
        yield effective_dir


def _needs_tokenless_fern_project(
    *,
    local_fern_action: bool,
    with_library_mdx: bool,
    cpp_api_present: bool,
) -> bool:
    """Return whether a local Fern action needs API config removed."""
    return local_fern_action and not with_library_mdx and not cpp_api_present


def _run_local_fern_action(args: argparse.Namespace, *, fern_dir: Path) -> int:
    if args.skip_fern_check and not args.preview:
        _print_section("Pipeline finished successfully")
        return 0

    if not _fern_assets_ok(fern_dir):
        print(
            f"error: missing dist theme assets at {fern_dir / 'dist'}",
            file=sys.stderr,
        )
        return 1

    fern_exe = _fern_exe()
    if not fern_exe:
        print("error: the `fern` CLI was not found on PATH.", file=sys.stderr)
        return 127

    if args.preview:
        _print_section("Local preview")
        _print_detail(f"Working directory: {fern_dir}")
        _print_detail("Running: fern docs dev")
        _run(
            [fern_exe, "docs", "dev"],
            cwd=fern_dir,
            show_command=args.verbose,
        )
    else:
        _print_section("Fern validation")
        _print_detail(f"Working directory: {fern_dir}")
        _print_detail("Running: fern check --local --warnings")
        _run(
            [fern_exe, "check", "--local", "--warnings"],
            cwd=fern_dir,
            show_command=args.verbose,
        )

    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    global BUILD_CONTEXT_PREFIX

    fern_dir = args.fern_dir.expanduser().resolve()
    preview_url_file = (
        args.preview_url_file.expanduser().resolve() if args.preview_url_file is not None else None
    )
    generated_docs = fern_dir / "generated"
    cpp_api_docs = generated_docs / "api-reference" / "cpp"

    if not SOURCE.is_dir():
        print(f"error: missing source directory: {SOURCE}", file=sys.stderr)
        return 1

    fern_exe = _fern_exe() if args.with_library_mdx else None
    if args.with_library_mdx:
        if not fern_exe:
            print(
                "error: the `fern` CLI was not found on PATH; install it or omit "
                "--with-library-mdx.",
                file=sys.stderr,
            )
            return 127
        runtime_error = _local_library_runtime_error()
        if runtime_error is not None:
            print(f"error: {runtime_error}.", file=sys.stderr)
            return 1

    documentation_version = _documentation_version()
    BUILD_CONTEXT_PREFIX = f"[{args.build_role} {documentation_version}] "
    _print_section("Documentation build context")
    _print_detail(f"Source ref: {args.source_ref or _checkout_source_ref()}")
    _print_detail(f"Purpose:    {_build_purpose(args)}")

    _print_section("Documentation build inputs")
    _print_detail(f"Repository: {REPO}")
    _print_detail(f"Source:     {SOURCE}")
    _print_detail(f"Fern:       {fern_dir}")
    _print_detail(f"Fern CLI:   {_configured_fern_version(fern_dir)}")

    if args.with_library_mdx:
        _print_section("C++ API generation: input")
        _print_detail("Library:       holoscan-cpp")
        _print_detail(f"Configuration: {fern_dir / 'docs.yml'}")
        _print_detail(f"Output:        {cpp_api_docs}")

    if not args.no_clean:
        _print_section(
            "C++ API generation: cleanup"
            if args.with_library_mdx
            else "C++ API generation: reuse"
        )
        _clean_generated_api_docs(
            generated_docs=generated_docs,
            preserve=not args.with_library_mdx,
        )

    if args.with_library_mdx:
        assert fern_exe is not None
        if not FIX_LIBRARY_MDX.is_file():
            print(f"error: missing post-process script: {FIX_LIBRARY_MDX}", file=sys.stderr)
            return 2
        _print_section("C++ API generation")
        _print_detail(f"Working directory: {DOCS_ROOT}")
        _print_detail("Running: fern docs md generate --local")
        tmpdir = Path(tempfile.mkdtemp(prefix=".fern-library-", dir=DOCS_ROOT))
        env = os.environ.copy()
        env["TMPDIR"] = str(tmpdir)
        result = _run(
            [fern_exe, "docs", "md", "generate", "--local"],
            cwd=DOCS_ROOT,
            env=env,
            check=False,
            show_command=args.verbose,
        )
        if result.returncode:
            print(
                f"error: fern docs md generate --local exited with {result.returncode}.",
                file=sys.stderr,
            )
            print(f"Fern temporary files retained at: {tmpdir}", file=sys.stderr)
            return result.returncode
        shutil.rmtree(tmpdir)

    cpp_api_present = cpp_api_docs.is_dir() and any(
        path.is_file() for path in cpp_api_docs.rglob("*.mdx")
    )
    if args.publish and not cpp_api_present:
        print(
            "error: production publication requires a non-empty generated C++ API "
            f"MDX tree at {cpp_api_docs}. Run with --with-library-mdx first.",
            file=sys.stderr,
        )
        return 1

    if generated_docs.is_dir() and not FIX_LIBRARY_MDX.is_file():
        print(f"error: missing post-process script: {FIX_LIBRARY_MDX}", file=sys.stderr)
        return 2

    if generated_docs.is_dir():
        _print_section("C++ API generation: post-processing")
        _print_detail(f"Working directory: {generated_docs}")
        _print_detail(f"Running: {FIX_LIBRARY_MDX.name}")
        fix_command = [
            sys.executable,
            str(FIX_LIBRARY_MDX),
            "--root",
            str(generated_docs),
        ]
        if args.publish and args.skip_library_mdx:
            fix_command.append("--check")
        _run(fix_command, indent_output=True, show_command=args.verbose)

    if _should_check_doc_links(args):
        if not CHECK_SOURCE_LINKS.is_file():
            print(f"error: missing source-link check script: {CHECK_SOURCE_LINKS}", file=sys.stderr)
            return 2
        source_link_command = [
            sys.executable,
            str(CHECK_SOURCE_LINKS),
            "--docs-root",
            str(SOURCE),
        ]
        if args.allow_rendered_source_links:
            source_link_command.append("--allow-rendered")
        for browser_url in args.allowed_source_browser_url:
            source_link_command.extend(["--allowed-source-browser-url", browser_url])
        for source_ref in args.allowed_source_ref:
            source_link_command.extend(["--allowed-source-ref", source_ref])
        source_link_check = _run(
            source_link_command,
            check=False,
            indent_output=True,
            show_command=args.verbose,
        )
        if source_link_check.returncode:
            print(
                "error: SDK source links did not validate "
                "(pass --skip-link-check to bypass).",
                file=sys.stderr,
            )
            return source_link_check.returncode
        if not CHECK_DOC_LINKS.is_file():
            print(f"error: missing link check script: {CHECK_DOC_LINKS}", file=sys.stderr)
            return 2
        _print_section("Internal link checks")
        _print_detail(f"Working directory: {REPO}")
        _print_detail(f"Running: {CHECK_DOC_LINKS.name}")
        link_check = _run(
            [
                sys.executable,
                str(CHECK_DOC_LINKS),
                "--docs-root",
                str(SOURCE),
                "--fern-dir",
                str(fern_dir),
            ],
            check=False,
            indent_output=True,
            show_command=args.verbose,
        )
        if link_check.returncode:
            print(
                "error: internal documentation links did not validate "
                "(pass --skip-link-check to bypass).",
                file=sys.stderr,
            )
            return link_check.returncode

    needs_remote_publish = args.publish or args.publish_preview or args.delete_preview_id
    if needs_remote_publish:
        action = "Production Fern publication" if args.publish else "Remote Fern preview"
        if (rc := _require_fern_auth(action=action)) is not None:
            return rc
        fern_exe = _fern_exe()
        if not fern_exe:
            print("error: the `fern` CLI was not found on PATH.", file=sys.stderr)
            return 127
        if not _fern_assets_ok(fern_dir):
            print(
                f"error: missing dist theme assets at {fern_dir / 'dist'}",
                file=sys.stderr,
            )
            return 1
        _print_section(action)
        if args.publish:
            _print_detail(f"Working directory: {fern_dir}")
            _print_detail("Running: fern generate --docs")
            _publish_production(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                verbose=args.verbose,
            )
        if args.delete_preview_id:
            _print_detail(f"Working directory: {fern_dir}")
            _print_detail(f"Running: fern docs preview delete --id {args.delete_preview_id}")
            _delete_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.delete_preview_id,
                verbose=args.verbose,
            )
        if args.publish_preview:
            if not args.preview_id:
                print("error: --publish-preview requires --preview-id.", file=sys.stderr)
                return 2
            if preview_url_file is not None:
                preview_url_file.parent.mkdir(parents=True, exist_ok=True)
                preview_url_file.write_text("", encoding="utf-8")
            _print_detail(f"Working directory: {fern_dir}")
            preview_command = f"fern generate --docs --preview --id {args.preview_id}"
            if args.force:
                preview_command += " --force"
            _print_detail(f"Running: {preview_command}")
            preview_url = _publish_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.preview_id,
                force=args.force,
                preview_url_file=preview_url_file,
                verbose=args.verbose,
            )
            if preview_url_file is not None and not preview_url:
                print(
                    f"error: Fern preview URL was not written to {preview_url_file}",
                    file=sys.stderr,
                )
                return 1
        _print_section("Pipeline finished successfully")
        return 0

    local_fern_action = args.preview or not args.skip_fern_check
    if _needs_tokenless_fern_project(
        local_fern_action=local_fern_action,
        with_library_mdx=args.with_library_mdx,
        cpp_api_present=cpp_api_present,
    ):
        _print_section("Tokenless local Fern project")
        _print_detail("Generated C++ API: not found; omitting API reference")
        try:
            with _tokenless_fern_project(fern_dir) as effective_fern_dir:
                _print_detail(f"Effective Fern: {effective_fern_dir}")
                return _run_local_fern_action(args, fern_dir=effective_fern_dir)
        except ValueError as exc:
            print(f"error: unable to prepare tokenless Fern project: {exc}", file=sys.stderr)
            return 2

    return _run_local_fern_action(args, fern_dir=fern_dir)


def _docker_image_tag() -> str:
    version_path = DOCS_ROOT.parent / "VERSION"
    version = (
        version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "local"
    )
    return f"holoscan-docs:{version}"


def _mount_host_fern_auth(*, docker_home: Path, docker_mounts: list[str]) -> None:
    """Bind-mount host Fern login credentials into the container HOME.

    Only ``token`` and ``id`` are shared. Logs, app-preview caches, and other
    mutable Fern state stay in the container HOME so ``fern docs dev`` does not
    watch and reload on files written under the mounted directory.
    """
    host_fern_dir = Path.home() / ".fern"
    if not host_fern_dir.is_dir():
        return
    container_fern_dir = docker_home / ".fern"
    container_fern_dir.mkdir(parents=True, exist_ok=True)
    for name in ("token", "id"):
        host_file = host_fern_dir / name
        if host_file.is_file():
            docker_mounts.append(f"{host_file}:{container_fern_dir / name}")


def _active_docker_socket() -> str:
    """Return the Unix socket used by the active Docker context."""
    docker_host = os.environ.get("DOCKER_HOST", "").strip()
    if not docker_host:
        result = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        docker_host = result.stdout.strip()
        if result.returncode or not docker_host:
            detail = result.stderr.strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise RuntimeError(f"could not determine the active Docker socket{suffix}")

    if not docker_host.startswith("unix://"):
        raise RuntimeError(f"the active Docker endpoint is not a Unix socket: {docker_host}")
    socket_path = docker_host.removeprefix("unix://")
    if not socket_path.startswith("/"):
        raise RuntimeError(f"the active Docker socket path is not absolute: {socket_path}")
    return socket_path


def _container_docker_options(*, image: str, with_library_mdx: bool) -> list[str]:
    """Return docker-run options that let Fern start its local parser."""
    if not with_library_mdx:
        return []

    socket_mount = f"{_active_docker_socket()}:/var/run/docker.sock"
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            socket_mount,
            image,
            "stat",
            "-c",
            "%g",
            "/var/run/docker.sock",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    socket_gid = result.stdout.strip()
    if result.returncode or not socket_gid.isdigit():
        detail = result.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"could not inspect the Docker daemon socket{suffix}")
    return [
        "-eDOCKER_HOST=unix:///var/run/docker.sock",
        "--group-add",
        socket_gid,
        "-v",
        socket_mount,
    ]


@contextmanager
def _external_docker_home(
    *, watched_roots: tuple[Path, ...], docker_mounts: list[str]
) -> Iterator[Path]:
    """Yield a mounted per-run HOME outside Fern's watched content roots."""
    with TemporaryDirectory(prefix="holoscan-docs-home-") as tmpdir:
        docker_home = Path(tmpdir).resolve()
        overlapping_root = next(
            (root for root in watched_roots if _is_under(docker_home, root.resolve())),
            None,
        )
        if overlapping_root is not None:
            raise ValueError(
                f"temporary Docker HOME {docker_home} overlaps watched content "
                f"root {overlapping_root.resolve()}"
            )
        docker_mounts.append(f"{docker_home}:{docker_home}")
        yield docker_home


def _run_docker_container(
    args: argparse.Namespace,
    *,
    image: str,
    repo_root: Path,
    script_repo: Path,
    fern_dir: Path,
    docker_mounts: list[str],
    docker_options: list[str],
    shell_cmd: str,
) -> int:
    with _external_docker_home(
        watched_roots=(repo_root, script_repo, fern_dir.parent),
        docker_mounts=docker_mounts,
    ) as docker_home:
        docker_env = [f"-eHOME={docker_home}"]
        if os.environ.get("CI"):
            docker_env.append("-eCI=1")

        # Share Fern authentication with the container. Prefer an explicit
        # ``FERN_TOKEN`` and otherwise mount only the host credential files.
        # Mutable Fern state remains in the external per-run HOME so its own
        # debug log cannot retrigger a docs-tree watcher.
        if os.environ.get("FERN_TOKEN"):
            docker_env.append("-eFERN_TOKEN")
        else:
            _mount_host_fern_auth(docker_home=docker_home, docker_mounts=docker_mounts)

        docker_cmd = [
            "docker",
            "run",
            *docker_env,
            *docker_options,
            "--rm",
            "--net",
            "host",
            "--name",
            args.container_name,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
        ]
        for mount in docker_mounts:
            docker_cmd.extend(["-v", mount])
        docker_cmd.extend(
            [
                "-w",
                str(repo_root),
                image,
                "sh",
                "-c",
                shell_cmd,
            ]
        )

        if (args.preview or args.publish or args.publish_preview) and sys.stdin.isatty():
            docker_cmd.insert(2, "-ti")

        if args.verbose:
            print("+ " + shlex.join(docker_cmd), flush=True)
        return subprocess.run(docker_cmd, cwd=REPO, check=False).returncode


def _run_in_docker(args: argparse.Namespace) -> int:
    if not DOCKERFILE.is_file():
        print(f"Missing Dockerfile: {DOCKERFILE}", file=sys.stderr)
        return 1

    image = _docker_image_tag()
    _run(
        [
            "docker",
            "build",
            "--network=host",
            "-t",
            image,
            "-f",
            str(DOCKERFILE),
            str(DOCS_ROOT),
        ],
        show_command=args.verbose,
    )

    repo_root = REPO.resolve()
    script_repo = SCRIPT_REPO.resolve()
    docs_root = DOCS_ROOT.resolve()
    fern_dir = args.fern_dir.expanduser().resolve()
    docker_mounts, container_fern_dir = _container_mounts(
        repo_root=repo_root,
        script_repo=script_repo,
        docs_root=docs_root,
        fern_dir=fern_dir,
    )
    try:
        docker_mounts.extend(_external_git_metadata_mounts(repo_root))
    except ValueError as exc:
        print(f"error: unable to resolve Git metadata for Docker: {exc}", file=sys.stderr)
        return 2

    container_preview_url_file = None
    if args.preview_url_file is not None:
        host_preview_url_file = args.preview_url_file.expanduser().resolve()
        if not _is_under(fern_dir, repo_root) and _is_under(host_preview_url_file, fern_dir):
            container_preview_url_file = container_fern_dir / host_preview_url_file.relative_to(
                fern_dir
            )
        elif _is_under(host_preview_url_file, repo_root) or _is_under(
            host_preview_url_file, script_repo
        ):
            container_preview_url_file = host_preview_url_file
        else:
            host_preview_url_file.parent.mkdir(parents=True, exist_ok=True)
            host_preview_url_file.touch(exist_ok=True)
            docker_mounts.append(f"{host_preview_url_file}:{EXTERNAL_PREVIEW_URL_FILE_MOUNT}")
            container_preview_url_file = EXTERNAL_PREVIEW_URL_FILE_MOUNT

    pipeline_args = [
        "python3",
        "-u",
        str(SCRIPTS / "build_holoscan_docs.py"),
        "--no-docker",
        "--repository-checkout",
        str(repo_root),
        "--docs-root",
        str(docs_root),
        "--fern-dir",
        str(container_fern_dir),
        "--build-role",
        args.build_role,
    ]
    if args.source_ref:
        pipeline_args.extend(["--source-ref", args.source_ref])
    if container_preview_url_file is not None:
        pipeline_args.extend(["--preview-url-file", str(container_preview_url_file)])
    if args.no_clean:
        pipeline_args.append("--no-clean")
    if args.with_library_mdx:
        pipeline_args.append("--with-library-mdx")
    if args.skip_library_mdx:
        pipeline_args.append("--skip-library-mdx")
    if args.skip_fern_check:
        pipeline_args.append("--skip-fern-check")
    if args.skip_link_check:
        pipeline_args.append("--skip-link-check")
    if args.allow_rendered_source_links:
        pipeline_args.append("--allow-rendered-source-links")
    for browser_url in args.allowed_source_browser_url:
        pipeline_args.extend(["--allowed-source-browser-url", browser_url])
    for source_ref in args.allowed_source_ref:
        pipeline_args.extend(["--allowed-source-ref", source_ref])
    if args.preview:
        pipeline_args.append("--preview")
    if args.publish:
        pipeline_args.append("--publish")
    if args.publish_preview:
        pipeline_args.append("--publish-preview")
    if args.preview_id:
        pipeline_args.extend(["--preview-id", args.preview_id])
    if args.delete_preview_id:
        pipeline_args.extend(["--delete-preview-id", args.delete_preview_id])
    if args.force:
        pipeline_args.append("--force")
    if args.verbose:
        pipeline_args.append("--verbose")

    pipeline_cmd = " ".join(shlex.quote(part) for part in pipeline_args)
    shell_cmd = (
        f"git config --global --add safe.directory {shlex.quote(str(REPO))} 2>/dev/null || true; "
        f"{pipeline_cmd}"
    )
    try:
        docker_options = _container_docker_options(
            image=image,
            with_library_mdx=args.with_library_mdx,
        )
    except RuntimeError as exc:
        print(
            f"error: Docker-container C++ API generation {exc}; "
            "use a local Docker daemon or run with --no-docker.",
            file=sys.stderr,
        )
        return 1
    try:
        return _run_docker_container(
            args,
            image=image,
            repo_root=repo_root,
            script_repo=script_repo,
            fern_dir=fern_dir,
            docker_mounts=docker_mounts,
            docker_options=docker_options,
            shell_cmd=shell_cmd,
        )
    except ValueError as exc:
        print(f"error: unable to prepare external Docker HOME: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, validate, preview, or publish Holoscan SDK Fern documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES,
    )
    parser.add_argument(
        "--fern-dir",
        type=Path,
        default=None,
        help="Fern project directory (default: <detected-documentation-root>/fern).",
    )
    parser.add_argument(
        "--repository-checkout",
        type=Path,
        default=SCRIPT_REPO,
        help=(
            "HSDK checkout supplying the documentation source "
            "(default: the checkout containing this script)."
        ),
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        help=(
            "Documentation source tree (default: public/docs or docs in the selected "
            "repository checkout). Release publication uses this for its rendered "
            "staging copy."
        ),
    )
    parser.add_argument(
        "--build-role",
        choices=("archive", "current"),
        default="current",
        help="Label this run as an archive or current documentation build.",
    )
    parser.add_argument(
        "--source-ref",
        help="Source ref to report for an orchestrated build (default: derive from checkout).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip removing prior C++ API MDX under generated/.",
    )
    parser.add_argument(
        "--with-library-mdx",
        action="store_true",
        help=(
            "Generate fresh C++ API MDX in the selected local fern/generated tree "
            "using the configured library input. The configured public Git repository "
            "is parsed on Fern's servers and requires authentication."
        ),
    )
    parser.add_argument(
        "--skip-library-mdx",
        action="store_true",
        help=(
            "Reuse the existing local fern/generated C++ API MDX tree. "
            "The tree remains included in Fern output. Production publication "
            "requires it to be non-empty and fully post-processed."
        ),
    )
    parser.add_argument(
        "--skip-fern-check",
        action="store_true",
        help="Run the pipeline only; skip fern check / preview.",
    )
    parser.add_argument(
        "--skip-link-check",
        action="store_true",
        help="Skip source-link and internal-link validation.",
    )
    parser.add_argument(
        "--allow-rendered-source-links",
        action="store_true",
        help=(
            "Accept one consistently rendered release source target instead of "
            "requiring committed GitHub/main links."
        ),
    )
    parser.add_argument(
        "--allowed-source-browser-url",
        action="append",
        default=[],
        help=(
            "Additional Holoscan SDK repository browser accepted when validating "
            "rendered source links (repeatable)."
        ),
    )
    parser.add_argument(
        "--allowed-source-ref",
        action="append",
        default=[],
        help="Additional exact source ref accepted for rendered source links (repeatable).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run fern docs dev after the pipeline (live preview).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the selected checkout's Fern documentation to production.",
    )
    parser.add_argument(
        "--publish-preview",
        action="store_true",
        help="Publish a remote Fern docs preview (fern generate --docs --preview).",
    )
    parser.add_argument(
        "--preview-id",
        help="Stable preview id for --publish-preview (typically a branch name).",
    )
    parser.add_argument(
        "--preview-url-file",
        type=Path,
        help=(
            "Write the published preview URL to this file. When provided, "
            "publication fails if a URL cannot be determined."
        ),
    )
    parser.add_argument(
        "--delete-preview-id",
        help="Delete a remote Fern preview with this id before publishing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip overwrite confirmation for remote preview publish (CI).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print commands and arguments for all documentation subprocesses.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Run on the host instead of the docs container.",
    )
    parser.add_argument(
        "--container-name",
        default="docs",
        help="Docker container name (default: docs).",
    )
    args = parser.parse_args()

    try:
        _configure_repository_checkout(args.repository_checkout.expanduser().resolve())
    except ValueError as exc:
        print(f"error: invalid --repository-checkout: {exc}", file=sys.stderr)
        return 2

    if args.docs_root is not None:
        try:
            _configure_docs_root(args.docs_root)
        except ValueError as exc:
            print(f"error: invalid --docs-root: {exc}", file=sys.stderr)
            return 2

    if args.fern_dir is None:
        args.fern_dir = DOCS_ROOT / "fern"

    if args.publish and (args.preview or args.publish_preview or args.delete_preview_id):
        print(
            "error: --publish cannot be combined with preview publication or deletion options.",
            file=sys.stderr,
        )
        return 2

    if args.skip_library_mdx:
        args.with_library_mdx = False
    elif not args.with_library_mdx:
        args.with_library_mdx = _resolve_with_library_mdx(None)

    if args.no_docker:
        return run_pipeline(args)
    return _run_in_docker(args)


if __name__ == "__main__":
    raise SystemExit(main())
