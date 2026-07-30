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
2. With ``--with-library-mdx``: ``fern docs md generate`` + ``fix_generated_library_mdx.py``.
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
  python3 docs/scripts/build_holoscan_docs.py --delete-preview-id my-branch --publish-preview --preview-id main --force
  python3 /path/to/build_holoscan_docs.py --repository-checkout /path/to/release-checkout --no-docker --publish --skip-library-mdx

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
from pathlib import Path

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
DOCKERFILE = DOCS_ROOT / "Dockerfile"

FERN_API_VERSION = "5.82.0"
EXTERNAL_FERN_DIR_MOUNT = Path("/mnt/fern_dir")
EXTERNAL_PREVIEW_URL_FILE_MOUNT = Path("/mnt/fern_preview_url")
PREVIEW_URL_FILE = ".fern-preview-url"
_PREVIEW_URL_RE = re.compile(
    r"https://[^\s\"'<>)\]]+\.docs\.buildwithfern\.com[^\s\"'<>)\]]*",
    re.IGNORECASE,
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
    --with-library-mdx    generate fresh local output from the configured
                          library input
    --skip-library-mdx    reuse the existing local output tree

  docs.yml configures https://github.com/nvidia-holoscan/holoscan-sdk with
  subpath include/holoscan as the C++ API source. Fern parses that public
  source on its servers (authentication required) and writes the returned
  MDX into the selected local fern/generated directory. A preview ID names a
  deployment; it does not select a library Git revision.

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


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd or REPO, check=check)


def _fern_exe() -> str | None:
    return shutil.which("fern")


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


def _delete_remote_preview(*, fern_exe: str, fern_dir: Path, preview_id: str) -> None:
    print(f"Deleting Fern preview: {preview_id}", flush=True)
    result = _run(
        [fern_exe, "docs", "preview", "delete", "--id", preview_id],
        cwd=fern_dir,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"warning: fern docs preview delete --id {preview_id!r} exited "
            f"{result.returncode} (preview may already be gone).",
            file=sys.stderr,
        )


def _publish_remote_preview(
    *,
    fern_exe: str,
    fern_dir: Path,
    preview_id: str,
    force: bool,
    preview_url_file: Path | None = None,
) -> str | None:
    cmd = [fern_exe, "generate", "--docs", "--preview", "--id", preview_id]
    if force:
        cmd.append("--force")
    print("+ " + " ".join(cmd), flush=True)
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


def _publish_production(*, fern_exe: str, fern_dir: Path) -> None:
    """Publish the selected checkout's HSDK User Guide to production."""
    _run([fern_exe, "generate", "--docs"], cwd=fern_dir)


def _repo_relative(path: Path) -> Path | str:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def _clean_generated_api_docs(*, generated_docs: Path, preserve: bool = False) -> None:
    if not generated_docs.is_dir():
        return
    if preserve:
        print(
            f"reusing existing local C++ API MDX: {_repo_relative(generated_docs)}; "
            "it remains part of Fern validation, preview, or publication "
            "(pass --with-library-mdx to regenerate it)",
            flush=True,
        )
        return
    shutil.rmtree(generated_docs)
    print(f"removed: {_repo_relative(generated_docs)}")


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


def run_pipeline(args: argparse.Namespace) -> int:
    fern_dir = args.fern_dir.expanduser().resolve()
    preview_url_file = (
        args.preview_url_file.expanduser().resolve()
        if args.preview_url_file is not None
        else None
    )
    generated_docs = fern_dir / "generated"
    cpp_api_docs = generated_docs / "api-reference" / "cpp"

    if not SOURCE.is_dir():
        print(f"Missing source dir: {SOURCE}", file=sys.stderr)
        return 1

    print("Holoscan docs pipeline")
    print(f"  Repo: {REPO}")
    print(f"  Source: {SOURCE}")
    print(f"  Fern: {fern_dir}")

    if not args.no_clean:
        _clean_generated_api_docs(
            generated_docs=generated_docs,
            preserve=not args.with_library_mdx,
        )

    if args.with_library_mdx:
        fern_exe = _fern_exe()
        if not fern_exe:
            print(
                "The `fern` CLI was not found on PATH; install it or omit --with-library-mdx.",
                file=sys.stderr,
            )
            return 127
        if not FIX_LIBRARY_MDX.is_file():
            print(f"Missing post-process script: {FIX_LIBRARY_MDX}", file=sys.stderr)
            return 2
        _run([fern_exe, "docs", "md", "generate"], cwd=DOCS_ROOT)

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
        print(f"Missing post-process script: {FIX_LIBRARY_MDX}", file=sys.stderr)
        return 2

    if generated_docs.is_dir():
        fix_command = [
            sys.executable,
            str(FIX_LIBRARY_MDX),
            "--root",
            str(generated_docs),
        ]
        if args.publish and args.skip_library_mdx:
            fix_command.append("--check")
        _run(fix_command)

    if _should_check_doc_links(args):
        if not CHECK_DOC_LINKS.is_file():
            print(f"Missing link check script: {CHECK_DOC_LINKS}", file=sys.stderr)
            return 2
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
            print("The `fern` CLI was not found on PATH.", file=sys.stderr)
            return 127
        if not _fern_assets_ok(fern_dir):
            print(
                f"error: missing dist theme assets at {fern_dir / 'dist'}",
                file=sys.stderr,
            )
            return 1
        if args.publish:
            _publish_production(fern_exe=fern_exe, fern_dir=fern_dir)
        if args.delete_preview_id:
            _delete_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.delete_preview_id,
            )
        if args.publish_preview:
            if not args.preview_id:
                print("error: --publish-preview requires --preview-id.", file=sys.stderr)
                return 2
            if preview_url_file is not None:
                preview_url_file.parent.mkdir(parents=True, exist_ok=True)
                preview_url_file.write_text("", encoding="utf-8")
            preview_url = _publish_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.preview_id,
                force=args.force,
                preview_url_file=preview_url_file,
            )
            if preview_url_file is not None and not preview_url:
                print(
                    f"error: Fern preview URL was not written to {preview_url_file}",
                    file=sys.stderr,
                )
                return 1
        print("\nPipeline finished successfully.", flush=True)
        return 0

    if args.skip_fern_check and not args.preview:
        print("\nPipeline finished successfully.", flush=True)
        return 0

    if not _fern_assets_ok(fern_dir):
        print(
            f"error: missing dist theme assets at {fern_dir / 'dist'}",
            file=sys.stderr,
        )
        return 1

    fern_exe = _fern_exe()
    if not fern_exe:
        print("The `fern` CLI was not found on PATH.", file=sys.stderr)
        return 127

    if args.preview:
        _run([fern_exe, "docs", "dev"], cwd=fern_dir)
    else:
        _run([fern_exe, "check", "--local", "--warnings"], cwd=fern_dir)

    return 0


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
        ]
    )

    repo_root = REPO.resolve()
    script_repo = SCRIPT_REPO.resolve()
    fern_dir = args.fern_dir.expanduser().resolve()
    if _is_under(fern_dir, repo_root):
        container_fern_dir = fern_dir
        docker_mounts = [f"{repo_root}:{repo_root}"]
    else:
        container_fern_dir = EXTERNAL_FERN_DIR_MOUNT
        docker_mounts = [
            f"{repo_root}:{repo_root}",
            f"{fern_dir}:{EXTERNAL_FERN_DIR_MOUNT}",
        ]
    if script_repo != repo_root:
        docker_mounts.append(f"{script_repo}:{script_repo}")

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
            docker_mounts.append(
                f"{host_preview_url_file}:{EXTERNAL_PREVIEW_URL_FILE_MOUNT}"
            )
            container_preview_url_file = EXTERNAL_PREVIEW_URL_FILE_MOUNT

    pipeline_args = [
        "python3",
        str(SCRIPTS / "build_holoscan_docs.py"),
        "--no-docker",
        "--repository-checkout",
        str(repo_root),
        "--fern-dir",
        str(container_fern_dir),
    ]
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

    pipeline_cmd = " ".join(shlex.quote(part) for part in pipeline_args)
    shell_cmd = (
        f"git config --global --add safe.directory {shlex.quote(str(REPO))} 2>/dev/null || true; "
        f"{pipeline_cmd}"
    )

    docker_home = DOCS_ROOT / ".fern-docker-home"
    docker_home.mkdir(parents=True, exist_ok=True)

    docker_env = [f"-eHOME={docker_home}"]
    if os.environ.get("CI"):
        docker_env.append("-eCI=1")

    # Share Fern authentication with the container. Prefer an explicit ``FERN_TOKEN``
    # (CI or an exported organization token) and pass it straight through. Otherwise
    # bind-mount only the host ``~/.fern`` credential files (``token``, ``id``) so the
    # containerized Fern CLI reuses ``fern login`` without prompting for an interactive
    # browser login, while keeping logs and preview caches out of the mount.
    if os.environ.get("FERN_TOKEN"):
        docker_env.append("-eFERN_TOKEN")
    else:
        _mount_host_fern_auth(docker_home=docker_home, docker_mounts=docker_mounts)

    docker_cmd = [
        "docker",
        "run",
        *docker_env,
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

    if (args.preview or args.publish) and sys.stdin.isatty():
        docker_cmd.insert(2, "-ti")

    print("+ " + " ".join(docker_cmd), flush=True)
    return subprocess.run(docker_cmd, cwd=REPO, check=False).returncode


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
        help="Skip check_doc_links.py internal link validation.",
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

    if args.with_library_mdx and not _fern_auth_token():
        print(
            "error: C++ API generation requires Fern auth (run 'fern login' or set FERN_TOKEN).",
            file=sys.stderr,
        )
        return 1

    if args.no_docker:
        return run_pipeline(args)
    return _run_in_docker(args)


if __name__ == "__main__":
    raise SystemExit(main())
