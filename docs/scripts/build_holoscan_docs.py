#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Build and validate Holoscan SDK Fern documentation.

User guide pages are committed as ``*.mdx`` under ``public/docs/``. Fern config
(``fern.config.json``, ``docs.yml``, ``index.yml``, ``assets/``, ``dist/``) lives under
``public/docs/fern/``. C++ API pages are generated under ``public/docs/fern/generated/``
(gitignored).

1. Optionally remove prior C++ API MDX under ``generated/``.
2. With ``--with-library-mdx``: ``fern docs md generate`` + ``fix_generated_library_mdx.py``.
3. ``fern check --local --warnings`` (local validation), ``fern docs dev`` (``--preview``),
   or ``fern generate --docs --preview`` (``--publish-preview`` for CI).

Usage (from holoscan-sdk repo root):

  python3 public/docs/scripts/build_holoscan_docs.py
  python3 public/docs/scripts/build_holoscan_docs.py --preview
  python3 public/docs/scripts/build_holoscan_docs.py --no-docker
  python3 public/docs/scripts/build_holoscan_docs.py --with-library-mdx
  python3 public/docs/scripts/build_holoscan_docs.py --skip-fern-check
  python3 public/docs/scripts/build_holoscan_docs.py --publish-preview --preview-id my-branch --force
  python3 public/docs/scripts/build_holoscan_docs.py --delete-preview-id my-branch --publish-preview --preview-id main --force
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
DOCS_ROOT = SCRIPTS.parent
REPO = DOCS_ROOT.parent.parent
SOURCE = DOCS_ROOT
DEFAULT_FERN_DIR = DOCS_ROOT / "fern"
FIX_LIBRARY_MDX = SCRIPTS / "fix_generated_library_mdx.py"
DOCKERFILE = DOCS_ROOT / "Dockerfile"

ASSET_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
SKIP_SOURCE_DIRS = frozenset({"fern", "scripts", "vale", "_static", "_templates"})
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
FERN_API_VERSION = "5.44.1"
EXTERNAL_FERN_DIR_MOUNT = Path("/mnt/fern_dir")
PREVIEW_URL_FILE = ".fern-preview-url"
_PREVIEW_URL_RE = re.compile(
    r"https://[^\s\"'<>)\]]+\.docs\.buildwithfern\.com[^\s\"'<>)\]]*",
    re.IGNORECASE,
)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[: len(_LFS_POINTER_PREFIX)] == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _iter_source_assets() -> list[Path]:
    assets: list[Path] = []
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(SOURCE)
        if rel.parts and rel.parts[0] in SKIP_SOURCE_DIRS:
            continue
        if any(part in SKIP_SOURCE_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in ASSET_EXTS:
            assets.append(path)
    return assets


def _ensure_lfs_assets() -> bool:
    if not shutil.which("git-lfs"):
        return False
    try:
        _run(["git", "lfs", "pull", "--include=public/docs/**"])
        return True
    except subprocess.CalledProcessError:
        print("warning: git lfs pull failed; images may be missing.", file=sys.stderr)
        return False


def _report_lfs_pointers(paths: list[Path]) -> int:
    if not paths:
        return 0
    print(
        f"\nerror: {len(paths)} image(s) under public/docs/ are Git LFS pointers, not "
        "binary files. Fern cannot display them until LFS objects are downloaded.",
        file=sys.stderr,
    )
    print("Install git-lfs, then from the repository root run:", file=sys.stderr)
    print("  git lfs install", file=sys.stderr)
    print("  git lfs pull --include='public/docs/**'", file=sys.stderr)
    print("\nAffected files:", file=sys.stderr)
    for path in paths[:20]:
        print(f"  - {path.relative_to(REPO)}", file=sys.stderr)
    if len(paths) > 20:
        print(f"  ... and {len(paths) - 20} more", file=sys.stderr)
    return 1


def _check_source_assets(*, allow_lfs_pointers: bool) -> int:
    pointers = [p for p in _iter_source_assets() if _is_lfs_pointer(p)]
    if not pointers:
        return 0
    if allow_lfs_pointers:
        print(
            f"warning: {len(pointers)} image(s) are Git LFS pointers; docs preview may show "
            "broken images.",
            file=sys.stderr,
        )
        return 0
    return _report_lfs_pointers(pointers)


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


def _write_preview_url(*, fern_dir: Path, url: str) -> None:
    preview_url_path = fern_dir / PREVIEW_URL_FILE
    preview_url_path.write_text(f"{url}\n", encoding="utf-8")
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
        _write_preview_url(fern_dir=fern_dir, url=url)
    else:
        print("warning: could not determine Fern docs preview URL", file=sys.stderr)
    return url


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
            f"preserving: {_repo_relative(generated_docs)} "
            "(pass --with-library-mdx to regenerate C++ API docs)",
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


def run_pipeline(args: argparse.Namespace) -> int:
    fern_dir = args.fern_dir.expanduser().resolve()
    generated_docs = fern_dir / "generated"

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

    if not args.skip_lfs_pull:
        if _ensure_lfs_assets():
            print("Git LFS: pulled public/docs image assets.")
        elif shutil.which("git-lfs") is None:
            print("Git LFS: git-lfs not on PATH; skipping automatic pull.")

    rc = _check_source_assets(allow_lfs_pointers=args.allow_lfs_pointers)
    if rc:
        return rc

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
        _run([fern_exe, "docs", "md", "generate"], cwd=fern_dir)

    if generated_docs.is_dir() and FIX_LIBRARY_MDX.is_file():
        _run([sys.executable, str(FIX_LIBRARY_MDX)])

    needs_remote_publish = args.publish_preview or args.delete_preview_id
    if needs_remote_publish:
        if (rc := _require_fern_auth(action="Remote Fern preview")) is not None:
            return rc
        fern_exe = _fern_exe()
        if not fern_exe:
            print("The `fern` CLI was not found on PATH.", file=sys.stderr)
            return 127
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
            if not _fern_assets_ok(fern_dir):
                print(
                    "error: missing dist theme assets (commit public/docs/fern/dist/)",
                    file=sys.stderr,
                )
                return 1
            _publish_remote_preview(
                fern_exe=fern_exe,
                fern_dir=fern_dir,
                preview_id=args.preview_id,
                force=args.force,
            )
        print("\nPipeline finished successfully.", flush=True)
        return 0

    if args.skip_fern_check and not args.preview:
        print("\nPipeline finished successfully.", flush=True)
        return 0

    if not _fern_assets_ok(fern_dir):
        print(
            "error: missing dist theme assets (commit public/docs/fern/dist/)",
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
    version_path = REPO / "VERSION"
    version = (
        version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "local"
    )
    return f"holoscan-docs:{version}"


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

    pipeline_args = [
        "python3",
        str(SCRIPTS / "build_holoscan_docs.py"),
        "--no-docker",
        "--fern-dir",
        str(container_fern_dir),
    ]
    if args.no_clean:
        pipeline_args.append("--no-clean")
    if args.allow_lfs_pointers:
        pipeline_args.append("--allow-lfs-pointers")
    if args.skip_lfs_pull:
        pipeline_args.append("--skip-lfs-pull")
    if args.with_library_mdx:
        pipeline_args.append("--with-library-mdx")
    if args.skip_library_mdx:
        pipeline_args.append("--skip-library-mdx")
    if args.skip_fern_check:
        pipeline_args.append("--skip-fern-check")
    if args.preview:
        pipeline_args.append("--preview")
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
    token = _fern_auth_token()
    if token:
        docker_env.append("-eFERN_TOKEN")
        os.environ.setdefault("FERN_TOKEN", token)

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

    if args.preview and sys.stdin.isatty():
        docker_cmd.insert(2, "-ti")

    print("+ " + " ".join(docker_cmd), flush=True)
    return subprocess.run(docker_cmd, cwd=REPO, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate Holoscan SDK Fern documentation."
    )
    parser.add_argument(
        "--fern-dir",
        type=Path,
        default=DEFAULT_FERN_DIR,
        help="Fern project directory (default: public/docs/fern).",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip removing prior C++ API MDX under generated/.",
    )
    parser.add_argument(
        "--allow-lfs-pointers",
        action="store_true",
        help="Do not fail when public/docs images are unresolved Git LFS pointers.",
    )
    parser.add_argument(
        "--skip-lfs-pull",
        action="store_true",
        help="Do not run ``git lfs pull`` before validating image assets.",
    )
    parser.add_argument(
        "--with-library-mdx",
        action="store_true",
        help="Generate C++ API MDX (requires Fern auth).",
    )
    parser.add_argument(
        "--skip-library-mdx",
        action="store_true",
        help="Skip C++ API MDX generation even when Fern auth is available.",
    )
    parser.add_argument(
        "--skip-fern-check",
        action="store_true",
        help="Run the pipeline only; skip fern check / preview.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run fern docs dev after the pipeline (live preview).",
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
