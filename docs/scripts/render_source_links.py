#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate or render links to Holoscan SDK source files.

Committed documentation uses directly readable GitHub links to ``main``. Release
publication copies the documentation into an isolated staging directory and uses
this script to render every SDK source link with one release browser and ref.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Collection
from pathlib import Path
from urllib.parse import unquote, urlsplit

CANONICAL_SOURCE_BROWSER_URL = "https://github.com/nvidia-holoscan/holoscan-sdk"
CANONICAL_SOURCE_REF = "main"
ALLOWED_SOURCE_BROWSER_URLS = frozenset(
    {
        CANONICAL_SOURCE_BROWSER_URL,
        "https://gitlab.com/nvidia/holoscan/holoscan-sdk/-",
    }
)

_RELEASE_TAG_RE = re.compile(
    r"^v\d+\.\d+\.\d+(?:(?:\.post|\.)(?:\d+))?(?:-[A-Za-z0-9][A-Za-z0-9._-]*)?$"
)
_SOURCE_LINK_RE = re.compile(
    r"https://"
    r"(?P<host>[^/\s]+)/"
    r"(?P<repository>(?:[^/\s]+/){1,2}holoscan-sdk)"
    r"(?P<gitlab_segment>/-)?/"
    r"(?P<route>blob|tree)/"
    r"(?P<ref>[^/\s\])?#]+)"
    r"(?P<path>/[^\s\])?\"'<>]+)?"
)


class SourceLinkError(ValueError):
    """A source-link invariant violation."""


def normalize_source_browser_url(value: str) -> str:
    """Return a normalized GitHub or GitLab repository browser URL."""
    browser_url = value.strip().rstrip("/")
    parsed = urlsplit(browser_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise SourceLinkError("source browser URL must be an https URL")
    repository_path = parsed.path.rstrip("/").removesuffix("/-")
    if not repository_path.endswith("/holoscan-sdk"):
        raise SourceLinkError("source browser URL must identify a holoscan-sdk repository")
    if "gitlab" in parsed.hostname and not browser_url.endswith("/-"):
        browser_url += "/-"
    return browser_url


def validate_target(
    source_browser_url: str,
    source_ref: str,
    *,
    allowed_source_browser_urls: Collection[str] = ALLOWED_SOURCE_BROWSER_URLS,
    allowed_source_refs: Collection[str] = (),
) -> tuple[str, str]:
    """Validate a render target and enforce main-or-release-tag semantics."""
    browser_url = normalize_source_browser_url(source_browser_url)
    allowed_browser_urls = {
        normalize_source_browser_url(value) for value in allowed_source_browser_urls
    }
    if browser_url not in allowed_browser_urls:
        allowed = ", ".join(sorted(value.removesuffix("/-") for value in allowed_browser_urls))
        raise SourceLinkError(f"source browser URL must be one of: {allowed}")
    source_ref = source_ref.strip()
    if not source_ref:
        raise SourceLinkError("source ref must not be empty")
    if (
        source_ref != CANONICAL_SOURCE_REF
        and not _RELEASE_TAG_RE.fullmatch(source_ref)
        and source_ref not in allowed_source_refs
    ):
        raise SourceLinkError(
            f"source ref must be '{CANONICAL_SOURCE_REF}' or a release tag, got {source_ref!r}"
        )
    if source_ref == CANONICAL_SOURCE_REF and browser_url != CANONICAL_SOURCE_BROWSER_URL:
        raise SourceLinkError(
            f"source ref '{CANONICAL_SOURCE_REF}' must use {CANONICAL_SOURCE_BROWSER_URL}"
        )
    return browser_url, source_ref


def _is_current_source_file(docs_root: Path, path: Path) -> bool:
    """Return whether a path is a current authored Markdown input."""
    try:
        relative = path.relative_to(docs_root)
    except ValueError:
        return False
    if path.suffix.lower() not in {".md", ".mdx"} or "generated" in relative.parts:
        return False
    return not (
        len(relative.parts) >= 2
        and relative.parts[0] == "fern"
        and re.fullmatch(r"v\d+\.\d+", relative.parts[1])
    )


def source_files(docs_root: Path, paths: list[Path] | None = None) -> list[Path]:
    """Return selected or all current authored Markdown inputs."""
    if paths is None:
        candidates = [path for suffix in ("*.md", "*.mdx") for path in docs_root.rglob(suffix)]
    else:
        candidates = []
        for path in paths:
            candidate = path.expanduser()
            if not candidate.is_absolute():
                candidate = candidate.resolve()
            if candidate.is_file():
                candidates.append(candidate)
    return sorted({path for path in candidates if _is_current_source_file(docs_root, path)})


def _matched_target(match: re.Match[str]) -> tuple[str, str]:
    browser_url = match.group(0).split(f"/{match.group('route')}/", 1)[0]
    return normalize_source_browser_url(browser_url), match.group("ref")


def _expected_route(docs_root: Path, match: re.Match[str]) -> str | None:
    """Return the canonical browser route when the checkout target exists."""
    raw_path = match.group("path")
    if not raw_path:
        return None
    relative = Path(unquote(re.split(r"[?#]", raw_path, maxsplit=1)[0]).lstrip("/"))
    public_root = docs_root.resolve().parent
    target = (public_root / relative).resolve()
    try:
        target.relative_to(public_root)
    except ValueError:
        return None
    if target.is_dir():
        return "tree"
    if target.is_file():
        return "blob"
    return None


def audit_consistent_source_links(
    docs_root: Path,
    *,
    allow_empty: bool = False,
    paths: list[Path] | None = None,
    allowed_source_browser_urls: Collection[str] = ALLOWED_SOURCE_BROWSER_URLS,
    allowed_source_refs: Collection[str] = (),
) -> int:
    """Require a rendered documentation tree to use one valid source browser and ref."""
    targets: dict[tuple[str, str], list[str]] = {}
    count = 0
    findings: list[str] = []
    for path in source_files(docs_root, paths):
        relative = path.relative_to(docs_root)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _SOURCE_LINK_RE.finditer(line):
                count += 1
                target = _matched_target(match)
                targets.setdefault(target, []).append(f"{relative}:{lineno}")
                expected_route = _expected_route(docs_root, match)
                if expected_route and match.group("route") != expected_route:
                    findings.append(
                        f"{relative}:{lineno}: SDK source link uses "
                        f"{match.group('route')} for a {expected_route} target"
                    )
    if findings:
        raise SourceLinkError("\n".join(findings))
    if count == 0:
        if allow_empty:
            return 0
        raise SourceLinkError(f"no Holoscan SDK source links found under {docs_root}")
    if len(targets) != 1:
        summary = ", ".join(
            f"{browser}/.../{ref} ({len(locations)} links)"
            for (browser, ref), locations in sorted(targets.items())
        )
        raise SourceLinkError(f"SDK source links use mixed browser/ref targets: {summary}")
    browser_url, source_ref = next(iter(targets))
    validate_target(
        browser_url,
        source_ref,
        allowed_source_browser_urls=allowed_source_browser_urls,
        allowed_source_refs=allowed_source_refs,
    )
    return count


def audit_source_links(
    docs_root: Path,
    *,
    expected_browser_url: str = CANONICAL_SOURCE_BROWSER_URL,
    expected_ref: str = CANONICAL_SOURCE_REF,
    allow_empty: bool = False,
    paths: list[Path] | None = None,
    allowed_source_browser_urls: Collection[str] = ALLOWED_SOURCE_BROWSER_URLS,
    allowed_source_refs: Collection[str] = (),
) -> int:
    """Require every current SDK source link to use one browser and ref."""
    expected_browser_url, expected_ref = validate_target(
        expected_browser_url,
        expected_ref,
        allowed_source_browser_urls=allowed_source_browser_urls,
        allowed_source_refs=allowed_source_refs,
    )
    findings: list[str] = []
    count = 0
    for path in source_files(docs_root, paths):
        relative = path.relative_to(docs_root)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _SOURCE_LINK_RE.finditer(line):
                count += 1
                browser_url, source_ref = _matched_target(match)
                if (
                    expected_browser_url == CANONICAL_SOURCE_BROWSER_URL
                    and "gitlab" in match.group("host").lower()
                ):
                    findings.append(
                        f"{relative}:{lineno}: GitLab Holoscan SDK source links are not "
                        f"allowed in committed documentation; use {CANONICAL_SOURCE_BROWSER_URL}"
                    )
                elif browser_url != expected_browser_url or source_ref != expected_ref:
                    expected = (
                        f"{expected_browser_url}/{match.group('route')}/{expected_ref}"
                        f"{match.group('path') or ''}"
                    )
                    findings.append(
                        f"{relative}:{lineno}: SDK source link uses {match.group(0)}; "
                        f"expected {expected}"
                    )
                expected_route = _expected_route(docs_root, match)
                if expected_route and match.group("route") != expected_route:
                    findings.append(
                        f"{relative}:{lineno}: SDK source link uses "
                        f"{match.group('route')} for a {expected_route} target"
                    )
    if findings:
        raise SourceLinkError("\n".join(findings))
    if count == 0:
        if allow_empty:
            return 0
        raise SourceLinkError(f"no Holoscan SDK source links found under {docs_root}")
    return count


def write_committed_source_links(docs_root: Path, paths: list[Path] | None = None) -> int:
    """Normalize fixable GitHub source links to canonical routes at ``main``."""
    updated = 0
    for path in source_files(docs_root, paths):
        text = path.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            nonlocal updated
            if match.group("host").lower() != "github.com":
                return match.group(0)
            expected_route = _expected_route(docs_root, match)
            route = expected_route or match.group("route")
            replacement = (
                f"{CANONICAL_SOURCE_BROWSER_URL}/{route}/{CANONICAL_SOURCE_REF}"
                f"{match.group('path') or ''}"
            )
            if replacement != match.group(0):
                updated += 1
            return replacement

        rendered = _SOURCE_LINK_RE.sub(replace, text)
        if rendered != text:
            path.write_text(rendered, encoding="utf-8")
    return updated


def render_source_links(
    docs_root: Path,
    *,
    source_browser_url: str,
    source_ref: str,
    allowed_source_browser_urls: Collection[str] = ALLOWED_SOURCE_BROWSER_URLS,
    allowed_source_refs: Collection[str] = (),
) -> int:
    """Render canonical committed source links to one validated release target."""
    source_browser_url, source_ref = validate_target(
        source_browser_url,
        source_ref,
        allowed_source_browser_urls=allowed_source_browser_urls,
        allowed_source_refs=allowed_source_refs,
    )
    audit_source_links(docs_root)
    count = 0
    for path in source_files(docs_root):
        text = path.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return (
                f"{source_browser_url}/{match.group('route')}/{source_ref}"
                f"{match.group('path') or ''}"
            )

        rendered = _SOURCE_LINK_RE.sub(replace, text)
        if rendered != text:
            path.write_text(rendered, encoding="utf-8")
    audit_source_links(
        docs_root,
        expected_browser_url=source_browser_url,
        expected_ref=source_ref,
        allowed_source_browser_urls=allowed_source_browser_urls,
        allowed_source_refs=allowed_source_refs,
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, required=True)
    parser.add_argument(
        "--source-browser-url",
        default=CANONICAL_SOURCE_BROWSER_URL,
    )
    parser.add_argument("--source-ref", default=CANONICAL_SOURCE_REF)
    parser.add_argument(
        "--allowed-source-browser-url",
        action="append",
        default=[],
        help="Add a trusted publication browser URL to the built-in public allowlist.",
    )
    parser.add_argument(
        "--allowed-source-ref",
        action="append",
        default=[],
        help="Add an exact trusted source ref to the main-or-release-tag policy.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Normalize fixable committed GitHub links before checking them.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Rewrite canonical committed links to the selected release target.",
    )
    parser.add_argument(
        "--allow-rendered",
        action="store_true",
        help="Accept one consistent main or tagged render target instead of requiring GitHub/main.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="With --allow-rendered, accept a documentation tree with no SDK source links.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Markdown inputs to check (default: all current documentation inputs).",
    )
    args = parser.parse_args()
    if args.write and (args.render or args.allow_rendered):
        parser.error("--write cannot be combined with --render or --allow-rendered")
    docs_root = args.docs_root.expanduser().resolve()
    if not docs_root.is_dir():
        print(f"error: documentation root not found: {docs_root}", file=sys.stderr)
        return 2
    try:
        allowed_source_browser_urls = (
            *ALLOWED_SOURCE_BROWSER_URLS,
            *args.allowed_source_browser_url,
        )
        if args.write:
            updated = write_committed_source_links(docs_root, args.files or None)
            count = audit_source_links(
                docs_root,
                allow_empty=bool(args.files),
                paths=args.files or None,
            )
            print(
                f"Updated {updated} and checked {count} SDK source links against "
                f"{CANONICAL_SOURCE_BROWSER_URL}/.../{CANONICAL_SOURCE_REF}."
            )
        elif args.render:
            count = render_source_links(
                docs_root,
                source_browser_url=args.source_browser_url,
                source_ref=args.source_ref,
                allowed_source_browser_urls=allowed_source_browser_urls,
                allowed_source_refs=args.allowed_source_ref,
            )
            print(
                f"Rendered {count} SDK source links with "
                f"{normalize_source_browser_url(args.source_browser_url)}/.../{args.source_ref}."
            )
        elif args.allow_rendered:
            count = audit_consistent_source_links(
                docs_root,
                allow_empty=args.allow_empty,
                paths=args.files or None,
                allowed_source_browser_urls=allowed_source_browser_urls,
                allowed_source_refs=args.allowed_source_ref,
            )
            print(f"Checked {count} consistently rendered SDK source links.")
        else:
            count = audit_source_links(
                docs_root,
                allow_empty=bool(args.files),
                paths=args.files or None,
            )
            print(
                f"Checked {count} SDK source links against "
                f"{CANONICAL_SOURCE_BROWSER_URL}/.../{CANONICAL_SOURCE_REF}."
            )
    except SourceLinkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
