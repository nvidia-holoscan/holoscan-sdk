#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Validate links in the Holoscan SDK public Markdown and Fern user guide.

``fern check --local`` validates configuration and MDX syntax, but not the targets of
internal links. A link written as a bare relative path (``using-the-sdk/holoscan-core``)
is published verbatim, so the browser resolves it against the current page's directory
and lands on a path that does not exist. With ``hide-404-page: true`` in ``docs.yml``
those requests are redirected to the docs homepage, which looks like a working link.

Inter-page links must therefore be published site paths starting with ``/``
(``/holoscan/sdk-user-guide/...``), with ``#anchor`` alone for same-page links. Fern also
rewrites paths to committed page files (``sdk_installation.mdx``), but its documentation
lists that form as unsupported, so a CLI upgrade could publish those verbatim and break
them; this script reports them so the site paths stay canonical.

This script derives every published page path from the Fern navigation, collects the
anchors each page defines, and reports links that cannot resolve:

* relative link targets that Fern publishes unchanged,
* page-file (``.mdx``) paths that should be site paths,
* page paths with no matching navigation entry,
* ``#anchor`` fragments no heading, ``<Anchor id>``, or component title provides,
* image or asset paths missing from disk,
* ``docs.yml`` redirect destinations that do not resolve, and
* MyST/Sphinx markup left over from the Sphinx migration.

Usage (from the holoscan-sdk repo root):

  python3 docs/scripts/check_doc_links.py
  python3 docs/scripts/check_doc_links.py --docs-root public/docs
  python3 docs/scripts/check_doc_links.py --local-files-only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

SCRIPTS = Path(__file__).resolve().parent
SCRIPT_DOCS_ROOT = SCRIPTS.parent
DEFAULT_SITE_PREFIX = "/holoscan/sdk-user-guide"

# Fern derives anchors from these component titles in addition to headings.
_TITLED_COMPONENTS = ("Tab", "Accordion", "AccordionGroup", "Step", "Card")

_LINK_RE = re.compile(r"\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*(<[^>]+>|[^)\s]+)\s*(?:\"[^\"]*\")?\)")
_REFERENCE_LINK_RE = re.compile(r"^\s*\[(?!\^)[^]]+\]:\s*(?:<([^>]+)>|([^\s]+))", re.MULTILINE)
_HTML_TARGET_RE = re.compile(r'\b(?:href|src)="([^"]+)"')
_HEADING_RE = re.compile(r"#{1,6}\s+(.*)")
_FENCE_RE = re.compile(r"\s*(`{3,}|~{3,})")
_ASSET_RE = re.compile(r"\.(png|jpe?g|gif|svg|webp|pdf)$", re.IGNORECASE)
_EXTERNAL_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")
_PARENT_PREFIX_RE = re.compile(r"^(?:\.\./)+")
_ARCHIVED_VERSION_DIR_RE = re.compile(r"^v\d+\.\d+$")
# v4.3 and v4.4 predate the current canonical-link and MDX migration rules.
_STRICT_ARCHIVE_VERSION = (4, 5)

_SPHINX_ARTIFACTS = (
    ("MyST anchor label", re.compile(r"^\(#[^)]+\)=\s*$")),
    ("MyST cross-reference role", re.compile(r"\{(ref|doc|numref|eq|term)\}`")),
    ("Sphinx directive block", re.compile(r"^```\{[a-z]+\}")),
)


class Finding:
    """A single link problem, rendered as one ``file:line: message`` report line."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        self.path = path
        self.line = line
        self.message = message

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _slug_from_title(title: str) -> str:
    """Slugify a navigation title the way Fern does, splitting camelCase runs.

    ``Use iGPU with dGPU`` becomes ``use-i-gpu-with-d-gpu`` and ``Enabling GPUDirect
    RDMA`` becomes ``enabling-gpu-direct-rdma``.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", title)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9]+", "-", spaced).strip("-").lower())


def _slug_from_heading(text: str) -> str:
    """Slugify heading or component-title text the way Fern does.

    Unlike navigation titles, heading anchors keep camelCase runs together
    (``BlockMemoryPool`` becomes ``blockmemorypool``) and preserve underscores.
    """
    plain = re.sub(r"`([^`]*)`", r"\1", text)
    plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", plain)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = re.sub(r"\*+", "", plain)
    plain = re.sub(r"[^A-Za-z0-9\s_-]+", "", plain)
    return re.sub(r"[\s-]+", "-", plain).strip("-").lower()


def _strip_code(text: str) -> list[str]:
    """Blank out fenced code blocks, keeping line numbers aligned with the source."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        marker = _FENCE_RE.match(line)
        if fence is None and marker:
            fence = marker.group(1)
            lines.append("")
        elif fence is not None:
            if marker:
                candidate = marker.group(1)
                if (
                    candidate[0] == fence[0]
                    and len(candidate) >= len(fence)
                    and not line[marker.end() :].strip()
                ):
                    fence = None
            lines.append("")
        else:
            lines.append(line)
    return lines


def _yaml_entries(text: str) -> list[tuple[int, str]]:
    """Return ``(indent, content)`` for each significant line of a simple YAML file."""
    entries = []
    for raw in text.split("\n"):
        without_comment = re.sub(r"(?<!\S)#.*$", "", raw).rstrip()
        if not without_comment.strip():
            continue
        entries.append(
            (
                len(without_comment) - len(without_comment.lstrip()),
                without_comment.strip(),
            )
        )
    return entries


def _yaml_value(raw: str) -> str:
    return raw.strip().strip("'\"")


def _parse_nav(entries: list[tuple[int, str]], start: int, indent: int) -> tuple[list[dict], int]:
    """Parse one navigation list, returning its items and the index after them.

    Handles the ``docs.yml``/``index.yml`` subset in use: ``- section:``, ``- page:``,
    ``- folder:``, ``- link:`` items with ``path``, ``slug``, and nested ``contents``.
    """
    items: list[dict] = []
    i = start
    while i < len(entries):
        line_indent, content = entries[i]
        if line_indent < indent or not content.startswith("- "):
            break
        item: dict = {}
        key, _, value = content[2:].partition(":")
        item[key.strip()] = _yaml_value(value)
        i += 1
        while i < len(entries) and entries[i][0] > line_indent:
            child_indent, child = entries[i]
            child_key, _, child_value = child.partition(":")
            if child_key.strip() == "contents":
                item["contents"], i = _parse_nav(entries, i + 1, child_indent + 1)
                continue
            if child.startswith("- "):
                break
            item[child_key.strip()] = _yaml_value(child_value)
            i += 1
        items.append(item)
    return items, i


def _load_nav(path: Path) -> list[dict]:
    entries = _yaml_entries(path.read_text(encoding="utf-8"))
    for i, (indent, content) in enumerate(entries):
        if content.rstrip(":") == "navigation":
            return _parse_nav(entries, i + 1, indent + 1)[0]
    return []


def _load_redirects(docs_yml: Path) -> list[tuple[str, str]]:
    entries = _yaml_entries(docs_yml.read_text(encoding="utf-8"))
    for i, (indent, content) in enumerate(entries):
        if content.rstrip(":") == "redirects":
            items, _ = _parse_nav(entries, i + 1, indent + 1)
            return [
                (item.get("source", ""), item["destination"])
                for item in items
                if "destination" in item
            ]
    return []


def _site_prefix(docs_yml: Path) -> str:
    """Read the published site path from the configured Fern instance."""
    for _, content in _yaml_entries(docs_yml.read_text(encoding="utf-8")):
        for key in ("custom-domain", "url"):
            if content.startswith(f"- {key}:") or content.startswith(f"{key}:"):
                value = _yaml_value(content.split(":", 1)[1])
                _, _, suffix = value.partition("/")
                if suffix:
                    return f"/{suffix.rstrip('/')}"
    return DEFAULT_SITE_PREFIX


def _walk_nav(
    nodes: list[dict],
    prefix: str,
    fern_dir: Path,
    pages: dict[Path, str],
    folders: list[tuple[str, Path]],
) -> None:
    for node in nodes:
        if "section" in node:
            section_path = f"{prefix}/{node.get('slug') or _slug_from_title(node['section'])}"
            if node.get("path"):
                pages[(fern_dir / node["path"]).resolve()] = section_path
            _walk_nav(node.get("contents", []), section_path, fern_dir, pages, folders)
        elif "page" in node and node.get("path"):
            slug = node.get("slug") or _slug_from_title(node["page"])
            pages[(fern_dir / node["path"]).resolve()] = f"{prefix}/{slug}"
        elif "folder" in node:
            folder = Path(node["folder"])
            folders.append((f"{prefix}/{folder.name}", (fern_dir / folder).resolve()))


def _generated_pages(
    folders: list[tuple[str, Path]],
) -> tuple[set[str], list[str], dict[Path, str]]:
    """Map generated (C++ API) page paths, or note prefixes whose tree is absent.

    The generated tree is produced at build time and gitignored, so its links can only
    be verified when it is present; otherwise those prefixes are accepted unchecked.
    """
    known: set[str] = set()
    unchecked: list[str] = []
    paths: dict[Path, str] = {}
    for site_path, directory in folders:
        if not directory.is_dir():
            unchecked.append(site_path)
            continue
        # Fern redirects a folder's own path to its first child page.
        known.add(site_path)
        for mdx in directory.rglob("*.mdx"):
            relative = mdx.relative_to(directory).with_suffix("")
            generated_path = f"{site_path}/{'/'.join(part.lower() for part in relative.parts)}"
            known.add(generated_path)
            # Generated library navigation exposes intermediate directories as
            # folder routes even though they are not listed individually in
            # docs.yml. Fern redirects each route to its first generated page.
            for parent in relative.parents:
                if parent == Path("."):
                    continue
                known.add(f"{site_path}/{'/'.join(part.lower() for part in parent.parts)}")
            paths[mdx.resolve()] = generated_path
    return known, unchecked, paths


def _is_archived_version_page(path: Path, fern_dir: Path) -> bool:
    """Return whether ``path`` belongs to an injected ``vX.Y`` archive snapshot."""
    try:
        relative = path.resolve().relative_to(fern_dir.resolve())
    except ValueError:
        return False
    return bool(relative.parts and _ARCHIVED_VERSION_DIR_RE.fullmatch(relative.parts[0]))


def _archive_version(path: Path) -> tuple[int, int]:
    """Return the numeric version represented by a ``vX.Y`` archive directory."""
    match = _ARCHIVED_VERSION_DIR_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid archive version directory: {path}")
    major, minor = path.name[1:].split(".")
    return int(major), int(minor)


def _archive_slug(path: Path) -> str:
    """Return Fern's URL slug for a ``vX.Y`` archive directory."""
    major, minor = _archive_version(path)
    return f"{major}-{minor}"


def _collect_anchors(lines: list[str]) -> set[str]:
    """Collect every fragment the page provides: headings, anchors, and tab titles."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in lines:
        heading = _HEADING_RE.match(line)
        if heading:
            base = _slug_from_heading(heading.group(1))
            if base:
                # Fern disambiguates repeated headings with a numeric suffix.
                count = seen.get(base, 0)
                seen[base] = count + 1
                anchors.add(base if count == 0 else f"{base}-{count}")
        anchors.update(re.findall(r'\bid="([A-Za-z0-9][A-Za-z0-9_-]*)"', line))
        for component in _TITLED_COMPONENTS:
            for title in re.findall(rf'<{component}\b[^>]*\btitle="([^"]+)"', line):
                anchors.add(_slug_from_heading(title))
    return anchors


def _link_targets(lines: list[str]) -> list[tuple[int, str]]:
    """Return inline, reference, and HTML targets with their starting lines."""
    text = "\n".join(lines)
    targets = {
        (text.count("\n", 0, match.start()) + 1, match.group(1).strip("<>"))
        for match in _LINK_RE.finditer(text)
    }
    targets.update(
        (text.count("\n", 0, match.start()) + 1, match.group(1))
        for match in _HTML_TARGET_RE.finditer(text)
    )
    targets.update(
        (
            text.count("\n", 0, match.start()) + 1,
            match.group(1) or match.group(2),
        )
        for match in _REFERENCE_LINK_RE.finditer(text)
    )
    return sorted(targets)


def _markdown_sources(public_root: Path) -> list[Path]:
    """Return authored Markdown inputs from the public mirror tree."""
    excluded_parts = {".cache", ".local", "dist", "generated", "node_modules", "skills"}
    sources = []
    for suffix in ("*.md", "*.mdx"):
        for path in public_root.rglob(suffix):
            relative = path.relative_to(public_root)
            if any(part in excluded_parts for part in relative.parts):
                continue
            if any(
                re.fullmatch(r"(?:build|install|gxf-build)(?:-.*)?", part)
                for part in relative.parts
            ):
                continue
            sources.append(path.resolve())
    return sorted(set(sources))


def check_local_file_links(public_root: Path) -> list[Finding]:
    """Report Markdown targets missing from the public mirror tree.

    Published Fern routes are handled by :func:`check`; this pass checks paths that
    GitHub and local Markdown renderers resolve as files. Keeping it separate lets the
    pre-commit hook run for any public-tree change, including a target deletion or move.
    """
    public_root = public_root.resolve()
    sources = _markdown_sources(public_root)
    findings: list[Finding] = []

    for source in sources:
        relative = source.relative_to(public_root)
        lines = _strip_code(source.read_text(encoding="utf-8"))
        for lineno, target in _link_targets(lines):
            target = target.strip()
            if (
                not target
                or _EXTERNAL_RE.match(target)
                or target.startswith(("#", "?", DEFAULT_SITE_PREFIX))
                or any(character in target for character in "{}*$")
            ):
                continue

            page = unquote(re.split(r"[?#]", target, maxsplit=1)[0])
            if not page:
                continue
            candidate = (
                public_root / page.lstrip("/") if page.startswith("/") else source.parent / page
            ).resolve()
            try:
                candidate.relative_to(public_root)
            except ValueError:
                findings.append(
                    Finding(relative, lineno, f"local link escapes the public tree: {target}")
                )
                continue
            if not candidate.exists():
                findings.append(Finding(relative, lineno, f"missing local target: {target}"))

    print(f"Checked local file targets in {len(sources)} Markdown files.")
    return findings


def _check_navigation(
    *,
    source_root: Path,
    nav_file: Path,
    nav_root: Path,
    prefix: str,
    report_root: Path,
    allow_legacy_links: bool,
    exclude_archives: bool = False,
    redirects_file: Path | None = None,
) -> list[Finding]:
    """Validate one current or archived navigation tree."""
    if not nav_file.is_file():
        raise ValueError(f"missing Fern navigation: {nav_file}")

    pages: dict[Path, str] = {}
    folders: list[tuple[str, Path]] = []
    _walk_nav(_load_nav(nav_file), prefix, nav_root, pages, folders)
    if not pages:
        raise ValueError(f"no pages found in {nav_file}")
    generated, unchecked_prefixes, generated_by_path = _generated_pages(folders)
    redirects = _load_redirects(redirects_file) if redirects_file is not None else []
    redirect_sources = {
        source.rstrip("/")
        for source, _ in redirects
        if not re.search(r"/:[A-Za-z][A-Za-z0-9_]*\*?(?:/|$)", source)
    }

    sources = sorted(
        path.resolve()
        for path in source_root.rglob("*.mdx")
        if "generated" not in path.relative_to(source_root).parts
        and (not exclude_archives or not _is_archived_version_page(path, nav_root))
    )
    body_by_path = {path: _strip_code(path.read_text(encoding="utf-8")) for path in sources}
    anchors_by_site_path = {
        site_path: _collect_anchors(body_by_path[path])
        for path, site_path in pages.items()
        if path in body_by_path
    }

    findings: list[Finding] = []
    for path in sources:
        rel = path.relative_to(report_root.resolve())
        own = pages.get(path.resolve())
        for lineno, target in _link_targets(body_by_path[path]):
            if _EXTERNAL_RE.match(target) or target.startswith("?"):
                continue
            page, _, anchor = target.partition("#")

            if _ASSET_RE.search(page):
                if page and not (path.parent / page).resolve().is_file():
                    findings.append(Finding(rel, lineno, f"missing asset: {target}"))
                continue

            if not page:
                if (
                    not allow_legacy_links
                    and anchor
                    and own
                    and anchor not in anchors_by_site_path.get(own, set())
                ):
                    findings.append(Finding(rel, lineno, f"no anchor '{anchor}' on this page"))
                continue

            if page.endswith((".mdx", ".md")):
                target_path = (path.parent / page).resolve()
                resolved = pages.get(target_path) or generated_by_path.get(target_path)
                if resolved is None:
                    if allow_legacy_links:
                        continue
                    findings.append(
                        Finding(rel, lineno, f"link to a file outside the navigation: {target}")
                    )
                    continue
                if allow_legacy_links:
                    continue
                # Fern still rewrites page-file paths, but its documentation calls them
                # unsupported, so a CLI upgrade could publish them verbatim and break them.
                suggestion = f"{resolved}#{anchor}" if anchor else resolved
                findings.append(
                    Finding(rel, lineno, f"page-file link: use the site path {suggestion}")
                )
                continue
            elif page.startswith("/"):
                resolved = page.rstrip("/")
            else:
                candidate = f"{prefix}/{_PARENT_PREFIX_RE.sub('', page).rstrip('/')}"
                if candidate in anchors_by_site_path or candidate in generated:
                    hint = f"use {candidate}{'#' + anchor if anchor else ''}"
                else:
                    hint = f"no page at {candidate}; check the slugs in {nav_file.name}"
                findings.append(
                    Finding(
                        rel,
                        lineno,
                        f"relative link is published unchanged: {target} ({hint})",
                    )
                )
                continue

            if any(resolved.startswith(unchecked) for unchecked in unchecked_prefixes):
                continue
            if resolved in redirect_sources:
                continue
            if resolved in generated:
                continue
            if resolved not in anchors_by_site_path:
                findings.append(Finding(rel, lineno, f"no such page: {resolved}"))
            elif not allow_legacy_links and anchor and anchor not in anchors_by_site_path[resolved]:
                findings.append(Finding(rel, lineno, f"no anchor '{anchor}' on {resolved}"))

        if not allow_legacy_links:
            for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
                for label, pattern in _SPHINX_ARTIFACTS:
                    if pattern.search(line.strip()):
                        findings.append(Finding(rel, lineno, f"{label}: {line.strip()[:80]}"))

    if redirects_file is not None:
        for source, destination in redirects:
            if destination.startswith("http"):
                continue
            page = destination.split("#")[0].rstrip("/")
            if re.search(r"/:[A-Za-z][A-Za-z0-9_]*\*?(?:/|$)", page):
                continue
            if page in anchors_by_site_path or page in generated:
                continue
            if any(page.startswith(unchecked) for unchecked in unchecked_prefixes):
                continue
            findings.append(
                Finding(
                    redirects_file.resolve().relative_to(report_root.resolve()),
                    0,
                    f"redirect {source} -> unknown page {destination}",
                )
            )

    print(
        f"Checked {len(sources)} pages against {len(anchors_by_site_path)} "
        f"navigation entries from {nav_file.relative_to(report_root)}."
    )
    if allow_legacy_links:
        print("Legacy compatibility: page-file, anchor, and Sphinx migration checks skipped.")
    return findings


def check(docs_root: Path, fern_dir: Path) -> list[Finding]:
    docs_yml = fern_dir / "docs.yml"
    if not docs_yml.is_file():
        raise ValueError(f"missing Fern config: {docs_yml}")

    prefix = _site_prefix(docs_yml)
    findings = _check_navigation(
        source_root=docs_root,
        nav_file=fern_dir / "index.yml",
        nav_root=fern_dir,
        prefix=prefix,
        report_root=docs_root,
        allow_legacy_links=False,
        exclude_archives=True,
        redirects_file=docs_yml,
    )

    archive_dirs = sorted(
        path
        for path in fern_dir.iterdir()
        if path.is_dir() and _ARCHIVED_VERSION_DIR_RE.fullmatch(path.name)
    )
    for archive_dir in archive_dirs:
        version = _archive_version(archive_dir)
        findings.extend(
            _check_navigation(
                source_root=archive_dir,
                nav_file=archive_dir / "index.yml",
                nav_root=archive_dir,
                prefix=f"{prefix}/{_archive_slug(archive_dir)}",
                report_root=docs_root,
                allow_legacy_links=version < _STRICT_ARCHIVE_VERSION,
            )
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local Markdown targets or Fern user-guide links.",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=SCRIPT_DOCS_ROOT,
        help="Documentation root holding the MDX pages (default: the one containing this script).",
    )
    parser.add_argument(
        "--fern-dir",
        type=Path,
        default=None,
        help="Fern project directory (default: <docs-root>/fern).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Check local targets in every authored Markdown file under the public root.",
    )
    args = parser.parse_args()

    docs_root = args.docs_root.expanduser().resolve()
    if not docs_root.is_dir():
        print(f"error: documentation root not found: {docs_root}", file=sys.stderr)
        return 2

    try:
        if args.local_files_only:
            findings = check_local_file_links(docs_root.parent)
        else:
            fern_dir = (args.fern_dir or docs_root / "fern").expanduser().resolve()
            findings = check(docs_root, fern_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for finding in findings:
        print(f"error: {finding}", file=sys.stderr)
    if findings:
        print(
            f"\nerror: found {len(findings)} broken documentation link(s).",
            file=sys.stderr,
        )
        return 1
    print("No broken documentation links found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
