#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Validate internal links in the Holoscan SDK Fern user guide.

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
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SCRIPT_DOCS_ROOT = SCRIPTS.parent
DEFAULT_SITE_PREFIX = "/holoscan/sdk-user-guide"

# Fern derives anchors from these component titles in addition to headings.
_TITLED_COMPONENTS = ("Tab", "Accordion", "AccordionGroup", "Step", "Card")

_LINK_RE = re.compile(r"\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*(<[^>]+>|[^)\s]+)\s*(?:\"[^\"]*\")?\)")
_IMG_SRC_RE = re.compile(r'<img[^>]*\bsrc="([^"]+)"')
_HEADING_RE = re.compile(r"#{1,6}\s+(.*)")
_FENCE_RE = re.compile(r"\s*(`{3,}|~{3,})")
_ASSET_RE = re.compile(r"\.(png|jpe?g|gif|svg|webp|pdf)$", re.IGNORECASE)
_EXTERNAL_RE = re.compile(r"^(https?:|mailto:|tel:|ftp:)")
_PARENT_PREFIX_RE = re.compile(r"^(?:\.\./)+")

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
            fence = marker.group(1)[0] * 3
            lines.append("")
        elif fence is not None:
            if marker and marker.group(1)[0] * 3 == fence:
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


def _generated_pages(folders: list[tuple[str, Path]]) -> tuple[set[str], list[str]]:
    """Map generated (C++ API) page paths, or note prefixes whose tree is absent.

    The generated tree is produced at build time and gitignored, so its links can only
    be verified when it is present; otherwise those prefixes are accepted unchecked.
    """
    known: set[str] = set()
    unchecked: list[str] = []
    for site_path, directory in folders:
        if not directory.is_dir():
            unchecked.append(site_path)
            continue
        # Fern redirects a folder's own path to its first child page.
        known.add(site_path)
        for mdx in directory.rglob("*.mdx"):
            relative = mdx.relative_to(directory).with_suffix("")
            known.add(f"{site_path}/{'/'.join(part.lower() for part in relative.parts)}")
    return known, unchecked


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
    """Return link targets and their starting lines, including multiline markup."""
    text = "\n".join(lines)
    targets = [
        (text.count("\n", 0, match.start()) + 1, match.group(1).strip("<>"))
        for match in _LINK_RE.finditer(text)
    ]
    targets.extend(
        (text.count("\n", 0, match.start()) + 1, match.group(1))
        for match in _IMG_SRC_RE.finditer(text)
    )
    return sorted(targets)


def check(docs_root: Path, fern_dir: Path) -> list[Finding]:
    docs_yml = fern_dir / "docs.yml"
    if not docs_yml.is_file():
        raise ValueError(f"missing Fern config: {docs_yml}")

    prefix = _site_prefix(docs_yml)
    nav_file = fern_dir / "index.yml"
    if not nav_file.is_file():
        raise ValueError(f"missing Fern navigation: {nav_file}")

    pages: dict[Path, str] = {}
    folders: list[tuple[str, Path]] = []
    _walk_nav(_load_nav(nav_file), prefix, fern_dir, pages, folders)
    if not pages:
        raise ValueError(f"no pages found in {nav_file}")
    generated, unchecked_prefixes = _generated_pages(folders)

    sources = sorted(
        path
        for path in docs_root.rglob("*.mdx")
        if "generated" not in path.relative_to(docs_root).parts
    )
    body_by_path = {path: _strip_code(path.read_text(encoding="utf-8")) for path in sources}
    anchors_by_site_path = {
        site_path: _collect_anchors(body_by_path[path])
        for path, site_path in pages.items()
        if path in body_by_path
    }

    findings: list[Finding] = []
    for path in sources:
        rel = path.relative_to(docs_root)
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
                if anchor and own and anchor not in anchors_by_site_path.get(own, set()):
                    findings.append(Finding(rel, lineno, f"no anchor '{anchor}' on this page"))
                continue

            if page.endswith((".mdx", ".md")):
                resolved = pages.get((path.parent / page).resolve())
                if resolved is None:
                    findings.append(
                        Finding(rel, lineno, f"link to a file outside the navigation: {target}")
                    )
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
            if resolved in generated:
                continue
            if resolved not in anchors_by_site_path:
                findings.append(Finding(rel, lineno, f"no such page: {resolved}"))
            elif anchor and anchor not in anchors_by_site_path[resolved]:
                findings.append(Finding(rel, lineno, f"no anchor '{anchor}' on {resolved}"))

        for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for label, pattern in _SPHINX_ARTIFACTS:
                if pattern.search(line.strip()):
                    findings.append(Finding(rel, lineno, f"{label}: {line.strip()[:80]}"))

    for source, destination in _load_redirects(docs_yml):
        if destination.startswith("http"):
            continue
        page = destination.split("#")[0].rstrip("/")
        if page in anchors_by_site_path or page in generated:
            continue
        if any(page.startswith(unchecked) for unchecked in unchecked_prefixes):
            continue
        findings.append(
            Finding(
                docs_yml.relative_to(docs_root),
                0,
                f"redirect {source} -> unknown page {destination}",
            )
        )

    print(f"Checked {len(sources)} pages against {len(anchors_by_site_path)} navigation entries.")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate internal links, anchors, and assets in the Fern user guide.",
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
    args = parser.parse_args()

    docs_root = args.docs_root.expanduser().resolve()
    fern_dir = (args.fern_dir or docs_root / "fern").expanduser().resolve()
    if not docs_root.is_dir():
        print(f"error: documentation root not found: {docs_root}", file=sys.stderr)
        return 2

    try:
        findings = check(docs_root, fern_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for finding in findings:
        print(f"error: {finding}", file=sys.stderr)
    if findings:
        print(f"\nFound {len(findings)} broken documentation link(s).", file=sys.stderr)
        return 1
    print("No broken documentation links found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
