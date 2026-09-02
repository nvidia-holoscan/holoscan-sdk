#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize internal links in a flattened Fern documentation archive."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from check_doc_links import (
    _ARCHIVED_VERSION_DIR_RE,
    _LINK_RE,
    DEFAULT_SITE_PREFIX,
    _generated_pages,
    _load_nav,
    _strip_code,
    _walk_nav,
)

_VERSIONED_PREFIX_PATTERN = r"\d+-\d+/"


def _mask_fenced_code(text: str) -> str:
    """Blank fenced code while preserving offsets into the original text."""
    return "\n".join(
        line if stripped else " " * len(line)
        for line, stripped in zip(text.split("\n"), _strip_code(text), strict=True)
    )


def archive_slug(version: str) -> str:
    """Return Fern's URL slug for a ``vX.Y`` archive version."""
    if _ARCHIVED_VERSION_DIR_RE.fullmatch(version) is None:
        raise ValueError(f"invalid archive version: {version}")
    return version[1:].replace(".", "-")


def _navigation_paths(archive_dir: Path) -> dict[Path, str]:
    nav_file = archive_dir / "index.yml"
    if not nav_file.is_file():
        raise ValueError(f"missing Fern navigation: {nav_file}")

    pages: dict[Path, str] = {}
    folders: list[tuple[str, Path]] = []
    prefix = f"{DEFAULT_SITE_PREFIX}/{archive_slug(archive_dir.name)}"
    _walk_nav(_load_nav(nav_file), prefix, archive_dir, pages, folders)
    if not pages:
        raise ValueError(f"no pages found in {nav_file}")
    _, _, generated_by_path = _generated_pages(folders)
    pages.update(generated_by_path)
    return pages


def convert_page_file_links(archive_dir: Path, *, write: bool = True) -> tuple[int, int]:
    """Replace links to MDX files with their published archive site paths."""
    archive_dir = archive_dir.resolve()
    navigation = _navigation_paths(archive_dir)
    by_name: dict[str, list[str]] = defaultdict(list)
    for path, site_path in navigation.items():
        by_name[path.name].append(site_path)

    updated_files = 0
    updated_links = 0
    unresolved: list[str] = []
    updates: dict[Path, tuple[str, int]] = {}
    for path in archive_dir.rglob("*.mdx"):
        text = path.read_text(encoding="utf-8")
        replacements: list[tuple[int, int, str]] = []
        for match in _LINK_RE.finditer(_mask_fenced_code(text)):
            raw_target = match.group(1)
            target = raw_target.strip("<>")
            page, separator, anchor = target.partition("#")
            if not page.endswith(".mdx"):
                continue

            candidates = [
                (path.parent / page).resolve(),
                (archive_dir / re.sub(r"^(?:\./|\.\./)+", "", page)).resolve(),
            ]
            site_path = next((navigation[item] for item in candidates if item in navigation), None)
            if site_path is None:
                matches = by_name.get(Path(page).name, [])
                if len(matches) == 1:
                    site_path = matches[0]
            if site_path is None:
                unresolved.append(f"{path.relative_to(archive_dir)}: {target}")
                continue

            replacement = f"{site_path}{separator}{anchor}" if separator else site_path
            replacements.append((*match.span(1), replacement))

        if not replacements:
            continue
        for start, end, replacement in reversed(replacements):
            text = f"{text[:start]}{replacement}{text[end:]}"
        updates[path] = text, len(replacements)

    if unresolved:
        details = "\n".join(f"  {item}" for item in unresolved)
        raise ValueError(f"unresolved MDX page-file links:\n{details}")
    for path, (text, count) in updates.items():
        if write:
            path.write_text(text, encoding="utf-8")
        updated_files += 1
        updated_links += count
    return updated_files, updated_links


def version_site_links(archive_dir: Path, version: str, *, write: bool = True) -> tuple[int, int]:
    """Keep unversioned absolute SDK links within the selected archive."""
    archive_dir = archive_dir.resolve()
    base = f"{DEFAULT_SITE_PREFIX}/"
    versioned = f"{base}{archive_slug(version)}/"
    pattern = re.compile(
        rf"(?P<lead>\]\(\s*<?|(?:href|to)=[\"']){re.escape(base)}"
        rf"(?!{_VERSIONED_PREFIX_PATTERN})"
    )

    updated_files = 0
    updated_links = 0
    for path in archive_dir.rglob("*.mdx"):
        text = path.read_text(encoding="utf-8")
        replacements = [
            (*match.span(), f"{match.group('lead')}{versioned}")
            for match in pattern.finditer(_mask_fenced_code(text))
        ]
        if not replacements:
            continue
        updated = text
        for start, end, replacement in reversed(replacements):
            updated = f"{updated[:start]}{replacement}{updated[end:]}"
        if write:
            path.write_text(updated, encoding="utf-8")
        updated_files += 1
        updated_links += len(replacements)
    return updated_files, updated_links


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument(
        "--version",
        help="Archive version in vX.Y form (default: archive directory name).",
    )
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--convert-page-file-links", action="store_true")
    stages.add_argument("--version-site-links", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report required changes without modifying archive files.",
    )
    args = parser.parse_args()

    archive_dir = args.archive_dir.expanduser().resolve()
    version = args.version or archive_dir.name
    if not archive_dir.is_dir():
        print(f"error: archive directory not found: {archive_dir}", file=sys.stderr)
        return 2

    try:
        if args.convert_page_file_links:
            files, links = convert_page_file_links(archive_dir, write=not args.check)
            label = "page-file"
        else:
            files, links = version_site_links(archive_dir, version, write=not args.check)
            label = "versioned site"
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check:
        pages = sum(1 for _ in archive_dir.rglob("*.mdx"))
        if links:
            print(
                f"error: {links} {label} link(s) in {files} file(s) require "
                f"normalization under {archive_dir}",
                file=sys.stderr,
            )
            return 1
        print(f"Checked {pages} page(s) in {version}; all {label} links are normalized")
    else:
        print(f"Updated {links} {label} link(s) in {files} file(s) under {archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
