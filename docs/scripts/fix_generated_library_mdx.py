#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Post-process MDX emitted by `fern docs md generate` for C++ library docs.

1) ParamField `type="const T &"` / `type="T &&"`: bare `&` in JSX strings
   breaks MDX; escape as `&amp;` inside the type attribute only.

2) ParamField default=""..."" (Fern's encoding of a C++ string literal, including
   empty "") is invalid JSX; rewrite as default='"..."' with single-quoted attrs.

3) YAML frontmatter description="..." may contain <Type>; MDX treats that as JSX.
   Escape angle brackets inside the description value only.

4) Void HTML `<br>` tags must be self-closing in MDX (`<br />`). Normalize `<br/>`
   and bare `<br>` / `<br >` so Fern's MDX parser does not fail (e.g. in tables).

5) HTML comments `<!-- ... -->` outside ``` fences become MDX `{/* ... */}`; comments
   whose body contains `{` or `}` are removed.

6) Quoted C++ placeholders such as `"<code name>: &lt;message&gt;"` are not HTML
   elements. Escape their raw angle brackets so MDX does not parse them as JSX.

7) Fern cross-links transitive nvidia::gxf types to unpublished
   ``api-reference/nvidia/namespaces/gxf/...`` pages. There is no public GXF API
   reference, so unwrap those hyperlinks and keep the visible type names.

8) Fern can reuse a derived page's inheritance links for inherited signatures,
   leaving types in a signature unlinked or linked to the wrong page. Reconcile
   each C++ CodeBlock link map against uniquely named generated pages and the
   current page's inner-type headings. See "Generated C++ CodeBlock links" in
   ``docs/README.md`` for the input provenance, matching heuristic, ambiguity
   handling, and an example.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# ParamField ... type="VALUE" — VALUE may contain & and &&
PARAMFIELD_TYPE = re.compile(
    r'(<ParamField\b[^>]*?\btype=")([^"]*)(")',
    flags=re.DOTALL,
)

# Leave existing entities intact so the script is idempotent if run twice.
_AMPERSAND_ESCAPE = re.compile(
    r"&(?!amp;|lt;|gt;|quot;|apos;|#(?:[0-9]{1,7}|[xX][0-9A-Fa-f]{1,6});)"
)


def fix_line(line: str) -> str:
    def repl(m: re.Match[str]) -> str:
        prefix, inner, suffix = m.group(1), m.group(2), m.group(3)
        if "&" not in inner:
            return m.group(0)
        inner = _AMPERSAND_ESCAPE.sub("&amp;", inner)
        return prefix + inner + suffix

    return PARAMFIELD_TYPE.sub(repl, line)


# Fern emits C++ string defaults as default=""value"" (invalid JSX). Empty "" is
# encoded as default="""" which closes default="" early and leaves stray quotes.
_DEFAULT_STRING_LITERAL = re.compile(r'default=""((?:[^"\\]|\\.)*)""')

# Any remaining Fern encoding after fix_file_content indicates a missed rewrite.
_UNFIXED_CPP_STRING_DEFAULT = re.compile(r'\bdefault=""')

# Frontmatter description="..." may contain <Type> which MDX parses as JSX.
_DESCRIPTION_LINE = re.compile(r'^(description:\s*")(.*)("\s*)$')


def _fix_description_line(line: str) -> str:
    raw = line.rstrip("\r\n")
    ending = line[len(raw) :]
    m = _DESCRIPTION_LINE.match(raw)
    if not m:
        return line
    prefix, body, suffix = m.group(1), m.group(2), m.group(3)
    if "<" not in body:
        return line
    body = (
        body.replace("&lt;", "\x00lt\x00")
        .replace("&gt;", "\x00gt\x00")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    body = body.replace("\x00lt\x00", "&lt;").replace("\x00gt\x00", "&gt;")
    return prefix + body + suffix + ending


def fix_mdx_void_br_tags(text: str) -> str:
    """MDX requires void elements to be self-closing; bare <br> breaks parsing."""
    text = re.sub(r"<br\s*/\s*>", "<br />", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*>", "<br />", text, flags=re.IGNORECASE)
    return text


_QUOTED_CPP_PLACEHOLDER = re.compile(
    r'(?<=")<([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)+)>'
)

# Unpublished Fern routes for nvidia::gxf types (not holoscan::gxf wrappers).
_GXF_API_TARGET = (
    r"(?:(?:\.\./)+|/holoscan/sdk-user-guide/api-reference/(?:cpp/)?|"
    r"/api-reference/(?:cpp/)?)nvidia/namespaces/gxf/[^)\s\"'>]*"
)
_MD_GXF_API_LINK = re.compile(rf"\[([^\]]+)\]\({_GXF_API_TARGET}/?\)")
_HTML_GXF_API_LINK = re.compile(
    rf'<a\b[^>]*\bhref=["\']{_GXF_API_TARGET}/?["\'][^>]*>(.*?)</a>',
    flags=re.IGNORECASE | re.DOTALL,
)
_GXF_API_TARGET_RE = re.compile(rf"{_GXF_API_TARGET}/?")
_CODEBLOCK_TAG = re.compile(r"<CodeBlock\b[^>]*>", flags=re.DOTALL)
_CODEBLOCK_LINKS = re.compile(
    r'\s+links=\{(?P<mapping>\{(?:[^{}"]|"(?:\\.|[^"\\])*")*\})\}',
    flags=re.DOTALL,
)
_CPP_CODEBLOCK = re.compile(
    r"(?P<tag><CodeBlock\b[^>]*>)"
    r"(?P<body>\s*```cpp[^\n]*\n(?P<code>.*?)\n```\s*</CodeBlock>)",
    flags=re.DOTALL,
)
_FRONTMATTER_TITLE = re.compile(
    r"^title:\s*(?P<quote>[\"']?)(?P<title>.*?)(?P=quote)\s*$",
    flags=re.MULTILINE,
)
_HOLOSCAN_QUALIFIED = re.compile(r"holoscan(?:::[A-Za-z_][A-Za-z0-9_]*)+")
_CPP_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ATX_HEADING = re.compile(
    r"^(?P<level>#{1,6})\s+(?P<title>.*?)\s*#?\s*$",
    flags=re.MULTILINE,
)
_EXPLICIT_HEADING_ANCHOR = re.compile(r"\s+(?:\\)?\[#(?P<anchor>[A-Za-z0-9_-]+)\]\s*$")


def _remove_codeblock_gxf_api_links(tag: re.Match[str]) -> str:
    """Remove unpublished nvidia::gxf targets from one CodeBlock tag."""
    text = tag.group(0)
    links = _CODEBLOCK_LINKS.search(text)
    if links is None:
        return text

    try:
        mapping = json.loads(links.group("mapping"))
    except json.JSONDecodeError:
        return text
    if not isinstance(mapping, dict):
        return text

    filtered = {
        label: target
        for label, target in mapping.items()
        if not isinstance(target, str) or _GXF_API_TARGET_RE.fullmatch(target) is None
    }
    if filtered == mapping:
        return text

    replacement = (
        "" if not filtered else " links={" + json.dumps(filtered, ensure_ascii=False) + "}"
    )
    return text[: links.start()] + replacement + text[links.end() :]


def unwrap_gxf_api_links(text: str) -> str:
    """Replace unpublished nvidia::gxf hyperlinks with their visible labels."""
    text = _CODEBLOCK_TAG.sub(_remove_codeblock_gxf_api_links, text)
    text = _HTML_GXF_API_LINK.sub(lambda match: match.group(1), text)
    return _MD_GXF_API_LINK.sub(lambda match: match.group(1), text)


def collect_holoscan_api_targets(root: Path) -> dict[str, Path]:
    """Return uniquely titled generated ``holoscan::...`` pages under root.

    The frontmatter title is authoritative for every generated page kind; this
    deliberately does not infer class, struct, enum, or function kinds from paths.
    Duplicate titles are excluded because they cannot produce a deterministic target.
    """
    candidates: dict[str, list[Path]] = {}
    for path in root.rglob("*.mdx"):
        match = _FRONTMATTER_TITLE.search(path.read_text(encoding="utf-8"))
        if match is None:
            continue
        title = match.group("title")
        if not title.startswith("holoscan::"):
            continue
        candidates.setdefault(title, []).append(path)
    return {title: paths[0] for title, paths in candidates.items() if len(paths) == 1}


def _declaring_holoscan_page(
    qualified_name: str, targets: dict[str, Path]
) -> tuple[str, Path] | None:
    """Find the longest generated-page prefix of a qualified C++ name."""
    parts = qualified_name.split("::")
    while len(parts) > 1:
        candidate = "::".join(parts)
        if candidate in targets:
            return candidate, targets[candidate]
        parts.pop()
    return None


def _identifier_present(code: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", code) is not None


def _link_key_has_substring_collision(code: str, name: str) -> bool:
    """Return whether Fern could link ``name`` inside a longer C++ identifier.

    Fern applies a CodeBlock link key textually after syntax highlighting. A
    mapping for ``Arg`` can therefore also link the ``Arg`` prefix in ``ArgT``
    or ``ArgsT`` even though those are distinct C++ identifiers.
    """
    return any(
        identifier != name and name in identifier
        for identifier in set(_CPP_IDENTIFIER.findall(code))
    )


def _unqualified_type_present(code: str, name: str) -> bool:
    """Return whether an unqualified identifier is used in a type-like position.

    A unique generated short name is not sufficient by itself: an unrelated type
    can share its spelling with a qualified standard-library name or a parameter.
    Prefer conservative C++ declaration cues so ``std::type_info``, ``&message``,
    and ``&codec`` do not acquire links to same-named Holoscan pages.
    """
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for match in pattern.finditer(code):
        before = code[: match.start()]
        after = code[match.end() :]
        if before.endswith("::") or after.startswith("::"):
            continue

        previous = re.search(r"\S(?=\s*$)", before)
        if previous is not None and previous.group(0) in "*&":
            continue

        following = after.lstrip()
        if name[:1].isupper() or name.endswith("_t") or following.startswith("<"):
            return True
        if re.match(r"(?:[*&]+\s*)?[A-Za-z_][A-Za-z0-9_]*", following):
            return True
    return False


def _relative_api_target(path: Path, target_path: Path) -> str:
    return os.path.relpath(target_path.with_suffix(""), path.parent).replace(os.sep, "/").lower()


def _unique_short_api_targets(targets: dict[str, Path]) -> dict[str, Path]:
    candidates: dict[str, set[Path]] = {}
    for title, target_path in targets.items():
        if not {"classes", "structs", "enums", "typedefs", "unions"}.intersection(
            target_path.parts
        ):
            continue
        candidates.setdefault(title.rsplit("::", 1)[-1], set()).add(target_path)
    return {name: next(iter(paths)) for name, paths in candidates.items() if len(paths) == 1}


def _fern_heading_slug(title: str) -> str:
    """Approximate Fern's generated heading anchors for C++ API pages."""
    title = re.sub(r"<[^>]+>", "", title).casefold()
    title = re.sub(r"[^a-z0-9 -]", "", title)
    return re.sub(r"[ -]+", "-", title).strip("-")


def _page_local_type_targets(text: str) -> dict[str, str]:
    """Return unique page-local type headings and their collision-adjusted anchors."""
    without_fences = _FENCE_BLOCK.sub(
        lambda match: "\n" * match.group(0).count("\n"),
        text,
    )
    slug_counts: dict[str, int] = {}
    candidates: dict[str, set[str]] = {}
    in_page_local_types = False
    for match in _ATX_HEADING.finditer(without_fences):
        level = len(match.group("level"))
        title = match.group("title").strip()
        explicit_anchor = _EXPLICIT_HEADING_ANCHOR.search(title)
        if explicit_anchor is not None:
            title = title[: explicit_anchor.start()].rstrip()
            base_slug = explicit_anchor.group("anchor")
        else:
            base_slug = _fern_heading_slug(title)
        occurrence = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = occurrence + 1
        if explicit_anchor is not None:
            anchor = f"#{base_slug}"
        else:
            anchor = f"#{base_slug}" if occurrence == 0 else f"#{base_slug}-{occurrence}"

        if level == 2:
            section = title.casefold()
            in_page_local_types = section == "types" or section.startswith("inner ")
        elif level == 3 and in_page_local_types and _CPP_IDENTIFIER.fullmatch(title):
            candidates.setdefault(title, set()).add(anchor)

    return {name: next(iter(anchors)) for name, anchors in candidates.items() if len(anchors) == 1}


def reconcile_codeblock_holoscan_links(text: str, *, path: Path, targets: dict[str, Path]) -> str:
    """Align C++ CodeBlock link maps with resolvable Holoscan symbols in the code."""

    short_targets = _unique_short_api_targets(targets)
    local_type_targets = _page_local_type_targets(text)

    def repl(block: re.Match[str]) -> str:
        tag = block.group("tag")
        code = block.group("code")
        links = _CODEBLOCK_LINKS.search(tag)
        if links is None:
            mapping: dict[str, object] = {}
        else:
            try:
                decoded = json.loads(links.group("mapping"))
            except json.JSONDecodeError:
                return block.group(0)
            if not isinstance(decoded, dict):
                return block.group(0)
            mapping = decoded

        # Link maps affect only identifiers present in this code block. Dropping
        # inherited-page entries that are absent from the signature prevents a
        # stale base-class target from obscuring the actual declaring class.
        reconciled = {
            name: target
            for name, target in mapping.items()
            if isinstance(name, str) and _identifier_present(code, name)
        }

        desired: dict[str, set[str]] = {}
        for qualified in _HOLOSCAN_QUALIFIED.findall(code):
            declared = _declaring_holoscan_page(qualified, targets)
            if declared is None:
                continue
            title, target_path = declared
            token = title.rsplit("::", 1)[-1]
            relative_target = _relative_api_target(path, target_path)
            desired.setdefault(token, set()).add(relative_target)

        # Link unqualified types only when their short name identifies exactly
        # one generated Holoscan API page and occurs in a type-like declaration
        # position. This covers MemoryStorageType and expected<T, E> without
        # mistaking std::type_info or parameters named message/codec for types.
        for token in sorted(set(_CPP_IDENTIFIER.findall(code))):
            target_path = short_targets.get(token)
            if target_path is not None and _unqualified_type_present(code, token):
                desired.setdefault(token, set()).add(_relative_api_target(path, target_path))

        # CodeBlock maps are keyed by the short displayed token. When two
        # qualified symbols with the same token resolve to different pages in
        # one block, preserve an existing mapping instead of guessing.
        for token, candidate_targets in desired.items():
            if len(candidate_targets) == 1:
                reconciled[token] = next(iter(candidate_targets))

        # Same-page types take precedence over global short-name matches. Fern
        # emits enum-like types under "Types" and classes/structs under "Inner ...".
        # It also suffixes duplicate heading anchors, so a Config type following
        # a config() method resolves to #config-1 rather than #config.
        reconciled.update(
            {
                token: target
                for token, target in local_type_targets.items()
                if _identifier_present(code, token)
            }
        )

        # CodeBlock links are block-wide text mappings rather than references to
        # individual C++ identifier occurrences. Prefer leaving an exact symbol
        # unlinked when its key would also link part of a longer identifier.
        reconciled = {
            token: target
            for token, target in reconciled.items()
            if not _link_key_has_substring_collision(code, token)
        }

        if reconciled == mapping:
            return block.group(0)

        if reconciled:
            replacement = " links={" + json.dumps(reconciled, ensure_ascii=False) + "}"
            if links is None:
                new_tag = tag[:-1] + replacement + ">"
            else:
                new_tag = tag[: links.start()] + replacement + tag[links.end() :]
        elif links is None:
            new_tag = tag
        else:
            new_tag = tag[: links.start()] + tag[links.end() :]
        return new_tag + block.group("body")

    return _CPP_CODEBLOCK.sub(repl, text)


def fix_quoted_cpp_placeholders(text: str) -> str:
    """Escape multi-word angle placeholders that begin a quoted C++ format."""
    return _QUOTED_CPP_PLACEHOLDER.sub(lambda match: f"&lt;{match.group(1)}&gt;", text)


_FENCE_BLOCK = re.compile(r"^```[^\n]*\n[\s\S]*?^```\s*$", re.MULTILINE)


def fix_html_comments_for_mdx(text: str) -> str:
    """Replace `<!-- ... -->` outside ``` fences with `{/* ... */}` for MDX."""

    def repl(m: re.Match[str]) -> str:
        body = m.group(1).replace("*/", "* /")
        if "{" in body or "}" in body:
            return ""
        return "{/* " + body + " */}"

    out: list[str] = []
    pos = 0
    for m in _FENCE_BLOCK.finditer(text):
        chunk = text[pos : m.start()]
        chunk = re.sub(r"<!--([\s\S]*?)-->", repl, chunk)
        out.append(chunk)
        out.append(m.group(0))
        pos = m.end()
    tail = text[pos:]
    tail = re.sub(r"<!--([\s\S]*?)-->", repl, tail)
    out.append(tail)
    return "".join(out)


def fix_file_content(text: str) -> str:
    text = fix_html_comments_for_mdx(text)
    text = fix_mdx_void_br_tags(text)
    text = fix_quoted_cpp_placeholders(text)
    text = unwrap_gxf_api_links(text)
    text = _DEFAULT_STRING_LITERAL.sub(lambda m: "default='\"" + m.group(1) + "\"'", text)
    lines = text.splitlines(keepends=True)
    lines = [_fix_description_line(line) for line in lines]
    return "".join(fix_line(line) for line in lines)


def find_unfixed_cpp_string_defaults(text: str) -> list[tuple[int, str]]:
    """Return (line_no, line) for lines that still use Fern's default=\"\"…\"\" encoding."""
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        if _UNFIXED_CPP_STRING_DEFAULT.search(line):
            hits.append((i, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process C++ API MDX generated by Fern.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "fern" / "generated",
        help="Generated MDX root (default: public/docs/fern/generated beside this script).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail without writing if any generated MDX file still requires processing.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        prefix = "error" if args.check else "warning"
        print(f"{prefix}: generated MDX directory not found: {root}", file=sys.stderr)
        return 1 if args.check else 0

    holoscan_api_targets = collect_holoscan_api_targets(root)
    changed_files = 0
    changed_paths: list[str] = []
    validation_errors: list[str] = []
    for path in sorted(root.rglob("*.mdx")):
        text = path.read_text(encoding="utf-8")
        new_text = fix_file_content(text)
        new_text = reconcile_codeblock_holoscan_links(
            new_text, path=path, targets=holoscan_api_targets
        )
        if new_text != text:
            changed_files += 1
            changed_paths.append(str(path.relative_to(root)))
            if not args.check:
                path.write_text(new_text, encoding="utf-8")
            text = new_text
        for line_no, line in find_unfixed_cpp_string_defaults(text):
            rel = path.relative_to(root.parent)
            validation_errors.append(f"{rel}:{line_no}: {line}")

    action = "would update" if args.check else "updated"
    print(f"{action.capitalize()} {changed_files} file(s) under {root}.")
    failed = False
    if validation_errors:
        print(
            'error: still found invalid ParamField default=""…"" '
            f"encoding in {len(validation_errors)} location(s). Examples:",
            file=sys.stderr,
        )
        for err in validation_errors[:10]:
            print(f"  {err}", file=sys.stderr)
        if len(validation_errors) > 10:
            print(f"  … and {len(validation_errors) - 10} more", file=sys.stderr)
        failed = True
    if args.check and changed_paths:
        print(
            "error: generated MDX is not fully post-processed. "
            "Run without --check before publication. Examples:",
            file=sys.stderr,
        )
        for path in changed_paths[:10]:
            print(f"  {path}", file=sys.stderr)
        if len(changed_paths) > 10:
            print(f"  … and {len(changed_paths) - 10} more", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
