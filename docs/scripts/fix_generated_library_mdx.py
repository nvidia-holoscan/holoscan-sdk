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
"""

from __future__ import annotations

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
_DEFAULT_STRING_LITERAL = re.compile(r'default=""([^"]*)""')

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
    root = Path(__file__).resolve().parent.parent / "fern" / "generated"
    if not root.is_dir():
        print(f"Skip: no directory {root}", file=sys.stderr)
        return 0

    changed_files = 0
    validation_errors: list[str] = []
    for path in sorted(root.rglob("*.mdx")):
        text = path.read_text(encoding="utf-8")
        new_text = fix_file_content(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed_files += 1
            text = new_text
        for line_no, line in find_unfixed_cpp_string_defaults(text):
            rel = path.relative_to(root.parent)
            validation_errors.append(f"{rel}:{line_no}: {line}")

    print(f"fix-generated-library-mdx: updated {changed_files} file(s) under {root}")
    if validation_errors:
        print(
            'fix-generated-library-mdx: still found invalid ParamField default=""…"" '
            f"encoding in {len(validation_errors)} location(s). Examples:",
            file=sys.stderr,
        )
        for err in validation_errors[:10]:
            print(f"  {err}", file=sys.stderr)
        if len(validation_errors) > 10:
            print(f"  … and {len(validation_errors) - 10} more", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
