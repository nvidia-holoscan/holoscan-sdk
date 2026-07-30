#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from check_copyright import (
    APACHE_BOILERPLATE_LINES,
    check_copyright,
    shorten_license_header,
)


def long_hash_header():
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n",
        "# SPDX-License-Identifier: Apache-2.0\n",
        "#\n",
    ]
    lines.extend(f"# {line}\n" if line else "#\n" for line in APACHE_BOILERPLATE_LINES)
    return "".join(lines)


def long_slash_header():
    lines = [
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n",
        "// SPDX-License-Identifier: Apache-2.0\n",
        "//\n",
    ]
    lines.extend(f"// {line}\n" if line else "//\n" for line in APACHE_BOILERPLATE_LINES)
    return "".join(lines)


def long_html_header():
    lines = [
        "<!--\n",
        "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n",
        "SPDX-License-Identifier: Apache-2.0\n",
        "\n",
    ]
    lines.extend(f"{line}\n" if line else "\n" for line in APACHE_BOILERPLATE_LINES)
    lines.append("-->\n")
    return "".join(lines)


def long_legacy_block_header():
    lines = [
        "/*\n",
        " * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n",
        " *\n",
    ]
    lines.extend(f" * {line}\n" if line else " *\n" for line in APACHE_BOILERPLATE_LINES)
    lines.extend(
        [
            " *\n",
            " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
            "All rights reserved.\n",
            " * SPDX-License-Identifier: Apache-2.0\n",
            " */\n",
        ]
    )
    return "".join(lines)


def test_shortens_only_leading_header():
    header = long_hash_header()
    body = "int main() {}\n\n" + header

    updated, count = shorten_license_header(header + "\n" + body)

    assert count == 1
    assert "Licensed under the Apache License" not in updated[: updated.index("int main")]
    assert updated[updated.index("int main") :] == body


def test_stops_at_preprocessor_directive():
    header = long_hash_header()
    body = "#include <example.hpp>\n\n" + header

    updated, count = shorten_license_header(header + "\n" + body)

    assert count == 1
    assert updated[updated.index("#include") :] == body


def test_shortens_header_variants():
    for header in (
        long_hash_header(),
        long_slash_header(),
        long_html_header(),
        long_legacy_block_header(),
    ):
        updated, count = shorten_license_header(header)

        assert count == 1
        assert "Licensed under the Apache License" not in updated


def test_rejects_long_header_without_update_current_year(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(long_hash_header() + "\nprint('hello')\n", encoding="utf-8")

    errors = check_copyright(str(source), update_current_year=False)

    assert any("Deprecated long Apache boilerplate header detected" in err[2] for err in errors)
