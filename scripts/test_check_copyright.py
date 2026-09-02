#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from check_copyright import (
    APACHE_BOILERPLATE_LINES,
    NVIDIA_PROPRIETARY_BOILERPLATE_LINES,
    check_copyright,
    check_this_file,
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


def long_nvidia_proprietary_header():
    lines = [
        "/*!\n",
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n",
        " * SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n",
        " *\n",
    ]
    lines.extend(f" * {line}\n" for line in NVIDIA_PROPRIETARY_BOILERPLATE_LINES)
    lines.append(" */\n")
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
        long_nvidia_proprietary_header(),
    ):
        updated, count = shorten_license_header(header)

        assert count == 1
        assert "Licensed under the Apache License" not in updated


def test_rejects_long_header_without_update_current_year(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(long_hash_header() + "\nprint('hello')\n", encoding="utf-8")

    errors = check_copyright(str(source), update_current_year=False)

    assert any("Deprecated verbose license boilerplate header detected" in err[2] for err in errors)


def test_shortens_nvidia_proprietary_boilerplate(tmp_path):
    source = tmp_path / "example.css"
    source.write_text(long_nvidia_proprietary_header() + "\n:root {}\n", encoding="utf-8")

    errors = check_copyright(str(source), update_current_year=False)
    assert any("Deprecated verbose license boilerplate header detected" in err[2] for err in errors)

    update_errors = check_copyright(
        str(source), update_current_year=True, reject_nvidia_proprietary=True
    )
    assert any("NVIDIA proprietary SPDX license" in err[2] for err in update_errors)
    updated = source.read_text(encoding="utf-8")
    assert "retain all intellectual" not in updated
    assert "SPDX-License-Identifier: LicenseRef-NvidiaProprietary" in updated


def test_requires_spdx_license_identifier_in_leading_header(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 Example Authors\n\nprint('hello')\n",
        encoding="utf-8",
    )

    errors = check_copyright(str(source), update_current_year=False)

    assert any("SPDX-License-Identifier missing" in err[2] for err in errors)


def test_requires_spdx_copyright_in_leading_header(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(
        "# SPDX-License-Identifier: BSD-2-Clause\n\nprint('hello')\n",
        encoding="utf-8",
    )

    errors = check_copyright(str(source), update_current_year=False)

    assert any("SPDX-FileCopyrightText missing" in err[2] for err in errors)


def test_accepts_third_party_spdx_header(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2010 Example Authors\n"
        "# SPDX-License-Identifier: BSD-2-Clause\n\n"
        "print('hello')\n",
        encoding="utf-8",
    )

    assert check_copyright(str(source), update_current_year=False) == []


def test_accepts_multiple_spdx_copyright_tags(tmp_path):
    source = tmp_path / "example.cpp"
    source.write_text(
        "// SPDX-FileCopyrightText: Copyright 2024 Example Authors\n"
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n\n"
        "int main() {}\n",
        encoding="utf-8",
    )

    assert check_copyright(str(source), update_current_year=False) == []


def test_rejects_multiple_spdx_license_identifiers(tmp_path):
    source = tmp_path / "example.cpp"
    source.write_text(
        "// SPDX-FileCopyrightText: Copyright 2024 Example Authors\n"
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// SPDX-License-Identifier: MIT\n\n"
        "int main() {}\n",
        encoding="utf-8",
    )

    errors = check_copyright(str(source), update_current_year=False)

    assert any("Multiple SPDX-License-Identifier tags" in error[2] for error in errors)


def test_rejects_nvidia_proprietary_spdx_in_public_source(tmp_path):
    source = tmp_path / "example.css"
    source.write_text(
        "/*\n"
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        " * SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n"
        " */\n\n"
        ":root {}\n",
        encoding="utf-8",
    )

    errors = check_copyright(str(source), update_current_year=False, reject_nvidia_proprietary=True)

    assert any("NVIDIA proprietary SPDX license" in err[2] for err in errors)


def test_accepts_other_license_ref_for_nvidia_source(tmp_path):
    source = tmp_path / "example.css"
    source.write_text(
        "/*\n"
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        " * SPDX-License-Identifier: LicenseRef-SomeOtherLicense\n"
        " */\n\n"
        ":root {}\n",
        encoding="utf-8",
    )

    assert (
        check_copyright(str(source), update_current_year=False, reject_nvidia_proprietary=True)
        == []
    )


def test_accepts_nvidia_proprietary_spdx_outside_public_source(tmp_path):
    source = tmp_path / "example.css"
    source.write_text(
        "/*\n"
        " * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        " * SPDX-License-Identifier: LicenseRef-NvidiaProprietary\n"
        " */\n\n"
        ":root {}\n",
        encoding="utf-8",
    )

    assert check_copyright(str(source), update_current_year=False) == []


def test_spdx_metadata_in_body_does_not_satisfy_header(tmp_path):
    source = tmp_path / "example.py"
    source.write_text(
        "print('hello')\n"
        "# SPDX-FileCopyrightText: Copyright (c) 2026 Example Authors\n"
        "# SPDX-License-Identifier: BSD-2-Clause\n",
        encoding="utf-8",
    )

    errors = check_copyright(str(source), update_current_year=False)

    assert any("SPDX-FileCopyrightText missing" in err[2] for err in errors)
    assert any("SPDX-License-Identifier missing" in err[2] for err in errors)


def test_checks_idl_sources(tmp_path):
    idl = tmp_path / "example.idl"
    idl.write_text("interface Example {};\n", encoding="utf-8")

    assert check_this_file(str(idl)) is True


def test_checks_extensionless_shebang_scripts(tmp_path):
    script = tmp_path / "run_example"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    data = tmp_path / "VERSION"
    data.write_text("1.0.0\n", encoding="utf-8")

    assert check_this_file(str(script)) is True
    assert check_this_file(str(data)) is False
