#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from uuid import uuid4


def generate_uuid_c_string():
    v = uuid4().hex
    return f"0x{v[:16]}, 0x{v[16:]}"


def main():
    print(generate_uuid_c_string())


if __name__ == "__main__":
    main()
