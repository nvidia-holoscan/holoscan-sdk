# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Core operator authoring handles."""

import holoscan


class OperatorHandle:
    """A weak handle to one Graph-owned operator declaration."""

    @property
    def name(self) -> str: ...

    @property
    def type_name(self) -> str: ...

    @property
    def graph_valid(self) -> bool: ...

    @property
    def ports(self, /) -> dict[str, holoscan.Port]: ...

    def __getitem__(self, name: str) -> holoscan.Port: ...

    def __getattr__(self, name: str) -> holoscan.Port: ...

    def __repr__(self) -> str: ...
