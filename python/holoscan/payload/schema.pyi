# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Built-in schema payload identities."""

from holoscan.payload import PayloadType as _payload_PayloadType



Header: _payload_PayloadType = ...

String: _payload_PayloadType = ...

TensorPacket: _payload_PayloadType = ...

Time: _payload_PayloadType = ...

__all__: list[str] = ['Header', 'String', 'TensorPacket', 'Time']
