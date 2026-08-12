# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import holoscan.core


def test_hello_prints_holoscan_greeting(capfd):
    holoscan.core.hello()

    captured = capfd.readouterr()
    assert captured.out == "Hello Holoscan!\n"
    assert captured.err == ""
