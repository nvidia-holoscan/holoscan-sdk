"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp


class FrameOnceSource(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        frame = np.zeros((1, 1, 4), dtype=np.uint8)
        op_output.emit({"data": frame}, "out")


class CloseCallbackSmokeApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cb_calls = 0

    def on_close(self):
        self.cb_calls += 1

    def compose(self):
        source = FrameOnceSource(self, CountCondition(self, 1), name="source")

        holoviz = HolovizOp(
            self,
            name="viz",
            headless=True,
            width=16,
            height=16,
            window_close_callback=self.on_close,
            tensors=[dict(name="data", type="color")],
        )

        self.add_flow(source, holoviz, {("out", "receivers")})


def test_holoviz_window_close_callback_smoke(capfd):
    """Smoke test: ensure window_close_callback is accepted and app runs."""
    app = CloseCallbackSmokeApp()
    app.run()

    # callback should not be invoked during normal run
    assert app.cb_calls == 0

    captured = capfd.readouterr()
    assert "[error]" not in captured.err


def test_holoviz_default_window_close_callback_callable(capfd):
    """Ensure default_window_close_callback is exposed and callable from Python."""
    app = CloseCallbackSmokeApp()
    holoviz = HolovizOp(
        app,
        name="viz_default_cb",
        headless=True,
        width=16,
        height=16,
        tensors=[dict(name="data", type="color")],
    )

    # Calling the default callback directly should not raise and should not log errors.
    holoviz.default_window_close_callback()

    captured = capfd.readouterr()
    assert "[error]" not in captured.err
