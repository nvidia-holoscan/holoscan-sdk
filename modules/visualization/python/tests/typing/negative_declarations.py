# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Negative typing contracts for the typed Holoviz declarations shared by pyright and ty."""

from holoscan_visualization.operators import (
    GeometryLayer,
    HolovizOp,
    ImageLayer,
    ViewTransform,
    WindowSink,
)

import holoscan as hs

layer = ImageLayer(port_name="video", max_width=640, max_height=480)
layer.max_width = 1280  # typing-error: frozen-assignment
ImageLayer("video", 640, 480)  # typing-error: keyword-only
ImageLayer(
    kind="image",  # typing-error: classvar-kind
    port_name="video",
    max_width=640,
    max_height=480,
)
GeometryLayer(
    port_name="shape",
    topologies=["not_a_topology"],  # typing-error: topology-literal
    max_primitives=1,
)
WindowSink(present_mode="vsync")  # typing-error: present-mode-literal

# A typed view is lowered only inside ImageLayer, never inside a hand-authored layer dictionary.
# Both checkers reject the call; pyright also reports the layers argument itself.
HolovizOp(  # typing-error: dict-layer-typed-view
    hs.Graph("typing-visualization-negative"),
    "visualizer",
    layers=[  # typing-error[pyright]: dict-layer-typed-view-argument
        {
            "kind": "image",
            "port_name": "video",
            "max_width": 640,
            "max_height": 480,
            "view": ViewTransform(scale_x=0.5),
        }
    ],
    sink=WindowSink(),
)
