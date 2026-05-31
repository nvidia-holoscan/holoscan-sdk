"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import argparse
import ctypes
import logging
from pathlib import Path

import cupy as cp

import holoscan


class RawFrameConverterOp(holoscan.core.Operator):
    """Reinterprets a raw Bayer blob from V4L2 as a uint16 tensor shaped (height, width, 1).

    Pixel format, width, and height are read from V4L2 metadata on each frame so nothing
    needs to be hardcoded here.  Requires pass_through=True on the upstream V4L2 op.
    """

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("input")

        pixel_format = self.metadata.get("V4L2_pixel_format", "")
        width = self.metadata.get("V4L2_width", 0)
        height = self.metadata.get("V4L2_height", 0)

        if not pixel_format or width == 0 or height == 0:
            raise RuntimeError(
                "Missing V4L2 metadata (pixel_format/width/height). "
                "Ensure pass_through=True on the V4L2 capture operator."
            )

        out_message = {}
        for name, tensor in in_message.items():
            # Reinterpret the raw byte blob as 16-bit samples (one per Bayer pixel).
            pixel_data = cp.from_dlpack(tensor).view(cp.uint16)
            # The V4L2 driver may use a row stride wider than the active image width
            # (e.g. stride = full sensor width even for a cropped/binned mode).
            # Detect this by computing stride = total samples / height, then crop to
            # the active width so that valid rows and zero-padded stride bytes are
            # not interleaved before demosaic.
            stride = pixel_data.size // height
            arr = pixel_data.reshape(height, stride)[:, :width].reshape(height, width, 1).copy()
            out_message[name] = arr

        op_output.emit(out_message, "output")


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        use_exclusive_display,
        frame_limit,
        pixel_format=None,
        bayer_grid=None,
        raw_depth=None,
        device=None,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._use_exclusive_display = use_exclusive_display
        self._frame_limit = frame_limit
        self._pixel_format = pixel_format
        self._bayer_grid = bayer_grid
        self._raw_depth = raw_depth
        self._device = device

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(self, name="ok", enable_tick=True)
            condition = self._ok

        # Resolve pixel_format, bayer_grid, raw_depth: CLI > YAML > default.
        receiver_kwargs = self.kwargs("receiver")
        raw_image_processor_kwargs = self.kwargs("raw_image_processor")

        pixel_format = self._pixel_format or receiver_kwargs.get("pixel_format", "RG10")
        bayer_grid = (
            self._bayer_grid
            if self._bayer_grid is not None
            else raw_image_processor_kwargs.get("bayer_grid", 1)
        )
        raw_depth = (
            self._raw_depth
            if self._raw_depth is not None
            else raw_image_processor_kwargs.get("raw_depth", 1)
        )

        # Pool is sized from the receiver config dimensions so it is large enough regardless
        # of what resolution the V4L2 driver actually negotiates at runtime.
        pool_width = receiver_kwargs.get("width", 3840)
        pool_height = receiver_kwargs.get("height", 2160)

        # Pass the resolved pixel_format into receiver_kwargs so V4L2 uses the right format.
        receiver_kwargs["pixel_format"] = pixel_format
        if self._device is not None:
            receiver_kwargs["device"] = self._device

        receiver_operator = holoscan.operators.V4L2VideoCaptureOp(
            self,
            condition,
            name="receiver",
            pass_through=True,
            **receiver_kwargs,
        )

        probe = RawFrameConverterOp(self, name="raw_frame_converter")

        raw_image_processor = holoscan.operators.RawImageProcessorOp(
            self,
            name="raw_image_processor",
            pixel_format=raw_depth,
            bayer_format=bayer_grid,
            **{
                k: v
                for k, v in raw_image_processor_kwargs.items()
                if k not in ("bayer_grid", "raw_depth")
            },
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=pool_width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * pool_height,
            num_blocks=4,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_grid,
            interpolation_mode=0,
        )
        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            use_exclusive_display=self._use_exclusive_display,
            framebuffer_srgb=True,
        )
        #
        self.add_flow(receiver_operator, probe, {("signal", "input")})
        self.add_flow(probe, raw_image_processor, {("output", "input")})
        self.add_flow(raw_image_processor, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})


def main(config_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument(
        "--use-exclusive-display", action="store_true", help="Run in exclusive display mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="V4L2 device node (e.g. /dev/video0). Overrides YAML receiver.device.",
    )
    parser.add_argument(
        "--pixel-format",
        default=None,
        help="V4L2 FourCC pixel format (e.g. RG10, BG10). Overrides YAML receiver.pixel_format.",
    )
    parser.add_argument(
        "--bayer-grid",
        type=int,
        default=None,
        help=(
            "NppiBayerGridPosition: 0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG. "
            "Overrides YAML raw_image_processor.bayer_grid."
        ),
    )
    parser.add_argument(
        "--raw-depth",
        type=int,
        default=None,
        help=(
            "holoscan::csi::PixelFormat: 1=RAW_10, 2=RAW_12. "
            "Overrides YAML raw_image_processor.raw_depth."
        ),
    )
    args = parser.parse_args()

    logging.info("Initializing.")

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        args.use_exclusive_display,
        args.frame_limit,
        pixel_format=args.pixel_format,
        bayer_grid=args.bayer_grid,
        raw_depth=args.raw_depth,
        device=args.device,
    )
    # if the --config command line argument was provided, it will override this config_file
    application.config(str(config_file))

    # Run it.
    application.run()


if __name__ == "__main__":
    config_file = Path(__file__).parent / "v4l2_imx274_player.yaml"
    main(config_file=config_file)
