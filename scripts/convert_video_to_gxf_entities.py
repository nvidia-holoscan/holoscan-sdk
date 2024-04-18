#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

import numpy as np
from gxf_entity_codec import EntityWriter


def iter_input_frames(f, width, height, channels):
    frame_byte_count = width * height * channels
    while True:
        frame_bytes = f.read(frame_byte_count)
        if len(frame_bytes) != frame_byte_count:
            break
        array = np.frombuffer(frame_bytes, np.uint8)
        frame = array.reshape(height, width, channels)
        yield frame


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Command line utility for writing raw video frames in GXF Tensors for "
            "stream playback."
        )
    )
    parser.add_argument("--width", required=True, type=int, help="Input width in pixels")
    parser.add_argument("--height", required=True, type=int, help="Input height in pixels")
    parser.add_argument("--channels", default=3, type=int, help="Channel count")
    parser.add_argument("--framerate", default=30, type=int, help="Output frame rate")
    parser.add_argument("--basename", default="tensor", help="Basename for gxf entities to write")
    parser.add_argument("--directory", default="./", help="Directory for gxf entities to write")
    args = parser.parse_args()

    with EntityWriter(
        directory=args.directory, basename=args.basename, framerate=args.framerate
    ) as recorder:
        for frame in iter_input_frames(
            sys.stdin.buffer,
            width=args.width,
            height=args.height,
            channels=args.channels,
        ):
            recorder.add(frame)


if __name__ == "__main__":
    main()
