#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import argparse
import sys

from gxf_entity_codec import EntityReader


def iter_output_frames(entity_iter):
    for entity in entity_iter:
        component = entity.components[0]  # assume only one component with a tensor
        tensor = component.tensor
        array = tensor.array
        frame_bytes = array.tobytes()
        yield frame_bytes


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Command line utility for reading raw video frames in GXF Tensors for stream recording."
        )
    )
    parser.add_argument("--basename", default="tensor", help="Basename for gxf entities to read")
    parser.add_argument("--directory", default="./", help="Directory for gxf entities to read")
    args = parser.parse_args()

    with EntityReader(directory=args.directory, basename=args.basename) as reader:
        frame_rate = reader.get_framerate()
        print(f"Guessed frame rate: {frame_rate} fps", file=sys.stderr)
        frame_shape = reader.get_frame(0).shape
        print(
            f"Frame array shape: {frame_shape[0]}x{frame_shape[1]}x{frame_shape[2]}"
            " (height x width x channels)",
            file=sys.stderr,
        )
        entities = reader.get_entities()
        for frame_data in iter_output_frames(entities):
            sys.stdout.buffer.write(frame_data)


if __name__ == "__main__":
    main()
