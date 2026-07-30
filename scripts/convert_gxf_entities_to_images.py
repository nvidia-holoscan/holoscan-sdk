#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import argparse
import sys

from gxf_entity_codec import EntityReader
from PIL import Image


def iter_output_frames(entity_iter):
    for entity in entity_iter:
        component = entity.components[0]  # assume only one component with a tensor
        tensor = component.tensor
        array = tensor.array
        frame_bytes = array.tobytes()
        yield frame_bytes


def convert_gxf_entity_to_images(entity_dir, entity_basename, output_dir, output_name):
    with EntityReader(directory=entity_dir, basename=entity_basename) as reader:
        if reader.num_entities == 0:
            raise ValueError(
                f"No entities in GXF recording (directory={entity_dir!r}, "
                f"basename={entity_basename!r}). Index file is empty or missing. "
                "Ensure the application produced render output."
            )
        frame_shape = reader.get_frame(0).shape
        print(
            f"Frame array shape: {frame_shape[0]}x{frame_shape[1]}x{frame_shape[2]}"
            " (height x width x channels)",
            file=sys.stderr,
        )

        entities = reader.get_entities()
        for i, frame_data in enumerate(iter_output_frames(entities)):
            img = Image.frombytes("RGB", (frame_shape[1], frame_shape[0]), frame_data)
            img.save(str(output_dir + "/" + output_name + "{:04d}.png").format(i + 1))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Command line utility for exporting raw video frames from GXF Tensors files "
            "as PNG files."
        )
    )
    parser.add_argument("--basename", default="tensor", help="Basename for gxf entities to read")
    parser.add_argument("--directory", default="./", help="Directory for gxf entities to read")
    parser.add_argument("--outputname", default="tensor", help="Output name for images")
    parser.add_argument("--outputdir", default="./", help="Directory for output images")
    args = parser.parse_args()

    convert_gxf_entity_to_images(args.directory, args.basename, args.outputdir, args.outputname)


if __name__ == "__main__":
    main()
