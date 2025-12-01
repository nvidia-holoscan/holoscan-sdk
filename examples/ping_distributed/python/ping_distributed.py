"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import textwrap
from argparse import ArgumentParser

import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, Tracker
from holoscan.operators import PingTensorRxOp, PingTensorTxOp


class Fragment1(Fragment):
    def __init__(
        self,
        *args,
        gpu_tensor=False,
        count=10,
        shape=(32, 64),
        dtype="uint8_t",
        tensor_name="out",
        **kwargs,
    ):
        self.gpu_tensor = gpu_tensor
        self.count = count
        if len(shape) == 2:
            self.rows, self.columns = shape
        else:
            raise ValueError("expected shape of length 2")
        self.shape = shape
        self.dtype = dtype
        self.tensor_name = tensor_name
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        storage_type = "device" if self.gpu_tensor else "system"
        tx = PingTensorTxOp(
            self,
            CountCondition(self, self.count),
            storage_type=storage_type,
            rows=self.rows,
            columns=self.columns,
            dtype=self.dtype,
            tensor_name=self.tensor_name,
            name="tx",
        )

        # Add the operator (tx) to the fragment
        self.add_operator(tx)


class Fragment2(Fragment):
    def compose(self):
        rx = PingTensorRxOp(self, name="rx")

        # Add the operator (rx) to the fragment
        self.add_operator(rx)


class MyPingApp(Application):
    def __init__(self, *args, gpu_tensor=False, count=10, shape=(32, 64), dtype=np.uint8, **kwargs):
        self.gpu_tensor = gpu_tensor
        self.count = count
        self.shape = shape
        self.dtype = dtype
        super().__init__(*args, **kwargs)

    def compose(self):
        fragment1 = Fragment1(
            self,
            name="fragment1",
            gpu_tensor=self.gpu_tensor,
            count=self.count,
            shape=self.shape,
            dtype=self.dtype,
        )
        fragment2 = Fragment2(self, name="fragment2")

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(fragment1, fragment2, {("tx", "rx")})


def main(on_gpu=False, count=10, shape=(64, 32), dtype=np.uint8, data_flow_tracking_enabled=False):
    app = MyPingApp(gpu_tensor=on_gpu, count=count, shape=shape, dtype=dtype)

    if data_flow_tracking_enabled:
        with Tracker(
            app,
            num_start_messages_to_skip=0,
            num_last_messages_to_discard=0,
        ) as trackers:
            # set separate log files for each fragment
            for fragment_name, tracker in trackers.items():
                tracker.enable_logging(fragment_name + "_logger.log")
            app.run()
            print(f"{type(trackers)=}, {trackers=}")
            for fragment_name, tracker in trackers.items():
                print(f"Fragment: {fragment_name}")
                tracker.print()
    else:
        app.run()


if __name__ == "__main__":
    description = (
        "Distributed tensor-based ping application. The tensor will have shape ([batch], rows, "
        "[columns], [channels]) where dimensions listed in square brackets are only present when "
        "the corresponding parameter is not 0."
    )
    parser = ArgumentParser(description=description, add_help=False)
    # Application().argv will intercept, -h and --help. Can provide --info instead
    # to see the help strings for these additional custom options.
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use a GPU tensor instead of a host tensor",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="The number of times to transmit the tensor.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="The batch size of the tensor. If 0, no batch dimension will be present.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=32,
        help="The number of rows for the transmitted tensor.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=64,
        help=(
            "The number of columns for the transmitted tensor. If 0, no columns dimension will be "
            "present."
        ),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=0,
        help=(
            "The number of channels for the transmitted tensor. If 0, no channels dimension will "
            "be present."
        ),
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="uint8_t",
        help=(
            "The C++ data type of the tensor elements. It must be one of "
            "{'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', "
            " 'uint64_t', 'float', 'double', 'complex<float>', 'complex<double>'}"
        ),
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable data flow tracking for the distributed app",
    )
    # use parse_known_args to ignore other CLI arguments that may be used by Application
    args, remaining = parser.parse_known_args()

    # can print the parser's help here prior to the Application's help output
    if "-h" in remaining or "--help" in remaining:
        print("\nAdditional arguments supported by this application:")
        print(textwrap.indent(parser.format_help(), "  "))
    else:
        # validate the arguments
        max_int32 = 2**31 - 1
        max_int64 = 2**63 - 1
        if args.count < 0 or args.count > max_int64:
            raise ValueError(f"count must be in range [0, {max_int64}]")
        if args.batch < 0 or args.batch > max_int32:
            raise ValueError(f"batch must be in range [0, {max_int32}]")
        if args.rows < 1 or args.rows > max_int32:
            raise ValueError(f"rows must be in range [1, {max_int32}]")
        if args.columns < 0 or args.columns > max_int32:
            raise ValueError(f"columns must be in range [0, {max_int32}]")
        if args.channels < 0 or args.channels > max_int32:
            raise ValueError(f"channels must be in range [0, {max_int32}]")

        tensor_str = "GPU" if args.gpu else "host"
        print(f"Configuring application to use {tensor_str} tensors")

    # only non-zero dimensions are included in the shape
    shape = tuple(s for s in (args.batch, args.rows, args.columns, args.channels) if s > 0)

    main(
        on_gpu=args.gpu,
        count=args.count,
        shape=shape,
        dtype=args.data_type,
        data_flow_tracking_enabled=args.track,
    )
