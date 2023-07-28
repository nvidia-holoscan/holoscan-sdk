# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from argparse import ArgumentParser, HelpFormatter, _SubParsersAction
from typing import List

from ..common import argparse_types
from ..common.argparse_types import valid_existing_path

logger = logging.getLogger("runner")


def create_run_parser(
    subparser: _SubParsersAction, command: str, parents: List[ArgumentParser]
) -> ArgumentParser:
    parser: ArgumentParser = subparser.add_parser(
        command, formatter_class=HelpFormatter, parents=parents, add_help=False
    )

    parser.add_argument("map", metavar="<image[:tag]>", help="HAP/MAP image name.")

    parser.add_argument(
        "--address",
        dest="address",
        help="address ('[<IP or hostname>][:<port>]') of the App Driver. If not specified, "
        "the App Driver uses the default host address ('0.0.0.0') with the default port "
        "number ('8765').",
    )

    parser.add_argument(
        "--driver",
        dest="driver",
        action="store_true",
        default=False,
        help="run the App Driver on the current machine. Can be used together with the "
        "'--worker' option "
        "to run both the App Driver and the App Worker on the same machine.",
    )

    parser.add_argument(
        "-i",
        "--input",
        metavar="<input>",
        type=argparse_types.valid_dir_path,
        help="input data directory path.",
        required=False,
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="<output>",
        type=argparse_types.valid_dir_path,
        help="output data directory path.",
        required=False,
    )

    parser.add_argument(
        "-f",
        "--fragments",
        dest="fragments",
        help="comma-separated names of the fragments to be executed by the App Worker. "
        "If not specified, only one fragment (selected by the App Driver) will be executed. "
        "'all' can be used to run all the fragments.",
    )

    parser.add_argument(
        "--worker",
        dest="worker",
        action="store_true",
        default=False,
        help="run the App Worker.",
    )

    parser.add_argument(
        "--worker-address",
        dest="worker_address",
        help="address (`[<IP or hostname>][:<port>]`) of the App Worker. If not specified, the App "
        "Worker uses the default host address ('0.0.0.0') with the default port number "
        "randomly chosen from unused ports (between 10000 and 32767).",
    )

    advanced_group = parser.add_argument_group(title="advanced run options")

    advanced_group.add_argument(
        "--config",
        dest="config",
        type=valid_existing_path,
        help="path to the configuration file. This will override the configuration file embedded "
        "in the application.",
    )

    advanced_group.add_argument(
        "--name",
        dest="name",
        help="name and hostname of the container to create.",
    )
    advanced_group.add_argument(
        "-n",
        "--network",
        dest="network",
        default="host",
        help="name of the Docker network this application will be connected to.  (default: host)",
    )
    advanced_group.add_argument(
        "--nic",
        dest="nic",
        help="name of the network interface to use with a distributed multi-fragment application. "
        "This option sets UCX_NET_DEVICES environment variable with the value specified.",
    )
    advanced_group.add_argument(
        "-r",
        "--render",
        dest="render",
        action="store_true",
        default=False,
        help="enable rendering (default: False); runs the container with required flags to enable "
        "rendering of graphics.",
    )
    advanced_group.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="suppress the STDOUT and print only STDERR from the application. (default: False)",
    )
    advanced_group.add_argument(
        "--shm-size",
        dest="shm_size",
        help="sets the size of /dev/shm. The format is <number(int,float)>[MB|m|GB|g|Mi|MiB|Gi|GiB]."
        " Use 'config' to read the shared memory value defined in the app.json manifest."
        " If not specified, the container is launched using '--ipc=host' with host system's "
        "/dev/shm mounted.",
    )
    advanced_group.add_argument(
        "--terminal",
        dest="terminal",
        action="store_true",
        default=False,
        help="enters terminal with all configured volume mappings and environment variables. "
        "(default: False)",
    )

    user_group = parser.add_argument_group(title="security options")
    user_group.add_argument(
        "--uid",
        type=str,
        default=os.getuid(),
        help=f"runs the container with the UID. (default:{os.getuid()})",
    )
    user_group.add_argument(
        "--gid",
        type=str,
        default=os.getgid(),
        help=f"runs the container with the GID. (default:{os.getgid()})",
    )

    return parser
