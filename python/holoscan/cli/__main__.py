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

import argparse
import json
import logging
import logging.config
import os
from pathlib import Path
from typing import List, Optional, Union

from .common.enum_types import Platform, PlatformConfiguration

logging.getLogger("docker.api.build").setLevel(logging.WARNING)
logging.getLogger("docker.auth").setLevel(logging.WARNING)
logging.getLogger("docker.utils.config").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

LOG_CONFIG_FILENAME = "logging.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from .packager.package_command import create_package_parser
    from .runner.run_command import create_run_parser

    if argv is None:
        import sys

        argv = sys.argv
    argv = list(argv)  # copy argv for manipulation to avoid side-effects

    # We have intentionally not set the default using `default="INFO"` here so that the default
    # value from here doesn't override the value in `LOG_CONFIG_FILENAME` unless the user indends to do
    # so. If the user doesn't use this flag to set log level, this argument is set to "None"
    # and the logging level specified in `LOG_CONFIG_FILENAME` is used.

    command_name = os.path.basename(argv[0])
    program_name = "holoscan" if command_name == "__main__.py" else command_name
    parent_parser = argparse.ArgumentParser()

    parent_parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        help="set the logging level (default: INFO)",
    )

    parser = argparse.ArgumentParser(
        parents=[parent_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        prog=program_name,
    )

    subparser = parser.add_subparsers(dest="command")

    # Parser for `package` command
    create_package_parser(subparser, "package", parents=[parent_parser])

    # Parser for `run` command
    create_run_parser(subparser, "run", parents=[parent_parser])

    # Parser for `version` command
    subparser.add_parser(
        "version", formatter_class=argparse.HelpFormatter, parents=[parent_parser], add_help=False
    )
    args = parser.parse_args(argv[1:])
    args.argv = argv  # save argv for later use in runpy

    # Print help if no command is specified
    if args.command is None:
        parser.print_help()
        parser.exit()

    if args.command == "package":
        if args.platform[0] == Platform.X64Workstation:
            args.platform_config = PlatformConfiguration.dGPU
        elif args.platform_config is None:
            parser.error(f"--platform-config is required for '{args.platform[0].value}'")

    return args


def set_up_logging(level: Optional[str], config_path: Union[str, Path] = LOG_CONFIG_FILENAME):
    """Initializes the logger and sets up logging level.

    Args:
        level (str): A logging level (DEBUG, INFO, WARN, ERROR, CRITICAL).
        log_config_path (str): A path to logging config file.
    """
    # Default log config path
    log_config_path = Path(__file__).absolute().parent / LOG_CONFIG_FILENAME

    config_path = Path(config_path)

    # If a logging config file that is specified by `log_config_path` exists in the current folder,
    # it overrides the default one
    if config_path.exists():
        log_config_path = config_path

    config_dict = json.loads(log_config_path.read_bytes())

    if level is not None and "root" in config_dict:
        config_dict["root"]["level"] = level
    logging.config.dictConfig(config_dict)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    set_up_logging(args.log_level)

    if args.command == "package":
        from .packager.packager import execute_package_command

        execute_package_command(args)

    elif args.command == "run":
        from .runner.runner import execute_run_command

        execute_run_command(args)

    elif args.command == "version":
        from .common.artifact_sources import ArtifactSources
        from .version.version import execute_version_command

        artifact_sources = ArtifactSources()
        execute_version_command(args, artifact_sources)


if __name__ == "__main__":
    main()
