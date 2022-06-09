#
# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import logging
import os
import signal
import subprocess

def download(resource: str, short_name: str, args):
    # Run ngc command
    cmd = f"ngc registry resource download-version '{args.ngc_org_team}/{resource}:{args.version}' --dest {args.dest}"
    logging.info(f"> {cmd}")
    dl_proc = subprocess.Popen([cmd], shell=True)

    # Propagate callback
    def signal_handler(sig, frame):
        os.killpg(os.getpgid(dl_proc.pid), signal.SIGKILL)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for upload to end
    dl_proc.wait()

    # Move
    old = f"{args.dest}/{resource}_v{args.version}"
    new = f"{args.dest}/{short_name}"
    os.rename(old, new)


def main():
    # Logger
    logging.getLogger().setLevel(logging.INFO)

    # Parse args
    parser = argparse.ArgumentParser(
        description='Command line utility for downloading sample data from NGC.'
    )
    parser.add_argument(
        '-v', '--version', default='20220608', type=str, help='NGC tag version'
    )
    parser.add_argument(
        '-o', '--ngc_org_team', default='nvidia/clara-holoscan', type=str, help='NGC org/team'
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.realpath(f"{script_dir}/../test_data")
    parser.add_argument(
        '-d', '--dest', default=default_data_dir, help='Output directory'
    )
    args = parser.parse_args()

    # Create destination folder
    os.makedirs(args.dest, exist_ok=True)

    # Download
    download('holoscan_ultrasound_sample_data', 'ultrasound', args)
    download('holoscan_endoscopy_sample_data', 'endoscopy', args)

if __name__ == '__main__':
    main()
