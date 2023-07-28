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
import sys
from argparse import Namespace

from ..common.artifact_sources import ArtifactSources
from ..common.enum_types import SdkType
from ..common.sdk_utils import detect_holoscan_version, detect_monaideploy_version, detect_sdk

logger = logging.getLogger("version")


def execute_version_command(args: Namespace, artifact_sources: ArtifactSources):
    print(f"You are executing Holoscan CLI from: {os.path.dirname(os.path.abspath(sys.argv[0]))}\n")

    try:
        sdk = detect_sdk()
        try:
            sdk_version = detect_holoscan_version(artifact_sources)
            print(f"Holoscan SDK:           {sdk_version}")
        except Exception:
            print("Holoscan SDK:           N/A")

        if sdk == SdkType.MonaiDeploy:
            try:
                sdk_version = detect_monaideploy_version(artifact_sources)
                print(f"MONAI Deploy App SDK:   {sdk_version}")
            except Exception:
                print("MONAI Deploy App SDK:   N/A")

    except Exception as ex:
        logging.error("Error executing version command.")
        logger.debug(ex)
