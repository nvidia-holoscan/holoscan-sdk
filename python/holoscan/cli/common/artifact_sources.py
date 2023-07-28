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

import json
from pathlib import Path
from typing import Any, List, Optional

from .enum_types import Arch, Platform, PlatformConfiguration, SdkType
from .exceptions import InvalidSourceFileError


class ArtifactSources:
    """Provides default artifact source URLs with ability to override."""

    SectionVersion = "versions"
    SectionDebianPackages = "debian-packges"
    SectionBaseImages = "base-images"
    SectionBuildImages = "build-images"
    SectionHealthProbe = "health-probes"

    def __init__(self) -> None:
        self._data = {
            SdkType.MonaiDeploy.value: {ArtifactSources.SectionVersion: ["0.6.0"]},
            SdkType.Holoscan.value: {
                ArtifactSources.SectionVersion: ["0.6.0"],
                ArtifactSources.SectionDebianPackages: {
                    "0.6.0": {
                        Arch.amd64.value: "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/holoscan_0.6.0.3-1_amd64.deb",  # noqa: E501
                        Arch.arm64.value: {
                            PlatformConfiguration.iGPU.value: "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/holoscan_0.6.0.3-1_arm64.deb",  # noqa: E501
                            PlatformConfiguration.dGPU.value: "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/sbsa/holoscan_0.6.0.3-1_arm64.deb",  # noqa: E501
                        },
                    }
                },
                ArtifactSources.SectionBaseImages: {
                    PlatformConfiguration.iGPU.value: {
                        Platform.JetsonAgxOrinDevKit.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-igpu"
                        },
                        Platform.IGXOrinDevIt.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-igpu"
                        },
                    },
                    PlatformConfiguration.dGPU.value: {
                        Platform.X64Workstation.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        },
                        Platform.ClaraAGXDevKit.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        },
                        Platform.IGXOrinDevIt.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        },
                    },
                    PlatformConfiguration.iGPUAssist.value: {
                        Platform.ClaraAGXDevKit.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/l4t-compute-assist:r34.1.0-r8.4.0-runtime"  # noqa: E501
                        },
                        Platform.IGXOrinDevIt.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/l4t-compute-assist:r35.3.0-r8.5.2-runtime"  # noqa: E501
                        },
                    },
                    PlatformConfiguration.CPU.value: {
                        Platform.X64Workstation.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        }
                    },
                },
                ArtifactSources.SectionBuildImages: {
                    PlatformConfiguration.iGPU.value: {
                        Platform.JetsonAgxOrinDevKit.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-igpu"
                        },
                        Platform.IGXOrinDevIt.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-igpu"
                        },
                    },
                    PlatformConfiguration.dGPU.value: {
                        Platform.X64Workstation.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        },
                        Platform.ClaraAGXDevKit.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        },
                        Platform.IGXOrinDevIt.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        },
                    },
                    PlatformConfiguration.iGPUAssist.value: {
                        Platform.ClaraAGXDevKit.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-igpu"
                        },
                        Platform.IGXOrinDevIt.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-igpu"
                        },
                    },
                    PlatformConfiguration.CPU.value: {
                        Platform.X64Workstation.value: {
                            "0.6.0": "nvcr.io/nvidia/clara-holoscan/holoscan:v0.6.0-dgpu"
                        }
                    },
                },
                ArtifactSources.SectionHealthProbe: {
                    Arch.amd64.value: {
                        "0.6.0": "https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.19/grpc_health_probe-linux-amd64"  # noqa: E501
                    },
                    Arch.arm64.value: {
                        "0.6.0": "https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.19/grpc_health_probe-linux-arm64"  # noqa: E501
                    },
                },
            },
        }
        self.validate(self._data)

    @property
    def monai_deploy_versions(self) -> List[str]:
        return self._data[SdkType.MonaiDeploy.value]["versions"]

    @property
    def holoscan_versions(self) -> List[str]:
        return self._data[SdkType.Holoscan.value]["versions"]

    @property
    def base_images(self) -> List[Any]:
        return self._data[SdkType.Holoscan.value][ArtifactSources.SectionBaseImages]

    @property
    def build_images(self) -> List[Any]:
        return self._data[SdkType.Holoscan.value][ArtifactSources.SectionBuildImages]

    @property
    def health_prob(self) -> List[Any]:
        return self._data[SdkType.Holoscan.value][ArtifactSources.SectionHealthProbe]

    def load(self, file: Path):
        """Overrides the default values from a given JOSN file.
           Validates top-level attributes to ensure file is valid

        Args:
            file (Path): Path to JSON file
        """
        temp = json.loads(file.read_text())

        try:
            self.validate(temp)
        except Exception as ex:
            raise InvalidSourceFileError(f"{file} is missing required data: {ex}")

        self._data = temp

    def validate(self, data: Any):
        assert SdkType.MonaiDeploy.value in data
        assert SdkType.Holoscan.value in data

        assert ArtifactSources.SectionVersion in data[SdkType.MonaiDeploy.value]

        assert ArtifactSources.SectionVersion in data[SdkType.Holoscan.value]
        assert ArtifactSources.SectionDebianPackages in data[SdkType.Holoscan.value]
        assert ArtifactSources.SectionBaseImages in data[SdkType.Holoscan.value]
        assert ArtifactSources.SectionBuildImages in data[SdkType.Holoscan.value]

        assert "0.6.0" in data[SdkType.Holoscan.value][ArtifactSources.SectionDebianPackages]
        assert (
            Arch.amd64.value
            in data[SdkType.Holoscan.value][ArtifactSources.SectionDebianPackages]["0.6.0"]
        )
        assert "0.6.0" in data[SdkType.Holoscan.value][ArtifactSources.SectionDebianPackages]
        assert (
            Arch.arm64.value
            in data[SdkType.Holoscan.value][ArtifactSources.SectionDebianPackages]["0.6.0"]
        )

        for config in PlatformConfiguration:
            assert config.value in data[SdkType.Holoscan.value][ArtifactSources.SectionBaseImages]
            assert config.value in data[SdkType.Holoscan.value][ArtifactSources.SectionBuildImages]

    def debian_packages(
        self, version: str, architecture: Arch, platform_configuration: PlatformConfiguration
    ) -> Optional[str]:
        """Gets the URI of a debian package based on the version,
        the architecture and the platform configuration.

        Args:
            version (str): version of package
            architecture (Arch): architecture oif the package
            platform_configuration (PlatformConfiguration): platform configuration of the package

        Returns:
            Optional[str]: _description_
        """
        debian_sources = self._data[SdkType.Holoscan.value][ArtifactSources.SectionDebianPackages]
        if version not in debian_sources:
            return None

        if architecture == Arch.amd64:
            if architecture.value in debian_sources[version]:
                return debian_sources[version][architecture.value]
        elif architecture == Arch.arm64:
            if (
                architecture.value in debian_sources[version]
                and platform_configuration.value in debian_sources[version][architecture.value]
            ):
                return debian_sources[version][architecture.value][platform_configuration.value]

        return None
