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
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..common.artifact_sources import ArtifactSources
from ..common.constants import Constants
from ..common.dockerutils import image_exists
from ..common.enum_types import ApplicationType, SdkType
from ..common.exceptions import IncompatiblePlatformConfiguration, InvalidSdk
from ..common.sdk_utils import detect_sdk, detect_sdk_version
from .parameters import PlatformParameters
from .sdk_downloader import download_health_probe_file, download_sdk_debian_file


class Platform:
    def __init__(self, artifact_sources: ArtifactSources) -> None:
        """
        Args:
            args (Namespace): Input arguments for Packager from CLI
        """
        self._logger = logging.getLogger("packager")
        self._artifact_sources = artifact_sources

    def configure_platforms(
        self,
        args: Namespace,
        temp_dir: str,
        version: str,
        application_type: ApplicationType,
    ) -> Tuple[SdkType, str, List[PlatformParameters]]:
        """Configures a list of platforms that need to be built.
        1. Detect the SDK to use
        2. Detect the version of the SDK to use
        3. Ensure user provided arguments are valid and do not conflict
        4. Builds a list of platforms to be built where each platform includes the following:
          a. SDK distribution file(s) to use
          b. Base image to use
          c. Build image to use if application type is C++

        Args:
            args (Namespace): user provided arguments
            temp_dir (str): temporary path for storing downloaded artifacts
            version (str): application version
            application_type (ApplicationType): application type

        Returns:
            Tuple[SdkType, str, List[PlatformParameters]]: A tuple contains SDK, SDK version and list
            of platforms to be built
        """
        sdk = detect_sdk(args.sdk)
        sdk_version = detect_sdk_version(sdk, self._artifact_sources, args.sdk_version)

        self._validate_platform_options(args, sdk)

        platforms = []
        for platform in args.platform:
            platform_config = args.platform_config

            platform_parameters = PlatformParameters(platform, platform_config, args.tag, version)

            (
                platform_parameters.holoscan_sdk_file,
                platform_parameters.monai_deploy_sdk_file,
            ) = self._select_sdk_file(
                platform_parameters,
                temp_dir,
                sdk,
                sdk_version,
                application_type,
                args.holoscan_sdk_file,
                args.monai_deploy_sdk_file,
            )
            platform_parameters.base_image = self._find_base_image(
                platform_parameters, sdk_version, args.base_image
            )
            platform_parameters.build_image = self._find_build_image(
                platform_parameters, sdk_version, application_type, args.build_image
            )

            if sdk is SdkType.Holoscan:
                platform_parameters.health_probe = download_health_probe_file(
                    sdk_version,
                    platform_parameters.platform_arch,
                    temp_dir,
                    self._logger,
                    self._artifact_sources,
                )

            platforms.append(platform_parameters)

        return (sdk, sdk_version, platforms)

    def _validate_platform_options(self, args: Namespace, sdk: SdkType):
        """Validates user requests.
        - Packager accepts a single SDK distribution file, if user requests to build x64 and arm64
            at the same time, raise exception
        - If Packager was called using 'holoscan' command while user provides a MONAI Deploy APP
            SDK file, raise exception

        Args:
            args (Namespace): user provided arguments
            sdk (SdkType): SDK type

        Raises:
            IncompatiblePlatformConfiguration: when validation fails
        """
        if (args.holoscan_sdk_file is not None or args.monai_deploy_sdk_file is not None) and len(
            args.platform
        ) > 1:
            raise IncompatiblePlatformConfiguration(
                "Validation error: '--sdk-file' cannot be used with multiple platforms."
            )
        if sdk == SdkType.Holoscan and args.monai_deploy_sdk_file is not None:
            raise IncompatiblePlatformConfiguration(
                "--monai-deploy-sdk-file was used. Did you mean to use "
                "'monai-deploy package' command instead?"
            )

    def _find_base_image(
        self,
        platform_parameters: PlatformParameters,
        sdk_version: str,
        base_image: Optional[str] = None,
    ) -> str:
        """
        Ensure user provided base image exists in Docker or locate the base image to use based on
        request platform.

        Args:
            platform_parameters (PlatformParameters): target platform parameters
            sdk_version (str): SDK version
            base_image (Optional[str]): user provided base image

        Returns:
            (str): base image for building the image based on the given platform and SDK version.
        """
        if base_image is not None:
            if image_exists(base_image):
                return base_image
            else:
                raise InvalidSdk(f"Specified base image cannot be found: {base_image}")

        try:
            return self._artifact_sources.base_images[platform_parameters.platform_config.value][
                platform_parameters.platform.value
            ][sdk_version]
        except Exception:
            raise IncompatiblePlatformConfiguration(
                f"""No base image found for the selected configuration:
                        Platform: {platform_parameters.platform}
                        Configuration: {platform_parameters.platform_config}
                        Version: {sdk_version}"""
            )

    def _find_build_image(
        self,
        platform_parameters: PlatformParameters,
        sdk_version: str,
        application_type: ApplicationType,
        build_image: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ensure user provided build image exists or locate the build image to use based on the
        requested platform.

        Args:
            platform_parameters (PlatformParameters): target platform parameters
            sdk_version (str): SDK version
            application_type (ApplicationType): application type
            build_image (Optional[str]): user provided build image

        Returns:
            (str): build image for building the image based on the given platform and SDK version.
        """
        if build_image is not None:
            if image_exists(build_image):
                return build_image
            else:
                raise InvalidSdk(f"Specified build image cannot be found: {build_image}")

        if application_type == ApplicationType.CppCMake:
            try:
                return self._artifact_sources.build_images[
                    platform_parameters.platform_config.value
                ][platform_parameters.platform.value][sdk_version]
            except Exception:
                raise IncompatiblePlatformConfiguration(
                    f"No build image found for the selected configuration:"
                    f"\n   Platform: {platform_parameters.platform.value}"
                    f"\n   Configuration: {platform_parameters.platform_config.value}"
                    f"\n   Version: {sdk_version}"
                )
        else:
            return None

    def _select_sdk_file(
        self,
        platform_parameters: PlatformParameters,
        temp_dir: str,
        sdk: SdkType,
        sdk_version: str,
        application_type: ApplicationType,
        holoscan_sdk_file: Optional[Path] = None,
        monai_deploy_sdk_file: Optional[Path] = None,
    ) -> Tuple[Union[Path, str], Union[Path, str, None]]:
        """
        Detects the SDK distributable to use based on internal mapping or user input.

        - C++ & binary applications, attempt to download the SDK file.
        - Python application, use Holoscan or MONAI Deploy App SDK PyPI package.

        Args:
            platform_parameters (PlatformParameters): target platform parameters
            temp_dir (str): temporary location for storing downloaded files.
            sdk (SdkType): SDK to use
            sdk_version (str): SDK version
            application_type (ApplicationType): application type
            holoscan_sdk_file (Optional[Path]): path to the user specified Holoscan SDK file
            monai_deploy_sdk_file (Optional[Path]): path to the user specified MONAI Deploy
                                                    App SDK file

        Returns:
            (Tuple[Union[Path, str], Union[Path, str, None]]): A tuple where the first value contains
            Holoscan SDK and the second value contains MONAI Deploy App SDK.

        Raises:
            InvalidSdk: when user specified SDK file does not pass validation.
        """
        if sdk == SdkType.Holoscan:
            return (
                self._get_holoscan_sdk(
                    platform_parameters,
                    temp_dir,
                    SdkType.Holoscan,
                    sdk_version,
                    application_type,
                    holoscan_sdk_file,
                ),
                None,
            )
        elif sdk == SdkType.MonaiDeploy:
            return (
                self._get_holoscan_sdk(
                    platform_parameters,
                    temp_dir,
                    SdkType.Holoscan,
                    sdk_version,
                    application_type,
                    holoscan_sdk_file,
                ),
                self._get_monaideploy_sdk(monai_deploy_sdk_file),
            )
        return (None, None)

    def _get_holoscan_sdk(
        self,
        platform_parameters: PlatformParameters,
        temp_dir: str,
        sdk: SdkType,
        sdk_version: str,
        application_type: ApplicationType,
        sdk_file: Optional[Path] = None,
    ) -> Union[Path, str]:
        """
        Validates Holoscan SDK redistributable file if specified.
        Otherwise, attempt to download the SDK file from internet.

        Args:
            platform_parameters (PlatformParameters): Platform parameters
            temp_dir (str): Temporary location for storing downloaded distribution files.
            sdk (SdkType): SDK for building the application
            sdk_version (str): SDK version. Defaults to None.
            application_type (ApplicationType): application type
            sdk_file (Optional[Path], optional): SDK file from user input. Defaults to None.

        Raises:
            InvalidSdk: when an invalid SDK file is provided or unable to find matching SDK file.

        Returns:
            Union[Path, str]: Path to the SDK redistributable file.
        """
        assert sdk is SdkType.Holoscan

        if sdk_file is not None:
            if application_type in [
                ApplicationType.PythonModule,
                ApplicationType.PythonFile,
            ]:
                if sdk_file.suffix not in Constants.PYPI_FILE_EXTENSIONS:
                    raise InvalidSdk(
                        "Invalid SDK file format, must be a PyPI wheel file with .whl file extension."
                    )
                return sdk_file
            elif application_type in [
                ApplicationType.CppCMake,
                ApplicationType.Binary,
            ]:
                if sdk_file.suffix != Constants.DEBIAN_FILE_EXTENSION:
                    raise InvalidSdk(
                        "Invalid SDK file format, must be a Debian package file with .deb "
                        "file extension."
                    )
                return sdk_file

            raise InvalidSdk(f"Unknown application type: {application_type.value}")
        else:
            if application_type in [
                ApplicationType.PythonModule,
                ApplicationType.PythonFile,
            ]:
                return Constants.PYPI_INSTALL_SOURCE
            elif application_type in [
                ApplicationType.CppCMake,
                ApplicationType.Binary,
            ]:
                debian_package_source = self._artifact_sources.debian_packages(
                    sdk_version,
                    platform_parameters.platform_arch,
                    platform_parameters.platform_arch,
                )
                if debian_package_source is not None:
                    return download_sdk_debian_file(
                        debian_package_source,
                        sdk_version,
                        platform_parameters.platform_arch,
                        temp_dir,
                        self._logger,
                        self._artifact_sources,
                    )
                else:
                    raise InvalidSdk(
                        f"No match Debian packages found for Holoscan SDK v{sdk_version}. Try using "
                        "`--sdk-file` instead."
                    )

            raise InvalidSdk(f"Unknown application type: {application_type.value}")

    def _get_monaideploy_sdk(
        self,
        sdk_file: Optional[Path] = None,
    ) -> Union[Path, str]:
        """
        Validates MONAI Deploy SDK redistributable file if specified.
        Otherwise, Docker build stage will install the SDK from PyPI.

        Args:
            sdk_file (Optional[Path], optional): SDK file from user input. Defaults to None.

        Raises:
            InvalidSdk: when an invalid SDK file is provided or unable to find matching SDK file.

        Returns:
            Union[Path, str]: Path to the SDK redistributable file.
        """

        if sdk_file is not None:
            if sdk_file.suffix not in Constants.PYPI_FILE_EXTENSIONS:
                raise InvalidSdk(
                    "Invalid SDK file format, must be a PyPI wheel file with .whl file extension."
                )
            return sdk_file

        return Constants.PYPI_INSTALL_SOURCE
