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

import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

from ..common.artifact_sources import ArtifactSources
from ..common.constants import Constants
from ..common.dockerutils import image_exists
from ..common.enum_types import ApplicationType, SdkType
from ..common.exceptions import IncompatiblePlatformConfigurationError, InvalidSdkError
from ..common.sdk_utils import detect_sdk, detect_sdk_version
from .parameters import PlatformParameters


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
    ) -> tuple[SdkType, str, str, list[PlatformParameters]]:
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
            Tuple[SdkType, str, str, List[PlatformParameters]]: A tuple contains SDK, Holoscan SDK
            version, MONAI Deploy App SDK version and list of platforms to be built
        """
        sdk = detect_sdk(args.sdk)
        holoscan_sdk_version, monai_deploy_app_sdk_version = detect_sdk_version(
            sdk, self._artifact_sources, args.sdk_version
        )

        self._validate_platform_options(args, sdk)

        platforms = []
        for platform in args.platform:
            platform_config = args.platform_config

            platform_parameters = PlatformParameters(platform, platform_config, args.tag, version)

            (
                platform_parameters.custom_base_image,
                platform_parameters.base_image,
            ) = self._find_base_image(platform_parameters, holoscan_sdk_version, args.base_image)

            platform_parameters.build_image = self._find_build_image(
                platform_parameters, holoscan_sdk_version, application_type, args.build_image
            )

            (
                (
                    platform_parameters.custom_holoscan_sdk,
                    platform_parameters.holoscan_sdk_file,
                ),
                (
                    platform_parameters.custom_monai_deploy_sdk,
                    platform_parameters.monai_deploy_sdk_file,
                ),
            ) = self._select_sdk_file(
                platform_parameters,
                temp_dir,
                sdk,
                holoscan_sdk_version,
                monai_deploy_app_sdk_version,
                application_type,
                args.holoscan_sdk_file,
                args.monai_deploy_sdk_file,
            )

            if sdk is SdkType.Holoscan:
                platform_parameters.health_probe = self._artifact_sources.health_probe(
                    holoscan_sdk_version
                )[platform_parameters.platform_arch.value]

            platforms.append(platform_parameters)

        return (sdk, holoscan_sdk_version, monai_deploy_app_sdk_version, platforms)

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
            IncompatiblePlatformConfigurationError: when validation fails
        """
        if (args.holoscan_sdk_file is not None or args.monai_deploy_sdk_file is not None) and len(
            args.platform
        ) > 1:
            raise IncompatiblePlatformConfigurationError(
                "Validation error: '--sdk-file' cannot be used with multiple platforms."
            )
        if sdk == SdkType.Holoscan and args.monai_deploy_sdk_file is not None:
            raise IncompatiblePlatformConfigurationError(
                "--monai-deploy-sdk-file was used. Did you mean to use "
                "'monai-deploy package' command instead?"
            )

    def _find_base_image(
        self,
        platform_parameters: PlatformParameters,
        sdk_version: str,
        base_image: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Ensure user provided base image exists in Docker or locate the base image to use based on
        request platform.

        Args:
            platform_parameters (PlatformParameters): target platform parameters
            sdk_version (str): SDK version
            base_image (Optional[str]): user provided base image

        Returns:
            (Tuple(bool, str)): bool: True if using user provided image.
                                str: base image for building the image based on the given
                                     platform and SDK version.
        """
        if base_image is not None:
            if image_exists(base_image):
                return (True, base_image)
            else:
                raise InvalidSdkError(f"Specified base image cannot be found: {base_image}")

        try:
            return (
                False,
                self._artifact_sources.base_image(sdk_version)[
                    platform_parameters.platform_config.value
                ],
            )
        except Exception as ex:
            raise IncompatiblePlatformConfigurationError(
                f"""No base image found for the selected configuration:
                        Platform: {platform_parameters.platform}
                        Configuration: {platform_parameters.platform_config}
                        Version: {sdk_version}"""
            ) from ex

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
                raise InvalidSdkError(f"Specified build image cannot be found: {build_image}")

        if application_type == ApplicationType.CppCMake:
            try:
                return self._artifact_sources.build_images(sdk_version)[
                    platform_parameters.platform_config.value
                ][platform_parameters.platform.value]
            except Exception as ex:
                raise IncompatiblePlatformConfigurationError(
                    f"No build image found for the selected configuration:"
                    f"\n   Platform: {platform_parameters.platform.value}"
                    f"\n   Configuration: {platform_parameters.platform_config.value}"
                    f"\n   Version: {sdk_version}"
                ) from ex
        else:
            return None

    def _select_sdk_file(
        self,
        platform_parameters: PlatformParameters,
        temp_dir: str,
        sdk: SdkType,
        holoscan_sdk_version: str,
        monai_deploy_app_sdk_version: Optional[str],
        application_type: ApplicationType,
        holoscan_sdk_file: Optional[Path] = None,
        monai_deploy_sdk_file: Optional[Path] = None,
    ) -> tuple[
        tuple[bool, Union[Path, str]],
        tuple[Union[Optional[Path], Optional[str]], Union[Optional[Path], Optional[str]]],
    ]:
        """
        Detects the SDK distributable to use based on internal mapping or user input.

        - C++ & binary applications, attempt to download the SDK file.
        - Python application, use Holoscan or MONAI Deploy App SDK PyPI package.

        Args:
            platform_parameters (PlatformParameters): target platform parameters
            temp_dir (str): temporary location for storing downloaded files.
            sdk (SdkType): SDK to use
            holoscan_sdk_version (str): Holoscan SDK version
            monai_deploy_app_sdk_version (Optional[str]): MONAI Deploy SDK version
            application_type (ApplicationType): application type
            holoscan_sdk_file (Optional[Path]): path to the user specified Holoscan SDK file
            monai_deploy_sdk_file (Optional[Path]): path to the user specified MONAI Deploy
                                                    App SDK file

        Returns:
            (Tuple[Union[Path, str], Union[Path, str, None]]): A tuple where the first value
                contains Holoscan SDK and the second value contains MONAI Deploy App SDK.

        Raises:
            InvalidSdkError: when user specified SDK file does not pass validation.
        """
        if sdk == SdkType.Holoscan:
            return (
                self._get_holoscan_sdk(
                    platform_parameters,
                    temp_dir,
                    SdkType.Holoscan,
                    holoscan_sdk_version,
                    application_type,
                    holoscan_sdk_file,
                ),
                (None, None),
            )
        elif sdk == SdkType.MonaiDeploy:
            if monai_deploy_app_sdk_version is None:
                raise InvalidSdkError("MONAI Deploy App SDK version missing")
            return (
                self._get_holoscan_sdk(
                    platform_parameters,
                    temp_dir,
                    SdkType.Holoscan,
                    holoscan_sdk_version,
                    application_type,
                    holoscan_sdk_file,
                ),
                self._get_monai_deploy_sdk(monai_deploy_app_sdk_version, monai_deploy_sdk_file),
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
    ) -> tuple[bool, Union[Path, str]]:
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
            InvalidSdkError: when an invalid SDK file is provided or unable to find matching SDK
                             file.

        Returns:
            Tuple[bool, Union[Path, str]]:
                bool: True when user provides SDk file. Otherwise, False.
                Union[Path, str]: User provided SDK file path or package version.
        """
        assert sdk is SdkType.Holoscan

        if sdk_file is not None:
            if application_type in [
                ApplicationType.PythonModule,
                ApplicationType.PythonFile,
            ]:
                if sdk_file.suffix not in Constants.PYPI_FILE_EXTENSIONS:
                    raise InvalidSdkError(
                        "Invalid SDK file format, must be a PyPI wheel file with .whl file "
                        "extension."
                    )
                return (True, sdk_file)
            elif application_type in [
                ApplicationType.CppCMake,
                ApplicationType.Binary,
            ]:
                if sdk_file.suffix != Constants.DEBIAN_FILE_EXTENSION:
                    raise InvalidSdkError(
                        "Invalid SDK file format, must be a Debian package file with .deb "
                        "file extension."
                    )
                return (True, sdk_file)

            raise InvalidSdkError(f"Unknown application type: {application_type.value}")
        else:
            if application_type in [
                ApplicationType.PythonModule,
                ApplicationType.PythonFile,
            ]:
                wheel_package_version = self._artifact_sources.wheel_package_version(sdk_version)

                if wheel_package_version is None:
                    raise InvalidSdkError(
                        "Unable to locate matching Holoscan SDK PyPI package with "
                        f"version {sdk_version}."
                    )

                return (False, wheel_package_version)
            elif application_type in [
                ApplicationType.CppCMake,
                ApplicationType.Binary,
            ]:
                debian_package_version = self._artifact_sources.debian_package_version(sdk_version)

                if debian_package_version is None:
                    raise InvalidSdkError(
                        "Unable to locate matching Holoscan SDK Debian package with "
                        f"version {sdk_version}."
                    )

                return (False, debian_package_version)

            raise InvalidSdkError(f"Unknown application type: {application_type.value}")

    def _get_monai_deploy_sdk(
        self, monai_deploy_app_sdk_version: Optional[str], sdk_file: Optional[Path] = None
    ) -> tuple[bool, Union[Optional[Path], Optional[str]]]:
        """
        Validates MONAI Deploy SDK redistributable file if specified.
        Otherwise, Docker build stage will install the SDK from PyPI.

        Args:
            sdk_file (Optional[Path], optional): SDK file from user input. Defaults to None.
            monai_deploy_app_sdk_version: (Optional[str]): MONAI Deploy App SDK version
        Raises:
            InvalidSdkError: when an invalid SDK file is provided or unable to find matching SDK
                             file.

        Returns:
            Union[Path, str]: Path to the SDK redistributable file.
        """

        if sdk_file is not None:
            if sdk_file.suffix not in Constants.PYPI_FILE_EXTENSIONS:
                raise InvalidSdkError(
                    "Invalid SDK file format, must be a PyPI wheel file with .whl file extension."
                )
            return (True, sdk_file)

        return (False, None)
