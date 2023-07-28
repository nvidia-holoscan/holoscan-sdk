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
import shutil
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..common.constants import Constants, DefaultValues
from ..common.dockerutils import build_docker_image, create_and_get_builder, docker_export_tarball
from ..common.enum_types import SdkType
from ..common.exceptions import WrongApplicationPathError
from .parameters import PackageBuildParameters, PlatformBuildResults, PlatformParameters


class BuilderBase:
    """
    Docker container image builder base class.
    Prepares files for building the docker image and calls Docker API to build the container image.
    """

    def __init__(
        self,
        build_parameters: PackageBuildParameters,
        temp_dir: str,
    ) -> None:
        """
        Copy the application, model files, and user documentations here in __init__ since they
        won't change when building different platforms.

        Args:
            build_parameters (PackageBuildParameters): general build parameters
            temp_dir (str): temporary directory to store files required for build

        """
        self._logger = logging.getLogger("packager.builder")
        self._build_parameters = build_parameters
        self._temp_dir = temp_dir
        self._copy_application()
        self._copy_model_files()
        self._copy_docs()
        _ = self._write_dockerignore()
        _ = self._copy_script()

    def build(self, platform_parameters: PlatformParameters) -> PlatformBuildResults:
        """Build a new container image for a specific platform.
        Copy supporting files, such as redistributables and generate Dockerfile for the build.

        Args:
            platform_parameters (PlatformParameters): platform parameters

        Returns:
            PlatformBuildResults: build results
        """
        self._copy_supporting_files(platform_parameters)
        self._copy_health_probe(platform_parameters)
        docker_file_path = self._write_dockerfile(platform_parameters)

        return self._build_internal(docker_file_path, platform_parameters)

    def _build_internal(
        self, dockerfile: str, platform_parameters: PlatformParameters
    ) -> PlatformBuildResults:
        """Prepare parameters for Docker buildx build

        Args:
            dockerfile (str): Path to Dockerfile to be built
            platform_parameters (PlatformParameters): platform parameters

        Returns:
            PlatformBuildResults: build results
        """
        self.print_build_info(platform_parameters)
        builder = create_and_get_builder(Constants.LOCAL_BUILDX_BUILDER_NAME)

        build_result = PlatformBuildResults(platform_parameters)

        cache_to = {"type": "local", "dest": self._build_parameters.build_cache}
        cache_from = [{"type": "local", "src": self._build_parameters.build_cache}]
        if platform_parameters.base_image is not None:
            cache_from.append({"type": "registry", "ref": platform_parameters.base_image})
        if platform_parameters.build_image is not None:
            cache_from.append({"type": "registry", "ref": platform_parameters.build_image})
        builds = {
            "builder": builder,
            "cache": not self._build_parameters.no_cache,
            "cache_from": cache_from,
            "cache_to": cache_to,
            "context_path": self._temp_dir,
            "file": dockerfile,
            "platforms": [platform_parameters.docker_arch],
            "progress": "plain" if self._logger.root.level == logging.DEBUG else "auto",
            "pull": True,
            "tags": [platform_parameters.tag],
        }

        export_to_tar_ball = False
        if self._build_parameters.tarball_output is not None:
            build_result.tarball_filenaem = str(
                self._build_parameters.tarball_output
                / f"{platform_parameters.tag}{Constants.TARBALL_FILE_EXTENSION}"
            ).replace(":", "-")

        # Make result image available on 'docker image' only if arch matches
        if platform_parameters.same_arch_as_system:
            builds["load"] = True
            build_result.docker_tag = platform_parameters.tag
            export_to_tar_ball = self._build_parameters.tarball_output is not None
        else:
            if shutil.which("update-binfmts") is None:
                build_result.succeeded = False
                build_result.error = (
                    "Skipped due to missing QEMU and its dependencies. "
                    "Please follow the link to install QEMU "
                    "https://docs.nvidia.com/datacenter/cloud-native/playground/x-arch.html#installing-qemu"  # noqa: E501
                )
                return build_result
            if self._build_parameters.tarball_output is not None:
                builds["output"] = {
                    "type": "oci",
                    "dest": build_result.tarball_filenaem,
                }
            else:
                build_result.succeeded = False
                build_result.error = (
                    "Skipped due to incompatible system architecture. "
                    "Use '--output' to write image to disk."
                )
                return build_result

        builds["build_args"] = {
            "UID": self._build_parameters.uid,
            "GID": self._build_parameters.gid,
            "UNAME": self._build_parameters.username,
        }

        self._logger.debug(f"Building Holoscan Application Package: tag={platform_parameters.tag}")

        try:
            build_docker_image(**builds)
            build_result.succeeded = True
            if export_to_tar_ball:
                try:
                    self._logger.info(
                        f"Saving {platform_parameters.tag} to {build_result.tarball_filenaem}..."
                    )
                    docker_export_tarball(build_result.tarball_filenaem, platform_parameters.tag)
                except Exception as ex:
                    build_result.error = f"Error saving tarball: {ex}"
                    build_result.succeeded = False
        except Exception:
            build_result.succeeded = False
            build_result.error = "Error building image: see Docker output for additional details."

        return build_result

    def print_build_info(self, platform_parameters):
        """Print build information for the platform."""
        self._logger.info(
            f"""
===============================================================================
Building image for:                 {platform_parameters.platform.value}
    Architecture:                   {platform_parameters.platform_arch.value}
    Base Image:                     {platform_parameters.base_image}
    Build Image:                    {platform_parameters.build_image if platform_parameters.build_image is not None else "N/A"}  
    Cache:                          {'Disabled' if self._build_parameters.no_cache else 'Enabled'}
    Configuration:                  {platform_parameters.platform_config.value}
    Holoiscan SDK Package:          {platform_parameters.holoscan_sdk_file if platform_parameters.holoscan_sdk_file is not None else "N/A"}
    MONAI Deploy App SDK Package:   {platform_parameters.monai_deploy_sdk_file if platform_parameters.monai_deploy_sdk_file is not None else "N/A"}
    gRPC Health Probe:              {platform_parameters.health_probe if platform_parameters.health_probe is not None else "N/A"}
    SDK Version:                    {self._build_parameters.sdk_version}
    SDK:                            {self._build_parameters.sdk.value}
    Tag:                            {platform_parameters.tag}
    """  # noqa: E501
        )

    def _write_dockerignore(self):
        """Copy .dockerignore file to temporary location."""
        # Write out .dockerignore file
        dockerignore_source_file_path = Path(__file__).parent / "templates" / "dockerignore"
        dockerignore_dest_file_path = os.path.join(self._temp_dir, ".dockerignore")
        shutil.copyfile(dockerignore_source_file_path, dockerignore_dest_file_path)
        return dockerignore_dest_file_path

    def _copy_script(self):
        """Copy HAP/MAP tools.sh script to temporary directory"""
        # Copy the tools script
        tools_script_file_path = Path(__file__).parent / "templates" / "tools.sh"
        tools_script_dest_file_path = os.path.join(self._temp_dir, "tools")
        shutil.copyfile(tools_script_file_path, tools_script_dest_file_path)
        return tools_script_dest_file_path

    def _write_dockerfile(self, platform_parameters: PlatformParameters):
        """Write Dockerfile temporary location"""
        docker_template_string = self._get_template(platform_parameters)
        self._logger.debug(
            f"""
========== Begin Dockerfile ==========
{docker_template_string}
=========== End Dockerfile ===========
"""
        )

        docker_file_path = os.path.join(self._temp_dir, DefaultValues.DOCKER_FILE_NAME)
        with open(docker_file_path, "w") as docker_file:
            docker_file.write(docker_template_string)

        return os.path.abspath(docker_file_path)

    def _copy_application(self):
        """Copy application to temporary location"""
        # Copy application files to temp directory (under 'app' folder)
        target_application_path = Path(os.path.join(self._temp_dir, "app"))
        if os.path.exists(target_application_path):
            shutil.rmtree(target_application_path)

        if not os.path.exists(self._build_parameters.application):
            raise WrongApplicationPathError(
                f'Directory "{self._build_parameters.application}" not found.'
            )

        if os.path.isfile(self._build_parameters.application):
            shutil.copytree(self._build_parameters.application.parent, target_application_path)
        else:
            shutil.copytree(self._build_parameters.application, target_application_path)

        target_config_file_path = Path(os.path.join(self._temp_dir, "app.config"))
        shutil.copyfile(self._build_parameters.config_file, target_config_file_path)

    def _copy_model_files(self):
        """Copy models to temporary location"""
        if self._build_parameters.models:
            target_models_root_path = os.path.join(self._temp_dir, "models")
            os.makedirs(target_models_root_path, exist_ok=True)

            for model in self._build_parameters.models.keys():
                target_model_path = os.path.join(target_models_root_path, model)
                if self._build_parameters.models[model].is_dir():
                    shutil.copytree(self._build_parameters.models[model], target_model_path)
                elif self._build_parameters.models[model].is_file():
                    shutil.copy(self._build_parameters.models[model], target_model_path)

    def _copy_health_probe(self, platform_parameters: PlatformParameters):
        if self._build_parameters.sdk is SdkType.Holoscan:
            target_path = os.path.join(self._temp_dir, "grpc_health_probe")
            shutil.copy2(platform_parameters.health_probe, target_path)

    def _copy_docs(self):
        """Copy user documentations to temporary location"""
        if self._build_parameters.docs is not None:
            target_path = os.path.join(self._temp_dir, "docs")
            shutil.copytree(self._build_parameters.docs, target_path)

    def _get_template(self, platform_parameters: PlatformParameters):
        """Generate Dockerfile using Jinja2 engine"""
        jinja_env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        jinja_template = jinja_env.get_template("Dockerfile.jinja2")
        return jinja_template.render(
            {**self._build_parameters.to_jina, **platform_parameters.to_jina}
        )

    def _copy_supporting_files(self, platform_parameters: PlatformParameters):
        """Abstract base function to copy supporting files"""
        return NotImplemented

    def __init_subclass__(cls):
        if cls._copy_supporting_files is BuilderBase._copy_supporting_files:
            raise NotImplementedError("{cls} has not overwritten method {_copy_supporting_files}!")


class PythonAppBuilder(BuilderBase):
    """A subclass of BuilderBase for Python-based applications.
    Copioes PyPI package and requirement.txt file
    """

    def __init__(
        self,
        build_parameters: PackageBuildParameters,
        temp_dir: str,
    ) -> None:
        BuilderBase.__init__(self, build_parameters, temp_dir)

    def _copy_supporting_files(self, platform_parameters: PlatformParameters):
        self._copy_sdk_file(platform_parameters.holoscan_sdk_file)
        self._copy_sdk_file(platform_parameters.monai_deploy_sdk_file)
        self._copy_pip_requirements()

    def _copy_pip_requirements(self):
        pip_folder = os.path.join(self._temp_dir, "pip")
        os.makedirs(pip_folder, exist_ok=True)
        pip_requirements_path = os.path.join(pip_folder, "requirements.txt")
        with open(pip_requirements_path, "w") as requirements_file:
            # Use local requirements.txt packages if provided, otherwise use sdk provided packages
            if self._build_parameters.requirements_file_path is not None:
                with open(self._build_parameters.requirements_file_path, "r") as lr:
                    for line in lr:
                        requirements_file.write(line)
                requirements_file.writelines("\n")

            if self._build_parameters.pip_packages:
                requirements_file.writelines("\n".join(self._build_parameters.pip_packages))

    def _copy_sdk_file(self, sdk_file: Optional[Path]):
        if sdk_file is not None and sdk_file != Constants.PYPI_INSTALL_SOURCE:
            dest = os.path.join(self._temp_dir, sdk_file.name)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.copyfile(sdk_file, dest)


class CppAppBuilder(BuilderBase):
    """A subclass of BuilderBase for C++ applications.
    Copies Debian.
    """

    def __init__(
        self,
        build_parameters: PackageBuildParameters,
        temp_dir: str,
    ) -> None:
        BuilderBase.__init__(self, build_parameters, temp_dir)

    def _copy_supporting_files(self, platform_parameters: PlatformParameters):
        """Copies the SDK file to the temporary directory"""
        if platform_parameters.holoscan_sdk_file is not None:
            dest = os.path.join(self._temp_dir, platform_parameters.holoscan_sdk_file.name)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.copyfile(
                platform_parameters.holoscan_sdk_file,
                dest,
            )
