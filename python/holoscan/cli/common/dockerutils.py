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

import json
import logging
import os
import posixpath
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from python_on_whales import docker

from ..common.utils import run_cmd_output
from .constants import DefaultValues, EnvironmentVariables
from .enum_types import PlatformConfiguration, SdkType
from .exceptions import InvalidManifestError, RunContainerError
from .utils import get_requested_gpus

logger = logging.getLogger("common")


def parse_docker_image_name_and_tag(image_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a given Docker image name and tag.

    Args:
        image_name (str): Docker image name and optionally a tag

    Returns:
        Tuple[Optional[str], Optional[str]]: a tuple with first item as the name of the image
        and tag as the second item
    """
    match = re.search(
        r"^(?P<name>([\w.\-_]+((:\d+|)(?=/[a-z0-9._-]+/[a-z0-9._-]+))|)(/?)([a-z0-9.\-_/]+(/[a-z0-9.\-_]+|)))(:(?P<tag>[\w.\-_]{1,127})|)$",
        image_name,
    )

    if match is None or match.group("name") is None:
        return None, None

    name = match.group("name")
    tag = match.group("tag") if match.group("tag") else None

    return (name, tag)


def create_or_use_network(network: Optional[str], image_name: Optional[str]) -> str:
    """Create a Docker network by the given name if not already exists.

    Args:
        network (Optional[str]): name of the network to create
        image_name (Optional[str]): name of the image used to generate a network name from

    Raises:
        RunContainerError: when unable to retrieve the specified network or failed to create one.

    Returns:
        str: network name
    """
    if network is None and image_name is not None:
        network = image_name.split(":")[0]
        network += "-network"

    assert network is not None

    try:
        networks = docker.network.list(filters={"name": f"^{network}$"})
        if len(networks) > 0:
            return networks[0].name
    except Exception as ex:
        raise RunContainerError(f"error retrieving network information: {ex}") from ex

    try:
        return docker.network.create(network, driver="bridge").name
    except Exception as ex:
        raise RunContainerError(f"error creating Docker network: {ex}") from ex


def image_exists(image_name: str) -> bool:
    """Checks if the Docker image exists.

    Args:
        image_name (str): name of the Docker image

    Returns:
        bool: whether the image exists or not.
    """
    if image_name is None:
        return False
    try:
        if not docker.image.exists(image_name):
            logger.info(f"Attempting to pull image {image_name}..")
            docker.image.pull(image_name)
        return docker.image.exists(image_name)
    except Exception as e:
        logger.error(str(e))
        return False


def docker_export_tarball(file: str, tag: str):
    """Exports the docker image to a file

    Args:
        file (str): name of the exported file
        tag (str): source Docker image tag
    """
    docker.image.save(tag, file)


def create_and_get_builder(builder_name: str):
    """Creates a Docker BuildX builder

    Args:
        builder_name (str): name of the builder to create

    Returns:
        _type_: name of the builder created
    """
    builders = docker.buildx.list()
    for builder in builders:
        if builder.name == builder_name:
            logger.info(f"Using existing Docker BuildKit builder `{builder_name}`")
            return builder_name

    logger.info(f"Creating Docker BuildKit builder `{builder_name}` using `docker-container`")
    builder = docker.buildx.create(
        name=builder_name, driver="docker-container", driver_options={"network": "host"}
    )
    return builder.name


def build_docker_image(**kwargs):
    """Builds a Docker image"""
    _ = docker.buildx.build(**kwargs)


def docker_run(
    name: str,
    image_name: str,
    input_path: Optional[Path],
    output_path: Optional[Path],
    app_info: dict,
    pkg_info: dict,
    quiet: bool,
    commands: List[str],
    network: str,
    network_interface: Optional[str],
    use_all_nics: bool,
    config: Optional[Path],
    render: bool,
    user: str,
    terminal: bool,
    devices: List[str],
    platform_config: str,
    shared_memory_size: str = "1GB",
    is_root: bool = False,
):
    """Creates and runs a Docker container

    `HOLOSCAN_HOSTING_SERVICE` environment variable is used for hiding the help message
    inside the tools.sh when the users run the container using holoscan run.

    Args:
        image_name (str): Docker image name
        input_path (Optional[Path]): input data path
        output_path (Optional[Path]): output data path
        app_info (dict): app manifest
        pkg_info (dict): package manifest
        quiet (bool): prints only stderr when True, otherwise, prints all logs
        commands (List[str]): list of arguments to provide to the container
        network (str): Docker network to associate the container with
        network_interface (Optional[str]): Name of the network interface for setting
            UCX_NET_DEVICES
        use_all_nics (bool): Sets UCX_CM_USE_ALL_DEVICES to 'y' if True
        config (Optional[Path]): optional configuration file for overriding the embedded one
        render (bool): whether or not to enable graphic rendering
        user (str): UID and GID to associate with the container
        terminal (bool): whether or not to enter bash terminal
        devices (List[str]): list of devices to be mapped into the container
        platformConfig (str): platform configuration value used when packaging the application,
        shared_memory_size (str): size of /dev/shm,
        is_root (bool): whether the user is root (UID = 0) or not
    """
    volumes = []
    environment_variables = {
        "NVIDIA_DRIVER_CAPABILITIES": "graphics,video,compute,utility,display",
        "HOLOSCAN_HOSTING_SERVICE": "HOLOSCAN_RUN",
        "UCX_CM_USE_ALL_DEVICES": "y" if use_all_nics else "n",
    }

    if network_interface is not None:
        environment_variables["UCX_NET_DEVICES"] = network_interface

    if logger.root.level == logging.DEBUG:
        environment_variables["UCX_LOG_LEVEL"] = "DEBUG"
        environment_variables["VK_LOADER_DEBUG"] = "all"

    if render:
        volumes.append(("/tmp/.X11-unix", "/tmp/.X11-unix"))
        display = os.environ.get("DISPLAY", None)
        if display is not None:
            environment_variables["DISPLAY"] = display

    gpu = None

    # If the image was built for iGPU but the system is configured for dGPU, attempt
    # targeting the system's iGPU using the CDI spec
    if platform_config == PlatformConfiguration.iGPU.value and not _host_is_native_igpu():
        environment_variables["NVIDIA_VISIBLE_DEVICES"] = "nvidia.com/igpu=0"
        logger.info(
            "Attempting to run an image for iGPU (integrated GPU) on a system configured "
            "with a dGPU (discrete GPU). If this is correct (ex: IGX Orin developer kit), "
            "make sure to enable iGPU on dGPU support as described in your developer kit "
            "user guide. If not, either rebuild the image for dGPU or run this image on a "
            "system configured for iGPU only (ex: Jetson AGX, Nano...)."
        )
    else:
        requested_gpus = get_requested_gpus(pkg_info)
        if requested_gpus > 0:
            gpu = "all"

    if "path" in app_info["input"]:
        mapped_input = Path(app_info["input"]["path"]).as_posix()
    else:
        mapped_input = DefaultValues.INPUT_DIR

    if not posixpath.isabs(mapped_input):
        mapped_input = posixpath.join(app_info["workingDirectory"], mapped_input)
    if input_path is not None:
        volumes.append((str(input_path), mapped_input))

    if "path" in app_info["output"]:
        mapped_output = Path(app_info["output"]["path"]).as_posix()
    else:
        mapped_output = DefaultValues.INPUT_DIR

    if not posixpath.isabs(mapped_output):
        mapped_output = posixpath.join(app_info["workingDirectory"], mapped_output)
    if output_path is not None:
        volumes.append((str(output_path), mapped_output))

    for env in app_info["environment"]:
        if env == EnvironmentVariables.HOLOSCAN_INPUT_PATH:
            environment_variables[env] = mapped_input
        elif env == EnvironmentVariables.HOLOSCAN_OUTPUT_PATH:
            environment_variables[env] = mapped_output
        else:
            environment_variables[env] = app_info["environment"][env]

        # always pass path to config file for Holoscan apps
        if (
            "sdk" in app_info
            and app_info["sdk"] == SdkType.Holoscan.value
            and env == EnvironmentVariables.HOLOSCAN_CONFIG_PATH
        ):
            commands.append("--config")
            commands.append(environment_variables[env])

    if config is not None:
        if EnvironmentVariables.HOLOSCAN_CONFIG_PATH not in app_info["environment"]:
            raise InvalidManifestError(
                "The application manifest does not contain a required "
                f"environment variable: '{EnvironmentVariables.HOLOSCAN_CONFIG_PATH}'"
            )
        volumes.append(
            (
                str(config),
                app_info["environment"][EnvironmentVariables.HOLOSCAN_CONFIG_PATH],
            )
        )
        logger.info(f"Using user provided configuration file: {config}")

    logger.debug(
        f"Environment variables: {json.dumps(environment_variables, indent=4, sort_keys=True)}"
    )
    logger.debug(f"Volumes: {json.dumps(volumes, indent=4, sort_keys=True)}")
    logger.debug(f"Shared memory size: {shared_memory_size}")

    ipc_mode = "host" if shared_memory_size is None else None
    ulimits = [
        "memlock=-1",
        "stack=67108864",
    ]
    additional_devices, group_adds = _additional_devices_to_mount(is_root)
    devices.extend(additional_devices)

    video_group = run_cmd_output('/usr/bin/cat /etc/group | grep "video" | cut -d: -f3').strip()
    if not is_root and video_group not in group_adds:
        group_adds.append(video_group)

    if terminal:
        _enter_terminal(
            name,
            image_name,
            app_info,
            network,
            user,
            volumes,
            environment_variables,
            gpu,
            shared_memory_size,
            ipc_mode,
            ulimits,
            devices,
            group_adds,
        )
    else:
        _start_container(
            name,
            image_name,
            app_info,
            quiet,
            commands,
            network,
            user,
            volumes,
            environment_variables,
            gpu,
            shared_memory_size,
            ipc_mode,
            ulimits,
            devices,
            group_adds,
        )


def _start_container(
    name,
    image_name,
    app_info,
    quiet,
    commands,
    network,
    user,
    volumes,
    environment_variables,
    gpu,
    shared_memory_size,
    ipc_mode,
    ulimits,
    devices,
    group_adds,
):
    container = docker.container.create(
        image_name,
        command=commands,
        envs=environment_variables,
        gpus=gpu,
        hostname=name,
        name=name,
        networks=[network],
        remove=True,
        shm_size=shared_memory_size,
        user=user,
        volumes=volumes,
        workdir=app_info["workingDirectory"],
        ipc=ipc_mode,
        cap_add=["CAP_SYS_PTRACE"],
        ulimit=ulimits,
        devices=devices,
        groups_add=group_adds,
        runtime="nvidia",
    )
    container_name = container.name
    container_id = container.id[:12]

    ulimit_str = ", ".join(
        f"{ulimit.name}={ulimit.soft}:{ulimit.hard}" for ulimit in container.host_config.ulimits
    )
    logger.info(
        f"Launching container ({container_id}) using image '{image_name}'..."
        f"\n    container name:      {container_name}"
        f"\n    host name:           {container.config.hostname}"
        f"\n    network:             {network}"
        f"\n    user:                {container.config.user}"
        f"\n    ulimits:             {ulimit_str}"
        f"\n    cap_add:             {', '.join(container.host_config.cap_add)}"
        f"\n    ipc mode:            {container.host_config.ipc_mode}"
        f"\n    shared memory size:  {container.host_config.shm_size}"
        f"\n    devices:             {', '.join(devices)}"
        f"\n    group_add:           {', '.join(group_adds)}"
    )
    logs = container.start(
        attach=True,
        stream=True,
    )

    for log in logs:
        if log[0] == "stdout":
            if not quiet:
                print(log[1].decode("utf-8"))
        elif log[0] == "stderr":
            print(str(log[1].decode("utf-8")))

    logger.info(f"Container '{container_name}'({container_id}) exited.")


def _enter_terminal(
    name,
    image_name,
    app_info,
    network,
    user,
    volumes,
    environment_variables,
    gpu,
    shared_memory_size,
    ipc_mode,
    ulimits,
    devices,
    group_adds,
):
    print("\n\nEntering terminal...")
    print(
        "\n".join(
            f"\t{k:25s}\t{v}"
            for k, v in sorted(environment_variables.items(), key=lambda t: str(t[0]))
        )
    )
    print("\n\n")
    docker.container.run(
        image_name,
        detach=False,
        entrypoint="/bin/bash",
        envs=environment_variables,
        gpus=gpu,
        hostname=name,
        interactive=True,
        name=name,
        networks=[network],
        remove=True,
        shm_size=shared_memory_size,
        tty=True,
        user=user,
        volumes=volumes,
        workdir=app_info["workingDirectory"],
        ipc=ipc_mode,
        cap_add=["CAP_SYS_PTRACE"],
        ulimit=ulimits,
        devices=devices,
        groups_add=group_adds,
        runtime="nvidia",
    )
    logger.info("Container exited.")


def _additional_devices_to_mount(is_root: bool):
    """Mounts additional devices"""
    devices = []
    group_adds = []

    # On iGPU, the /dev/dri/* devices (mounted by the NV container runtime) permissions require root
    # privilege or to be part of the `video` and `render` groups. The ID for these group names might
    # differ on the host system and in the container, so we need to pass the group ID instead of the
    # group name when running docker.
    if (
        os.path.exists("/sys/devices/platform/gpu.0/load")
        and os.path.exists("/usr/bin/tegrastats")
        and not is_root
    ):
        group = run_cmd_output('/usr/bin/cat /etc/group | grep "video" | cut -d: -f3').strip()
        group_adds.append(group)
        group = run_cmd_output('/usr/bin/cat /etc/group | grep "render" | cut -d: -f3').strip()
        group_adds.append(group)
    return (devices, group_adds)


def _host_is_native_igpu() -> bool:
    proc = subprocess.run(
        ["nvidia-smi --query-gpu name --format=csv,noheader | grep nvgpu -q"], shell=True
    )
    result = proc.returncode
    return result == 0
