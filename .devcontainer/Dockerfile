# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG HOLOSCAN_BASE_IMAGE="holoscan-sdk-dev"
FROM ${HOLOSCAN_BASE_IMAGE} as holoscan-vscode

############################################################
# VSCode devcontainer-compatible Dockerfile
# (https://code.visualstudio.com/docs/remote/create-dev-container)
############################################################

# Options for setup script
ARG INSTALL_ZSH="true"
# don't upgrade packages, some packages are pinned by the base image
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=holoscan-sdk
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY library-scripts/common-debian.sh /tmp/library-scripts/
RUN yes | unminimize 2>&1 \
    && bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# Additional tooling
RUN apt update \
    && apt install --no-install-recommends -y \
        ccache \
        gdb \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_GENERATOR=Ninja
