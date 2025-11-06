# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG HOLOSCAN_BUILD_IMAGE
FROM ${HOLOSCAN_BUILD_IMAGE} AS holoscan-vscode

############################################################
# VSCode devcontainer-compatible Dockerfile
# (https://code.visualstudio.com/docs/remote/create-dev-container)
############################################################

# Options for setup script
ARG INSTALL_ZSH="true"
# don't upgrade packages, some packages are pinned by the base image
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=holoscan
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# The 24.04 image added the non-root user 'ubuntu' with uid 1000 which needs to be removed first
# else creating our user will fail
# see https://bugs.launchpad.net/cloud-images/+bug/2005129
RUN if id "ubuntu" >/dev/null 2>&1; then touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu; fi
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY library-scripts/common-debian.sh /tmp/library-scripts/
# Skip running 'yes | unminimize 2>&1' here to avoid upgrading/reinstalling packages,
# which can result in version conflicts (e.g., cmake/cmake-data being updated from 3.x to 4.x).
RUN bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# Additional tooling
RUN apt update \
    && apt install --no-install-recommends -y \
        ccache \
        gdb \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*
ENV CMAKE_GENERATOR=Ninja

# Set the default user.
# (See https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user)
USER $USERNAME
