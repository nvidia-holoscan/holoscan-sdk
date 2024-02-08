# syntax=docker/dockerfile:1.2

# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG BASE_IMAGE=holoscan-sdk-build

################################################################
# Base docs image that installs docs dependencies
################################################################
FROM ubuntu:22.04 AS docs-base

ARG DEBIAN_FRONTEND=noninteractive

# Install apt & pip build dependencies
# Deadsnakes repo is added then package index files are updated
#  software-properties-common - Needed to use `add-apt-repository`
#  build-essential - Adds GNU/g++ compiler collection
#  curl - Used to download Doxygen and Node.js
#  python3-pip - Needed for pip installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-pip \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install up to date doxygen for better C++ parsing with a few cases like
# forward declaration of enums.
RUN cd /tmp/ \
    && curl -L -o doxygen-1.9.5.linux.bin.tar.gz 'https://sourceforge.net/projects/doxygen/files/rel-1.9.5/doxygen-1.9.5.linux.bin.tar.gz/download#' \
    && tar xvf doxygen-1.9.5.linux.bin.tar.gz \
    && cd doxygen-1.9.5 \
    && make install \
    && cd .. \
    && rm -rf doxygen*

# Install Node.js 20 using DEB packages
# https://github.com/nodesource/distributions#debian-and-ubuntu-based-distributions
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
    | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Mermaid CLI
RUN npm install --production -g @mermaid-js/mermaid-cli

# Install Python dependencies
# Pinned additional sphinxcontrib-* extensions to specific versions to avoid following error:
# "The sphinxcontrib.* extension used by this project needs at least Sphinx v5.0;"
RUN python3 -m pip install --no-cache-dir \
    exhale==0.3.5 \
    Sphinx==4.5.0 \
    sphinxcontrib-applehelp==1.0.2 \
    sphinxcontrib-devhelp==1.0.2 \
    sphinxcontrib-htmlhelp==2.0.0 \
    sphinxcontrib-serializinghtml==1.1.5 \
    sphinxcontrib-qthelp==1.0.3 \
    sphinx_rtd_theme==1.0.0 \
    sphinx-autobuild==2021.3.14 \
    myst-parser==0.17.2 \
    numpydoc==1.5.0 \
    sphinxcontrib-mermaid==0.7.1 \
    sphinx_design==0.3.0

################################################################
# HTML docs image that copies all the doc-specific packages
# over to holoscan-sdk-build
################################################################
FROM $BASE_IMAGE as docs-html

# Copy over installed denpendencies from docs-base
COPY --from=docs-base /usr/bin/curl /usr/bin/curl
COPY --from=docs-base /usr/bin/node /usr/bin/node
COPY --from=docs-base /usr/bin/npx /usr/bin/npx
COPY --from=docs-base /usr/bin/npm /usr/bin/npm
COPY --from=docs-base /usr/bin/mmdc /usr/bin/mmdc
COPY --from=docs-base /usr/bin/corepack /usr/bin/corepack
COPY --from=docs-base /usr/local/bin/doxygen /usr/local/bin/doxygen
COPY --from=docs-base /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=docs-base /usr/local/bin/sphinx-build /usr/local/bin/sphinx-build
COPY --from=docs-base /usr/local/bin/sphinx-autobuild /usr/local/bin/sphinx-autobuild
COPY --from=docs-base /usr/lib/node_modules /usr/lib/node_modules
COPY --from=docs-base /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=docs-base /usr/include/node /usr/include/node
COPY --from=docs-base /usr/share/doc/nodejs /usr/share/doc/nodejs

# Below logic needed due to copy cmds being used instead of local installs
# ------------------------------------------------------------------------
# Update npm executable to invoke the npm module's CLI script and pass the
# current Node.js process as an argument
RUN echo '#!/usr/bin/env node' > /usr/bin/npm \
    && echo "require('/usr/lib/node_modules/npm/lib/cli.js')(process)" >> /usr/bin/npm && \
    # Update mmdc (mermaid-cli) executable to set the current process title to 'mmdc',
    # invoke the mermaid module's CLI function, and print any error if one is encountered
    echo '#!/usr/bin/env node' > /usr/bin/mmdc \
    && echo "process.title = 'mmdc'; \
    import('/usr/lib/node_modules/@mermaid-js/mermaid-cli/src/index.js') \
    .then(({ cli, error }) => cli().catch((exception) => error(exception instanceof Error ? exception.stack : exception))) \
    .catch((err) => { \
    console.error(err); \
    process.exit(1); \
    });" >> /usr/bin/mmdc

#################################################################
# PDF docs image that installs pdf/latex dependencies to the base
# docs image
#################################################################
FROM docs-base as docs-pdf

RUN apt-get update \
    && apt-get install -y \
    latexmk \
    texlive-base \
    texlive-latex-extra \
    # For pdfcrop
    texlive-extra-utils \
    # For Chromium (used by Puppeteer)
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Add configuration for for puppeteer
RUN echo '{"args": ["--no-sandbox"]}' >> /usr/bin/puppeteer-config.json
