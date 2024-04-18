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
#  curl - Used to download Doxygen
#  python3-pip - Needed for pip installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-pip \
    gnupg \
    graphviz \
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
COPY --from=docs-base /usr/bin/dot /usr/bin/dot
COPY --from=docs-base /usr/local/bin/doxygen /usr/local/bin/doxygen
COPY --from=docs-base /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=docs-base /usr/local/bin/sphinx-build /usr/local/bin/sphinx-build
COPY --from=docs-base /usr/local/bin/sphinx-autobuild /usr/local/bin/sphinx-autobuild
COPY --from=docs-base /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=docs-base /usr/share/fonts  /usr/share/fonts
COPY --from=docs-base /lib/x86_64-linux-gnu/  /lib/x86_64-linux-gnu/

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
