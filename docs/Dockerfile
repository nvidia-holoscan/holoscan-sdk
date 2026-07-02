# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Documentation build image for Fern user guide.

ARG NODE_VERSION=22-bookworm-slim
FROM node:${NODE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    git-lfs \
    python3 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN npm install -g fern-api@5.44.1 pnpm

WORKDIR /workspace
