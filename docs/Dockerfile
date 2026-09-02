# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Documentation build image for Fern user guide.

ARG DOCKER_CLI_VERSION=29.2.1
ARG NODE_VERSION=22-bookworm-slim

FROM docker:${DOCKER_CLI_VERSION}-cli AS docker-cli
FROM node:${NODE_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g fern-api@5.89.3 pnpm@9.15.9

# Fern's local library parser runs as a sibling container through the Docker
# daemon supplied by the host or Jenkins dind environment.
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /workspace
