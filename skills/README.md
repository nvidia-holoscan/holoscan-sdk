<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan AI Coding-Assistant Skills

A growing collection of agent-readable skills that help you install, run, and explore
the Holoscan SDK from inside an AI coding assistant such as
[Claude Code](https://www.anthropic.com/claude-code),
[Codex](https://openai.com/codex/),
[Cursor](https://cursor.com),
or [GitHub Copilot CLI](https://github.com/github/copilot-cli).
Each skill is a self-contained playbook the assistant invokes by name to perform
a specific task on your behalf — for example, picking an install method that fits
your platform, then running the install steps and verifying the result.

The Holoscan team publishes SDK skills in the public
[Holoscan SDK skills directory](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/skills).
The [NVIDIA/skills](https://github.com/NVIDIA/skills) catalog mirrors published
skills from product repositories and makes them discoverable alongside skills
from other NVIDIA teams.

## Installation

Skills are installed with the [`skills` CLI](https://github.com/NVIDIA/skills), which
runs through `npx` and prompts you to choose a skill and an install destination —
no manual `git clone` or folder copying required:

```bash
npx skills add nvidia/skills
```

Follow the interactive prompts to select the skills you want. The CLI installs each
skill into the location appropriate for your AI assistant (project scope, user scope,
or global).

For advanced install options — including non-interactive flags, custom destinations,
and integration with multiple assistants — see the
[skills CLI advanced install guide](https://docs.nvidia.com/skills/advanced-install).

## What these skills do

These skills target **Holoscan SDK 5.x**. They help assess your host and choose
an installation method based on the requested release's available artifacts:

| Skill | Workflow |
| --- | --- |
| `holoscan-setup` | Check the host, release availability, and component needs |
| `holoscan-install-container` | CUDA 13 development container with C++ and Python |
| `holoscan-install-debian` | Native C++ metapackage or individual components |
| `holoscan-install-wheel` | Python metapackage or component wheels in a venv |
| `holoscan-install-source` | Build an explicit 5.x source ref with `./run` |
| `holoscan-install-conda` | Check release-specific Conda availability and select an alternative |

Holoscan SDK 5.0 has no Conda package. Binary options can be unavailable for a
requested release; the skills check the selected release before installing and
offer source when needed. See the
[SDK installation guide](https://docs.nvidia.com/holoscan/sdk-user-guide/setup/sdk-installation),
selecting the documentation version that matches your SDK. The
[checked-in installation guide](../docs/sdk_installation.mdx) provides the
platform matrix for this checkout.

Each published skill's `SKILL.md` describes its workflow and prerequisites.
Check its supported SDK versions and platform requirements before using it with
your SDK version.
Browse the published collection in
[`NVIDIA/skills`](https://github.com/NVIDIA/skills), or run the install command
above to select skills interactively in the CLI.

## Feedback

We're actively iterating on these skills and want to hear from users.
Share feedback, ideas, and skill-specific issues (bad install steps, outdated commands,
unclear prompts) through the
[Holoscan community channels](https://nvidia-holoscan.github.io/) — developer forum,
Discord, and more. Tell us which workflows you'd like a skill for, what worked, and what
didn't.

> **Please do not open issues or pull requests against
> [NVIDIA/skills](https://github.com/NVIDIA/skills) for Holoscan skills.** That
> repository is shared across many NVIDIA teams; reports filed there are unlikely
> to reach the Holoscan team. Route all Holoscan-specific feedback through the
> community channels above.

## Contributions

The skill collection is curated by the Holoscan team while the format and content
stabilize. At this stage, we are **not accepting external pull requests** that add,
remove, or modify skills — either in this directory or in the
[NVIDIA/skills](https://github.com/NVIDIA/skills) repository. This is a narrower
constraint than the SDK's general [Contributing Guide](../CONTRIBUTING.md), which
continues to welcome community contributions to the rest of the SDK.

Once the skill format is stable, we expect to open contributions. Until then, please
route ideas and improvements through the forum link above.

## What's in this directory

Each skill directory contains its `SKILL.md` and supporting artifacts, such as
evaluation cases, a skill card, benchmark results, and a detached signature
(`skill.oms.sig`). Keep each published skill package together and preserve its
signed content when installing or redistributing it.
