# Holoscan AI Coding-Assistant Skills

A growing collection of agent-readable skills that help you install, run, and explore
the Holoscan SDK from inside an AI coding assistant such as
[Claude Code](https://www.anthropic.com/claude-code),
[Cursor](https://cursor.com),
or [GitHub Copilot CLI](https://github.com/github/copilot-cli).
Each skill is a self-contained playbook the assistant invokes by name to perform
a specific task on your behalf — for example, picking an install method that fits
your platform, then running the install steps and verifying the result.

The canonical home for the skills themselves is the
[NVIDIA/skills](https://github.com/NVIDIA/skills) repository, where they are
organized alongside skills from other NVIDIA teams. This directory mirrors a curated
subset for local visibility.

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

Available skills cover a range of Holoscan SDK tasks. Today they focus on
installing the SDK — walking you through any supported install path (container,
Debian package, Python wheel, Conda, or from source) and helping you pick the
one that fits your platform.

The canonical list of skills — including each skill's full description,
prerequisites, and versioning — lives in
[`NVIDIA/skills`](https://github.com/NVIDIA/skills). Run the install command
above to browse them interactively in the CLI.

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

Today this directory contains only this README. The skill content lives in
[NVIDIA/skills](https://github.com/NVIDIA/skills) and is installed via the `npx`
command above. As skills stabilize, selected ones may be mirrored here for offline
browsing.
