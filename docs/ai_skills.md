<!-- markdownlint-disable MD041 -->
(ai-skills)=

# AI Coding-Assistant Skills

The Holoscan team publishes a collection of **agent-readable skills** that help you
install, run, and explore the Holoscan SDK from inside an AI coding assistant such
as [Claude Code](https://www.anthropic.com/claude-code),
[Cursor](https://cursor.com),
or [GitHub Copilot CLI](https://github.com/github/copilot-cli).
The skills are organized in the public
[NVIDIA/skills](https://github.com/NVIDIA/skills) repository, alongside skills from
other NVIDIA teams.

## What are skills?

A skill is a self-contained playbook — a description of *what it does*, *when to use
it*, and *the exact steps to follow* — that an AI assistant invokes by name. When
you ask the assistant to "install Holoscan in a container," the assistant matches
your request to the `holoscan-install-container` skill and follows its steps:
pulling the right NGC image for your platform, verifying GPU passthrough, and
running the bundled example tests on your behalf. Skills give the assistant a
reliable, reviewable procedure to follow instead of improvising.

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

:::{warning}
Please do not open issues or pull requests against
[NVIDIA/skills](https://github.com/NVIDIA/skills) for Holoscan skills. That repository
is shared across many NVIDIA teams; reports filed there are unlikely to reach the
Holoscan team. Route all Holoscan-specific feedback through the community channels
above.
:::

## Contributions

The skill collection is curated by the Holoscan team while the format and content
stabilize. At this stage, the Holoscan team is **not accepting external pull
requests** that add, remove, or modify skills — either in the SDK's local
[`skills/`](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/skills)
mirror or in the canonical [NVIDIA/skills](https://github.com/NVIDIA/skills)
repository. This is a narrower constraint than the SDK's general
[Contributing Guide](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/CONTRIBUTING.md),
which continues to welcome community contributions to the rest of the SDK.

Once the skill format is stable, we expect to open contributions. Until then, please
route ideas and improvements through the forum link above.
