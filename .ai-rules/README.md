<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared agent rules

Keep coding, documentation, and commit guidance here as Markdown with Cursor
YAML frontmatter (`description`, `globs`, and `alwaysApply`). Edit this source
rather than copying policy into each tool's configuration. Paths
and commands in these files are relative to the SDK source root containing
`AGENTS.md`, `run`, and `CMakeLists.txt`.

`.ai-rules` is a repository convention, not a directory these tools discover
automatically. Native entrypoints provide discovery and task selection:

| Entrypoint | Role |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | Repository instructions and directions to read each relevant rule. |
| [CLAUDE.md](../CLAUDE.md) | Imports `AGENTS.md` using Claude Code's supported syntax. |
| [.cursor/rules/](../.cursor/rules/) | `.mdc` symlinks to the shared files, including their activation metadata. |

Do not import every rule body at startup. Follow the task-specific directions
in `AGENTS.md`; Cursor uses the shared frontmatter for activation. These fields
do not give other agents Cursor's glob-based activation behavior.

Cursor symlinks point to `../../.ai-rules/<name>.md`. They resolve within this
SDK checkout, so no rule-generation step is needed.

## Maintaining rules

- Edit the canonical `.ai-rules/<name>.md`, not the symlink.
- Give each shared rule a `.cursor/rules/<name>.mdc` relative symlink to
  `../../.ai-rules/<name>.md`.
- Update `AGENTS.md` task routing when adding a rule for the other agents.
- Keep activation globs aligned with the rule's scope. Review scope changes
  explicitly when changing this metadata.
- Keep Git symlink support enabled. Clones with `core.symlinks=false` receive
  link-text files instead of working Cursor rules; this is a known limitation
  for Windows checkouts without symlink support. `AGENTS.md` and the regular
  `CLAUDE.md` import remain available independently.

## Supported mechanisms

Checked against vendor documentation on 2026-09-21:

- [Cursor project rules](https://cursor.com/docs/rules) support `.mdc` files,
  YAML frontmatter, and `AGENTS.md`. Plain `.md` filenames in `.cursor/rules/`
  are not project rules, so the symlinks use `.mdc` even though their targets
  use `.md`.
- [Dave Taubler's shared-rules experiment](https://itnext.io/a-tale-of-two-ai-codegen-tools-that-wouldnt-cooperation-3fb1bb550df4)
  reports Cursor following these `.mdc` symlinks and parsing the target's
  frontmatter. This is reported working behavior, not an end-to-end test of
  our installed Cursor version.
- [Claude Code shared instructions](https://code.claude.com/docs/en/memory#share-one-file-with-other-coding-tools)
  support importing `AGENTS.md` from `CLAUDE.md`. Regular files avoid relying on
  Git symlink support. Recent versions also discover `AGENTS.md` directly,
  depending on version and session settings; the import supports sessions
  where native discovery is unavailable.
- [Codex repository instructions](https://developers.openai.com/codex/guides/agents-md)
  use `AGENTS.md`; explicit read directions route it to the shared rules.
  No user-global fallback-name configuration is required.

This does not imply identical automatic activation across clients. After
changing rules or links, inspect Cursor's active rules and Claude Code's `/context`,
and confirm that each agent reads the relevant shared file before working.
File, projection, and lint checks alone do not validate running-client behavior.
