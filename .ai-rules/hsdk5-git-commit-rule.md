---
description: Use when creating, rewriting, or reviewing git commit messages.
alwaysApply: false
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HSDK 5 Git Commit Messages

For ordinary development, follow the Conventional Commits v1.0.0
specification:
<https://www.conventionalcommits.org/en/v1.0.0/#specification>.

Before drafting a message, inspect the target branch's recent history and any
release-specific procedure. Preserve an exact subject supplied or reviewed by
the human submitter unless it is inaccurate. If that subject conflicts with
this rule or the target branch convention, surface the conflict instead of
silently rewriting it.

## Commit Boundaries and History

- Group changes by a coherent purpose, not by file type or the order in which
  they were edited. Use a meaningful scope, such as `fix(sonar)` or
  `docs(sonar)`, following the repository's established type names. Keep the
  tests needed to demonstrate a fix with that fix; split independent cleanup,
  documentation, or behavior changes when they can be reviewed separately.
- Inspect the diff and history before proposing an amendment or fixup. Fold a
  correction into an earlier commit when it completes that commit's purpose
  and the requested workflow permits rewriting it. A separate follow-up is
  appropriate for an independent change or to preserve an already reviewed
  history; do not default to either always squashing or always adding commits.
- Check who authored the commit before rewriting it. Do not fold a fix into
  another contributor's commit or replace their authorship without explicit
  direction. A standalone corrective commit can preserve that provenance.
- After folding or amending, revise both the subject and body to describe the
  final patch. Remove superseded claims and re-check the staged diff. Local
  history editing does not authorize a push, force-push, or MR update.

## Format

- Use a conventional subject, such as `type(scope): summary`, for ordinary
  development changes. A reviewed release-branch prefix or release subject takes
  precedence when the applicable procedure requires it.
- Keep the subject concise; prefer 50 characters or fewer when that does not
  sacrifice a clear description of the outcome.
- Separate the subject and body with one blank line.
- Wrap body paragraphs at 72 characters.
- For non-trivial changes, explain the motivation, the resulting behavior, and
  non-obvious scope decisions or trade-offs in short prose paragraphs. Keep
  the rationale and necessary context, not a chronological account of the
  conversation or a file-by-file restatement of the diff.
- Do not add `Context:`, `Summary:`, `Changes:`, or `Verification:` headings by
  default. Use them only when the submitter or an applicable template asks for
  structured sections. Bullets may still help explain distinct consequences;
  they do not require a heading.
- Keep detailed validation commands, run timings, and CI links in the MR or
  validation report unless they are needed to understand the commit. When
  verification is mentioned, distinguish checks actually run from expected
  outcomes; never imply that an unrun check passed. A passing local regression
  check does not establish that the full CI workflow succeeded.
- Describe supported behavior precisely. Do not call a mode "legacy" or claim
  "compatibility" without identifying what remains supported and why. If a
  mode is retained for comparison or testing, say so; describe its removal
  only in the commit that actually removes it.
- Explain behavior-changing trade-offs rather than claiming equivalence or
  performance improvements without evidence. For example, changing a build
  configuration may also change assertions or security definitions even if
  optimization flags look similar. Update the rationale when the chosen
  approach changes.
- For a mechanical cherry-pick, re-rooting, or release projection, identify the
  reviewed source commit or range and any deliberately excluded paths. Do not
  alter the projected payload merely to conform it to current coding style.

## Trailers

- Do not add `Co-authored-by` for an AI agent; remove any that tooling
  injects. Record agent assistance only with `Assisted-by`.
- When an AI agent wrote or substantially assisted the change and its tool
  identity is known, add an `Assisted-by` trailer unless the human submitter
  asks to omit it. Name the tool, optionally followed by `:` and the model
  when the model is known; do not guess a model name. Do not include an email
  address: agent emails are specific to the hosting infrastructure. For
  example:

  ```text
  Assisted-by: Devin
  Assisted-by: Codex:GPT-5
  ```

- When asked to remove assistant attribution from a message, do not replace
  it with equivalent generated-by text. Do not remove other contributors'
  valid provenance trailers as part of that cleanup.
- Add `Co-authored-by` only for a substantive human contributor and only with
  their reviewed name and email. Preserve valid provenance trailers when
  adapting or cherry-picking an existing commit.
- Do not add `Signed-off-by`; only the human submitter can certify DCO.

## Commit Message Files

For multi-paragraph messages, write a meaningful temporary file under `/tmp`
and commit with `git commit -F "$msgfile"`:

```sh
msgfile="$(mktemp /tmp/hsdk-commit-message-holomq-XXXXXX.msg)"
git commit -F "$msgfile"
```

Do not place temporary commit message files inside the repository. Do not remove
the temporary file automatically; leaving it can help users inspect, edit, or
reuse the message.
