---
description: Use when authoring or editing Fern User Guide or API documentation, including Fern navigation/configuration, redirects, assets, diagrams, or documentation tooling.
globs: docs/**/*.mdx,docs/fern/**/*.yml,docs/fern/**/*.yaml,docs/fern/**/*.json,docs/scripts/*.py,docs/Dockerfile,docs/README.md,public/docs/**/*.mdx,public/docs/fern/**/*.yml,public/docs/fern/**/*.yaml,public/docs/fern/**/*.json,public/docs/scripts/*.py,public/docs/Dockerfile,public/docs/README.md
alwaysApply: false
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HSDK 5 Fern Documentation Conventions

## Status and Scope

- The Fern documentation system **is live**: the user guide is authored as committed `.mdx` pages
  under `docs/`. Fern configuration lives in `docs/fern/` (`docs.yml`, `index.yml`,
  `fern.config.json`, `assets/`, `dist/`, `versions/`); the build/preview/publish pipeline is
  `docs/scripts/build_holoscan_docs.py`; `docs/README.md` is the canonical build
  reference.
- Document only the release the guide describes. Never present superseded concepts from an
  earlier major version as current (for 5.x, that means 4.x-era GXF, Conditions, Schedulers,
  Fragments, string-keyed parameters, exceptions from `compute()`); prior-version mappings belong
  in the dedicated migration page. Match examples to the release's actual API surface (see the
  API-stability page) — do not invent examples for language bindings that do not exist.
- Do not apply Fern MDX syntax to C++ header comments; those follow
  `.ai-rules/hsdk5-coding-conventions.md`. Public-header Doxygen is the authoritative source
  for generated API pages; conceptual material belongs in hand-authored MDX.

## Building, previewing, and validating

- Canonical commands (from the SDK source root; do not substitute ad hoc `fern` invocations that skip the
  pipeline's generation/post-processing steps):
  - `python3 docs/scripts/build_holoscan_docs.py` — Docker build + `fern check --local --warnings`.
  - `python3 docs/scripts/build_holoscan_docs.py --preview` — live preview (`fern docs dev`,
    `http://localhost:3000`).
  - `python3 docs/scripts/build_holoscan_docs.py --no-docker` — host mode (needs pinned `fern` CLI).
- **Pass bar: 0 errors.** Exactly two warnings are expected and non-blocking: the auth-gated
  "Missing redirects check skipped" warning and the NVIDIA-green contrast-ratio warning. Treat
  navigation errors, broken links, MDX parse failures, and Library Reference failures as blocking.
  Never claim validation passed without running one of the commands above.
- `FERN_TOKEN` is needed only to publish (preview or production), not for local `fern check`.
  With `FERN_TOKEN` set, the script **auto-enables** C++ API generation unless
  `--skip-library-mdx` is passed.
- If `fern docs dev` dies with `ENOSPC ... file watchers`, raise the host limit
  (`sudo sysctl fs.inotify.max_user_watches=1048576`); it is an environment issue.

## Page template

- **No YAML frontmatter and no `#` H1** — Fern supplies the title from navigation. A page opens
  with 1–3 orientation paragraphs, then `##` sections.
- Concept, deep-dive, tutorial, and performance pages open with a `<Note title="You will learn">`
  bullet list and close with a `## Recap` section. Follow this skeleton for new pages of those
  kinds. Setup/hardware/reference pages are exempt.
- Established page formats: glossary = alphabetical `## A`-style letter headings with bold terms;
  FAQ = thematic `##` groups with `###` question headings; migration page = a
  `| old concept | replacement | What changes for you |` concept-map table followed by
  per-concept `##` sections; example pages open with the source path
  (`examples/<name>/` in the SDK tree) then `## What it builds` / graph shape / contracts.
- The guide threads **one running example** from Getting Started through the deep dives (a
  multi-rate sensor-fusion pipeline backed by a shipped example app); new pages should reuse the
  guide's running example rather than invent unrelated scenarios. Organize content thematically —
  never by internal work-item/task numbering.
- Benchmark/performance numbers require **provenance**: state machine, build, and run date in a
  "Provenance and caveats" section, present figures as evidence of cost structure ("not
  guarantees for your hardware"), and never compare serialized vs saturated results directly.
- Experimental surfaces live under the "Advanced & Experimental" nav section and carry an in-page
  callout using the established wording pattern "... may change in upcoming releases" /
  "signatures may change between early-access releases" (see `plan_tooling.mdx`,
  `determinism_replay.mdx`).

## Navigation, slugs, and redirects

- Site settings, versions, theme, the `libraries:` block, and **all redirects** live in
  `docs/fern/docs.yml`. Page titles, hierarchy, and inclusion live in `docs/fern/index.yml`.
- Every page and section in `index.yml` carries an explicit `slug:`; the slug chain is the
  published route (e.g. `getting-started/core-concepts`). A section landing page is a `section:`
  with its own `path:`. Use `collapsed: true` for secondary sections.
- **Any change to an established published route requires a redirect** in `docs.yml`: renames,
  restructures, and removed pages (point removals at the nearest surviving equivalent). Keep the
  existing table's style — grouped with a one-line comment explaining each restructure.
- `<Anchor id="..." />` exists to preserve legacy published fragments (line-1 anchors on retained
  setup pages, mid-page anchors like `jetpack-rt-kernel`). Never rename or remove one without a
  redirect plan; do not mint new anchors for content that has never been published.

## Links

- Inter-page links use published site-root routes with the full prefix:
  `[Core concepts](/holoscan/sdk-user-guide/getting-started/core-concepts)`. Never introduce
  file-relative `.mdx`, `./`, or `../` inter-page links. File-relative paths are correct only for
  media (`images/...`) and YAML `path:` entries.
- Link the generated C++ API reference by route: the section root is
  `/holoscan/sdk-user-guide/api-reference/holoscan`; deep links use
  `/holoscan/sdk-user-guide/api-reference/cpp/holoscan/classes/...`. Typeset C++ symbols in
  backticks (`holoscan::expected<T, Error>`); link them only when pointing at a reference page.
- Descriptive link text; no raw URLs, no "here". No MyST/Sphinx roles (`{cpp:func}` etc.).

## Components

- Component semantics: `<Warning>` for correctness traps and must-not-do rules;
  `<Note>`/`<Note title="...">` for supplementary context and the "You will learn" opener;
  `<Tip>` for optional guidance. `<Tabs>`/`<Tab title="...">` only for genuinely parallel
  alternatives (platform variants, install flavors) — never sequential steps. `Steps`, `Card`,
  and `Accordion` are not part of the house style; do not introduce them casually.
- MDX comments `{/* ... */}` only — never HTML comments. Self-close void elements. Prefer
  Markdown or a supported Fern component over custom HTML; use Markdown pipe tables for tabular
  content.

## Code and media

- Give every fence a language id, using one canonical identifier per language: `cpp` for the
  primary API language, `bash` for shell, `text` for output/logs, `cmake`, `yaml`. Do not use
  `sh` or other alternates; do not replicate legacy identifiers that may survive in retained
  setup pages.
- Examples compile as shown or clearly mark omitted setup. Avoid line-number-tied explanations.
- Images: self-closing `img` elements with a page-relative `src` and meaningful
  `alt` text — do not use Markdown image syntax. Assets live in `docs/images/`; keep the directory pruned of unused
  assets. Prefer inline SVG diagrams over raster where feasible.

## Diagrams (inline SVG house style)

- Theme-adaptive: `currentColor` for structural strokes/fills/labels (with `opacity` for
  de-emphasis), NVIDIA green `#76B900` only for the highlighted path; JSX camelCase attributes
  (`strokeWidth`, `textAnchor`, `markerEnd`) and object style syntax (`style={{width: "100%"}}`).
- Accessibility: every diagram gets `role="img"`, a `<title>` referenced by `aria-labelledby`,
  **and a `<desc>` referenced by `aria-describedby`** describing the structure. Avoid faint,
  small annotation text (for example 11px at low opacity); keep annotations readable in both
  themes. If a diagram you are editing lacks these, add them as part of the edit.

## Generated C++ Library Reference and the dist bundle

- The `libraries:` block in `docs.yml` and the "API reference" section in `index.yml` toggle
  **together**: a nav folder pointing at a missing `docs/fern/generated/` tree makes `fern docs dev`
  render a blank page. `generated/` is gitignored — never hand-edit it; fix headers, config, or
  `fix_generated_library_mdx.py`. Ensure `libraries.holoscan-cpp.input.git`/`subpath` matches the
  version being published before regenerating.
- `docs/fern/dist/` is a committed generated theme bundle: never hand-edit or let lint
  auto-fixers touch it; it is excluded from the copyright, codespell, and clang-format hooks.
  Beware the generic `dist/` gitignore rule silently swallowing new files there — after adding
  files under `docs/fern/dist/`, confirm they are tracked.
- Keep the Fern CLI pin in lockstep in all three places: `fern.config.json` (`version`), the docs
  `Dockerfile` (`fern-api@<version>`), and `FERN_API_VERSION` in `build_holoscan_docs.py`. When
  intentionally upgrading Fern, bump all three together.
