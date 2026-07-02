<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Holoscan SDK user guide (Fern)

User guide pages are committed as **MDX** (`.mdx`) in this directory, alongside images.
Fern configuration lives under `fern/` (`fern.config.json`, `docs.yml`, `index.yml`,
`assets/`, `dist/`). C++ API pages are generated at build time under `fern/generated/`
(gitignored).

## Edit documentation

1. Change `*.mdx` files here (and images alongside them).
2. Update sidebar entries in `fern/index.yml` when adding or renaming pages.
3. Ensure doc images are materialized when stored in Git LFS:

   ```bash
   git lfs install
   git lfs pull --include='public/docs/**'
   ```

4. Build and validate docs (default: Docker, then `fern check --local`):

   ```bash
   python3 public/docs/scripts/build_holoscan_docs.py
   ```

5. Preview locally with a dev server:

   ```bash
   python3 public/docs/scripts/build_holoscan_docs.py --preview
   ```

Use `--no-docker` to run on the host instead of the docs container.
C++ API pages are generated automatically when `fern login` credentials or `FERN_TOKEN`
are available; pass `--skip-library-mdx` to skip them. Pass `--skip-fern-check` to run
the pipeline only (no Fern validation).

CI publishes remote Fern previews instead of running `fern check --local`:

- **Merge pipeline** (MR with `docs` scope label): `fern generate --docs --preview --id <source-branch> --force`, then posts the preview link on the MR
- **Push pipeline** (merge to main/release): deletes the merged branch preview, then publishes `--id <target-branch>`

Set `FERN_TOKEN` in Jenkins (Secret text credential ID `FERN_TOKEN`) so C++ API pages are generated and previews can be published.

## Fern CLI

Install the pinned CLI version (matches `fern/fern.config.json`):

```bash
npm install -g fern-api@5.44.1
```

Run Fern commands from **`public/docs/fern`** (where `fern.config.json` lives):

```bash
cd public/docs/fern
fern check --local --warnings
fern docs dev
```

Or use the build script from the repo root (Docker or `--no-docker`).

## Publish preview

Remote preview (requires login):

```bash
cd public/docs/fern
fern generate --docs --preview
```
