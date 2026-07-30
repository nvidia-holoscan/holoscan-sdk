<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan SDK user guide (Fern)

User guide pages are committed as **MDX** (`.mdx`) in this directory, alongside images.
Fern configuration lives under `fern/` (`fern.config.json`, `docs.yml`, `index.yml`,
`assets/`, `dist/`). C++ API pages are generated at build time under `fern/generated/`
(gitignored).

Paths in this README use the public repository layout (`docs/`). The
`build_holoscan_docs.py` helper script also detects the internal source layout
(`public/docs/`).

## Edit documentation

1. Change `*.mdx` files here (and images alongside them).
2. Update sidebar entries in `fern/index.yml` when adding or renaming pages.
   Renaming a page changes its published URL unless you pin the old one with
   `slug:`; add a `redirects` entry in `fern/docs.yml` when a URL must change.
3. Write inter-page links as published site paths
   (`/holoscan/sdk-user-guide/setup/sdk-installation#python-wheel`) and same-page links as
   `#anchor`. Fern publishes any other relative path unchanged, which resolves against the
   current page's directory and silently lands on the docs homepage. Page-file paths
   (`sdk_installation.mdx`) are rewritten by the current CLI but are documented as
   unsupported, so the link check rejects them too. Run
   `python3 docs/scripts/check_doc_links.py` to verify links, anchors, and images;
   `fern check` does not resolve link targets.
4. Build and validate docs (default: Docker, then `check_doc_links.py` and `fern check --local`):

   ```bash
   python3 docs/scripts/build_holoscan_docs.py
   ```

5. Preview locally with a dev server:

   ```bash
   python3 docs/scripts/build_holoscan_docs.py --preview
   ```

Use `--no-docker` to run on the host instead of the docs container.
C++ API pages are generated automatically when `fern login` credentials or `FERN_TOKEN`
are available; pass `--skip-library-mdx` to reuse an existing local generated tree.
The tree is still post-processed and included in validation, preview, or publication.
Pass `--skip-fern-check` to run the pipeline only (no Fern validation), or
`--skip-link-check` to bypass internal link validation.

In Docker mode the helper reuses your host Fern authentication: if `FERN_TOKEN` is set
it is passed into the container, otherwise only the host `~/.fern/token` and `~/.fern/id`
files (from `fern login`) are bind-mounted so the containerized Fern CLI does not prompt
for an interactive login. Run `fern login` once on the host beforehand.

CI publishes remote Fern previews instead of running `fern check --local`:

- **Merge pipeline** (MR with the `scope::docs` label): runs
  `build_holoscan_docs.py --publish-preview --preview-id <source-branch>
  --force`. The helper script writes the resolved preview URL to the ignored
  `fern/.fern-preview-url` handoff file, which CI reads to post the MR comment.
- **Push pipeline** (merge to `main` or a release target branch): deletes the
  source-branch preview, then runs `build_holoscan_docs.py` with
  `--preview-id <target-branch>`.

The `build_holoscan_docs.py` helper script builds the Docker image containing the
pinned Fern CLI, runs the pipeline in that container, generates and post-processes
the C++ API MDX, and asks Fern to publish the preview. Set `FERN_TOKEN` in Jenkins
(Secret text credential ID `FERN_TOKEN`) so generation and preview publication can
authenticate.

The preview ID names the Fern deployment; it does not select a Git revision for the
C++ library input. The current `fern/docs.yml` tells `fern docs md generate` to have
Fern parse `https://github.com/nvidia-holoscan/holoscan-sdk`, under
`include/holoscan`, on Fern's servers. In MR CI, the helper script removes any
previous generated tree and invokes generation. The Fern CLI writes the raw MDX
returned by the service into the MR checkout's local
`fern/generated/api-reference/cpp` directory; the helper script then runs
`docs/scripts/fix_generated_library_mdx.py` on that local tree and includes
the post-processed files in the preview publication. The post-processor output is
therefore not ignored. However, C++ header changes that exist only in the MR checkout
are not reflected in the generated API pages because Fern reads the headers from the
configured public repository. This is a limitation of the current implementation.
See Fern's
[library reference documentation](https://buildwithfern.com/learn/docs/api-references/library-reference).

## Pipeline phases and terminology

`build_holoscan_docs.py` selects the execution environment first:

```text
build_holoscan_docs.py (default)
  -> build the repository docs image -> run the pipeline with its pinned tools

build_holoscan_docs.py --no-docker
  -> run the same pipeline with tools installed directly on the host
```

The docs Docker image is expected to supply the pinned Fern CLI and its supporting
tools. In Docker mode, the helper script builds that image from
`docs/Dockerfile` before each run; Docker may reuse its build cache.
`--container-name` controls only the transient runtime container name (`docs` by
default), not the image. Using the image avoids dependence on host-installed tool
versions and matches the CI environment. `--no-docker` is useful when equivalent
tools are already available on the host.

Within either environment, it generates or reuses the local C++ API MDX tree,
post-processes that tree, and then performs the selected final action:

```text
generate or reuse local C++ API MDX
  |
  v
post-process local MDX
  |
  v
check internal links (check_doc_links.py)
  |
  +-- no action flag ...... fern check --local --warnings
  +-- --preview ........... fern docs dev
  +-- --publish-preview ... fern generate --docs --preview
  `-- --publish ........... fern generate --docs
```

The internal link check runs in every mode, including preview and publication, and
fails the pipeline on unresolvable links, anchors, assets, or `docs.yml` redirect
destinations. It validates C++ API links only when a generated tree is present.

Here, **local validation** means `fern check --local --warnings`. The helper script
stops after Fern reports publication success; it does not perform the public URL
read-back. The separate HSDK GA release verifier requests each approved, plan-listed
public route after publication, follows redirects, requires a successful HTTP
response, and records the resolved URL. This confirms that the reviewed routes are
publicly reachable; the release reviewer confirms that the planned and resolved
routes are the expected release pages.

## Fern CLI

Install the pinned CLI version (matches `fern/fern.config.json`):

```bash
npm install -g fern-api@5.82.0
```

Run Fern commands from **`docs`**. The CLI discovers the configuration under
`fern/` and selects the version pinned by `fern/fern.config.json`:

```bash
cd docs
fern check --local --warnings
fern docs dev
```

Or use the build script from the repo root (Docker or `--no-docker`).

## Publish preview

Remote preview (requires login):

```bash
cd docs
fern generate --docs --preview
```

## Publish the production user guide

Production publication requires Fern authentication and a generated C++ API tree. To
regenerate, post-process, validate, and publish the documentation from the current
checkout:

```bash
python3 docs/scripts/build_holoscan_docs.py \
  --no-docker \
  --with-library-mdx \
  --publish
```

If the reviewed release checkout already contains the prepared API tree, reuse it for
publication with `--skip-library-mdx`:

```bash
python3 docs/scripts/build_holoscan_docs.py \
  --no-docker \
  --skip-library-mdx \
  --publish
```

For production reuse, the helper requires a non-empty generated C++ API tree and
runs the postprocessor in check-only mode across every generated MDX file. Any file
that still requires a rewrite blocks publication without modifying the tree.

An updated copy of the helper script from one checkout can operate on a separate
clean release checkout. The selected checkout supplies the documentation source,
Fern configuration, local generated API output path, Dockerfile, and version. When
C++ API generation is requested, Fern still obtains the headers from
the configured public GitHub repository:

```bash
python3 /path/to/holoscan-sdk/docs/scripts/build_holoscan_docs.py \
  --repository-checkout /path/to/holoscan-sdk-release \
  --no-docker \
  --skip-library-mdx \
  --publish
```
