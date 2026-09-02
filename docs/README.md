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
   unsupported, so the link check rejects them too. Run:

   ```bash
   python3 docs/scripts/render_source_links.py --docs-root docs --write
   python3 docs/scripts/check_doc_links.py
   python3 docs/scripts/check_doc_links.py --local-files-only
   ```

   The first command normalizes SDK source links to the public GitHub repository
   at `main`; the second verifies Fern links, anchors, and images; and the third checks
   local file targets across the public Markdown tree. `fern check` does not resolve
   link targets. These checks run through pre-commit. The local-file check always scans
   the authored public tree, so moving or deleting a linked file is covered even when
   the deleted target would not otherwise match a staged-file hook.
   The source-link hook passes only changed files and uses
   `--write` to normalize GitHub refs and `blob`/`tree` routes; `pre-commit run
   --all-files` checks and normalizes the complete authored documentation tree.
4. From the public repository root, build and validate docs (default: Docker,
   then `check_doc_links.py` and `fern check --local`):

   ```bash
   ./run build_docs
   ```

5. Preview locally with a dev server:

   ```bash
   ./run live_docs
   ```

The wrapper commands resolve the helper relative to the checkout. In the
internal source layout, run the same commands as `./public/run build_docs` and
`./public/run live_docs` from the repository root.

For these zero-argument commands, a nonempty `FERN_TOKEN` opts in to automatic
C++ API generation. When the variable is empty or unset, the wrapper selects
`--skip-library-mdx`: an existing generated tree is still post-processed and
included, while a clean checkout temporarily omits both the C++ library and
API-reference navigation blocks for the local validation or preview. The
checked-in Fern configuration is never changed, and the temporary project is
removed when the command exits or is interrupted.

Any explicit option suppresses that wrapper default and is forwarded unchanged.
Use `--with-library-mdx` to explicitly generate the API reference with Fern's
local parser, which requires a running Docker daemon, or `--skip-library-mdx`
to reuse a prepared local tree. Use `--no-docker` to run the documentation
pipeline on the host instead of in the docs container. On macOS with Colima,
keep the selected checkout and publishing `--work-dir` under the home directory,
which Colima shares by default. Pass `--skip-fern-check` to run the pipeline only
(no Fern validation), or `--skip-link-check` to bypass internal link validation.
Remote preview and production publication retain their authentication and
generated-content requirements; they never use the tokenless local fallback.

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
C++ library input. The current `fern/docs.yml` instead points at the selected
checkout's local `include/holoscan` directory. The helper removes any previous
generated tree and invokes `fern docs md generate --local`, then runs
`docs/scripts/fix_generated_library_mdx.py` on the resulting
`fern/generated/api-reference/cpp` tree. User-guide and C++ API pages therefore come
from the same checkout, including MR changes and internal release tags that are not
yet available from the public GitHub repository.
See Fern's
[library reference documentation](https://buildwithfern.com/learn/docs/api-references/library-reference).

### Generated C++ `CodeBlock` links

Fern supplies the initial link values. The `holoscan-cpp` library entry in
`fern/docs.yml` points to `../../include/holoscan`. When
`fern docs md generate --local` parses those headers, it writes the generated MDX
pages and each initial `CodeBlock links={{...}}` map. These mappings are parser
output; they do not come from a separately maintained table in this repository.

`fix_generated_library_mdx.py` treats the generated page tree as the authoritative
catalog and reconciles each C++ code block as follows:

1. Scan every generated MDX file for a frontmatter `title` beginning with
   `holoscan::`. Retain only titles that identify exactly one generated page.
2. For fully qualified `holoscan::...` names, treat the generated frontmatter title
   as authoritative without inferring whether the page represents a class, struct,
   enum, function, or another C++ entity.
3. For each qualified name, select the longest prefix that has a generated page.
   For example, `holoscan::gxf::GXFComponent::gxf_component` resolves to the page
   titled `holoscan::gxf::GXFComponent`.
4. Link an unqualified identifier only when its short name identifies exactly one
   generated class, struct, enum, typedef, or union page and its signature occurrence
   is type-like. This resolves `MemoryStorageType`, `expected<T, E>`, and `_t` aliases
   without mistaking `std::type_info` or parameters named `message` or `codec` for
   same-named Holoscan types.
5. Treat level-three headings under generated `## Types` and `## Inner ...`
   sections as page-local type targets. Explicit Fern heading annotations such as
   `### config \[#config]` participate in anchor collision counting, so an inner
   `### Config` heading can resolve to `#config-1`.
6. Use the final component of a resolved page title (`GXFComponent` in the example) as
   the `CodeBlock links` key. Compute its value as the relative path from the current
   MDX page to the target page, remove the `.mdx` suffix, normalize path separators,
   and lowercase the route.
7. Remove mapping keys whose identifier is absent from the code block. Before this
   reconciliation, also remove targets under the unpublished
   `api-reference/nvidia/namespaces/gxf/...` hierarchy while preserving their visible
   code tokens.
8. Omit a key when it is a strict substring of another C++ identifier in the same
   block. Fern applies `CodeBlock` keys as block-wide text mappings, so an `Arg`
   mapping can also link the unrelated template identifiers `ArgT` and `ArgsT`.
   This conservative rule leaves the exact `holoscan::Arg` occurrence unlinked in
   that block. The same rule leaves `holoscan::Resource` unlinked when
   `ResourceType` occurs alongside it; longer noncolliding identifiers such as
   `ArgList` and `ResourceType` retain their own links.
9. When the same short key resolves to multiple qualified pages in one code block,
   preserve an existing mapping rather than guessing. Ambiguous unqualified names
   are not added, including collisions between different page kinds such as a class
   and struct with the same short name.

For example, an inherited signature containing
`holoscan::gxf::GXFComponent::gxf_component()` is assigned a
`"GXFComponent": "../namespaces/gxf/classes/gxfcomponent"` mapping when that is
the relative route from the current page. Transitive `nvidia::gxf::Handle` remains
visible and syntax-colored, but it receives no link because its API page is not
published.

An unqualified `MemoryStorageType` token links to its generated enum page when that
short name is unique. A page-local `AppStatus` token links to the corresponding
`#appstatus` heading rather than requiring a standalone generated page.

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
Fresh C++ API generation also forwards the active Unix Docker socket into the
docs container so Fern can run its local parser as a sibling container. The
socket is not forwarded when the generated API tree is reused.
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
npm install -g fern-api@5.89.3
```

Keep this Fern pin aligned with `fern/fern.config.json`, the docs Dockerfile,
and `docs/scripts/build_holoscan_docs.py`. Committed SDK source links use the public
GitHub repository and `main`, so they work directly in raw Markdown/MDX. The release
publishing wrapper copies the documentation to an isolated staging directory and
uses `docs/scripts/render_source_links.py` to render every source link with one
validated release repository and tag before invoking Fern. The link check rejects
mixed repositories or refs.

`docs.yml` sets `substitute-env-vars: false`, so Fern leaves environment expressions
in documentation and code examples untouched.

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
