<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan SDK user guide (Fern)

User guide pages are committed as **MDX** (`.mdx`) in this directory, alongside images.
Fern configuration lives under `fern/` (`fern.config.json`, `docs.yml`, `index.yml`,
`assets/`, `dist/`). C++ and Python API pages are generated at build time under
`fern/generated/` (gitignored).

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
   link targets. These checks run through pre-commit. The local-file check also runs
   when any public target changes, so moving or deleting a linked file is covered.
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
library API generation. When the variable is empty or unset, the wrapper selects
`--skip-library-mdx`: an existing generated tree is still post-processed and
included, while a clean checkout temporarily omits both configured libraries and
API-reference navigation blocks for the local validation or preview. The
checked-in Fern configuration is never changed, and the temporary project is
removed when the command exits or is interrupted.

Any explicit option suppresses that wrapper default and is forwarded unchanged.
Use `--with-library-mdx` to explicitly generate the API reference with either
`FERN_TOKEN` or host `fern login` credentials, or `--skip-library-mdx` to reuse
a prepared local tree. Use `--no-docker` to run on the host instead of the docs
container. Pass `--skip-fern-check` to run the pipeline only (no Fern
validation), or `--skip-link-check` to bypass internal link validation. Remote
preview and production publication retain their authentication and generated-
content requirements; they never use the tokenless local fallback.

Library API generation includes the C++ parser and requires a running Docker
daemon. With a VM-backed runtime such as Colima (Linux or macOS), keep the
selected checkout and publishing `--work-dir` in directories shared with the
VM. Colima shares the home directory by default.
Without `--work-dir`, the publishing helper uses
`~/.cache/holoscan-docs/work/holoscan-docs-<TAG>` (or `dev-<commit>` in place of
the tag). Local previews retain this directory for reuse. Remote publication
uses a fresh directory under the same parent and cleans it up unless
`--keep-workdir` is requested. Explicit work directories retain their existing
ownership and cleanup policy. The host's `TMPDIR` no longer chooses the default
publishing work directory.

The publishing helper places isolated Fern HOME directories under
`<work-dir>/fern-state`, beside rendered documentation so Fern does not watch
its own state files. Standalone Python-helper runs use `~/.cache/holoscan-docs/`
as this state location, not as their documentation work directory; inputs and
generated output remain in the selected checkout/docs tree.

Only local `--preview` reuses Fern HOME and its `.fern/` cache. Repeated
`--preview --docker` runs with the same work directory can reuse cached bundles.
Each invocation gets fresh `tmp/parser-<id>` scratch space, removed on exit or
interruption. Validation, remote `--publish-preview`, publication, and deletion
keep a fresh `run-<id>` HOME that is removed afterward, preserving CI isolation.

An advisory lock rejects simultaneous local previews using the same HOME. Stop the
existing run or select a different work directory to run another preview.
The `.lock` file remains after exit, but its lock is released automatically;
do not delete the file to bypass an active run.

Generate both APIs and preview them in one containerized run; no host `TMPDIR`
setting is needed:

```bash
python3 docs/scripts/build_holoscan_docs.py --with-library-mdx --preview
```

The helper first probes the active socket path from the daemon's filesystem,
preserving native and rootless Docker paths. If that path cannot be mounted as
a socket, it tries the daemon-side `/var/run/docker.sock`. This also handles
Colima's host proxy without depending on the host OS. The helper shares its
isolated container HOME and parser files at identical paths and sets the
container's `TMPDIR` automatically. Use the publishing helper's
`--work-dir <shared-directory> --docker --preview` options for a local preview.
For a standalone run with a custom VM sharing configuration, set
`HOLOSCAN_DOCS_FERN_STATE_DIR` to another shared directory outside the checkout.
The publishing helper sets this variable from `--work-dir`; the host's general
`TMPDIR` does not select the Fern state directory.

Docker previews default to `NODE_OPTIONS=--max-old-space-size=4096` to give
large documentation trees a 4 GiB Node.js old-space heap limit. This is a
project default, not a Fern requirement or a reservation of VM memory. Leave
additional Docker VM memory for other Node processes, native allocations, and
the operating system. An explicit `NODE_OPTIONS` value is forwarded unchanged
by both entry points; an empty value disables this default. Validation and
publication do not receive a default heap override. For example:

```bash
NODE_OPTIONS="--max-old-space-size=2048" \
  python3 docs/scripts/build_holoscan_docs.py --skip-library-mdx --preview
```

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
the C++ and Python API MDX, and asks Fern to publish the preview. Set `FERN_TOKEN` in Jenkins
(Secret text credential ID `FERN_TOKEN`) so generation and preview publication can
authenticate.

The preview ID names the Fern deployment; it does not select a Git revision for
either library input. The current `fern/docs.yml` points at the selected checkout's
local `include/holoscan` headers and a generated parser tree projected from the
public facade stubs under `python/holoscan`. The private `_core` extension inventory
is excluded. The helper removes any previous generated tree and invokes
`fern docs md generate --local`, then runs
`docs/scripts/fix_generated_library_mdx.py` across the resulting generated API tree.
User-guide, C++, and Python API pages therefore come from the same checkout,
including MR changes and internal release tags that are not yet available from the
public GitHub repository.
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
Fresh library API generation also forwards the active Unix Docker socket into the
docs container so Fern can run its local parser as a sibling container. The
socket is not forwarded when the generated API tree is reused.
`--container-name` controls only the transient runtime container name (`docs` by
default), not the image. Using the image avoids dependence on host-installed tool
versions and matches the CI environment. `--no-docker` is useful when equivalent
tools are already available on the host.

Within either environment, it generates or reuses the local API MDX trees,
post-processes that tree, and then performs the selected final action:

```text
generate or reuse local C++ and Python API MDX
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
destinations. It validates library API links only when generated trees are present.
After API generation, when reusing complete API trees, and before remote publication,
the check also requires every navigation folder to contain MDX pages. Source-only
local checks continue to allow absent generated output.

Fern initially nests the Python package under its library ID
(`python/holoscan-python/holoscan`). Post-processing flattens this to
`python/holoscan`, matching `cpp/holoscan` and the configured navigation folders.
Generated navigation page IDs follow the moved files. Python page slugs and
cross-links use `/holoscan/sdk-user-guide/python/api-reference/holoscan/`; C++ keeps
`/holoscan/sdk-user-guide/api-reference/holoscan/`. The generated Python tree appears
once, under **Python → API reference**, beside the Python guides. The shared
**API reference → Python API** entry links directly to it in the same tab; it
does not duplicate the generated tree or require a redirect. Archive assembly
rewrites that shortcut and other internal links to use the selected archive's
version prefix. Source-only local previews omit both generated references and
the shortcut when API output is unavailable.
Reusing an already normalized tree is supported;
publication with `--skip-library-mdx` checks the tree without modifying it.

Here, **local validation** means `fern check --local --warnings`. The helper script
stops after Fern reports publication success; it does not perform the public URL
read-back. The separate HSDK GA release verifier requests each approved, plan-listed
public route after publication, follows redirects, requires a successful HTTP
response, and records the resolved URL. This confirms that the reviewed routes are
publicly reachable; the release reviewer confirms that the planned and resolved
routes are the expected release pages.

### Validate rendered links after hydration

In web development, **hydration** is the process of taking static,
server-rendered HTML and attaching JavaScript behavior and event listeners to
make the page interactive. React may reconcile and update the document during
this process. As a result, the initial response returned by `curl` or shown by
**View Page Source** can differ from the live DOM shown in the browser's
**Elements** inspector after Fern's JavaScript runs.

Validate interactive Fern behavior against the hydrated DOM. In particular,
do not classify a generated `CodeBlock` token as linked or unlinked solely from
the initial HTTP response. For each representative change:

1. Open the same API page and signature in the baseline and branch previews.
2. Inspect the token in the browser's **Elements** panel after the page finishes
   loading.
3. For a positive link, confirm that the token is inside an
   `<a class="fern-code-link">` element with the expected `href`, then follow the
   link and verify the destination page or anchor.
4. For a removed link, confirm that the syntax-colored token remains visible
   but is not inside an anchor.

For example, a successfully hydrated generated-type link has this form:

```html
<a href="../enums/memorystoragetype" class="fern-code-link">
  <span>MemoryStorageType</span>
</a>
```

A headless browser that waits for JavaScript execution can automate this check.
Fetching the page with `curl` alone validates only the initial server response,
not the final interactive DOM used by readers.

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
mixed repositories or refs. The shared source target also accepts
`source_path_prefix` for a repository that keeps the documented source tree in a
subdirectory. It defaults to the repository root and is independent of the host.
For rendered SDK links, pass `--source-path-prefix` to both the renderer and the
builder's rendered-link validation; the release publisher supplies its selected
layout automatically. Prefixes do not change authored links or archived public
release layouts.

### Tooling-owned `docs.yml` configuration

Paired `begin` and `end` comments in `fern/docs.yml` delimit configuration owned by
documentation tooling. Keep every pair unique, ordered, and intact; edit content
inside a pair only according to the ownership described by its surrounding comment
or publishing workflow.

The release publisher copies the complete publishing-owned configuration block from
the active publishing branch into the staged release configuration. This lets a site
republish use current site-wide configuration while pages and release-specific
settings still come from the immutable release tag. Only that explicitly delimited
block is synchronized: other marked regions and unmarked settings retain their
existing behavior. The synchronization affects assembled site staging only and does
not rebuild or modify documentation archive snapshots.

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

Production publication requires Fern authentication and generated C++ and Python API
trees. To regenerate, post-process, validate, and publish the documentation from the
current checkout:

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

For production reuse, the helper requires non-empty generated C++ and Python API trees
and runs the postprocessor in check-only mode across every generated MDX file. Any
file that still requires a rewrite blocks publication without modifying the tree.

An updated copy of the helper script from one checkout can operate on a separate
clean release checkout. The selected checkout supplies the documentation source,
Fern configuration, local generated API output paths, Dockerfile, and version. When
API generation is requested, Fern reads both libraries from that selected checkout:

```bash
python3 /path/to/holoscan-sdk/docs/scripts/build_holoscan_docs.py \
  --repository-checkout /path/to/holoscan-sdk-release \
  --no-docker \
  --skip-library-mdx \
  --publish
```

## Include module documentation

The builder imports registered modules under **Holoscan Modules** in the SDK
guide. Module repositories maintain their own Markdown or MDX and a navigation
fragment at `docs/fern/index.yml`. Camera is currently the only registered module;
unknown module names fail until their source contract and metadata are added.

From the SDK source root, select a local checkout with `--module-source NAME=PATH`.
For example:

```bash
python3 docs/scripts/build_holoscan_docs.py --skip-library-mdx \
  --module-source camera=/path/to/holoscan-camera
```

Repeat `--module-source` to include multiple registered modules, or supply
`--module-sources-file /path/to/modules.json` with a JSON object mapping module
names to absolute checkout paths, such as `{"camera": "/path/to/holoscan-camera"}`.
These two selection forms are mutually exclusive. Each checkout is mounted
read-only in Docker; host Git credentials are not forwarded into the container.
Existing Camera source-selection arguments remain supported; select a module
only once and do not combine the manifest with Camera overrides.

The builder copies the selected SDK documentation into a temporary sibling tree
and imports each module's `page`/`path`/`slug` navigation fragment. Pages can be
grouped under `section`/`slug`/`contents` entries with `skip-slug: true`; these
sidebar groups preserve page URLs. Flat fragments remain supported. Markdown
sources generate MDX during assembly; authored MDX pages can also be imported.
Titles come from navigation, and generated copies are not committed to the module.
See [Module content processing](#module-content-processing) for conversion rules
and [Registering another module](#registering-another-module) for source mappings.

Internal links and Fern syntax are checked together with the SDK guide. SDK
source-ref consistency is checked before the module overlay: a module may
intentionally document a different SDK baseline tag. These module-owned source
links retain their authored targets. Local image/PDF assets may live anywhere
inside the module checkout; relative links to other source files use the module
source-link target described below. Generated API output and imported pages are
temporary; authored SDK navigation and module checkouts stay unchanged.
The preview URL handoff is retained at the original output location.

Add `--preview` for a local preview. With Camera selected, open
`http://localhost:3000/holoscan/sdk-user-guide/modules/camera/overview`.
The assembled guide is a snapshot: stop and restart the command after editing
module sources or navigation. Without `--preview`, the command validates only.
`--skip-library-mdx` avoids regenerating the C++ and Python API pages; a guide-only
check can report warnings for API or archived pages absent from the local tree.

### Module source-link targets

SDK and module links share `source_browser_url`, `source_ref`, and an optional
`source_path_prefix` through the same URL construction. Each source tree has its own target: an SDK release tag does not
select a module revision. The SDK renderer retains its release/ref validation;
module CI retains its repository allowlist and exact-checkout verification.

For a local checkout, source-file links default to the registered public
repository at `main`. To pin those links, supply both options:

```bash
CAMERA_SHA=$(git -C /path/to/holoscan-camera rev-parse HEAD)
python3 docs/scripts/build_holoscan_docs.py \
  --module-source camera=/path/to/holoscan-camera \
  --module-source-browser-url camera=https://github.com/nvidia-holoscan/holoscan-camera \
  --module-source-ref camera="$CAMERA_SHA"
```

An optional `--module-source-path-prefix camera=directory` selects a source
subdirectory in that browser repository. It defaults to empty (repository root),
independently of the SDK's `--source-path-prefix`. Camera's internal and public
repositories both use that empty module prefix.

These options affect source-file and directory links, not Fern page routes or
already absolute URLs. They are carried into Docker with the read-only module
mount; the container does not need access to the module's Git metadata. Use a
repository that contains the selected revision: a private preview can use its
GitLab browser, while public documentation needs a revision available publicly.
The explicit target identifies the source links; it does not verify the local
checkout against that ref or test remote URL availability.

`--module-sources-file` accepts both the existing name-to-path mapping and entries
with the three required fields below and an optional `source_path_prefix`. CI
supplies the target from its verified checkout:

```json
{
  "camera": {
    "path": "/path/to/holoscan-camera",
    "source_browser_url": "https://github.com/nvidia-holoscan/holoscan-camera",
    "source_ref": "<camera-commit-sha>"
  }
}
```

### Fetching Camera documentation

Camera also has repository and revision options backed by its SDK dependency
helper. To fetch the declared dependency revision instead of using a local checkout:

```bash
python3 docs/scripts/build_holoscan_docs.py --with-camera-docs
```

This uses `cmake -P cmake/external/HoloscanCameraDependency.cmake` to fetch sources
without configuring the SDK or Camera. The repository and revision default to the
existing Camera dependency declaration. Source links use the fetched HTTPS
repository and resolved commit, including when a branch or tag was requested.
For a fetch using another transport, supply an explicit source browser/ref pair.
The selected revision must contain
`docs/fern/index.yml` and its referenced READMEs or MDX pages; an older revision fails with an actionable
error. Until the Camera documentation integration is available at the SDK dependency pin,
ordinary builds retain the Camera repository link. Enable inclusion explicitly
when that prerequisite is met.

For a coordinated documentation change, use `--camera-source-dir /path/to/camera`
to assemble a local checkout without fetching, or `--camera-ref <commit>` to fetch
a commit, tag, or branch from the configured Camera repository. To review a change
from another remote, pair `--camera-ref` with `--camera-repository <git-url>`.
These options imply `--with-camera-docs` and leave the SDK build dependency pin
unchanged. Use a full commit SHA for reproducible validation and publication.

For example, from the SDK source root, with host Git authentication configured:

```bash
python3 docs/scripts/build_holoscan_docs.py --no-docker --skip-library-mdx \
  --camera-repository <camera-git-url> --camera-ref <full-camera-commit-sha>
```

Host builds require CMake 4.x, Git, and the pinned Fern tools. The docs container
supplies those tools. For an authenticated remote, fetch and check out the exact
Camera commit on the host, then use `--camera-source-dir /path/to/camera` with the
default Docker build. The checkout is mounted read-only; the builder does not
forward host Git credentials into Docker. `--camera-source-dir` is mutually
exclusive with the repository and revision overrides.

### Module content processing

The public builder implements this processing in `docs/scripts/modules_docs.py`.
The same rules apply to every registered module in local and CI builds.

| Input | Processing during module documentation assembly | Result or limitation |
| --- | --- | --- |
| Any `README.md` listed in `docs/fern/index.yml`, including nested application guides and Python setup | Generate MDX in the temporary SDK docs tree. | The README stays the maintained source; generated files are not committed. |
| Operator `benchmarks.md` listed in the fragment | Generate MDX using the same conversion as READMEs. | Reports stay beside their operators; existing benchmark slugs are preserved. |
| Any `.md` page inside the module checkout listed in the fragment | Generate MDX using the same conversion as READMEs. | Markdown stays the maintained source. |
| Authored `.mdx` page listed in the fragment | Import it and process supported links. | Authored MDX must already be valid; Markdown normalization does not apply. |
| Leading Markdown H1 | Remove it; use the navigation title. | Other headings remain, and navigation slugs determine published routes. |
| Simple inline relative link to a listed guide | Resolve it against the source file, then map it to the guide route. | Uses `/holoscan/sdk-user-guide/modules/<module>/<slug>`; fragments are preserved. |
| Absolute URL, including public repository URLs | Leave it unchanged. | Selecting a module commit does not pin absolute URLs or convert them to Fern routes. |
| Relative link to another existing repository file or directory, when absent from navigation | Use the module's selected source browser/ref with a `blob` or `tree` path; a local checkout without an explicit target defaults to public `main`. | Content absent from navigation is not imported automatically. A listed root README maps to the Overview route, including setup fragments. |
| Relative PNG, JPG, JPEG, GIF, SVG, WebP or PDF link | Copy the file from inside the checkout and rewrite its relative asset path. | Assets can live beside component READMEs or under `docs/`. |
| Same-page fragment or site-absolute path | Leave it unchanged. | Internal destinations are checked during combined-guide validation. |
| Marked `holoscan-module-internal` blocks in Markdown | Remove the entire block before normalizing prose or rewriting links. | Internal setup content is excluded; malformed or unbalanced markers fail assembly. |
| Markdown HTML comments, prose braces and `<br>` tags | Omit comments, escape braces and make line-break tags self-closing. | This normalization applies to Markdown prose, not authored MDX pages. |
| Fenced examples and inline code | Preserve them during conversion and link rewriting. | Example links and code are not rewritten or executed. |
| Reference-style links, HTML `href`/`src` and other markup | No dedicated link conversion. | Use supported inline Markdown for relative repository links; other markup must be MDX compatible. |
| Missing relative target, path escaping the checkout, unlisted `.mdx` target, or invalid/duplicate navigation entry | Fail assembly with an error. | Symlink targets must also stay inside the checkout. |
| External website availability | No remote URL reachability check. | Passing local link validation does not prove a GitHub page or external URL exists. |
| Archived SDK docs and source checkouts | Apply the module overlay only to the current guide's temporary tree. | Archives and maintained source files are unchanged. |

#### Worked example: Camera

For example, when importing `applications/sipl_frame_saver/README.md`, with the
SIPL operator listed in Camera's navigation fragment, this source link:

```markdown
[Operator](../../operators/sipl_capture_op/README.md#tests)
```

becomes this link in the generated MDX:

```markdown
[Operator](/holoscan/sdk-user-guide/modules/camera/sipl-capture#tests)
```

The following examples use the same README as their starting point and a local
checkout without an explicit source-link target:

| README input | Generated MDX or behavior |
| --- | --- |
| Link to `../../operators/sipl_capture_op/benchmarks.md#recorded-results` | Link to `/holoscan/sdk-user-guide/modules/camera/sipl-benchmarks#recorded-results` |
| Link to `../../Dockerfile` | Link to `https://github.com/nvidia-holoscan/holoscan-camera/blob/main/Dockerfile`; source files absent from navigation stay on GitHub. |
| Link to `../../operators/sipl_capture_op/README.md` | Link to `/holoscan/sdk-user-guide/modules/camera/sipl-capture` |
| Link to `../../operators/sipl_capture_op/README.md#tests` | Link to `/holoscan/sdk-user-guide/modules/camera/sipl-capture#tests` |
| Link to `../../README.md#holoscan-sdk-50-setup` | Link to `/holoscan/sdk-user-guide/modules/camera/overview#holoscan-sdk-50-setup` when the root README is listed as Overview; older fragments without that entry use the GitHub `blob/main/README.md` URL. |
| Link to `https://github.com/nvidia-holoscan/holoscan-camera/blob/main/README.md` | Unchanged; an absolute GitHub URL stays on GitHub, even when the selected Camera source is a different commit. |
| Markdown image targeting `diagram.png` when that file exists beside the README | Image target becomes `../assets/applications/sipl_frame_saver/diagram.png`; the image is copied into the temporary docs tree. |
| `Choose {mode}<br>` in ordinary prose | `Choose &#123;mode&#125;<br />` |
| Link to `missing.md` when that target does not exist | Assembly fails; no replacement URL is invented. |

For a Python application, list both `applications/holoscan_camera_v4l2/python/README.md`
and `python/README.md` in the fragment. Then the operator's relative link to the
Python application becomes its declared Fern route, and the application's
`../../../python/README.md#build-against-an-installed-sdk` link becomes the Python
support route with that fragment. Without those entries, existing Markdown
files use repository source links. For an exact preview, select the module
repository and commit as its source-link target; the public `main` default may
return 404 for files that have not been published there.

Link examples inside fenced code or inline code remain literal examples and are
not rewritten.

### Registering another module

`docs/scripts/modules_docs.py` owns the shared `markdown_to_mdx`, `fetch_module`,
`import_module`, and `assembled_module_docs` functions. Its SDK-owned `MODULES`
registry supplies each module's name, navigation title, public repository,
CMake dependency script/variable prefix, and fetched checkout directory.
The importer reads module documentation as data; it does not run scripts from
module checkouts. Fetching runs the selected SDK's reviewed CMake dependency helper.

The fragment is the explicit page list: each `path` is resolved relative to
`docs/fern/index.yml` and must name an existing `.md` or `.mdx` file inside the
module checkout. Directory depth and component names do not restrict inclusion.
Unlisted documents are not discovered or imported automatically.

The shared source contract is:

| Maintained source listed in `docs/fern/index.yml` | Generated path below `<module>/` |
| --- | --- |
| Root `README.md` | `overview.mdx` |
| Markdown or MDX under `docs/` | The same path relative to `docs/`, with an `.mdx` extension |
| Any `<directory>/README.md` outside `docs/` | `<directory>.mdx`; for example `python/README.md` becomes `python.mdx`, and `applications/capture/python/README.md` becomes `applications/capture/python.mdx` |
| `operators/<name>/benchmarks.md` | `benchmarks/<name>_benchmarks.mdx` (existing report layout) |
| Other Markdown or MDX anywhere in the checkout | The repository-relative path with an `.mdx` extension |

Navigation supplies the titles and published slugs. Output paths, routes, and
copied assets are isolated by module name. Relative guide links resolve within
the selected module; relative links escaping its checkout fail. Absolute links
remain unchanged. SDK version-sync blocks use `module-docs:sdk-source` markers;
these comments remain in the maintained README and are removed from generated
MDX. Complete `holoscan-module-internal` blocks are excluded before link processing.

To add Multimedia, first provide its navigation fragment and confirm that its
sources fit this contract. Then register its reviewed metadata and SDK dependency
helper, and validate its real pages together with Camera. If its source layout
needs a new mapping, add that mapping and tests before registration. Camera is
the only registered module today; the second-module tests use a synthetic fixture
and do not qualify Multimedia documentation or builds.

### Validation from a module's CI

A module job can validate its checked-out documentation using a pinned SDK
checkout containing this builder. It needs no SDK application build or publishing
token. For example, from the Camera checkout, with `HSDK_DOCS_SDK_CHECKOUT` pointing to a public
SDK checkout at the chosen immutable commit:

```bash
python3 "$HSDK_DOCS_SDK_CHECKOUT/docs/scripts/build_holoscan_docs.py" \
  --repository-checkout "$HSDK_DOCS_SDK_CHECKOUT" \
  --module-source "camera=$PWD" \
  --skip-library-mdx --container-name camera-docs-ci
```

The SDK checkout supplies the theme, navigation, tool versions, and importer;
the module checkout supplies the proposed module content. This validates the
module inside the combined SDK guide, not a separate module website. Record both
checkout SHAs in CI and retain the command output as the validation artifact.
Select the SDK commit through CI configuration rather than an unpinned branch
lookup. Docker is the default; use `--no-docker` only on a runner provisioned with
the SDK's documentation tools. Guide-only checks may report omitted API/archive
warnings. Add `--preview` for a local interactive preview.

Module-CI job wiring and hosted preview publication remain separate integration
steps. A publishing job should use a module-specific preview ID such as
`camera-mr-83`, distinct from an SDK MR preview, and obtain publishing credentials
only in the trusted publishing workflow. Validation alone does not publish or
replace any preview. Production documentation remains owned by SDK publication.
