# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble module Markdown and MDX into an isolated SDK documentation tree."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from check_doc_links import _load_nav, _site_prefix
from fix_generated_library_mdx import fix_mdx_void_br_tags
from source_link_target import SourceLinkTarget


@dataclass(frozen=True)
class ModuleDocs:
    """Reviewed SDK-owned metadata; module checkouts supply documentation only."""

    name: str
    title: str
    public_repository: str
    dependency_script: str
    cmake_prefix: str
    checkout_directory: str

    @property
    def repository_link(self) -> str:
        return f"      - link: {self.title}\n        href: {self.public_repository}\n"


@dataclass(frozen=True)
class ModuleSource:
    """Use a local checkout, or fetch through the SDK's pinned dependency."""

    path: Path | None = None
    revision: str | None = None
    repository: str | None = None
    source_links: SourceLinkTarget | None = None


# Register a module only after its navigation fragment and source layout have
# been validated. The CI repository allowlist remains a separate trust boundary.
MODULES = {
    "camera": ModuleDocs(
        name="camera",
        title="Camera",
        public_repository="https://github.com/nvidia-holoscan/holoscan-camera",
        dependency_script="cmake/external/HoloscanCameraDependency.cmake",
        cmake_prefix="HOLOSCAN_CAMERA",
        checkout_directory="holoscan-camera-src",
    ),
}
LINK = re.compile(r"\]\(([^\s)]+)\)")
CODE = re.compile(
    r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]{0,3}(?P=fence)[ \t]*$"
    r"|(?P<ticks>`+)[^\n]*?(?P=ticks)(?!`)",
    re.M | re.S,
)


def transform_prose(text, transform):
    """Preserve fenced examples and inline code while transforming Markdown prose."""
    chunks = []
    start = 0
    for match in CODE.finditer(text):
        chunks.extend((transform(text[start : match.start()]), match[0]))
        start = match.end()
    chunks.append(transform(text[start:]))
    return "".join(chunks)


def markdown_to_mdx(text: str) -> str:
    """Use the navigation title and normalize prose for MDX without changing code."""

    # Match the Holoscan module public-source projection. These markers are reserved for
    # internal-only content, including examples, and must be balanced.
    public_lines = []
    internal = False
    for line in text.splitlines(keepends=True):
        marker = line.strip()
        if marker == "<!-- holoscan-module-internal:begin -->":
            if internal:
                raise ValueError("nested module internal documentation block")
            internal = True
        elif marker == "<!-- holoscan-module-internal:end -->":
            if not internal:
                raise ValueError("module internal documentation block end without begin")
            internal = False
        elif "holoscan-module-internal:" in line:
            raise ValueError("malformed module internal documentation marker")
        elif not internal:
            public_lines.append(line)
    if internal:
        raise ValueError("unclosed module internal documentation block")
    text = "".join(public_lines)

    def normalize(prose):
        prose = re.sub(r"<!--[\s\S]*?-->", "", prose)
        prose = prose.replace("{", "&#123;").replace("}", "&#125;")
        return fix_mdx_void_br_tags(prose)

    text = transform_prose(text, normalize)
    return re.sub(r"\A\s*# [^\n]*\n", "", text, count=1).lstrip()


def fetch_module(
    module: ModuleDocs,
    checkout: Path,
    destination: Path,
    revision: str | None,
    repository: str | None = None,
) -> Path:
    public_root = checkout / "public" if (checkout / "public").is_dir() else checkout
    dependency = public_root / module.dependency_script
    command = ["cmake"]
    if revision:
        command.append(f"-D{module.cmake_prefix}_GIT_TAG={revision}")
    if repository:
        command.append(f"-D{module.cmake_prefix}_GIT_REPOSITORY={repository}")
    command.extend(["-P", str(dependency)])
    subprocess.run(command, cwd=destination, check=True)
    return destination / module.checkout_directory


def _navigation_pages(nodes: list[dict]) -> list[dict]:
    """Validate route-preserving groups and return their pages in display order."""
    pages = []
    for node in nodes:
        if set(node) == {"page", "path", "slug"}:
            pages.append(node)
        elif (
            set(node) == {"section", "slug", "skip-slug", "contents"}
            and node["skip-slug"] == "true"
            and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", node["slug"])
            and node["contents"]
        ):
            pages.extend(_navigation_pages(node["contents"]))
        else:
            raise ValueError(
                "Module navigation requires page/path/slug entries or nonempty "
                "section/slug/contents groups with skip-slug: true"
            )
    return pages


def _render_navigation(
    module: ModuleDocs, nodes: list[dict], manifest: Path, outputs: dict, indent: int = 10
) -> str:
    """Render validated groups without changing the existing page routes."""
    lines = []
    pad = " " * indent
    for node in nodes:
        if "section" in node:
            lines.append(
                f"{pad}- section: {json.dumps(node['section'])}\n"
                f"{pad}  slug: {node['slug']}\n"
                f"{pad}  skip-slug: true\n"
                f"{pad}  contents:\n"
            )
            lines.append(
                _render_navigation(module, node["contents"], manifest, outputs, indent + 4)
            )
        else:
            relative = outputs[(manifest.parent / node["path"]).resolve()]
            lines.append(
                f"{pad}- page: {json.dumps(node['page'])}\n"
                f"{pad}  path: ../{module.name}/{relative.as_posix()}\n"
                f"{pad}  slug: {node['slug']}\n"
            )
    return "".join(lines)


def import_module(
    module: ModuleDocs, checkout: Path, docs: Path, source_links: SourceLinkTarget | None = None
) -> None:
    """Generate component MDX from READMEs and import authored docs pages."""
    checkout = checkout.resolve()
    source = (checkout / "docs").resolve()
    manifest = source / "fern/index.yml"
    if not manifest.is_file():
        raise ValueError(
            f"{module.title} revision has no Fern navigation: {manifest}. "
            "Select a revision with docs/fern/index.yml or pass a local --module-source."
        )
    nodes = _load_nav(manifest)
    entries = _navigation_pages(nodes)
    if not entries:
        raise ValueError(f"{module.title} navigation must contain at least one page")
    routes = {}
    outputs = {}
    slugs = set()
    prefix = _site_prefix(docs / "fern/docs.yml") + f"/modules/{module.name}"
    for entry in entries:
        path = (manifest.parent / entry["path"]).resolve()
        slug = entry["slug"]
        if not path.is_relative_to(checkout) or not path.is_file():
            raise ValueError(
                f"{module.title} page must be an existing file inside its checkout: {path}"
            )
        if path.suffix not in (".md", ".mdx"):
            raise ValueError(f"{module.title} pages must be Markdown or MDX: {path}")
        relative = path.relative_to(checkout)
        if relative == Path("README.md"):
            output = Path("overview.mdx")
        elif path.is_relative_to(source) and path.suffix in (".md", ".mdx"):
            output = path.relative_to(source).with_suffix(".mdx")
        elif relative.name == "README.md":
            output = relative.parent.with_name(relative.parent.name + ".mdx")
        elif (
            len(relative.parts) == 3
            and relative.parts[0] == "operators"
            and relative.name == "benchmarks.md"
        ):
            output = Path("benchmarks") / (relative.parts[1] + "_benchmarks.mdx")
        else:
            output = relative.with_suffix(".mdx")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or slug in slugs:
            raise ValueError(f"invalid or duplicate {module.title} slug: {slug}")
        if path in routes:
            raise ValueError(f"duplicate {module.title} page: {path}")
        if output in outputs.values():
            raise ValueError(f"duplicate {module.title} output: {output}")
        routes[path] = f"{prefix}/{slug}"
        outputs[path] = output
        slugs.add(slug)

    source_links = source_links or SourceLinkTarget(module.public_repository, "main")
    target = docs / module.name
    if target.exists():
        raise ValueError(f"{module.title} import destination already exists: {target}")
    target.mkdir()
    navigation = (
        f"      - section: {json.dumps(module.title)}\n"
        f"        slug: {module.name}\n        contents:\n"
    )
    for entry in entries:
        path = (manifest.parent / entry["path"]).resolve()
        relative = outputs[path]
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            text = markdown_to_mdx(text)

        def rewrite(match: re.Match[str], path: Path = path, relative: Path = relative) -> str:
            url = urlsplit(match[1])
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                return match[0]
            linked = (path.parent / unquote(url.path)).resolve()
            if linked in routes:
                suffix = ("#" + url.fragment) if url.fragment else ""
                return "](" + routes[linked] + suffix + ")"
            if not linked.is_relative_to(checkout) or not linked.exists():
                raise ValueError(
                    f"{module.title} link is missing or outside its checkout: {match[1]}"
                )
            if linked.suffix == ".mdx":
                raise ValueError(
                    f"{module.title} link is outside its navigation: {path}: {match[1]}"
                )
            suffix = ("?" + url.query if url.query else "") + (
                "#" + url.fragment if url.fragment else ""
            )
            if linked.suffix.lower() not in (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".webp",
                ".pdf",
            ):
                url = source_links.url(
                    linked.relative_to(checkout).as_posix(),
                    directory=linked.is_dir(),
                    query=url.query,
                    fragment=url.fragment,
                )
                return f"]({url})"
            asset = target / "assets" / linked.relative_to(checkout)
            asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(linked, asset)
            return (
                "]("
                + Path(os.path.relpath(asset, (target / relative).parent)).as_posix()
                + suffix
                + ")"
            )

        text = transform_prose(text, lambda prose: LINK.sub(rewrite, prose))
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    navigation += _render_navigation(module, nodes, manifest, outputs)
    index = docs / "fern/index.yml"
    text = index.read_text(encoding="utf-8")
    repository_link = module.repository_link
    if text.count(repository_link) == 1:
        text = text.replace(repository_link, navigation)
    elif repository_link not in text:
        # Release content can predate the module placeholder. Insert only into
        # its existing modules section; never replace a release's navigation.
        sections = re.split(r"(?=^  - )", text, flags=re.M)
        matches = [i for i, section in enumerate(sections) if "    slug: modules\n" in section]
        if len(matches) != 1 or "    contents:\n" not in sections[matches[0]]:
            raise ValueError(
                f"SDK navigation needs one modules section for {module.title} documentation"
            )
        section = sections[matches[0]]
        if f"        slug: {module.name}\n" in section:
            raise ValueError(f"SDK navigation already contains {module.title} documentation")
        sections[matches[0]] = section.replace("    contents:\n", "    contents:\n" + navigation, 1)
        text = "".join(sections)
    else:
        raise ValueError(f"expected exactly one {module.title} repository link in SDK navigation")
    index.write_text(text, encoding="utf-8")


@contextmanager
def assembled_module_docs(checkout: Path, docs: Path, sources: dict[str, ModuleSource]):
    """Overlay selected modules once; preserve SDK and module source checkouts."""
    unsupported = set(sources) - MODULES.keys()
    if unsupported:
        raise ValueError(f"unsupported documentation modules: {sorted(unsupported)}")
    for name, source in sources.items():
        if source.path is not None and (source.revision or source.repository):
            raise ValueError(f"{name}: local source cannot be combined with fetch overrides")
    with TemporaryDirectory(prefix=".module-docs-", dir=docs.parent) as temporary:
        assembled = Path(temporary)
        shutil.copytree(
            docs,
            assembled,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".fern", ".fern-*", ".preview", "node_modules", ".next", "out", "*.log"
            ),
        )
        for name, source in sources.items():
            module = MODULES[name]
            if source.path is None:
                with TemporaryDirectory(prefix=f"{name}-source-") as fetch_dir:
                    path = fetch_module(
                        module, checkout, Path(fetch_dir), source.revision, source.repository
                    )
                    links = source.source_links
                    if links is None:
                        # The fetched checkout resolves symbolic refs to the actual commit.
                        def git(*args, path=path):
                            return subprocess.check_output(
                                ["git", "-C", str(path), *args], text=True
                            ).strip()

                        links = SourceLinkTarget(
                            git("remote", "get-url", "origin"), git("rev-parse", "HEAD^{commit}")
                        )
                    import_module(module, path, assembled, links)
            else:
                import_module(
                    module, source.path.expanduser().resolve(), assembled, source.source_links
                )
        yield assembled
