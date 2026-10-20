# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository-neutral source browser URLs; callers own repository/ref selection policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote, urlsplit


class SourceLinkError(ValueError):
    """A source-link invariant violation."""


def normalize_source_browser_url(value: str) -> str:
    """Normalize a credential-free HTTPS repository URL for GitHub or GitLab."""
    if not isinstance(value, str) or re.search(r"[\s\\]", value):
        raise SourceLinkError("source browser URL must be a credential-free https repository URL")
    parsed = urlsplit(value.rstrip("/"))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SourceLinkError("source browser URL must be a credential-free https repository URL")
    repository = parsed.path.removesuffix("/-").removesuffix(".git")
    parts = repository.lstrip("/").split("/")
    if len(parts) < 2 or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in (".", "..") for p in parts
    ):
        raise SourceLinkError("source browser URL must identify a repository")
    has_gitlab_segment = parsed.path.endswith("/-")
    is_gitlab = (
        has_gitlab_segment
        or parsed.hostname == "gitlab.com"
        or parsed.hostname.startswith(("gitlab.", "gitlab-"))
    )
    if parsed.hostname == "github.com" and (len(parts) != 2 or has_gitlab_segment):
        raise SourceLinkError("GitHub source browser URL must identify an owner/repository")
    return f"https://{parsed.hostname}{repository}" + ("/-" if is_gitlab else "")


@dataclass(frozen=True)
class SourceLinkTarget:
    """A repository and revision for one source tree, independent of other trees."""

    source_browser_url: str
    source_ref: str
    source_path_prefix: str = ""

    def __post_init__(self):
        object.__setattr__(
            self, "source_browser_url", normalize_source_browser_url(self.source_browser_url)
        )
        if (
            not isinstance(self.source_ref, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", self.source_ref)
            or any(p in ("", ".", "..") for p in self.source_ref.split("/"))
        ):
            raise SourceLinkError("source ref must be a nonempty branch, tag, or commit")

        if (
            not isinstance(self.source_path_prefix, str)
            or self.source_path_prefix.startswith("/")
            or "\\" in self.source_path_prefix
            or any(p in ("", ".", "..") for p in self.source_path_prefix.split("/"))
        ) and self.source_path_prefix != "":
            raise SourceLinkError("source path prefix must be a relative repository directory")

    def url(
        self, path: str, *, directory: bool = False, query: str = "", fragment: str = ""
    ) -> str:
        """Encode a checkout-relative path without conflating the revision and path."""
        if path.startswith("/") or "\\" in path or any(p in (".", "..") for p in path.split("/")):
            raise SourceLinkError("source path must stay inside its repository")
        path = "/".join(part for part in (self.source_path_prefix, path) if part)
        route = "tree" if directory else "blob"
        suffix = ("?" + query if query else "") + ("#" + fragment if fragment else "")
        return (
            f"{self.source_browser_url}/{route}/{quote(self.source_ref, safe='')}"
            + ("/" + quote(path, safe="/") if path else "")
            + suffix
        )
