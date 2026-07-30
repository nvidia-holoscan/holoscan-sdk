"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import os
import re
import subprocess
from subprocess import CalledProcessError


def is_file_empty(f):
    return os.stat(f).st_size == 0


def __git(*opts):
    """Runs a git command and returns its output"""
    ret = subprocess.check_output(["git"] + list(opts))
    return ret.decode("UTF-8").rstrip("\n")


def __gitdiff(*opts):
    """Runs a git diff command with no pager set"""
    return __git("--no-pager", "diff", *opts)


def branch():
    """Returns the name of the current branch"""
    name = __git("rev-parse", "--abbrev-ref", "HEAD")
    name = name.rstrip()
    return name


def dir_():
    """Returns the top level directory of the repository"""
    git_dir = __git("rev-parse", "--show-toplevel")
    git_dir = git_dir.rstrip()
    return git_dir


def ref_exists(ref):
    """True if ``ref`` resolves to a commit (local branch, remote-tracking, tag, sha, …)."""
    r = subprocess.run(
        ["git", "rev-parse", "--verify", ref + "^{commit}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return r.returncode == 0


def get_staged_files_absolute():
    """Paths staged for commit (absolute), for comparing with pre-commit's file list."""
    out = subprocess.check_output(
        [
            "git",
            "--no-pager",
            "diff",
            "--staged",
            "--name-only",
            "--diff-filter=ACMRTUXB",
            "-z",
        ]
    )
    names = [n for n in out.decode("UTF-8").split("\0") if n]
    root = dir_()
    return [os.path.join(root, n) for n in names]


def repo_version():
    """
    Determines the version of the repo by using `git describe`

    Returns
    -------
    str
        The full version of the repo in the format 'v#.#.#{a|b|rc}'
    """
    return __git("describe", "--tags", "--abbrev=0")


def repo_version_major_minor():
    """
    Determines the version of the repo using `git describe` and returns only
    the major and minor portion

    Returns
    -------
    str
        The partial version of the repo in the format '{major}.{minor}'
    """

    full_repo_version = repo_version()

    match = re.match(r"^v?(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?", full_repo_version)

    if match is None:
        print(
            "   [DEBUG] Could not determine repo major minor version. "
            f"Full repo version: {full_repo_version}."
        )
        return None

    out_version = match.group("major")

    if match.group("minor"):
        out_version += "." + match.group("minor")

    return out_version


def uncommitted_files():
    """
    Returns a list of all changed files that are not yet committed. This
    means both untracked/unstaged as well as uncommitted files too.
    """
    files = __git("status", "-u", "-s")
    ret = []
    for f in files.splitlines():
        f = f.strip(" ")
        f = re.sub(r"\s+", " ", f)
        tmp = f.split(" ", 1)
        # only consider staged files or uncommitted files
        # in other words, ignore untracked files
        if tmp[0] == "M" or tmp[0] == "A":
            ret.append(tmp[1])
    return ret


def changed_files_between(base_ref, new_ref):
    """
    Returns a list of files changed between base_ref and new_ref
    """
    files = __gitdiff("--name-only", "--ignore-submodules", f"{base_ref}..{new_ref}")
    return [ln for ln in files.splitlines() if ln]


def changed_files_in_ref_range(old_ref, new_ref, absolute_path=False):
    """
    Files changed between old_ref and new_ref using git's three-dot diff, with two-dot
    fallback — same rules as pre-commit's get_changed_files().
    """
    try:
        files = __gitdiff("--name-only", "--ignore-submodules", f"{old_ref}...{new_ref}")
    except CalledProcessError:
        files = __gitdiff("--name-only", "--ignore-submodules", f"{old_ref}..{new_ref}")
    lines = [ln for ln in files.split("\n") if ln]
    if absolute_path:
        git_dir = dir_()
        return [os.path.join(git_dir, fn) for fn in lines]
    return lines


def changes_in_file_between(file, b1, b2, filter=None):  # noqa: A002
    """Filters the changed lines to a file between the branches b1 and b2"""
    # Diff directly without checkout - git diff works on refs without modifying working tree
    diffs = __gitdiff("--ignore-submodules", "-w", "--minimal", "-U0", f"{b1}...{b2}", "--", file)
    return [line for line in diffs.splitlines() if (filter is None or filter(line))]


def modified_files(target=None, absolute_path=False):
    """
    If ``target`` is passed, list files changed between that ref and ``HEAD``.

    Uses :func:`changed_files_in_ref_range` (three-dot ``REF...HEAD``, with
    two-dot fallback inside that helper). With no ``target``, uses
    :func:`uncommitted_files` (staged changes only; see that function).
    """
    if not target:
        all_files = uncommitted_files()
    else:
        all_files = changed_files_in_ref_range(target, "HEAD", absolute_path=False)

    if absolute_path:
        git_dir = dir_()
        return [os.path.join(git_dir, fn) for fn in all_files]
    return all_files
