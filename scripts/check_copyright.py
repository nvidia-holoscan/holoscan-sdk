"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""  # noqa: E501

# This file is modified from the RAPIDS RAFT project which is under the
# Apache 2.0 license.
# (https://github.com/rapidsai/raft/blob/branch-22.08/ci/checks/copyright.py)

# Our latest guideline is captured at
# https://confluence.nvidia.com/display/LEG/Apache+2.0

import argparse
import datetime
import itertools
import os
import re
import subprocess
import sys
from subprocess import CalledProcessError

SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))

# Add the scripts dir for gitutils
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, "../scripts")))

# Now import gitutils. Ignore flake8 error here since there is no other way to
# set up imports
import gitutils  # noqa: E402

FilesToCheck = [
    re.compile(r"[.](cmake|cpp|css|cu|cuh|h|hpp|sh|pxd|py|pyx|yaml)$"),
    re.compile(r"CMakeLists[.]txt$"),
    re.compile(r"Dockerfile$"),
    re.compile(r"[.]dockerfile$"),
    re.compile(r"CMakeLists_standalone[.]txt$"),
    re.compile(r"setup[.]cfg$"),
    re.compile(r"[.]flake8[.]cython$"),
    re.compile(r"meta[.]yaml$"),
]
ExemptFiles = ["FindEigen3.cmake"]

# this will break starting at year 10000, which is probably OK :)
CheckSimple = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) *(\d{4}),? NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)
CheckDouble = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) *(\d{4})-(\d{4}),? NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved."
)


def check_this_file(f):
    # This check covers things like symlinks which point to files that DNE
    if not (os.path.exists(f)):
        return False
    if gitutils and gitutils.is_file_empty(f):
        return False
    for exempt in ExemptFiles:
        if exempt.search(f):
            return False
    return any(checker.search(f) for checker in FilesToCheck)


def get_copyright_years(line):
    res = CheckSimple.search(line)
    if res:
        return (int(res.group(1)), int(res.group(1)))
    res = CheckDouble.search(line)
    if res:
        return (int(res.group(1)), int(res.group(2)))
    return (None, None)


def replace_current_year(line, start, end):
    # first turn a simple regex into double (if applicable). then update years
    res = CheckSimple.sub(
        r"SPDX-FileCopyrightText: Copyright (c) \1-\1 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.",
        line,
    )
    res = CheckDouble.sub(
        f"SPDX-FileCopyrightText: Copyright (c) {start}-{end} NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.",
        res,
    )
    return res


def check_copyright(f, update_current_year):
    """
    Checks for copyright headers and their years
    """
    errs = []
    this_year = datetime.datetime.now().year
    line_num = 0
    cr_found = False
    year_matched = False
    with open(f, encoding="utf-8") as fp:
        lines = fp.readlines()
    for line in lines:
        line_num += 1
        start, end = get_copyright_years(line)
        if start is None:
            continue
        cr_found = True
        if start > end:
            e = [
                f,
                line_num,
                "First year after second year in the copyright header (manual fix required)",
                None,
            ]
            errs.append(e)
        if this_year < start or this_year > end:
            e = [f, line_num, "Current year not included in the copyright header", None]
            if this_year < start:
                e[-1] = replace_current_year(line, this_year, end)
            if this_year > end:
                e[-1] = replace_current_year(line, start, this_year)
            errs.append(e)
        else:
            year_matched = True
    # copyright header itself not found
    if not cr_found:
        e = [
            f,
            0,
            "Copyright header missing or formatted incorrectly (manual fix required)",
            None,
        ]
        errs.append(e)
    # even if the year matches a copyright header, make the check pass
    if year_matched:
        errs = []

    if update_current_year:
        errs_update = [x for x in errs if x[-1] is not None]
        if len(errs_update) > 0:
            print(
                "File: {}. Changing line(s) {}".format(
                    f, ", ".join(str(x[1]) for x in errs if x[-1] is not None)
                )
            )
            for _, line_num, __, replacement in errs_update:
                lines[line_num - 1] = replacement
            with open(f, "w", encoding="utf-8") as out_file:
                for new_line in lines:
                    out_file.write(new_line)
        errs = [x for x in errs if x[-1] is None]

    return errs


def get_all_files_under_dir(root):
    ret_list = []
    for dirpath, _, filenames in os.walk(root):
        ret_list.extend([os.path.join(dirpath, fn) for fn in filenames])
    return ret_list


def _normalize_repo_path(path):
    """Absolute, resolved path for stable set comparisons with git output."""
    return os.path.normcase(os.path.normpath(os.path.abspath(os.path.realpath(path))))


def expand_input_paths(paths):
    """
    Each path may be a file or a directory. Directories are walked recursively.
    Missing paths are skipped (e.g. deleted files still listed by a caller).
    """
    out = []
    for p in paths:
        ap = _normalize_repo_path(p)
        if not os.path.exists(ap):
            continue
        if os.path.isfile(ap):
            out.append(ap)
        elif os.path.isdir(ap):
            out.extend(get_all_files_under_dir(ap))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in out:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def _intersect_with_changed_files(all_files, changed_abs_paths, had_input_paths):
    """
    Narrow all_files to those in changed_abs_paths, or use changed_abs_paths if no paths
    were passed in.
    """
    changed_set = {_normalize_repo_path(p) for p in changed_abs_paths}
    if had_input_paths:
        return [f for f in all_files if _normalize_repo_path(f) in changed_set]
    seen = set()
    out = []
    for p in changed_abs_paths:
        n = _normalize_repo_path(p)
        if n not in seen:
            seen.add(n)
            out.append(p)
    return out


def _explicit_intersect_base_ref(args):
    """Base ref for three-dot intersect: --intersect-since-ref or HOLOSCAN_COPYRIGHT_BASE_REF."""
    if args.intersect_since_ref:
        return args.intersect_since_ref
    return os.environ.get("HOLOSCAN_COPYRIGHT_BASE_REF")


def _passed_paths_are_subset_of_staged(all_files):
    """
    True when every path to check is staged (typical ``git commit`` hook). False for
    ``pre-commit run --all-files``, which includes many unstaged tracked paths.
    """
    if not all_files:
        return True
    try:
        staged = gitutils.get_staged_files_absolute()
    except (CalledProcessError, OSError, subprocess.SubprocessError):
        return True
    staged_set = {_normalize_repo_path(f) for f in staged}
    passed_set = {_normalize_repo_path(f) for f in all_files}
    return passed_set <= staged_set


def _auto_intersect_base_ref():
    """
    When a wide file list is passed and no explicit base is set: intersect with
    ``REF...HEAD`` only if ``origin/main`` / ``main`` or ``origin/release/latest`` /
    ``release/latest`` exists. If neither line exists, return None (check all paths as-is).
    On branch ``main`` only main-line refs are considered; on ``release/latest`` only release.
    On other branches, try main line first, then release.
    """
    try:
        br = gitutils.branch().strip()
    except (CalledProcessError, OSError, subprocess.SubprocessError):
        return None

    def main_line_ref():
        for ref in ("origin/main", "main"):
            if gitutils.ref_exists(ref):
                return ref
        return None

    def release_line_ref():
        for ref in ("origin/release/latest", "release/latest"):
            if gitutils.ref_exists(ref):
                return ref
        return None

    if br == "main":
        return main_line_ref()
    if br == "release/latest":
        return release_line_ref()
    return main_line_ref() or release_line_ref()


def check_copyright_main():
    """Checks copyright headers on given paths (files or directories)."""
    ret_val = 0
    global ExemptFiles

    argparser = argparse.ArgumentParser(
        description=(
            "NVIDIA SPDX copyright check. Pre-commit passes filenames when pass_filenames is "
            "set. --git-modified-only limits to git changes (ref..HEAD two-dot with ref; no ref: "
            "staged per gitutils). On a wide file list (e.g. --all-files), paths are intersected "
            "with REF...HEAD when origin/main or main exists, else origin/release/latest or "
            "release/latest; if neither exists, all passed paths are checked. Normal commits only "
            "pass staged files, so auto-intersect is skipped. Override with --intersect-since-ref "
            "or HOLOSCAN_COPYRIGHT_BASE_REF."
        )
    )
    argparser.add_argument(
        "--update-current-year",
        dest="update_current_year",
        action="store_true",
        required=False,
        help="If set, update the current year if a header is already present and well formatted.",
    )
    argparser.add_argument(
        "--git-modified-only",
        dest="git_modified_only",
        action="store",
        type=str,
        nargs="?",
        default=None,
        const="no-target",
        required=False,
        help="Restrict to files changed in git (see gitutils.modified_files). "
        "With explicit paths, intersects with that set. "
        "Without a ref, uses staged-only listing; with REF, uses git three-dot "
        "REF...HEAD (two-dot fallback).",
    )
    argparser.add_argument(
        "--intersect-since-ref",
        dest="intersect_since_ref",
        default=None,
        metavar="REF",
        help="Intersect explicit paths with git three-dot diff REF...HEAD (overrides env).",
    )
    argparser.add_argument(
        "--exclude",
        dest="exclude",
        action="append",
        required=False,
        default=[],
        help=("Exclude the paths specified (regexp). Can be specified multiple times."),
    )
    argparser.add_argument(
        "paths",
        nargs="*",
        default=[],
        help="Files and/or directories to consider (directories are scanned recursively).",
    )

    args = argparser.parse_args()
    try:
        ExemptFiles = ExemptFiles + [pathName for pathName in args.exclude]
        ExemptFiles = [re.compile(file) for file in ExemptFiles]
    except re.error as reException:
        print("Regular expression error:")
        print(reException)
        return 1

    had_input_paths = bool(args.paths)
    all_files = []
    if args.paths:
        all_files = expand_input_paths(args.paths)

    if args.git_modified_only:
        target_branch = None
        if args.git_modified_only != "no-target":
            target_branch = args.git_modified_only
        modified_files = gitutils.modified_files(target_branch, True)
        modified_set = {_normalize_repo_path(p) for p in modified_files}
        if args.paths:
            all_files = [f for f in all_files if _normalize_repo_path(f) in modified_set]
        else:
            all_files = modified_files
    else:
        base_ref = _explicit_intersect_base_ref(args)
        if not base_ref and had_input_paths and not _passed_paths_are_subset_of_staged(all_files):
            base_ref = _auto_intersect_base_ref()
        if base_ref:
            changed = gitutils.changed_files_in_ref_range(base_ref, "HEAD", absolute_path=True)
            all_files = _intersect_with_changed_files(all_files, changed, had_input_paths)

    files = [f for f in all_files if check_this_file(f)]
    errors = tuple(itertools.chain(*[check_copyright(f, args.update_current_year) for f in files]))
    if errors:
        print("Copyright headers incomplete in some of the files!")
        for file_name, line_no, err_msg, _ in errors:
            print(f"  {file_name}:{line_no} Issue: {err_msg}")
        print("")
        n_fixable = sum(1 for e in errors if e[-1] is not None)
        if n_fixable > 0:
            print(
                f"You can run `python3 {' '.join(sys.argv)} --update-current-year` to fix "
                f"{n_fixable} of these errors."
            )
        ret_val = 1
    else:
        print(f"✅ All copyright headers are complete. {len(files)} file(s) successfully checked.")

    return ret_val


if __name__ == "__main__":
    sys.exit(check_copyright_main())
