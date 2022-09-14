# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import sys
import re
import os
import argparse


IncludeRegex = re.compile(r"\s*#include\s*(\S+)")
RemoveComments = re.compile(r"//.*")

exclusion_regex_list = [
    re.compile(r".*_deps.*"),
    re.compile(r".*third_?party.*"),
    re.compile(r".*\.cache.*"),
    re.compile(r".*public/install*"),
    re.compile(r".*public/build*"),
]


def parse_args():
    argparser = argparse.ArgumentParser(
        "Checks for a consistent '#include' syntax")
    argparser.add_argument("--regex", type=str,
                           default=r"[.](cu|cuh|h|hpp|hxx|cpp)$",
                           help="Regex string to filter in sources")
    argparser.add_argument("dirs", type=str, nargs="*",
                           help="List of dirs where to find sources")
    argparser.add_argument("--include",
                           dest="include",
                           action="append",
                           required=False,
                           default=["include"],
                           help=("Specify include paths (default: ['include']). "
                                 "Can be specified multiple times."))
    argparser.add_argument("-v", "--verbose",
                           dest="verbose",
                           action="store_true",
                           help="verbose listing of files checked")
    argparser.add_argument("--allow",
                           dest="allow",
                           action="append",
                           required=False,
                           default=["common", "gxf", "gxf_extensions", "holoscan", "holoviz"],
                           help=("Allow prefixes specified (default: ['gxf', 'holoscan', 'holoviz']). "
                                 "Can be specified multiple times."))
    args = argparser.parse_args()
    args.regex_compiled = re.compile(args.regex)
    return args


def list_all_source_file(file_regex, srcdirs):
    all_files = []
    for srcdir in srcdirs:
        for root, dirs, files in os.walk(srcdir):
            for f in files:
                if not any(re.search(exr, root) for exr in exclusion_regex_list) and re.search(file_regex, f):
                    src = os.path.join(root, f)
                    all_files.append(src)
    return all_files


def check_includes_in(src, include_folder_list, allow_prefix_set):
    errs = []
    curr_dir = os.path.dirname(src)
    prefixes = [curr_dir]
    prefixes.extend(include_folder_list)
    for line_number, line in enumerate(open(src)):
        line = RemoveComments.sub("", line)
        match = IncludeRegex.search(line)
        if match is None:
            continue
        val = match.group(1)
        inc_file = val[1:-1]  # strip out " or <

        include_exists = any(
            map(os.path.exists, (os.path.join(p, inc_file) for p in prefixes)))

        line_num = line_number + 1
        if val[0] == "\"" and not include_exists:
            if inc_file.split("/")[0] not in allow_prefix_set:
                errs.append("Line:%d use #include <...>" % line_num)
        elif val[0] == "<" and include_exists:
            if inc_file.split("/")[0] not in allow_prefix_set:
                errs.append("Line:%d use #include \"...\"" % line_num)
    return errs


def main():
    args = parse_args()
    all_files = list_all_source_file(args.regex_compiled, args.dirs)
    include_folder_list = args.include
    allow_prefix_set = set(args.allow)
    all_errs = {}
    for f in all_files:
        if args.verbose:
            print(f"checking: {f}")
        errs = check_includes_in(f, include_folder_list, allow_prefix_set)
        if len(errs) > 0:
            all_errs[f] = errs
    if len(all_errs) == 0:
        print("include-check PASSED")
    else:
        print("include-check FAILED! See below for errors...")
        for f, errs in all_errs.items():
            print("File: %s" % f)
            for e in errs:
                print("  %s" % e)
        sys.exit(-1)


if __name__ == "__main__":
    main()
