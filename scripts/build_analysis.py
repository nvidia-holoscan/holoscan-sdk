#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure where SDK build time goes.

Subcommands read an existing Ninja build directory. ``ninja-log``, ``deps``, ``duplicates``,
``closure``, and ``trace-report`` are pure analysis and run anywhere. ``header-cost``,
``ftime-trace``, and ``ftime-report`` invoke the compilers recorded in ``compile_commands.json`` and
therefore run inside the SDK build container
(``./run launch <build-dir> --run-cmd "python3 ../scripts/build_analysis.py ..."``).
See scripts/README.md.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parent.parent
MAX_JOBS = 8
CORE_OBJECT_DIRS = (
    "CMakeFiles/core.dir/",
    "CMakeFiles/holoscan_compiler_private_objects.dir/",
    "CMakeFiles/holoscan_lifecycle_plan_private_objects.dir/",
    "CMakeFiles/holoscan_runtime_private_objects.dir/",
)
SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu")


def _jobs(value: str) -> int:
    jobs = int(value)
    if not 1 <= jobs <= MAX_JOBS:
        raise argparse.ArgumentTypeError(f"jobs must be between 1 and {MAX_JOBS}")
    return jobs


def _emit(result: dict[str, Any], json_path: Path | None) -> None:
    if json_path is not None:
        json_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


def area(output: str) -> str:
    """Map a Ninja output path to a reporting area."""
    if output.startswith(CORE_OBJECT_DIRS) or "libholoscan_core.so" in output:
        return "libholoscan_core"
    if "_deps/" in output or output.startswith("rust/"):
        return "third-party"
    first, _, rest = output.partition("/")
    if first == "tests":
        return "tests:" + rest.split("/", 1)[0]
    if first == "modules":
        return "module:" + rest.split("/", 1)[0]
    if first == "python":
        return "python"
    if first in ("benchmark", "examples"):
        return first
    if first == "CMakeFiles":
        return "sdk-lib:" + rest.split("/", 1)[0].removesuffix(".dir")
    return "other:" + first


# ---------------------------------------------------------------------------------------------
# ninja-log
# ---------------------------------------------------------------------------------------------


def read_ninja_log(build_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {output: (start_ms, end_ms)} keeping the last record per output."""
    records: dict[str, tuple[int, int]] = {}
    with (build_dir / ".ninja_log").open() as log:
        log.readline()
        for line in log:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 5:
                records[fields[3]] = (int(fields[0]), int(fields[1]))
    return records


def _split_ninja(text: str) -> list[str]:
    tokens, current, index = [], "", 0
    while index < len(text):
        char = text[index]
        if char == "$" and index + 1 < len(text):
            current += text[index + 1]
            index += 2
            continue
        if char == " ":
            if current:
                tokens.append(current)
            current = ""
        else:
            current += char
        index += 1
    if current:
        tokens.append(current)
    return tokens


def read_build_edges(build_dir: Path) -> list[tuple[list[str], str, list[str]]]:
    """Return (outputs, rule, inputs) for every build statement, following include/subninja."""
    statements: list[str] = []

    def read(path: Path) -> None:
        for raw in path.read_text().replace("$\n", " ").split("\n"):
            if raw.startswith(("include ", "subninja ")):
                included = Path(raw.split(None, 1)[1].strip())
                read(included if included.is_absolute() else build_dir / included)
            elif raw.startswith("build "):
                statements.append(raw[len("build ") :])

    read(build_dir / "build.ninja")
    edges = []
    for body in statements:
        index = 0
        while index < len(body) and body[index] != ":":
            index += 2 if body[index] == "$" else 1
        outputs = [token for token in _split_ninja(body[:index]) if token != "|"]
        rest = _split_ninja(body[index + 1 :])
        inputs = [token for token in rest[1:] if token not in ("|", "||", "|@")]
        edges.append((outputs, rest[0] if rest else "", inputs))
    return edges


def ninja_log_report(build_dir: Path) -> dict[str, Any]:
    records = read_ninja_log(build_dir)
    edges = read_build_edges(build_dir)
    steps = []
    for outputs, rule, _ in edges:
        seen = [records[output] for output in outputs if output in records]
        if not seen:
            continue
        start, end = min(s for s, _ in seen), max(e for _, e in seen)
        kind = "compile" if "COMPILER" in rule else "link" if "LINKER" in rule else "other"
        steps.append(
            {
                "output": outputs[0],
                "area": area(outputs[0]),
                "kind": kind,
                "start": start,
                "end": end,
            }
        )
    for step in steps:
        step["seconds"] = (step["end"] - step["start"]) / 1000
    total = sum(step["seconds"] for step in steps)
    wall = (max(s["end"] for s in steps) - min(s["start"] for s in steps)) / 1000

    areas: dict[str, dict[str, Any]] = {}
    for step in steps:
        entry = areas.setdefault(step["area"], {"steps": 0, "seconds": 0.0, "compile": []})
        entry["steps"] += 1
        entry["seconds"] += step["seconds"]
        if step["kind"] == "compile":
            entry["compile"].append(step["seconds"])
    for entry in areas.values():
        values = sorted(entry.pop("compile"))
        entry["compile_units"] = len(values)
        entry["median"] = statistics.median(values) if values else 0.0
        entry["p90"] = values[min(len(values) - 1, int(0.9 * len(values)))] if values else 0.0
        entry["max"] = values[-1] if values else 0.0

    events = sorted([(s["start"], 1) for s in steps] + [(s["end"], -1) for s in steps])
    busy: collections.Counter[int] = collections.Counter()
    running, last = 0, events[0][0]
    for moment, delta in events:
        busy[running] += moment - last
        running, last = running + delta, moment

    producer = {output: i for i, (outputs, _, _) in enumerate(edges) for output in outputs}

    def duration(i: int) -> float:
        seen = [records[o] for o in edges[i][0] if o in records]
        return max((e - s for s, e in seen), default=0) / 1000

    sys.setrecursionlimit(max(10000, 4 * len(edges)))
    memo: dict[int, tuple[float, int | None]] = {}

    def longest(i: int) -> tuple[float, int | None]:
        if i not in memo:
            best, parent = 0.0, None
            for dependency in edges[i][2]:
                j = producer.get(dependency)
                if j is not None and j != i and longest(j)[0] > best:
                    best, parent = longest(j)[0], j
            memo[i] = (best + duration(i), parent)
        return memo[i]

    tail = max(range(len(edges)), key=lambda i: longest(i)[0])
    chain: list[int] = []
    node: int | None = tail
    while node is not None:
        chain.append(node)
        node = longest(node)[1]
    return {
        "cpu_seconds": total,
        "wall_seconds": wall,
        "areas": areas,
        "top_steps": sorted(
            ({k: s[k] for k in ("output", "area", "seconds")} for s in steps),
            key=lambda s: -s["seconds"],
        )[:40],
        "busy_seconds_by_jobs": {str(k): v / 1000 for k, v in sorted(busy.items())},
        "critical_path_seconds": longest(tail)[0],
        "critical_path": [
            {"output": edges[i][0][0], "seconds": duration(i)} for i in reversed(chain)
        ],
    }


def print_ninja_log(result: dict[str, Any]) -> None:
    print(
        f"CPU {result['cpu_seconds']:.0f} s, wall {result['wall_seconds']:.0f} s, "
        f"critical path {result['critical_path_seconds']:.0f} s"
    )
    print("\n| Area | Steps | CPU s | % | Compile units | Median s | p90 s | Max s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, entry in sorted(result["areas"].items(), key=lambda item: -item[1]["seconds"]):
        print(
            f"| {name} | {entry['steps']} | {entry['seconds']:.0f} | "
            f"{100 * entry['seconds'] / result['cpu_seconds']:.1f} | {entry['compile_units']} | "
            f"{entry['median']:.1f} | {entry['p90']:.1f} | {entry['max']:.1f} |"
        )
    print("\nSlowest steps:")
    for step in result["top_steps"][:20]:
        print(f"  {step['seconds']:7.1f}  {step['area']:24} {step['output']}")
    print("\nCritical path (steps >= 0.5 s):")
    for step in result["critical_path"]:
        if step["seconds"] >= 0.5:
            print(f"  {step['seconds']:7.1f}  {step['output']}")


# ---------------------------------------------------------------------------------------------
# deps
# ---------------------------------------------------------------------------------------------


def read_deps(deps_file: Path) -> dict[str, set[str]]:
    """Parse ``ninja -t deps`` output into {output: {dependency, ...}}."""
    deps: dict[str, set[str]] = {}
    current = None
    for line in deps_file.read_text(errors="replace").splitlines():
        if not line.strip():
            current = None
        elif not line.startswith(" "):
            match = re.match(r"^(\S+): #deps", line)
            current = match.group(1) if match else None
            if current:
                deps[current] = set()
        elif current:
            deps[current].add(line.strip())
    return deps


def _normalize(path: str) -> str:
    return path.replace("/workspace/holoscan-sdk/", "").replace(str(SOURCE_ROOT) + "/", "")


def deps_report(
    build_dir: Path, deps_file: Path, headers: list[str], top: int, exclude: str
) -> dict[str, Any]:
    records = read_ninja_log(build_dir)
    seconds = {o: (e - s) / 1000 for o, (s, e) in records.items()}
    excluded = re.compile(exclude) if exclude else None
    deps = {
        output: {_normalize(d) for d in values}
        for output, values in read_deps(deps_file).items()
        if output in seconds
        and output.endswith(".o")
        and not (excluded and excluded.search(output))
    }
    total = sum(seconds[o] for o in deps)
    counts: collections.Counter[str] = collections.Counter()
    cost: collections.Counter[str] = collections.Counter()
    for output, values in deps.items():
        for dependency in values:
            if not dependency.endswith(SOURCE_SUFFIXES):
                counts[dependency] += 1
                cost[dependency] += seconds[output]
    wanted = (
        headers or [h for h, _ in counts.most_common() if h.startswith(("include/", "src/"))][:top]
    )
    rows = []
    for header in wanted:
        key = header if header in counts else f"include/{header}"
        dependents = [o for o, values in deps.items() if key in values]
        by_area = collections.Counter(area(o) for o in dependents)
        rows.append(
            {
                "header": key,
                "objects": len(dependents),
                "cpu_seconds": sum(seconds[o] for o in dependents),
                "rebuild_seconds_j8": max(
                    sum(seconds[o] for o in dependents) / MAX_JOBS,
                    max((seconds[o] for o in dependents), default=0.0),
                ),
                "areas": dict(by_area.most_common(4)),
            }
        )
    return {"objects": len(deps), "compile_cpu_seconds": total, "headers": rows}


def print_deps(result: dict[str, Any]) -> None:
    print(f"{result['objects']} objects, {result['compile_cpu_seconds']:.0f} compile CPU-s")
    print("\n| Header | Dependent objects | CPU s | % of build | Rebuild at -j8 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in result["headers"]:
        print(
            f"| `{row['header']}` | {row['objects']} | {row['cpu_seconds']:.0f} | "
            f"{100 * row['cpu_seconds'] / result['compile_cpu_seconds']:.1f} | "
            f"{row['rebuild_seconds_j8'] / 60:.1f} min |"
        )


# ---------------------------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------------------------


def _compile_database(build_dir: Path) -> list[dict[str, Any]]:
    return json.loads((build_dir / "compile_commands.json").read_text())


def _target_of(output: str) -> str:
    return output.split("CMakeFiles/")[-1].split(".dir")[0]


def duplicates_report(build_dir: Path, prefix: str) -> dict[str, Any]:
    seconds = {o: (e - s) / 1000 for o, (s, e) in read_ninja_log(build_dir).items()}
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in _compile_database(build_dir):
        source = _normalize(entry["file"])
        if not source.startswith(prefix):
            continue
        arguments = shlex.split(entry["command"])
        output = _normalize(entry["output"]).removeprefix(_normalize(str(build_dir)) + "/")
        by_source[source].append(
            {
                "target": _target_of(entry["output"]),
                "flags": sorted(a for a in arguments if a.startswith(("-D", "-f", "-O", "-std"))),
                "seconds": seconds.get(output, 0.0),
            }
        )
    rows = []
    for source, compiles in sorted(by_source.items()):
        if len(compiles) < 2:
            continue
        variants: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
        for compile_ in compiles:
            variants[tuple(compile_["flags"])].append(compile_["target"])
        ordered = sorted(c["seconds"] for c in compiles)
        rows.append(
            {
                "source": source,
                "compiles": len(compiles),
                "flag_variants": len(variants),
                "targets": [sorted(targets) for targets in variants.values()],
                "extra_seconds": sum(ordered[1:]),
            }
        )
    return {
        "prefix": prefix,
        "duplicates": rows,
        "extra_seconds": sum(row["extra_seconds"] for row in rows),
    }


def print_duplicates(result: dict[str, Any]) -> None:
    print(
        f"{len(result['duplicates'])} files under {result['prefix']} compiled more than once; "
        f"{result['extra_seconds']:.0f} extra CPU-s"
    )
    for row in result["duplicates"]:
        groups = " | ".join(",".join(targets) for targets in row["targets"])
        print(
            f"  x{row['compiles']} ({row['flag_variants']} flag variants, "
            f"{row['extra_seconds']:.0f} s extra) {row['source']}: {groups}"
        )


# ---------------------------------------------------------------------------------------------
# closure
# ---------------------------------------------------------------------------------------------

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


def include_graph(include_root: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    edges: dict[str, list[str]] = {}
    lines: dict[str, int] = {}
    for path in include_root.rglob("*"):
        if path.suffix not in (".h", ".hpp"):
            continue
        name = path.relative_to(include_root).as_posix()
        text = path.read_text(errors="replace").splitlines()
        lines[name] = len(text)
        edges[name] = [m.group(1) for line in text if (m := INCLUDE_RE.match(line))]
    return edges, lines


def closure_report(include_root: Path, headers: list[str]) -> dict[str, Any]:
    edges, lines = include_graph(include_root)
    rows = []
    for header in headers:
        seen: set[str] = set()
        stack = [header]
        while stack:
            for dependency in edges.get(stack.pop(), []):
                if dependency in edges and dependency not in seen:
                    seen.add(dependency)
                    stack.append(dependency)
        external = sorted({d for h in seen | {header} for d in edges.get(h, []) if d not in edges})
        rows.append(
            {
                "header": header,
                "lines": lines.get(header, 0),
                "closure_headers": len(seen),
                "closure_lines": sum(lines[h] for h in seen) + lines.get(header, 0),
                "closure": sorted(seen),
                "external": external,
            }
        )
    return {"headers": rows}


def print_closure(result: dict[str, Any]) -> None:
    print("| Header | Lines | Closure headers | Closure lines |")
    print("| --- | ---: | ---: | ---: |")
    for row in result["headers"]:
        print(
            f"| `{row['header']}` | {row['lines']} | {row['closure_headers']} | "
            f"{row['closure_lines']} |"
        )


# ---------------------------------------------------------------------------------------------
# Compiler-driven measurements (run inside the build container)
# ---------------------------------------------------------------------------------------------


def compile_arguments(build_dir: Path, source: str) -> tuple[list[str], str]:
    """Return (arguments without "-o <obj>" and "-c <src>", absolute source) for a source file."""
    for entry in _compile_database(build_dir):
        if _normalize(entry["file"]) == source or entry["file"].endswith("/" + source):
            arguments, skip = [], False
            for argument in shlex.split(entry["command"]):
                if skip:
                    skip = False
                elif argument in ("-o", "-c"):
                    skip = True
                else:
                    arguments.append(argument)
            return arguments, entry["file"]
    raise SystemExit(f"{source}: not found in compile_commands.json")


def _timed(command: list[str]) -> tuple[float, subprocess.CompletedProcess[str]]:
    started = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        process = subprocess.CompletedProcess(command, 127, stdout="", stderr=str(error))
    return time.perf_counter() - started, process


def header_cost(
    build_dir: Path, reference: str, entries: list[str], jobs: int, repeats: int
) -> dict[str, Any]:
    arguments, _ = compile_arguments(build_dir, reference)
    compiler, flags = arguments[0], [a for a in arguments[1:] if not a.endswith("_EXPORTS")]
    work = Path(tempfile.mkdtemp(prefix="header-cost-", dir=build_dir))

    def measure(entry: str) -> dict[str, Any]:
        unit = work / (re.sub(r"[^A-Za-z0-9]+", "_", entry)[:180] + ".cpp")
        unit.write_text("".join(f"#include <{h}>\n" for h in entry.split("+")))
        times = []
        for _ in range(repeats):
            elapsed, process = _timed([compiler, *flags, "-fsyntax-only", str(unit)])
            if process.returncode:
                return {
                    "include": entry,
                    "returncode": process.returncode,
                    "error": process.stderr[-2000:],
                }
            times.append(elapsed)
        _, preprocessed = _timed([compiler, *flags, "-E", "-P", str(unit)])
        _, dependencies = _timed([compiler, *flags, "-M", str(unit)])
        for process in (preprocessed, dependencies):
            if process.returncode:
                return {
                    "include": entry,
                    "returncode": process.returncode,
                    "error": process.stderr[-2000:],
                }
        return {
            "include": entry,
            "seconds": min(times),
            "preprocessed_lines": sum(
                1 for line in preprocessed.stdout.splitlines() if line.strip()
            ),
            "dependencies": len(dependencies.stdout.replace("\\\n", " ").split()) - 2,
        }

    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        rows = list(pool.map(measure, entries))
    return {"reference": reference, "jobs": jobs, "repeats": repeats, "includes": rows}


def print_header_cost(result: dict[str, Any]) -> None:
    print("| Include | Parse s | Preprocessed lines | Dependencies |")
    print("| --- | ---: | ---: | ---: |")
    for row in sorted(result["includes"], key=lambda r: -r.get("seconds", 0)):
        if "error" in row:
            print(f"| `{row['include']}` | error | | |")
        else:
            print(
                f"| `{row['include']}` | {row['seconds']:.2f} | {row['preprocessed_lines']} | "
                f"{row['dependencies']} |"
            )


def ftime_trace(
    build_dir: Path, sources: list[str], out_dir: Path, jobs: int, clang: str, granularity: int
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def trace(source: str) -> dict[str, Any]:
        arguments, absolute = compile_arguments(build_dir, source)
        target = out_dir / (source.replace("/", "__") + ".json")
        elapsed, process = _timed(
            [
                clang,
                *arguments[1:],
                "-c",
                "-O0",
                "-o",
                os.devnull,
                "-Wno-everything",
                f"-ftime-trace-granularity={granularity}",
                f"-ftime-trace={target}",
                absolute,
            ]
        )
        return {
            "source": source,
            "seconds": elapsed,
            "returncode": process.returncode,
            "error": process.stderr[-1000:] if process.returncode else "",
        }

    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        return {"runs": list(pool.map(trace, sources)), "trace_dir": str(out_dir)}


def _strip_template_arguments(name: str) -> str:
    depth, out = 0, []
    for char in name:
        if char == "<":
            depth += 1
            if depth == 1:
                out.append("<>")
        elif char == ">":
            depth -= 1
        elif depth == 0:
            out.append(char)
    return "".join(out)


def trace_report(trace_dir: Path, top: int) -> dict[str, Any]:
    header_time: collections.Counter[str] = collections.Counter()
    header_units: collections.Counter[str] = collections.Counter()
    template_time: collections.Counter[str] = collections.Counter()
    units = []
    for path in sorted(trace_dir.glob("*.json")):
        events = [e for e in json.loads(path.read_text())["traceEvents"] if e.get("ph") == "X"]
        totals = {e["name"]: e["dur"] for e in events if e["name"].startswith("Total ")}
        frontend = totals.get("Total Frontend", 0)
        sources = sorted(
            (e for e in events if e["name"] == "Source"), key=lambda e: (e["ts"], -e["dur"])
        )
        top_level, end, seen = 0, -1, set()
        for event in sources:
            header = _normalize(event["args"]["detail"])
            if header not in seen:
                seen.add(header)
                header_time[header] += event["dur"]
                header_units[header] += 1
            if event["ts"] >= end:
                top_level += event["dur"]
                end = event["ts"] + event["dur"]
        for event in events:
            if event["name"] in ("InstantiateClass", "InstantiateFunction"):
                template_time[_strip_template_arguments(event["args"].get("detail", ""))] += event[
                    "dur"
                ]
        units.append(
            {
                "unit": path.stem.replace("__", "/"),
                "frontend": frontend / 1e6,
                "headers": top_level / 1e6,
            }
        )
    return {
        "units": units,
        "headers": [
            {"header": h, "units": header_units[h], "seconds": t / 1e6}
            for h, t in header_time.most_common(top)
        ],
        "templates": [
            {"template": n, "seconds": t / 1e6} for n, t in template_time.most_common(top)
        ],
    }


def print_trace_report(result: dict[str, Any]) -> None:
    frontend = sum(u["frontend"] for u in result["units"])
    headers = sum(u["headers"] for u in result["units"])
    print(
        f"{len(result['units'])} units, front end {frontend:.0f} s, header parsing {headers:.0f} s "
        f"({100 * headers / frontend if frontend else 0:.0f}%)"
    )
    print("\n| Header (first inclusion, inclusive) | Units | Total s | Avg s |")
    print("| --- | ---: | ---: | ---: |")
    for row in result["headers"]:
        print(
            f"| `{row['header']}` | {row['units']} | {row['seconds']:.0f} | "
            f"{row['seconds'] / row['units']:.2f} |"
        )
    print("\nTemplate families (inclusive; nested entries double count):")
    for row in result["templates"]:
        print(f"  {row['seconds']:8.0f} s  {row['template'][:140]}")


PHASES = (
    "phase setup",
    "phase parsing",
    "phase lang. deferred",
    "phase opt and generate",
    "phase last asm",
    "phase finalize",
)


def ftime_report(build_dir: Path, sources: list[str], jobs: int) -> dict[str, Any]:
    def report(source: str) -> dict[str, Any]:
        arguments, absolute = compile_arguments(build_dir, source)
        elapsed, process = _timed([*arguments, "-ftime-report", "-c", "-o", os.devnull, absolute])
        phases = {}
        for phase in (*PHASES, "template instantiation"):
            pattern = (
                rf"^\s*{re.escape(phase)}\s*:\s*[\d.]+\s*\(\s*\d+%\)\s*[\d.]+\s*\(\s*\d+%\)"
                r"\s*([\d.]+)"
            )
            match = re.search(pattern, process.stderr, re.M)
            phases[phase] = float(match.group(1)) if match else 0.0
        return {
            "source": source,
            "seconds": elapsed,
            "returncode": process.returncode,
            "error": process.stderr[-2000:] if process.returncode else "",
            "phases": phases,
        }

    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        return {"units": list(pool.map(report, sources))}


def print_ftime_report(result: dict[str, Any]) -> None:
    print(
        "| Source | Wall s | Parsing | Deferred | Opt + generate | Template inst. | Front-end % |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for unit in result["units"]:
        if unit["returncode"]:
            print(f"| `{unit['source']}` | error | | | | | |")
            continue
        phases = unit["phases"]
        total = sum(phases[p] for p in PHASES) or 1.0
        frontend = phases["phase parsing"] + phases["phase lang. deferred"]
        print(
            f"| `{unit['source']}` | {unit['seconds']:.1f} | {phases['phase parsing']:.1f} | "
            f"{phases['phase lang. deferred']:.1f} | {phases['phase opt and generate']:.1f} | "
            f"{phases['template instantiation']:.1f} | {100 * frontend / total:.0f} |"
        )


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write machine-readable results here")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser(
        "ninja-log", help="areas, slowest steps, concurrency, critical path"
    )
    command.add_argument("build_dir", type=Path)

    command = commands.add_parser("deps", help="header fan-out and rebuild cost")
    command.add_argument("build_dir", type=Path)
    command.add_argument(
        "--deps-file", type=Path, required=True, help="output of `ninja -C <build_dir> -t deps`"
    )
    command.add_argument(
        "--header",
        action="append",
        default=[],
        help="header relative to the source root or include/ (repeatable)",
    )
    command.add_argument("--top", type=int, default=40)
    command.add_argument(
        "--exclude",
        default="holoscan_core_(private_)?header_self_containment",
        help="regular expression for objects to ignore (default: header self-containment units)",
    )

    command = commands.add_parser("duplicates", help="sources compiled more than once")
    command.add_argument("build_dir", type=Path)
    command.add_argument("--prefix", default="src/")

    command = commands.add_parser("closure", help="static Holoscan include closure of headers")
    command.add_argument("headers", nargs="+", help="for example holoscan/core/graph.hpp")
    command.add_argument("--include-root", type=Path, default=SOURCE_ROOT / "include")

    command = commands.add_parser("header-cost", help="standalone parse cost of includes")
    command.add_argument("build_dir", type=Path)
    command.add_argument("includes", nargs="+", help="header, or several joined with '+'")
    command.add_argument(
        "--reference-source",
        default="src/core/graph.cpp",
        help="compile command whose flags are reused",
    )
    command.add_argument("--jobs", type=_jobs, default=4)
    command.add_argument("--repeats", type=int, default=3)

    command = commands.add_parser("ftime-trace", help="capture clang -ftime-trace for sources")
    command.add_argument("build_dir", type=Path)
    command.add_argument("sources", nargs="+")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--jobs", type=_jobs, default=MAX_JOBS)
    command.add_argument("--clang", default="clang++-18")
    command.add_argument("--granularity", type=int, default=300, help="microseconds")

    command = commands.add_parser("trace-report", help="aggregate captured -ftime-trace files")
    command.add_argument("trace_dir", type=Path)
    command.add_argument("--top", type=int, default=40)

    command = commands.add_parser("ftime-report", help="GCC front-end/back-end phase split")
    command.add_argument("build_dir", type=Path)
    command.add_argument("sources", nargs="+")
    command.add_argument("--jobs", type=_jobs, default=4)

    args = parser.parse_args(argv)
    if args.command == "ninja-log":
        result = ninja_log_report(args.build_dir)
        print_ninja_log(result)
    elif args.command == "deps":
        result = deps_report(args.build_dir, args.deps_file, args.header, args.top, args.exclude)
        print_deps(result)
    elif args.command == "duplicates":
        result = duplicates_report(args.build_dir, args.prefix)
        print_duplicates(result)
    elif args.command == "closure":
        result = closure_report(args.include_root, args.headers)
        print_closure(result)
    elif args.command == "header-cost":
        result = header_cost(
            args.build_dir, args.reference_source, args.includes, args.jobs, args.repeats
        )
        print_header_cost(result)
    elif args.command == "ftime-trace":
        result = ftime_trace(
            args.build_dir, args.sources, args.out, args.jobs, args.clang, args.granularity
        )
        failed = [run["source"] for run in result["runs"] if run["returncode"]]
        print(f"traced {len(result['runs']) - len(failed)} units into {args.out}; failed: {failed}")
    elif args.command == "trace-report":
        result = trace_report(args.trace_dir, args.top)
        print_trace_report(result)
    else:
        result = ftime_report(args.build_dir, args.sources, args.jobs)
        print_ftime_report(result)
    _emit(result, args.json)
    measurement_rows = {
        "header-cost": ("includes", "include"),
        "ftime-trace": ("runs", "source"),
        "ftime-report": ("units", "source"),
    }
    if args.command in measurement_rows:
        rows_key, identity_key = measurement_rows[args.command]
        failed = [row for row in result[rows_key] if row.get("returncode", 0)]
        for row in failed:
            print(
                f"{row[identity_key]}: measurement failed (exit {row['returncode']})",
                file=sys.stderr,
            )
            if row["error"]:
                print(row["error"].rstrip(), file=sys.stderr)
        return int(bool(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
