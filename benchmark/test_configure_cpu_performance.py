# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("configure_cpu_performance.sh")


def write_value(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")


def run_script(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "HOLOSCAN_CPU_SYSFS_ROOT": str(root)}
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        check=check,
        capture_output=True,
        env=environment,
        text=True,
    )


def make_sysfs(root: Path) -> dict[str, Path]:
    paths = {
        "c1e": root / "cpu0/cpuidle/state2/disable",
        "c6": root / "cpu1/cpuidle/state3/disable",
        "c1": root / "cpu2/cpuidle/state1/disable",
        "governor": root / "cpufreq/policy0/scaling_governor",
    }
    write_value(paths["c1e"].with_name("name"), "C1E")
    write_value(paths["c1e"].with_name("latency"), "10")
    write_value(paths["c1e"], "0")
    write_value(paths["c6"].with_name("name"), "CC7")
    write_value(paths["c6"].with_name("latency"), "133")
    write_value(paths["c6"], "1")
    write_value(paths["c1"].with_name("name"), "C1")
    write_value(paths["c1"].with_name("latency"), "0")
    write_value(paths["c1"], "0")
    write_value(paths["governor"], "schedutil")
    write_value(paths["governor"].with_name("scaling_available_governors"), "schedutil performance")
    return paths


def write_cpu_topology(
    root: Path,
    cpu: int,
    package: int,
    core: int,
    *,
    online: bool = True,
    thread_siblings: str | None = None,
) -> None:
    cpu_dir = root / f"cpu{cpu}"
    write_value(cpu_dir / "topology/physical_package_id", str(package))
    write_value(cpu_dir / "topology/core_id", str(core))
    if thread_siblings is not None:
        write_value(cpu_dir / "topology/thread_siblings_list", thread_siblings)
    if cpu != 0:
        write_value(cpu_dir / "online", "1" if online else "0")


def test_recommend_selects_distinct_online_cores_from_one_socket(tmp_path: Path) -> None:
    # Key setup: expose sibling threads, one offline core, and two sockets.
    sysfs = tmp_path / "sysfs"
    write_cpu_topology(sysfs, 0, 0, 0)
    write_cpu_topology(sysfs, 1, 0, 1)
    write_cpu_topology(sysfs, 2, 0, 0)
    write_cpu_topology(sysfs, 3, 0, 2, online=False)
    write_cpu_topology(sysfs, 4, 1, 0)
    write_cpu_topology(sysfs, 5, 1, 1)
    write_cpu_topology(sysfs, 6, 1, 2)

    completed = run_script(sysfs, "recommend", "3")

    # Key assertion: the fuller socket wins and contributes one thread per physical core.
    assert completed.stdout.strip() == "4,5,6"
    assert "3 distinct physical cores on socket 1" in completed.stderr


def test_recommend_uses_thread_siblings_when_core_ids_are_ambiguous(tmp_path: Path) -> None:
    # Thor exposes singleton sibling sets even when every CPU reports package/core (0, 0).
    sysfs = tmp_path / "sysfs"
    for cpu in range(8):
        write_cpu_topology(sysfs, cpu, 0, 0, thread_siblings=str(cpu))

    completed = run_script(sysfs, "recommend", "8")

    assert completed.stdout.strip() == "0,1,2,3,4,5,6,7"
    assert "8 distinct physical cores on socket 0" in completed.stderr


def test_recommend_returns_available_cores_when_socket_has_fewer_than_requested(
    tmp_path: Path,
) -> None:
    # Key setup: model the six-core Ampere topology from the reduced-hardware report.
    sysfs = tmp_path / "sysfs"
    for cpu in range(6):
        write_cpu_topology(sysfs, cpu, 0, cpu, thread_siblings=str(cpu))

    completed = run_script(sysfs, "recommend", "8")

    # Key assertion: the command remains usable by docs-core-runtime and explains the reduction.
    assert completed.stdout.strip() == "0,1,2,3,4,5"
    assert "Requested 8 cores, but socket 0 exposes only 6" in completed.stderr
    assert "Selected 6 distinct physical cores on socket 0" in completed.stderr


def test_recommend_selects_only_isolated_physical_cores(tmp_path: Path) -> None:
    # Key setup: mirror an 18-core SMT host with physical cores 2-9 isolated.
    sysfs = tmp_path / "sysfs"
    write_value(sysfs / "isolated", "2-9,20-27")
    for cpu in range(36):
        core = cpu % 18
        write_cpu_topology(
            sysfs,
            cpu,
            0,
            core,
            thread_siblings=f"{core},{core + 18}",
        )

    completed = run_script(sysfs, "recommend", "8")

    # Key assertion: housekeeping CPUs 0-1 are not mixed into the recommendation.
    assert completed.stdout.strip() == "2,3,4,5,6,7,8,9"
    assert "Selected 8 distinct isolated physical cores on socket 0" in completed.stderr


def test_recommend_does_not_fill_a_short_isolated_set_with_housekeeping_cpus(
    tmp_path: Path,
) -> None:
    # Key setup: expose six physical cores but reserve only three for pinned work.
    sysfs = tmp_path / "sysfs"
    write_value(sysfs / "isolated", "2-4")
    for cpu in range(6):
        write_cpu_topology(sysfs, cpu, 0, cpu, thread_siblings=str(cpu))

    completed = run_script(sysfs, "recommend", "8")

    # Key assertion: reduced-core planning receives only the isolated CPU set.
    assert completed.stdout.strip() == "2,3,4"
    assert "isolated CPU set on socket 0 exposes only 3 distinct physical cores" in (
        completed.stderr
    )
    assert "Selected 3 distinct isolated physical cores on socket 0" in completed.stderr


def test_apply_and_restore_preserve_exact_cpu_state(tmp_path: Path) -> None:
    # Key setup: expose two high-latency states, one zero-latency state, and one governor policy.
    sysfs = tmp_path / "sysfs"
    paths = make_sysfs(sysfs)
    state_file = tmp_path / "cpu.state"

    run_script(sysfs, "apply", str(state_file))

    # Key assertion: only states at or above the latency threshold are disabled.
    assert paths["c1e"].read_text(encoding="utf-8").strip() == "1"
    assert paths["c6"].read_text(encoding="utf-8").strip() == "1"
    assert paths["c1"].read_text(encoding="utf-8").strip() == "0"
    assert paths["governor"].read_text(encoding="utf-8").strip() == "performance"
    saved = state_file.read_text(encoding="utf-8")
    assert f"idle\t{paths['c1e']}\t0" in saved
    assert f"idle\t{paths['c6']}\t1" in saved
    assert f"governor\t{paths['governor']}\tschedutil" in saved

    run_script(sysfs, "restore", str(state_file))

    # Key assertion: restore uses saved values, including the initially-disabled named state.
    assert paths["c1e"].read_text(encoding="utf-8").strip() == "0"
    assert paths["c6"].read_text(encoding="utf-8").strip() == "1"
    assert paths["governor"].read_text(encoding="utf-8").strip() == "schedutil"


def test_apply_rejects_overwriting_an_existing_state_file(tmp_path: Path) -> None:
    # Key setup: retain a prior state file whose provenance must not be destroyed.
    sysfs = tmp_path / "sysfs"
    make_sysfs(sysfs)
    state_file = tmp_path / "cpu.state"
    state_file.write_text("existing\n", encoding="utf-8")

    completed = run_script(sysfs, "apply", str(state_file), check=False)

    # Key assertion: a second apply fails before changing the retained state file.
    assert completed.returncode != 0
    assert "state file already exists" in completed.stderr
    assert state_file.read_text(encoding="utf-8") == "existing\n"


def test_restore_rejects_paths_outside_cpu_sysfs(tmp_path: Path) -> None:
    # Key setup: forge an otherwise well-formed state entry targeting a writable external file.
    sysfs = tmp_path / "sysfs"
    make_sysfs(sysfs)
    external = tmp_path / "external"
    write_value(external, "unchanged")
    state_file = tmp_path / "cpu.state"
    state_file.write_text(
        f"# holoscan-benchmark-cpu-state-v1\ngovernor\t{external}\tperformance\n",
        encoding="utf-8",
    )

    completed = run_script(sysfs, "restore", str(state_file), check=False)

    # Key assertion: restore rejects the path before privileged replay can modify it.
    assert completed.returncode != 0
    assert "outside the CPU sysfs root" in completed.stderr
    assert external.read_text(encoding="utf-8").strip() == "unchanged"
