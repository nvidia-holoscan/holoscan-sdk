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
) -> None:
    cpu_dir = root / f"cpu{cpu}"
    write_value(cpu_dir / "topology/physical_package_id", str(package))
    write_value(cpu_dir / "topology/core_id", str(core))
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
