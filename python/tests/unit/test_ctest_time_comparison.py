# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ctest_time_comparison.py"
_spec = importlib.util.spec_from_file_location("ctest_time_comparison", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _run(tmp_path, monkeypatch, capsys, *args):
    cost = tmp_path / "CTestCost"
    cost.write_text("slow 100\nfast 20\n")
    monkeypatch.setattr(sys, "argv", ["ctest_time_comparison.py", str(cost), *args])
    _mod.main()
    return capsys.readouterr().out


def test_less_uses_numeric_order(tmp_path, monkeypatch, capsys):
    # "100" < "20" as text. 100 is not less than 20.
    out = _run(tmp_path, monkeypatch, capsys, "slow", "LESS", "fast")
    assert "does not match" in out


def test_smaller_cost_is_less(tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, monkeypatch, capsys, "fast", "LESS", "slow")
    assert "matches expectations" in out
