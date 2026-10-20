# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise testing bundled in an installed Core wheel without GPU Python packages."""

from pathlib import Path

import holoscan as hs
import holoscan.testing as testing


class Scale(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def compute(self, context):
        self.out.emit(self.inp.receive_data() * 2)


def main():
    with testing.OperatorTestHarness(Scale) as harness:
        harness.push_many("in", [1, 2, 3])
        result = harness.run()
    assert result.values("out") == [2, 4, 6]
    package = Path(testing.__file__).parent
    assert (package / "py.typed").is_file()
    assert (package / "__init__.pyi").is_file()
    print("Installed holoscan.testing scalar smoke passed")


if __name__ == "__main__":
    main()
