# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the supported native install that omits the Sensor I/O Python component."""

import holoscan as hs


def missing_sensor_value_message(name: str) -> str:
    """Return the optional-component diagnostic for one absent sensor value."""
    try:
        getattr(hs.schema, name)
    except AttributeError as error:
        return str(error)
    raise AssertionError(f"holoscan.schema.{name} unexpectedly available")


def main() -> None:
    """Check that Core schemas remain usable and optional sensor values explain their absence."""
    assert hs.schema.Time(time_ns=13).time_ns == 13
    assert hs.payload.schema.Time.name == "holoscan.schema.Time"

    for name in ("Image", "Imu"):
        assert name not in hs.schema.__all__
        assert name not in dir(hs.schema)
        assert "holoscan-sensor-io-python_libs" in missing_sensor_value_message(name)


if __name__ == "__main__":
    main()
