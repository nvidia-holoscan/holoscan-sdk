# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy must be installed to run this example.")


class SignalGeneratorOp(Operator):
    """Generate a time-varying impulse.

    Transmits an array of zeros with a single non-zero entry of a
    specified `height`. The position of the non-zero entry shifts
    to the right (in a periodic fashion) each time `compute` is
    called.

    Parameters
    ----------
    fragment : holoscan.core.Fragment
        The Fragment (or Application) the operator belongs to.
    height : number
        The height of the signal impulse.
    size : number
        The total number of samples in the generated 1d signal.
    dtype : numpy.dtype or str
        The data type of the generated signal.
    """

    def __init__(self, fragment, *args, height=1, size=10, dtype=np.int32, **kwargs):
        self.count = 0
        self.height = height
        self.dtype = dtype
        self.size = size
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("signal")

    def compute(self, op_input, op_output, context):
        # single sample wide impulse at a time-varying position
        signal = np.zeros((self.size,), dtype=self.dtype)
        signal[self.count % signal.size] = self.height
        self.count += 1

        op_output.emit(signal, "signal")


class ConvolveOp(Operator):
    """Apply convolution to a tensor.

    Convolves an input signal with a "boxcar" (i.e. "rect") kernel.

    Parameters
    ----------
    fragment : holoscan.core.Fragment
        The Fragment (or Application) the operator belongs to.
    width : number
        The width of the boxcar kernel used in the convolution.
    unit_area : bool, optional
        Whether or not to normalize the convolution kernel to unit area.
        If False, all samples have implitude one and the dtype of the
        kernel will match that of the signal. When True the sum over
        the kernel is one and a 32-bit floating point data type is used
        for the kernel.
    """

    def __init__(self, fragment, *args, width=4, unit_area=False, **kwargs):
        self.count = 0
        self.width = width
        self.unit_area = unit_area
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("signal_in")
        spec.output("signal_out")

    def compute(self, op_input, op_output, context):
        signal = op_input.receive("signal_in")
        assert isinstance(signal, np.ndarray)

        if self.unit_area:
            kernel = np.full((self.width,), 1 / self.width, dtype=np.float32)
        else:
            kernel = np.ones((self.width,), dtype=signal.dtype)

        convolved = np.convolve(signal, kernel, mode="same")

        op_output.emit(convolved, "signal_out")


class PrintSignalOp(Operator):
    """Print the received signal to the terminal."""

    def setup(self, spec: OperatorSpec):
        spec.input("signal")

    def compute(self, op_input, op_output, context):
        signal = op_input.receive("signal")
        print(signal)


class ConvolveApp(Application):
    """Minimal signal processing application.

    Generates a time-varying impulse, convolves it with a boxcar kernel, and
    prints the result to the terminal.

    A `CountCondition` is applied to the generate to terminate execution
    after a specific number of steps.
    """

    def compose(self):
        signal_generator = SignalGeneratorOp(
            self,
            CountCondition(self, count=24),
            name="generator",
            height=1,
            size=20,
            dtype="int32",
        )
        convolver = ConvolveOp(
            self,
            name="conv",
            width=4,
            unit_area=False,
        )
        printer = PrintSignalOp(self, name="printer")
        self.add_flow(signal_generator, convolver)
        self.add_flow(convolver, printer)


if __name__ == "__main__":
    set_log_level(LogLevel.WARN)

    app = ConvolveApp()
    app.run()
