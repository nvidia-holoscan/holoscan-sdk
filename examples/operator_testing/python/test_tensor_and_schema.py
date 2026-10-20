# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning captures from the installed Core and optional testing packages."""

import ctypes
from typing import cast

import numpy as np

import holoscan as hs
from holoscan.schema.image import Image
from holoscan.schema.string import String
from holoscan.testing import OperatorTestHarness, SchemaSnapshot


class ImageSource(hs.Operator):
    out = hs.Output("out", Image)

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
            ),
            hs.tensor.byte_span(4),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        request = hs.tensor.LoanRequest(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1))
        with self.out.allocate_tensors(Image(width=4, height=1), {"data": request}) as loan:
            (ctypes.c_uint8 * 4).from_address(loan.tensors["data"].data)[:] = (10, 20, 30, 40)


class StringEcho(hs.Operator):
    inp = hs.Input("in", String)
    out = hs.Output("out", String)

    def compute(self, context):
        self.out.emit(self.inp.receive_data())


def test_schema_value_copy():
    source = String(data="original")
    with OperatorTestHarness(StringEcho) as harness:
        harness.push("in", source)
        source.data = "later"
        result = harness.run()
    assert cast(String, result.value("out")).data == "original"


def test_schema_tensor_snapshot_survives_close():
    with OperatorTestHarness(ImageSource) as harness:
        result = harness.run()
    snapshot = result.snapshot("out")
    assert isinstance(snapshot, SchemaSnapshot)
    metadata = cast(Image, snapshot.metadata)
    assert metadata.width == 4
    assert snapshot.has_tensor("data")
    assert snapshot.tensor("data").bytes == bytes((10, 20, 30, 40))


class TensorSum(hs.Operator):
    inp = hs.Input("in", hs.payload.Tensor)
    out = hs.Output("out", hs.payload.uint32)

    def compute(self, context):
        tensor = cast(hs.Tensor, self.inp.receive_data())
        values = (ctypes.c_uint8 * tensor.nbytes).from_address(tensor.data)
        self.out.emit(sum(values))


def test_stage_host_tensor():
    with OperatorTestHarness(TensorSum) as harness:
        harness.push("in", hs.Tensor.as_tensor(np.array([1, 2, 3], dtype=np.uint8)))
        assert harness.run().value("out") == 6
