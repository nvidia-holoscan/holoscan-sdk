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

import decimal
from collections.abc import Generator, Sequence

import numpy as np
import pytest

from holoscan.conditions import BooleanCondition
from holoscan.core import (
    Arg,
    ArgContainerType,
    ArgElementType,
    ArgList,
    arglist_to_kwargs,
    kwargs_to_arglist,
)
from holoscan.core._core import arg_to_py_object, py_object_to_arg
from holoscan.operators import HolovizOp
from holoscan.resources import UnboundedAllocator


@pytest.mark.parametrize(
    "value, container_type, element_type",
    [
        # Python float
        (5.0, ArgContainerType.NATIVE, ArgElementType.FLOAT64),
        # strings and sequences of strings
        ("abcd", ArgContainerType.NATIVE, ArgElementType.STRING),
        # tuple, list set also work
        (("ab", "cd"), ArgContainerType.VECTOR, ArgElementType.STRING),
        (["ab", "cd"], ArgContainerType.VECTOR, ArgElementType.STRING),
        ({"ab", "cd"}, ArgContainerType.VECTOR, ArgElementType.STRING),
        # Python int gets cast to signed 64-bit int
        (5, ArgContainerType.NATIVE, ArgElementType.INT64),
        # range works
        (range(8), ArgContainerType.VECTOR, ArgElementType.INT64),
        # generator works
        ((a + 1 for a in range(3)), ArgContainerType.VECTOR, ArgElementType.INT64),
        ((3, 5, -1), ArgContainerType.VECTOR, ArgElementType.INT64),
        (False, ArgContainerType.NATIVE, ArgElementType.BOOLEAN),
        (True, ArgContainerType.NATIVE, ArgElementType.BOOLEAN),
        ((False, True), ArgContainerType.VECTOR, ArgElementType.BOOLEAN),
        # numpy dtypes get cast to the respective C++ types
        (np.uint8(3), ArgContainerType.NATIVE, ArgElementType.UNSIGNED8),
        (np.uint16(3), ArgContainerType.NATIVE, ArgElementType.UNSIGNED16),
        (np.uint32(3), ArgContainerType.NATIVE, ArgElementType.UNSIGNED32),
        (np.uint64(3), ArgContainerType.NATIVE, ArgElementType.UNSIGNED64),
        (np.int8(3), ArgContainerType.NATIVE, ArgElementType.INT8),
        (np.int16(3), ArgContainerType.NATIVE, ArgElementType.INT16),
        (np.int32(3), ArgContainerType.NATIVE, ArgElementType.INT32),
        (np.int64(3), ArgContainerType.NATIVE, ArgElementType.INT64),
        (np.float16(3), ArgContainerType.NATIVE, ArgElementType.FLOAT32),
        (np.float32(3), ArgContainerType.NATIVE, ArgElementType.FLOAT32),
        (np.float64(3), ArgContainerType.NATIVE, ArgElementType.FLOAT64),
        (np.bool_(3), ArgContainerType.NATIVE, ArgElementType.BOOLEAN),
        (
            np.arange(5, dtype=np.float16),
            ArgContainerType.VECTOR,
            ArgElementType.FLOAT32,  # float16 will be promoted to float32
        ),
        (
            np.arange(5, dtype=np.float32),
            ArgContainerType.VECTOR,
            ArgElementType.FLOAT32,
        ),
        (
            np.arange(5, dtype=np.uint8),
            ArgContainerType.VECTOR,
            ArgElementType.UNSIGNED8,
        ),
    ],
)
def test_py_object_to_arg(value, container_type, element_type):
    arg = py_object_to_arg(value, name="alpha")
    assert arg.name == "alpha"
    assert arg.has_value
    assert arg.arg_type.container_type == container_type
    assert arg.arg_type.element_type == element_type

    # check expected round-trip conversion via arg_to_py_object
    if np.isscalar(value):
        value_dtype = getattr(value, "dtype", None)
        numpy_scalar = value_dtype is not None
        if isinstance(object, bool):
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, bool)
            assert round_trip_value == value
        elif isinstance(object, int) or (numpy_scalar and value_dtype.kind in "iu"):
            # all integer types become a Python int
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, int)
            assert round_trip_value == value
        elif isinstance(object, float) or (numpy_scalar and value_dtype.kind in "f"):
            # all floating point types become a Python float
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, float)
            assert round_trip_value == value
        elif isinstance(object, str):
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, str)
            assert round_trip_value == value
    else:
        if isinstance(value, Generator):
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, list)
            return

        val_list = list(value)
        v0 = val_list[0]
        value_dtype = getattr(v0, "dtype", None)
        numpy_scalar = value_dtype is not None
        if isinstance(v0, bool):
            print("bool case testing")
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, list)
            assert isinstance(round_trip_value[0], bool)
            assert all(r == v for r, v in zip(round_trip_value, val_list))
        elif isinstance(v0, int) or (numpy_scalar and value_dtype.kind in "iu"):
            print("int case testing")
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, list)
            assert isinstance(round_trip_value[0], int)
            assert all(r == v for r, v in zip(round_trip_value, val_list))
        elif isinstance(v0, float) or (numpy_scalar and value_dtype.kind in "f"):
            print("float case testing")
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, list)
            assert isinstance(round_trip_value[0], float)
            assert all(r == v for r, v in zip(round_trip_value, val_list))
        elif isinstance(v0, str):
            print("str case testing")
            round_trip_value = arg_to_py_object(arg)
            assert isinstance(round_trip_value, list)
            assert isinstance(round_trip_value[0], str)
            assert all(r == v for r, v in zip(round_trip_value, val_list))


@pytest.mark.parametrize(
    "value, container_type, element_type",
    [
        # list of lists
        (
            [[3.0, 3.0, 3.0], [4.0, 4.0, 5.0]],
            ArgContainerType.VECTOR,
            ArgElementType.FLOAT64,
        ),
        ([[3, 3, 3], [4, 4, 5]], ArgContainerType.VECTOR, ArgElementType.INT64),
        (
            [[False, True], [True, True]],
            ArgContainerType.VECTOR,
            ArgElementType.BOOLEAN,
        ),
        # list of tuple
        ([(3, 3, 3), (4, 4, 5)], ArgContainerType.VECTOR, ArgElementType.INT64),
        # sequence of sequence of mixed type
        (([3, 3, 3], (4, 4, 5)), ArgContainerType.VECTOR, ArgElementType.INT64),
        # 2d array cases
        (
            np.ones((5, 8), dtype=bool),
            ArgContainerType.VECTOR,
            ArgElementType.BOOLEAN,
        ),
        (
            np.arange(5 * 8, dtype=np.float32).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.FLOAT32,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.float64).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.FLOAT64,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.int8).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.INT8,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.int16).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.INT16,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.int32).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.INT32,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.int64).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.INT64,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.uint8).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.UNSIGNED8,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.uint16).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.UNSIGNED16,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.uint32).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.UNSIGNED32,
        ),  # noqa
        (
            np.arange(5 * 8, dtype=np.uint64).reshape(5, 8),
            ArgContainerType.VECTOR,
            ArgElementType.UNSIGNED64,
        ),  # noqa
        (
            [["str1", "string2"], ["s3", "st4"]],
            ArgContainerType.VECTOR,
            ArgElementType.STRING,
        ),
    ],
)
def test_py_object_to_arg_2d(value, container_type, element_type):
    arg = py_object_to_arg(value, name="alpha")
    assert arg.name == "alpha"
    assert arg.has_value
    assert arg.arg_type.container_type == container_type
    assert arg.arg_type.element_type == element_type

    if isinstance(value, np.ndarray):
        val_list = value.tolist()
    else:
        val_list = [list(v) for v in value]
    v0 = val_list[0][0]
    value_dtype = getattr(v0, "dtype", None)
    numpy_scalar = value_dtype is not None
    if isinstance(v0, bool):
        print("bool case testing")
        round_trip_value = arg_to_py_object(arg)
        assert isinstance(round_trip_value, list)
        assert isinstance(round_trip_value[0][0], bool)
        assert all(r == v for r, v in zip(round_trip_value, val_list))
    elif isinstance(v0, int) or (numpy_scalar and value_dtype.kind in "iu"):
        print("int case testing")
        round_trip_value = arg_to_py_object(arg)
        assert isinstance(round_trip_value, list)
        assert isinstance(round_trip_value[0][0], int)
        assert all(r == v for r, v in zip(round_trip_value, val_list))
    elif isinstance(v0, float) or (numpy_scalar and value_dtype.kind in "f"):
        print("float case testing")
        round_trip_value = arg_to_py_object(arg)
        assert isinstance(round_trip_value, list)
        assert isinstance(round_trip_value[0][0], float)
        assert all(r == v for r, v in zip(round_trip_value, val_list))
    elif isinstance(v0, str):
        print("str case testing")
        round_trip_value = arg_to_py_object(arg)
        assert isinstance(round_trip_value, list)
        assert isinstance(round_trip_value[0][0], str)
        assert all(r == v for r, v in zip(round_trip_value, val_list))


@pytest.mark.parametrize("as_vector", [False, True])
def test_condition_to_arg(app, as_vector):
    cond = BooleanCondition(app)
    if not as_vector:
        arg = py_object_to_arg(cond)
        assert arg.arg_type.container_type == ArgContainerType.NATIVE
    else:
        arg = py_object_to_arg([cond] * 4)
        assert arg.arg_type.container_type == ArgContainerType.VECTOR
    assert arg.arg_type.element_type == ArgElementType.CONDITION


@pytest.mark.parametrize("as_vector", [False, True])
def test_resource_to_arg(app, as_vector):
    pool = UnboundedAllocator(app)
    if not as_vector:
        arg = py_object_to_arg(pool)
        assert arg.arg_type.container_type == ArgContainerType.NATIVE
    else:
        arg = py_object_to_arg([pool] * 4)
        assert arg.arg_type.container_type == ArgContainerType.VECTOR
    assert arg.arg_type.element_type == ArgElementType.RESOURCE


@pytest.mark.skipif(not hasattr(np, "float128"), reason="float128 dtype not available")
def test_unsupported_numpy_dtype_raises():
    """Test that unknown dtypes raise an error."""
    with pytest.raises(RuntimeError):
        py_object_to_arg(np.float128(3))


def test_unknown_scalar_numeric_type():
    """Test that unknown scalar types get cast to double."""
    arg = py_object_to_arg(decimal.Decimal(3))
    assert arg.arg_type.container_type == ArgContainerType.NATIVE
    assert arg.arg_type.element_type == ArgElementType.FLOAT64


@pytest.mark.parametrize(
    "value",
    [
        # complex values aren't supported
        3 + 5j,
        np.ones((3,), dtype=np.complex64),
        # 0-length sequences aren't supported
        [],
        [[]],
        # nested sequences of depth > 2 aren't supported
        [[[1, 2]]],
        # arbitrary Python objectds aren't supported
        dict(a=6),
    ],
)
def test_py_object_to_arg_error(value):
    with pytest.raises(RuntimeError):
        py_object_to_arg(value)


@pytest.mark.parametrize(
    "value",
    [
        # arrays with ndim > 2 are unsupported
        np.zeros((3, 3, 3)),
        np.zeros((3, 3, 3, 3)),
        # sequences with > 2 levels of nesting are unsupported
        [[[1, 2], [3, 4]]],
        (((1, 2), (3, 4)),),
        # sequence with inconsistent element types
        (1, 2.0),
        # nested sequence with inconsistent element types
        [[1, 2.0], [3.0, 4]],
    ],
)
def test_py_object_to_arg_invalid_sequence(value):
    # Currently, only 1d or 2d arrays or nested lists of <= 2 levels are
    # supported
    with pytest.raises(RuntimeError):
        py_object_to_arg(value)


@pytest.mark.parametrize(
    "value",
    [
        (1, 2.0, 3),  # can't mix types in a sequence
        ["abc", 3],  # can't mix types in a sequence
    ],
)
def test_py_object_to_arg_invalid(value):
    with pytest.raises(RuntimeError):
        py_object_to_arg(value)


def test_arglist_to_kwargs():
    kwargs = dict(
        alpha=5,
        beta=[3, 2, 1],
        gamma="abcd",
        delta=["abc", "def"],
        epsilon=np.arange(12).reshape(4, 3),
        verbose=False,
        zeta=(False, True, False),
        eta=[[0.7, 0.2, 0.3, 1.0], [0.2, 0.7, 0.01, 1.0]],
    )
    arglist = kwargs_to_arglist(**kwargs)
    assert isinstance(arglist, ArgList)
    assert arglist.size == len(kwargs)

    round_trip_kwargs = arglist_to_kwargs(arglist)
    assert isinstance(round_trip_kwargs, dict)
    for k, v in kwargs.items():
        round_trip_v = round_trip_kwargs[k]
        if isinstance(v, np.ndarray):
            assert round_trip_v == v.tolist()
        elif isinstance(v, str):
            round_trip_v == v
        elif isinstance(v, Sequence):
            round_trip_v == list(v)
        else:
            round_trip_v == v


def test_arglist_from_kwargs():
    kwargs = dict(
        alpha=5.0,
        beta=np.float32(3),
        offsets=(1, 0, 3),
        names=["abc", "def"],
        verbose=False,
        flags=np.array([0, 1, 1, 0, 1], dtype=bool),
    )
    arglist = kwargs_to_arglist(**kwargs)
    assert isinstance(arglist, ArgList)
    assert arglist.size == len(kwargs)
    assert len(arglist.args) == len(kwargs)
    assert all(isinstance(arg, Arg) for arg in arglist.args)

    # also test __repr__ method of ArgList here
    rstr = repr(arglist)
    assert (
        rstr
        == """name: arglist
args:
  - name: alpha
    type: double
    value: 5
  - name: beta
    type: float
    value: 3
  - name: offsets
    type: std::vector<int64_t>
    value:
      - 1
      - 0
      - 3
  - name: names
    type: std::vector<std::string>
    value:
      - abc
      - def
  - name: verbose
    type: bool
    value: false
  - name: flags
    type: std::vector<bool>
    value:
      - false
      - true
      - true
      - false
      - true"""
    )


def test_arg_to_py_object_unsupported(fragment):
    op = HolovizOp(fragment)
    # retrieve an Arg containing unsupported type IOSpec* from HolovizOp
    iospec_arg = [arg for arg in op.args if arg.name == "render_buffer_input"][0]
    # no converter defined for this type so RuntimeError will be raised
    with pytest.raises(RuntimeError):
        arg_to_py_object(iospec_arg)
