"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

from collections.abc import Sequence

import numpy as np
import pytest

from holoscan.core import MetadataDictionary, MetadataPolicy


@pytest.mark.parametrize("name", ["REJECT", "UPDATE", "RAISE"])
def test_policy_type(name):
    assert hasattr(MetadataPolicy, name)


class TestMetadataDictionary:
    def test_init(self, capfd):
        d = MetadataDictionary()
        assert len(d) == 0
        assert d.size() == 0

    def test_invalid_init(self):
        # cannot initialize with a Python dict
        with pytest.raises(TypeError):
            MetadataDictionary({"a": 1, "b": 2})

    def test_container_dunder_methods(self):
        d = MetadataDictionary()

        # test __setitem__ and __getitem__
        # Note: __setitem__ always stores values as Python objects.
        d["key1"] = 5
        d["key2"] = "value2"
        pydict = {"a": 1, "b": 2}
        d["key3"] = pydict

        # test __getitem__
        assert isinstance(d["key1"], int)
        assert d["key1"] == 5
        assert isinstance(d["key2"], str)
        assert d["key2"] == "value2"
        assert isinstance(d["key3"], dict)
        assert d["key3"] == pydict
        with pytest.raises(KeyError):
            d["key4"]

        # updating the pydict dictionary also affects the stored value in MetadataDictionary
        pydict["c"] = [1, 2, 3]
        assert len(d["key3"]) == 3
        assert d["key3"] == pydict

        # test __len__
        assert len(d) == 3 == d.size()

        # test __contains__
        assert not d.has_key("key4")
        assert "key4" not in d
        assert "key1" in d
        assert "key2" in d
        assert "key3" in d

        d.erase("key2")
        assert len(d) == 2
        assert "key2" not in d

        # test __delitem__
        del d["key1"]
        assert len(d) == 1
        assert "key1" not in d

    def test_pop(self):
        d = MetadataDictionary()

        # test __setitem__ and __getitem__
        # Note: __setitem__ always stores values as Python objects.
        val2 = "value2"
        d["key1"] = 5
        d["key2"] = val2
        popped = d.pop("key2")
        assert popped == val2
        assert "key2" not in d
        assert len(d) == 1

        with pytest.raises(KeyError):
            d.pop("key2")

        popped = d.pop("key2", 10)
        assert len(d) == 1
        assert popped == 10

        popped = d.pop("key2", None)
        assert len(d) == 1
        assert popped is None

    def test_clear(self):
        d = MetadataDictionary()

        d["key1"] = 1
        d["key2"] = 2
        d["key3"] = 3
        assert len(d) == 3
        d.clear()
        assert len(d) == 0

    def test_keys(self):
        d = MetadataDictionary()

        d["k2"] = 2
        d["k1"] = 1
        d["k3"] = 3
        # underlying storage is unordered_map, so sort keys to compare
        assert sorted(d.keys()) == sorted(["k2", "k1", "k3"])

    @pytest.mark.parametrize(
        "method, policy",
        [
            ("insert", None),
            ("merge", None),
            ("update", MetadataPolicy.REJECT),
            ("update", MetadataPolicy.UPDATE),
            ("update", None),  # policy is RAISE if not otherwise specified
            ("update", MetadataPolicy.RAISE),
        ],
    )
    def test_insert_merge_update_swap(self, method, policy):
        d = MetadataDictionary()
        d["key1"] = 1
        d["key2"] = 2
        d["key3"] = 3
        assert len(d) == 3

        d2 = MetadataDictionary()
        d2["key1"] = 10
        d2["key4"] = 40
        d2["key5"] = 50
        assert len(d2) == 3

        if policy is not None:
            d.policy = policy
            d2.policy = policy

        if method == "insert":
            d.insert(d2)
            assert len(d) == 5
            assert len(d2) == 3
            assert d["key1"] == 1  # kept the original value
            assert "key5" in d
        elif method == "merge":
            d.merge(d2)
            assert len(d) == 5
            assert len(d2) == 1  # merged items are no longer in
            assert d["key1"] == 1  # kept the original value
            assert "key5" in d
        elif method == "update":
            if policy is None or policy == MetadataPolicy.RAISE:
                with pytest.raises(RuntimeError):
                    d.update(d2)
            else:
                d.update(d2)
                assert len(d) == 5
                assert len(d2) == 3
                if policy == MetadataPolicy.REJECT:
                    assert d["key1"] == 1  # kept the original value
                else:
                    assert d["key1"] == 10  # updated to the new value
                assert "key5" in d

        # swap contents
        size1 = len(d)
        size2 = len(d2)
        d2.swap(d)
        assert len(d2) == size1
        assert len(d) == size2

    def test_item_type_names(self):
        d = MetadataDictionary()
        d.set("key1", 1, cast_to_cpp=True)
        d.set("key2", 1.5, cast_to_cpp=True)
        d["key3"] = dict(a=1, b=2)
        type_dict = d.type_dict()
        assert len(type_dict) == 3
        assert isinstance(type_dict, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in type_dict.items())

    def test_items(self):
        d = MetadataDictionary()
        d["key1"] = 1
        d["key2"] = [2, 3]
        d["key3"] = dict(a=1, b=2)
        assert len(d) == 3

        count = 0
        for k, v in d.items():
            count += 1
            if k == "key1":
                assert v == 1
            elif k == "key2":
                assert v == [2, 3]
            elif k == "key3":
                assert v == dict(a=1, b=2)
        assert count == 3
        assert len(d) == 3

    @pytest.mark.parametrize(
        "dtype",
        [
            np.dtype("bool"),
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
            np.dtype("float32"),
            np.dtype("float64"),
            np.dtype("complex64"),
            np.dtype("complex128"),
        ],
    )
    @pytest.mark.parametrize("cast_to_cpp", [True, False])
    def test_dtype(self, dtype, cast_to_cpp):
        d = MetadataDictionary()
        value = 5
        d.set("x", value, dtype=dtype, cast_to_cpp=cast_to_cpp)

        if cast_to_cpp:
            x = d["x"]
            dtype = np.dtype(dtype)
            if dtype.kind == "b":
                assert x == bool(value)
            else:
                assert x == value

            if dtype.kind == "b":
                assert isinstance(x, bool)
            elif dtype.kind in "iu":
                assert isinstance(x, int)
            elif dtype.kind in "efg":
                assert isinstance(x, float)
            elif dtype.kind in "FG":
                assert isinstance(x, complex)
        else:
            # with cast_to_cpp = False, dtype has no effect. The Python int object is stored.
            x = d["x"]
            assert isinstance(x, int)
            assert x == value

    @pytest.mark.parametrize("value", [(1, 2, 3), [1, 2, 3], {1, 2, 3}, range(1, 4)])
    @pytest.mark.parametrize("cast_to_cpp", [True, False])
    def test_iterable(self, value, cast_to_cpp):
        d = MetadataDictionary()
        d.set("x", value, cast_to_cpp=cast_to_cpp)

        if cast_to_cpp:
            x = d["x"]
            assert isinstance(x, list)
            assert all(isinstance(x[i], int) for i in range(len(value)))
            assert x == [1, 2, 3]

        else:
            # with cast_to_cpp = False, dtype has no effect. The Python object is stored.
            x = d["x"]
            assert isinstance(x, type(value))
            assert x == value
            assert x is value

    @pytest.mark.parametrize(
        "value",
        [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1, 2, 3], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.complex64),
        ],
    )
    @pytest.mark.parametrize("cast_to_cpp", [True, False])
    def test_numpy_array_values(self, value, cast_to_cpp):
        d = MetadataDictionary()
        d.set("x", value, cast_to_cpp=cast_to_cpp)

        if cast_to_cpp:
            x = d["x"]
            assert isinstance(x, list)
            if value.dtype.kind in "iu":
                assert all(isinstance(x[i], int) for i in range(len(value)))
            elif value.dtype.kind in "fg":
                assert all(isinstance(x[i], float) for i in range(len(value)))
            elif value.dtype.kind in "FG":
                assert all(isinstance(x[i], complex) for i in range(len(value)))
            assert x == [1, 2, 3]
        else:
            # with cast_to_cpp = False, dtype has no effect. The Python object is stored.
            x = d["x"]
            assert isinstance(x, type(value))
            np.testing.assert_array_equal(x, value)
            assert x is value

    @pytest.mark.parametrize(
        "value",
        [
            "abc",
            ("abc", "def", "ghi"),
            (("a", "b", "c"), ["d", "e"], {"f", "g", "h", "i"}),
        ],
    )
    @pytest.mark.parametrize("cast_to_cpp", [True, False])
    def test_string_vector(self, value, cast_to_cpp):
        d = MetadataDictionary()
        d.set("x", value, cast_to_cpp=cast_to_cpp)

        if cast_to_cpp:
            x = d["x"]
            if isinstance(value, str):
                assert isinstance(x, str)
                assert x == value
            elif isinstance(value, Sequence) and isinstance(value[0], str):
                # stored as vector<string> in C++ -> becomes list[str] in Python
                assert isinstance(x, list)
                assert all(isinstance(x[i], str) for i in range(len(value)))
                assert x == list(value)
            elif isinstance(value, Sequence) and isinstance(value[0], Sequence):
                # stored as vector<vector<string>> in C++ -> becomes list[list[str]] in Python
                assert isinstance(x, list)
                assert all(isinstance(x[i], list) for i in range(len(value)))
                assert x == list(list(v) for v in value)
        else:
            # with cast_to_cpp = False, dtype has no effect. The Python object is stored.
            x = d["x"]
            assert isinstance(x, type(value))
            assert x == value
            assert x is value
