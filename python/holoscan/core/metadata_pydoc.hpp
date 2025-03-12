/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_CORE_METADATA_PYDOC_HPP
#define HOLOSCAN_CORE_METADATA_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace MetadataPolicy {

PYDOC(MetadataPolicy, R"doc(
Enum to define the policy for handling behavior of MetadataDictionary::set when a key already
exists.

The supported policies are:

- `MetadataPolicy.REJECT`: Reject the new value if the key already exists
- `MetadataPolicy.UPDATE`: Replace existing value with the new one if the key already exists
- `MetadataPolicy.INPLACE_UPDATE`: Update the value stored within an existing MetadataObject in-place
  if the key already exists (in contrast to UPDATE which always replaces the existing MetadataObject
  with a new one).
- `MetadataPolicy.RAISE`: Raise an exception if the key already exists
)doc")
}  // namespace MetadataPolicy

namespace MetadataDictionary {

PYDOC(MetadataDictionary, R"doc(
Class representing a holoscan metadata dictionary.
)doc")

PYDOC(has_key, R"doc(
Determine if an item with the given key exists in the dictionary.

Parameters
----------
key : str
    The key to check for in the dictionary.

Returns
-------
bool
    True if the key exists in the dictionary, False otherwise.
)doc")

PYDOC(keys, R"doc(
Get a list of the metadata keys in the dictionary.

Returns
-------
List[str]
    A list of the keys in the dictionary.
)doc")

PYDOC(pop, R"doc(
Pop the specified item from the dictionary.

Parameters
----------
key : str
    The key to pop from the dictionary.
default : object, optional
    The value to return if the key is not found in the dictionary. If not provided, a KeyError will
    be raised if the specified key does not exist.

Returns
-------
value : object
    The value stored in the dictionary with the given key.
)doc")

PYDOC(items, R"doc(
Returns a list of (key, value) tuples for all items in the dictionary.

Returns
-------
items : List[Tuple[str, object]]
    A list of (key, value) tuples for all items in the dictionary.
)doc")

PYDOC(type_dict, R"doc(
Returns a list of dictionary of C++ `std::type_info` names corresponding to the values.

Returns
-------
type_dict : Dict[str, str]
    The keys will match those of this MetadataDictionary while the values are the C++ type names
    corresponding to the values. These type names are mainly relevant for the items stored as C++
    types. All items with values that are Python objects, will have the name
    `typeid(GILGuardedPythonObject).name()`.
)doc")

PYDOC(get, R"doc(
Get the item with the given key from the dictionary.

Returns
-------
object
    The value stored in the dictionary with the given key.
)doc")

PYDOC(set, R"doc(
Store the given value in the dictionary with the given key.

Parameters
----------
key : str
    The key to store the value under.
value : object
    The value to set. By default the Python object is directly stored. If the metadata will be sent
    to a downstream operator that wraps a C++ operator, it may be desirable to instead cast the
    data to a C++ type. This can be done by setting `cast_to_cpp` to True.
dtype : numpy.dtype, optional
    When `cast_to_cpp` is True, the `dtype` argument can be used to indicate what numeric type
    the values should be cast to. If not provided, the default C++ type will be `double` for a
    Python `float` and `int64_t` for a Python `int`.
cast_to_cpp : bool, optional
    If True, the Python object will be converted to a corresponding C++ type, if possible. If
    False, the Python object will be stored directly. The types that can be cast are `str`, `bool`
    and various floating point and integer types. Iterables or sequences with uniform element type
    will become a std::vector of the contained type.
)doc")

PYDOC(size, R"doc(
Get the size of the metadata dictionary.

Returns
-------
size : int
    The number of items in the dictionary.
)doc")

PYDOC(erase, R"doc(
Remove the item with the given key from the dictionary.

Parameters
----------
key : str
    The key to check for in the dictionary.

Returns
-------
bool
    True if the key was found and removed, False otherwise.
)doc")

PYDOC(clear, R"doc(
Clear all items from the dictionary
)doc")

PYDOC(insert, R"doc(
Insert items from another MetadataDictionary into this dictionary.

Parameters
----------
other : MetadataDictionary
    Insert items from other into this dictionary. If a key already exists in this dictionary, the
    value will not be updated.
)doc")

PYDOC(merge, R"doc(
Merge items from another MetadataDictionary into this dictionary.

Parameters
----------
other : MetadataDictionary
    Merge items from `other` into this dictionary. If a key already exists in this dictionary, the
    value will not be updated. Any items inserted into this dictionary will be removed from `other`.
)doc")

PYDOC(swap, R"doc(
Swap the contents of this MetadataDictionary with another one.

Parameters
----------
other : MetadataDictionary
    The metadata dictionary to swap contents with.
)doc")

PYDOC(update, R"doc(
Update items in this dictionary with items from another MetadataDictionary.

Parameters
----------
other : MetadataDictionary
    Insert items from other into this dictionary. If a key already exists in this dictionary, the
    value will be updated in accordance with this dictionary's metadata policy.
)doc")

PYDOC(policy, R"doc(
Metadata policy property that governs the behavior of the `set` and `update` methods.

The supported policies are:

- `MetadataPolicy.REJECT`: Reject the new value if the key already exists
- `MetadataPolicy.UPDATE`: Replace existing value with the new one if the key already exists
- `MetadataPolicy.INPLACE_UPDATE`: Update the value stored within an existing MetadataObject in-place
  if the key already exists (in contrast to UPDATE which always replaces the existing MetadataObject
  with a new one).
- `MetadataPolicy.RAISE`: Raise an exception if the key already exists
)doc")

}  // namespace MetadataDictionary

}  // namespace holoscan::doc

#endif /* HOLOSCAN_CORE_METADATA_PYDOC_HPP */
