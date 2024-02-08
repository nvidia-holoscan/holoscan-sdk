..
   Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

.. _serializationExtension:

SerializationExtension
----------------------

Extension for serializing messages.

* UUID: :code:`bc573c2f-89b3-d4b0-8061-2da8b11fe79a`
* Version: :code:`2.0.0`
* Author: :code:`NVIDIA`
* License: :code:`LICENSE`

Interfaces
~~~~~~~~~~~~

nvidia::gxf::ComponentSerializer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interface for serializing components.

* Component ID: :code:`8c76a828-2177-1484-f841-d39c3fa47613`
* Base Type: :code:`nvidia::gxf::Component`
* Defined in: :code:`gxf/serialization/component_serializer.hpp`

Components
~~~~~~~~~~~~

nvidia::gxf::EntityRecorder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Serializes incoming messages and writes them to a file.

* Component ID: :code:`9d5955c7-8fda-22c7-f18f-ea5e2d195be9`
* Base Type: :code:`nvidia::gxf::Codelet`

Parameters
++++++++++++

**receiver**

Receiver channel to log.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_HANDLE`
* Handle Type: :code:`nvidia::gxf::Receiver`

|

**serializers**

List of component serializers to serialize entities.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::ComponentSerializer>>`

|

**directory**

Directory path for storing files.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_STRING`

|

**basename**

User specified file name without extension.

* Flags: :code:`GXF_PARAMETER_FLAGS_OPTIONAL`
* Type: :code:`GXF_PARAMETER_TYPE_STRING`

|

**flush_on_tick**

Flushes output buffer on every tick when true.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_BOOL`

nvidia::gxf::EntityReplayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

De-serializes and publishes messages from a file.

* Component ID: :code:`fe827c12-d360-c63c-8094-32b9244d83b6`
* Base Type: :code:`nvidia::gxf::Codelet`

Parameters
++++++++++++

**transmitter**

Transmitter channel for replaying entities.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_HANDLE`
* Handle Type: :code:`nvidia::gxf::Transmitter`

|

**serializers**

List of component serializers to serialize entities.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::ComponentSerializer>>`

|

**directory**

Directory path for storing files.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_STRING`

|

**batch_size**

Number of entities to read and publish for one tick.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_UINT64`

|

**ignore_corrupted_entities**

If an entity could not be de-serialized, it is ignored by default; otherwise a failure is generated.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_BOOL`

nvidia::gxf::StdComponentSerializer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Serializer for Timestamp and Tensor components.

* Component ID: :code:`c0e6b36c-39ac-50ac-ce8d-702e18d8bff7`
* Base Type: :code:`nvidia::gxf::ComponentSerializer`

Parameters
++++++++++++

**allocator**

Memory allocator for tensor components.

* Flags: :code:`GXF_PARAMETER_FLAGS_OPTIONAL`
* Type: :code:`GXF_PARAMETER_TYPE_HANDLE`
* Handle Type: :code:`nvidia::gxf::Allocator`
