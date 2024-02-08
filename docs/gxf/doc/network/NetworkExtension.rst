..
   Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

.. _networkExtension:

NetworkExtension
----------------------

Extension for communications external to a computation graph.

* UUID: :code:`f50665e5-ade2-f71b-de2a-2380614b1725`
* Version: :code:`1.0.0`
* Author: :code:`NVIDIA`
* License: :code:`LICENSE`

Interfaces
~~~~~~~~~~~~

Components
~~~~~~~~~~~~

nvidia::gxf::TcpClient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Codelet that functions as a client in a TCP connection.

* Component ID: :code:`9d5955c7-8fda-22c7-f18f-ea5e2d195be9`
* Base Type: :code:`nvidia::gxf::Codelet`

Parameters
++++++++++++

**receivers**

List of receivers to receive entities from.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::Receiver>>`

|

**transmitters**

List of transmitters to publish entities to.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::Transmitter>>`

|

**serializers**

List of component serializers to serialize and de-serialize entities.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::ComponentSerializer>>`

|

**address**

Address of TCP server.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_STRING`

|

**port**

Port of TCP server.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_INT32`

|


**timeout_ms**

Time in milliseconds to wait before retrying connection to TCP server.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_UINT64`

|

**maximum_attempts**

Maximum number of attempts for I/O operations before failing.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_UINT64`

nvidia::gxf::TcpServer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Codelet that functions as a server in a TCP connection.

* Component ID: :code:`a3e0e42d-e32e-73ab-ef83-fbb311310759`
* Base Type: :code:`nvidia::gxf::Codelet`

Parameters
++++++++++++

**receivers**

List of receivers to receive entities from.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::Receiver>>`

|

**transmitters**

List of transmitters to publish entities to.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::Transmitter>>`

|

**serializers**

List of component serializers to serialize and de-serialize entities.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_CUSTOM`
* Custom Type: :code:`std::vector<nvidia::gxf::Handle<nvidia::gxf::ComponentSerializer>>`

|

**address**

Address of TCP server.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_STRING`

|

**port**

Port of TCP server.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_INT32`

|


**timeout_ms**

Time in milliseconds to wait before retrying connection to TCP client.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_UINT64`

|

**maximum_attempts**

Maximum number of attempts for I/O operations before failing.

* Flags: :code:`GXF_PARAMETER_FLAGS_NONE`
* Type: :code:`GXF_PARAMETER_TYPE_UINT64`
