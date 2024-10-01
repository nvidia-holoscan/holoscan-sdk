..
   Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

Graph Specification TimeStamping
---------------------------------

Message Passing
~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the graph is built, the communication between various entities occur by passing around messages (messages are entities themselves). Specifically, one component/codelet can publish a message entity and another can receive it. When publishing, a message should always have an associated :code:`Timestamp` component with the name `"timestamp"`. A :code:`Timestamp` component
contains two different time values (See the ``gxf/std/timestamp.hpp`` header file for more information.):

1. :code:`acqtime` - This is the time when the message entity is acquired; for instance, this would generally be the driver time of the camera when it captures an image. You must provide this timestamp if you are publishing a message in a codelet.

2. :code:`pubtime` - This is the time when the message entity is published by a node in the graph. This will automatically get updated using the clock of the scheduler.

In a codelet, when publishing message entities using a :code:`Transmitter (tx)`, there are two ways to add the required :code:`Timestamp`:

1. :code:`tx.publish(Entity message)`: You can manually add a component of type :code:`Timestamp` with the name "timestamp" and set the ``acqtime``. The ``pubtime`` in this case should be set to ``0``. The message is published using the :code:`publish(Entity message)`. **This will be deprecated in the next release.**

2. :code:`tx.publish(Entity message, int64_t acqtime)`: You can simply call :code:`publish(Entity message, int64_t acqtime)` with the ``acqtime``. Timestamp will be added automatically.
