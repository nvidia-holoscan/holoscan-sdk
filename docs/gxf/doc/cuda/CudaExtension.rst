..
   Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

.. _cudaExtension:

CudaExtension
--------------

Extension for CUDA operations.

* UUID: d63a98fa-7882-11eb-a917-b38f664f399c
* Version: 2.0.0
* Author: NVIDIA
* License: LICENSE

Components
~~~~~~~~~~

nvidia::gxf::CudaStream
^^^^^^^^^^^^^^^^^^^^^^^

Holds and provides access to native ``cudaStream_t``.

:code:`nvidia::gxf::CudaStream` handle must be allocated by :code:`nvidia::gxf::CudaStreamPool`. Its lifecycle is valid until explicitly recycled through :code:`nvidia::gxf::CudaStreamPool.releaseStream()` or implicitly until :code:`nvidia::gxf::CudaStreamPool` is deactivated.

You may call :code:`stream()` to get the native ``cudaStream_t`` handle, and to submit GPU operations. After the submission, GPU takes over the input tensors/buffers and keeps them in use. To prevent the host carelessly releasing these in-use buffers, CUDA Codelet needs to call :code:`record(event, input_entity, sync_cb)` to extend :code:`input_entity`'s lifecycle until the GPU completely consumes it.
Alternatively, you may call :code:`record(event, event_destroy_cb)` for native ``cudaEvent_t`` operations and free in-use resource via :code:`event_destroy_cb`.

It is required to have a :code:`nvidia::gxf::CudaStreamSync` in the graph pipeline after all the CUDA operations. See more details in :code:`nvidia::gxf::CudaStreamSync`.

* Component ID: 5683d692-7884-11eb-9338-c3be62d576be
* Defined in: gxf/cuda/cuda_stream.hpp

nvidia::gxf::CudaStreamId
^^^^^^^^^^^^^^^^^^^^^^^^^

Holds CUDA stream Id to deduce ``nvidia::gxf::CudaStream`` handle.

:code:`stream_cid` should be :code:`nvidia::gxf::CudaStream` component id.

* Component ID: 7982aeac-37f1-41be-ade8-6f00b4b5d47c
* Defined in: gxf/cuda/cuda_stream_id.hpp

nvidia::gxf::CudaEvent
^^^^^^^^^^^^^^^^^^^^^^

Holds and provides access to native ``cudaEvent_t`` handle.

When a :code:`nvidia::gxf::CudaEvent` is created, you'll need to initialize a native ``cudaEvent_t`` through :code:`init(flags, dev_id)`,  or set third party event through :code:`initWithEvent(event, dev_id, free_fnc)`. The event keeps valid until :code:`deinit` is called explicitly otherwise gets recycled in destructor.

* Component ID: f5388d5c-a709-47e7-86c4-171779bc64f3
* Defined in: gxf/cuda/cuda_event.hpp

nvidia::gxf::CudaStreamPool
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CudaStream`` allocation.

You must explicitly call :code:`allocateStream()` to get a valid :code:`nvidia::gxf::CudaStream` handle. This component would hold all the its allocated :code:`nvidia::gxf::CudaStream` entities until :code:`releaseStream(stream)` is called explicitly or the :code:`CudaStreamPool` component is deactivated.

* Component ID: 6733bf8b-ba5e-4fae-b596-af2d1269d0e7
* Base Type: nvidia::gxf::Allocator


Parameters
++++++++++

**dev_id**

GPU device id.

* Flags: GXF_PARAMETER_FLAGS_NONE
* Type: GXF_PARAMETER_TYPE_INT32
* Default Value: 0

|

**stream_flags**

Flag values to create CUDA streams.

* Flags: GXF_PARAMETER_FLAGS_NONE
* Type: GXF_PARAMETER_TYPE_INT32
* Default Value: 0

|

**stream_priority**

Priority values to create CUDA streams.

* Flags: GXF_PARAMETER_FLAGS_NONE
* Type: GXF_PARAMETER_TYPE_INT32
* Default Value: 0

|

**reserved_size**

User-specified file name without extension.

* Flags: GXF_PARAMETER_FLAGS_NONE
* Type: GXF_PARAMETER_TYPE_INT32
* Default Value: 1

**max_size**

Maximum Stream Size.

* Flags: GXF_PARAMETER_FLAGS_NONE
* Type: GXF_PARAMETER_TYPE_INT32
* Default Value: 0, no limitation.

nvidia::gxf::CudaStreamSync
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronize all CUDA streams which are carried by message entities.

This codelet is required to get connected in the graph pipeline after all CUDA ops codelets. When a message entity is received, it would find all of the :code:`nvidia::gxf::CudaStreamId` in that message, and extract out each :code:`nvidia::gxf::CudaStream`. With each ``CudaStream`` handle, it synchronizes all previous :code:`nvidia::gxf::CudaStream.record()` events, along with all submitted GPU operations before this point.

.. note::
     ``CudaStreamSync`` must be set in the graph when :code:`nvidia::gxf::CudaStream.record()` is used, otherwise it may cause a memory leak.

* Component ID: 0d1d8142-6648-485d-97d5-277eed00129c
* Base Type: nvidia::gxf::Codelet

Parameters
++++++++++

**rx**

Receiver to receive all messages carrying :code:`nvidia::gxf::CudaStreamId`.

* Flags: GXF_PARAMETER_FLAGS_NONE
* Type: GXF_PARAMETER_TYPE_HANDLE
* Handle Type: nvidia::gxf::Receiver

|

**tx**

Transmitter to send messages to downstream.

* Flags: GXF_PARAMETER_FLAGS_OPTIONAL
* Type: GXF_PARAMETER_TYPE_HANDLE
* Handle Type: nvidia::gxf::Transmitter


