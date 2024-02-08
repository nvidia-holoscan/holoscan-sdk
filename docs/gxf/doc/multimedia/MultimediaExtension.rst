..
   Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

.. _multimediaExtension:

MultimediaExtension
----------------------

Extension for multimedia related data types, interfaces and components in GXF Core.

* UUID: :code:`6f2d1afc-1057-481a-9da6-a5f61fed178e`
* Version: :code:`2.0.0`
* Author: :code:`NVIDIA`
* License: :code:`LICENSE`

Components
~~~~~~~~~~~~

nvidia::gxf::AudioBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AudioBuffer is similar to Tensor component in the standard extension and holds memory and metadata corresponding to an audio buffer.

* Component ID: :code:`a914cac6-5f19-449d-9ade-8c5cdcebe7c3`

``AudioBufferInfo`` structure captures the following metadata:

+-----------------------+----------------------------------------+
| Field                 |       Description                      |
+=======================+========================================+
| channels              | Number of channels in an audio frame   |
+-----------------------+----------------------------------------+
| samples               | Number of samples in an audio frame    |
+-----------------------+----------------------------------------+
| sampling_rate         | sampling rate in Hz                    |
+-----------------------+----------------------------------------+
| bytes_per_sample      | Number of bytes required per sample    |
+-----------------------+----------------------------------------+
| audio_format          | AudioFormat of an audio frame          |
+-----------------------+----------------------------------------+
| audio_layout          | AudioLayout of an audio frame          |
+-----------------------+----------------------------------------+

Supported ``AudioFormat`` types:

+-------------------------+--------------------------------------+
| AudioFormat             |     Description                      |
+=========================+======================================+
| GXF_AUDIO_FORMAT_S16LE  | 16-bit signed PCM audio              |
+-------------------------+--------------------------------------+
| GXF_AUDIO_FORMAT_F32LE  | 32-bit floating-point audio          |
+-------------------------+--------------------------------------+

Supported ``AudioLayout`` types:

+----------------------------------+--------------------------------------------------------------+
| AudioLayout                      |    Description                                               |
+==================================+==============================================================+
| GXF_AUDIO_LAYOUT_INTERLEAVED     | Data from all the channels to be interleaved - LRLRLR        |
+----------------------------------+--------------------------------------------------------------+
| GXF_AUDIO_LAYOUT_NON_INTERLEAVED | Data from all the channels not to be interleaved - LLLRRR    |
+----------------------------------+--------------------------------------------------------------+

nvidia::gxf::VideoBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VideoBuffer is similar to Tensor component in the standard extension and holds memory and metadata corresponding to a video buffer.

* Component ID: :code:`16ad58c8-b463-422c-b097-61a9acc5050e`

``VideoBufferInfo`` structure captures the following metadata:

+-----------------------+-----------------------------------------------+
| Field                 |       Description                             |
+=======================+===============================================+
| width                 | width of a video frame                        |
+-----------------------+-----------------------------------------------+
| height                | height of a video frame                       |
+-----------------------+-----------------------------------------------+
| color_format          | VideoFormat of a video frame                  |
+-----------------------+-----------------------------------------------+
| color_planes          | ColorPlane(s) associated with the VideoFormat |
+-----------------------+-----------------------------------------------+
| surface_layout        | SurfaceLayout of the video frame              |
+-----------------------+-----------------------------------------------+

Supported VideoFormat types:

+--------------------------------------+-------------------------------------------------------+
| VideoFormat                          | Description                                           |
+======================================+=======================================================+
|  GXF_VIDEO_FORMAT_YUV420             | BT.601 multi planar 4:2:0 YUV                         |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_YUV420_ER          | BT.601 multi planar 4:2:0 YUV ER                      |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_YUV420_709         | BT.709 multi planar 4:2:0 YUV                         |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_YUV420_709_ER      | BT.709 multi planar 4:2:0 YUV ER                      |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_NV12               | BT.601 multi planar 4:2:0 YUV with interleaved UV     |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_NV12_ER            | BT.601 multi planar 4:2:0 YUV ER with interleaved UV  |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_NV12_709           | BT.709 multi planar 4:2:0 YUV with interleaved UV     |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_NV12_709_ER        | BT.709 multi planar 4:2:0 YUV ER with interleaved UV  |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_RGBA               | RGBA-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_BGRA               | BGRA-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_ARGB               | ARGB-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_ABGR               | ABGR-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_RGBX               | RGBX-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_BGRX               | BGRX-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_XRGB               | XRGB-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_XBGR               | XBGR-8-8-8-8 single plane                             |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_RGB                | RGB-8-8-8 single plane                                |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_BGR                | BGR-8-8-8 single plane                                |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_R8_G8_B8           | RGB - unsigned 8 bit multiplanar                      |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_B8_G8_R8           | BGR - unsigned 8 bit multiplanar                      |
+--------------------------------------+-------------------------------------------------------+
|  GXF_VIDEO_FORMAT_GRAY               | 8 bit GRAY scale single plane                         |
+--------------------------------------+-------------------------------------------------------+


Supported SurfaceLayout types:

+--------------------------------------+-------------------------------------------------------+
|  SurfaceLayout                       | Description                                           |
+======================================+=======================================================+
|  GXF_SURFACE_LAYOUT_PITCH_LINEAR     | pitch linear surface memory                           |
+--------------------------------------+-------------------------------------------------------+
|  GXF_SURFACE_LAYOUT_BLOCK_LINEAR     | block linear surface memory                           |
+--------------------------------------+-------------------------------------------------------+
