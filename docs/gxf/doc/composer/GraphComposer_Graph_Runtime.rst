..
   Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

Graph Execution Engine
======================

Graph Execution Engine is used to execute AI application graphs. It accepts multiple graph files as input, and all graphs are executed in same process context. It also needs manifest files as input which includes list of extensions to load. It must list all extensions required for the graph.


::

      gxe --help
        Flags from gxf/gxe/gxe.cpp:
          -app (GXF app file to execute. Multiple files can be comma-separated)
            type: string default: ""
          -graph_directory (Path to a directory for searching graph files.)
            type: string default: ""
          -log_file_path (Path to a file for logging.) type: string default: ""
          -manifest (GXF manifest file with extensions. Multiple files can be
            comma-separated) type: string default: ""
          -severity (Set log severity levels: 0=None, 1=Error, 2=Warning, 3=Info,
            4=Debug. Default: Info) type: int32 default: 3
