# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(network_contexts
    network_contexts.cpp
)

if(TARGET holoscan::pubsub_fastdds)
    target_link_libraries(network_contexts_python
        PRIVATE
            holoscan::pubsub_fastdds
    )
    target_compile_definitions(network_contexts_python
        PRIVATE
            HOLOSCAN_PYTHON_HAS_FASTDDS_PUBSUB_CONTEXT=1
    )
endif()
