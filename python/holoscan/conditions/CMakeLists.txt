# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(conditions
    asynchronous.cpp
    boolean.cpp
    conditions.cpp
    count.cpp
    cuda_buffer_available.cpp
    cuda_event.cpp
    cuda_stream.cpp
    downstream_message_affordable.cpp
    expiring_message.cpp
    memory_available.cpp
    message_available.cpp
    multi_message_available.cpp
    multi_message_available_timeout.cpp
    periodic.cpp
    publisher_available.cpp
    subscriber_available.cpp
)

target_link_libraries(conditions_python
    PRIVATE
        holoscan::pubsub_runtime
)

if(TARGET holoscan::pubsub_fastdds)
    target_sources(conditions_python
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/pending_export_condition.cpp
    )
    target_link_libraries(conditions_python
        PRIVATE
            holoscan::pubsub_fastdds
    )
    target_compile_definitions(conditions_python
        PRIVATE
            HOLOSCAN_PYTHON_HAS_PENDING_EXPORT_CONDITION=1
    )
endif()
