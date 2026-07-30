# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(resources
    allocators.cpp
    clocks.cpp
    component_serializers.cpp
    condition_combiners.cpp
    entity_serializers.cpp
    gxf_component_resource.cpp
    receivers.cpp
    resources.cpp
    serialization_buffers.cpp
    std_entity_serializer.cpp
    system_resources.cpp
    transmitters.cpp
)
