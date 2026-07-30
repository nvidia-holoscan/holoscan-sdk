# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(
    core
    application.cpp
    arg.cpp
    cli.cpp
    clock.cpp
    component.cpp
    condition.cpp
    core.cpp
    data_logger.cpp
    dataflow_tracker.cpp
    dl_converter.cpp
    emitter_receiver_registry.cpp
    execution_context.cpp
    executor.cpp
    fragment.cpp
    fragment_service.cpp
    io_context.cpp
    io_spec.cpp
    kwarg_handling.cpp
    messagelabel.cpp
    metadata.cpp
    network_context.cpp
    operator.cpp
    resource.cpp
    scheduler.cpp
    subgraph.cpp
    tensor.cpp
    ../gxf/entity.cpp
)

# Needed for the Holoviz InputSpec, Inference ActivationSpec workaround in PyOutputContext::py_emit()
target_link_libraries(core_python
    PRIVATE
        holoscan::ops::holoviz
        holoscan::infer::headers
)

# Copy headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include/holoscan/python
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "holoscan-core"
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*pydoc.hpp" EXCLUDE
)
