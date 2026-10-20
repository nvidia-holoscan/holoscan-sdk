# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the owning object-API type and the Holoscan package traits used by
# Input<app::CustomControlT> and Output<app::CustomControlT>.
holoscan_add_flatbuffer_schema(
    TARGET custom_schema_flatbuffers_generated
    SCHEMA custom_control.fbs
    MAX_SERIALIZED_SIZE 512
)
