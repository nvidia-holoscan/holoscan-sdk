# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(schedulers
    event_based_scheduler.cpp
    greedy_scheduler.cpp
    multithread_scheduler.cpp
    schedulers.cpp
)
