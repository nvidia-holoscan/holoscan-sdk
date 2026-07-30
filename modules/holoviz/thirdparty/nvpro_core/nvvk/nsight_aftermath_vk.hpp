/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>

namespace nvvk {

class GpuCrashTracker
{
public:
  GpuCrashTracker();
  ~GpuCrashTracker();

  void Initialize();  // Initialize the GPU crash dump tracker.

private:
  class GpuCrashTrackerImpl *m_pimpl;
};

}  //namespace nvvk
