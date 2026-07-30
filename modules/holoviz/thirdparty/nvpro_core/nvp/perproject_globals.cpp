/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvp/perproject_globals.hpp>

std::string getProjectName()
{
  return PROJECT_NAME;
}

bool isAftermathAvailable()
{
#ifdef NVVK_SUPPORTS_AFTERMATH
  return true;
#else
  return false;
#endif
}
