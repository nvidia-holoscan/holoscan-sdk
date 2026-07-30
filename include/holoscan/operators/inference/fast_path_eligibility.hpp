/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef HOLOSCAN_OPERATORS_INFERENCE_FAST_PATH_ELIGIBILITY_HPP
#define HOLOSCAN_OPERATORS_INFERENCE_FAST_PATH_ELIGIBILITY_HPP

#include <map>
#include <set>
#include <string>
#include <vector>

namespace holoscan::ops::inference_eligibility {

/**
 * @brief Returns true when `dev_map` routes any model to a device other than
 * the default data-transfer GPU ("0"). The reserved key "gpu-dt" identifies
 * the data-transfer device itself and is not treated as an off-device
 * assignment.
 *
 * A single-entry device_map like `{model: "1"}` is still cross-GPU relative
 * to gpu-dt=0 and must disqualify the fast path — checking `size() > 1` alone
 * is insufficient.
 */
inline bool has_off_gpu_dt_assignment(const std::map<std::string, std::string>& dev_map) {
  for (const auto& [k, v] : dev_map) {
    if (k == "gpu-dt") {
      continue;
    }
    if (!v.empty() && v != "0") {
      return true;
    }
  }
  return false;
}

/**
 * @brief Returns true when any tensor produced by some model (present as a
 * value in `inference_map`) is consumed as an input by some other model
 * (present as a value in `pre_processor_map`). Chained models cannot use the
 * multi-model fast paths — kSeqFast iterates in std::map key order, and
 * kParFast fans out with no cross-model waits.
 */
inline bool has_chained_models(
    const std::map<std::string, std::vector<std::string>>& pre_processor_map,
    const std::map<std::string, std::vector<std::string>>& inference_map) {
  std::set<std::string> produced;
  for (const auto& [_, outs] : inference_map) {
    for (const auto& t : outs) {
      produced.insert(t);
    }
  }
  for (const auto& [_, ins] : pre_processor_map) {
    for (const auto& t : ins) {
      if (produced.count(t)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace holoscan::ops::inference_eligibility

#endif  // HOLOSCAN_OPERATORS_INFERENCE_FAST_PATH_ELIGIBILITY_HPP
