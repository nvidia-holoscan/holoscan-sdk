# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Operator subtrees under the inference module.
#   inference/           - HoloInfer + op_inference (TRT-backed InferenceOp).
#   tensor_preprocessor/ - NPP-based image preprocessing for inference pipelines.

add_subdirectory(inference)
add_subdirectory(tensor_preprocessor)
