"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import torch


class IdentityModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: list[torch.Tensor]):
        # For rcnn, output in (losses, detections)
        return (0, [{"output": torch.zeros(1, dtype=torch.float32, device=x[0].device)}])


module = IdentityModule()
script_module = torch.jit.script(module)
script_module.save("identity_model.pt")
