# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
.. autosummary::
    :toctree: _autosummary

    common
    packager
    runner
    version
"""

import os
import sys

__title__ = "holoscan.cli"
_current_dir = os.path.abspath(os.path.dirname(__file__))
if sys.path and os.path.abspath(sys.path[0]) != _current_dir:
    sys.path.insert(0, _current_dir)
del _current_dir
