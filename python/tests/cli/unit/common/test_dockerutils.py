"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.cli.common.dockerutils import parse_docker_image_name_and_tag


class TestParseDockerImageNameAndTag:
    @pytest.mark.parametrize(
        "image_name,expected_name,expected_tag",
        [
            ("holoscan", "holoscan", None),
            ("holoscan:1.0", "holoscan", "1.0"),
            ("holoscan:latest", "holoscan", "latest"),
            ("_/holoscan", "_/holoscan", None),
            ("_/holoscan:latest", "_/holoscan", "latest"),
            ("my/holoscan:2.5", "my/holoscan", "2.5"),
            ("my/holoscan:latest", "my/holoscan", "latest"),
            (
                "localhost:5000/holoscan/holoscan-sdk/dev",
                "localhost:5000/holoscan/holoscan-sdk/dev",
                None,
            ),
            (
                "localhost:5000/holoscan/holoscan-sdk/dev:089167e159571cb3cef625a8b6b1011094c1b292",
                "localhost:5000/holoscan/holoscan-sdk/dev",
                "089167e159571cb3cef625a8b6b1011094c1b292",
            ),
            ("holoscan-sdk/dev", "holoscan-sdk/dev", None),
            ("holoscan-sdk/dev:100", "holoscan-sdk/dev", "100"),
            (
                "holoscan/holoscan-sdk/dev:089167e159571cb3cef625a8b6b1011094c1b292",
                "holoscan/holoscan-sdk/dev",
                "089167e159571cb3cef625a8b6b1011094c1b292",
            ),
            (
                ":",
                None,
                None,
            ),
            (
                ":latest",
                None,
                None,
            ),
            (
                ":1.0",
                None,
                None,
            ),
            (
                "my-image:1.0:beta",
                None,
                None,
            ),
        ],
    )
    def test_parsing_docker_name_tags(self, image_name, expected_name, expected_tag):
        name, tag = parse_docker_image_name_and_tag(image_name)

        assert name == expected_name
        assert tag == expected_tag
