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

import socket
from collections import namedtuple

import psutil

from holoscan.cli.common.utils import compare_versions, get_host_ip_addresses


class TestCompareVersions:
    def test_left_greater_than_left(self):
        assert compare_versions("1.2.3", "1.1.0") == 1
        assert compare_versions("1.2", "1.1.0") == 1

    def test_left_less_than_left(self):
        assert compare_versions("1.2.3", "3.1.0") == -1
        assert compare_versions("1.2.3", "3.1") == -1

    def test_left_equals_to_left(self):
        assert compare_versions("1.2.3", "1.2.3") == 0
        assert compare_versions("1.2.0", "1.2") == 0


class TestGetHostIpAddress:
    snicaddr = namedtuple("snicaddr", ["family", "address", "netmask", "broadcast", "ptp"])
    sample_data = dict(
        [
            (
                "lo",
                [
                    snicaddr(
                        family=socket.AF_INET,
                        address="127.0.0.1",
                        netmask="255.0.0.0",
                        broadcast=None,
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_INET6,
                        address="::1",
                        netmask="ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
                        broadcast=None,
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_PACKET,
                        address="00:00:00:00:00:00",
                        netmask=None,
                        broadcast=None,
                        ptp=None,
                    ),
                ],
            ),
            (
                "eth0",
                [
                    snicaddr(
                        family=socket.AF_INET,
                        address="10.20.12.100",
                        netmask="255.255.254.0",
                        broadcast="10.20.13.200",
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_INET6,
                        address="fe80::e3cb:9e06:5545:817c%eth0",
                        netmask="ffff:ffff:ffff:ffff::",
                        broadcast=None,
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_PACKET,
                        address="48:b0:2d:e8:ce:02",
                        netmask=None,
                        broadcast="ff:ff:ff:ff:ff:ff",
                        ptp=None,
                    ),
                ],
            ),
            (
                "eth1",
                [
                    snicaddr(
                        family=socket.AF_INET,
                        address="10.20.12.101",
                        netmask="255.255.254.0",
                        broadcast="10.20.13.255",
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_INET6,
                        address="fe80::87d5:3875:5f6d:749f%eth1",
                        netmask="ffff:ffff:ffff:ffff::",
                        broadcast=None,
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_PACKET,
                        address="48:b0:2d:e8:ce:03",
                        netmask=None,
                        broadcast="ff:ff:ff:ff:ff:ff",
                        ptp=None,
                    ),
                ],
            ),
            (
                "docker0",
                [
                    snicaddr(
                        family=socket.AF_INET,
                        address="172.17.0.1",
                        netmask="255.255.0.0",
                        broadcast="172.17.255.255",
                        ptp=None,
                    ),
                    snicaddr(
                        family=socket.AF_PACKET,
                        address="02:42:56:18:3d:8d",
                        netmask=None,
                        broadcast="ff:ff:ff:ff:ff:ff",
                        ptp=None,
                    ),
                ],
            ),
            (
                "wlan0",
                [
                    snicaddr(
                        family=socket.AF_PACKET,
                        address="b4:8c:9d:1c:5c:6d",
                        netmask=None,
                        broadcast="ff:ff:ff:ff:ff:ff",
                        ptp=None,
                    )
                ],
            ),
            (
                "usb0",
                [
                    snicaddr(
                        family=socket.AF_PACKET,
                        address="7a:65:51:3c:1f:91",
                        netmask=None,
                        broadcast="ff:ff:ff:ff:ff:ff",
                        ptp=None,
                    )
                ],
            ),
        ]
    )

    def test_ips_are_returns_all_matching_ips(self, monkeypatch):
        monkeypatch.setattr(psutil, "net_if_addrs", lambda: TestGetHostIpAddress.sample_data)

        ipv4, ipv6 = get_host_ip_addresses()

        assert ipv4 is not None
        assert ipv6 is not None
        assert len(ipv4) == 4
        assert len(ipv6) == 3
        assert ("lo", "127.0.0.1") in ipv4
        assert ("eth0", "10.20.12.100") in ipv4
        assert ("eth1", "10.20.12.101") in ipv4
        assert ("docker0", "172.17.0.1") in ipv4

        assert ("lo", "::1") in ipv6
        assert ("eth0", "fe80::e3cb:9e06:5545:817c%eth0") in ipv6
        assert ("eth1", "fe80::87d5:3875:5f6d:749f%eth1") in ipv6
        assert "docker0" not in [item[0] for item in ipv6]
