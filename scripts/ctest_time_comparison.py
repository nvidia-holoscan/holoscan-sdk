#!/usr/bin/env python3
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

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    # Define the valid operators
    operators = ["LESS", "GREATER", "EQUAL"]

    # Read the CTestCost file
    with open(sys.argv[1]) as f:
        ctest_costs = [section.split() for section in f.read().split("\n")]
        test_times = {}
        for _, row in enumerate(ctest_costs):
            if len(row) == 2:
                test_times[row[0]] = row[1]

        if (len(sys.argv) - 2) % 3 != 0:
            print("arguments should be a multiple of three")
            return

        valid_output = True
        index = 0
        operator = ""
        time1 = 0
        time2 = 0
        for i in range(2, len(sys.argv)):
            arg = sys.argv[i]
            if arg in test_times:
                if index == 0:
                    time1 = test_times[arg]
                else:
                    time2 = test_times[arg]
            elif arg in operators:
                operator = arg
            else:
                valid_output = False
                print("Argument " + arg + " is not recognized")
                break
            index += 1

            if index == 3:
                index = 0
                if (
                    (operator == "LESS" and time1 >= time2)
                    or (operator == "EQUAL" and time1 != time2)
                    or (operator == "GREATER" and time1 <= time2)
                ):
                    valid_output = False
                    break
                operator = ""
                time1 = 0
                time2 = 0

        if valid_output:
            print("Timing for tests matches expectations")
        else:
            print("Timing for tests does not match expectations")


if __name__ == "__main__":
    main()
