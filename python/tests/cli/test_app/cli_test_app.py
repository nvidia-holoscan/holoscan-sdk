"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import yaml

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec


class PrintAppArguments(Operator):
    """
    An operator that prints all arguments passed into the application.
    This operator has no ports.
    """

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("======================================================")
        print("Printing arguments:")
        for arg in sys.argv:
            print(arg)
        print("PrintAppArguments completed.")
        print("======================================================")


class ListFilesOp(Operator):
    """
    An operator that lists files from the path defined in HOLOSCAN_INPUT_PATH.
    This operator has no ports.
    """

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("======================================================")
        path = os.getenv("HOLOSCAN_INPUT_PATH")
        dirs = os.listdir(path)
        print(f"Listing directories and files in {path}:")
        for d in dirs:
            if os.path.isdir(os.path.join(path, d)):
                print(f"{d}/")
            else:
                print(f"{d}")
        print("ListFilesOp completed.")
        print("======================================================")


class ListModelsOp(Operator):
    """
    An operator that lists models from the path defined in HOLOSCAN_MODEL_PATH.
    This operator has no ports.
    """

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("======================================================")
        models_root_path = os.getenv("HOLOSCAN_MODEL_PATH")
        print(f"Listing models in {models_root_path}:")
        pkg_file_path = os.getenv("HOLOSCAN_PKG_MANIFEST_PATH")
        with open(pkg_file_path) as pkg_manifest_file:
            pkg_manifest = yaml.load(pkg_manifest_file, yaml.SafeLoader)
            for model in pkg_manifest["models"]:
                if os.path.exists(pkg_manifest["models"][model]):
                    print(f"Model '{model}' found in {pkg_manifest['models'][model]}")
                else:
                    print(f"Error: model '{model}' missing from {pkg_manifest['models'][model]}")

        print("ListModelsOp completed.")
        print("======================================================")


class ReadConfigOp(Operator):
    """
    An operator that read the configuration file from value defined in HOLOSCAN_CONFIG_PATH.
    This operator has no ports.
    """

    def setup(self, spec: OperatorSpec):
        pass

    def compute(self, op_input, op_output, context):
        print("======================================================")
        path = os.getenv("HOLOSCAN_CONFIG_PATH")
        print(f"Reading application configuration file from {path}...")

        try:
            with open(path) as app_manifest_file:
                config_object = yaml.load(app_manifest_file, yaml.SafeLoader)
                print(f"Title = {config_object['application']['title']}")
        except Exception as ex:
            print(f"Error reading configuration file: {ex}")

        print("ReadConfigOp completed.")
        print("======================================================")


class CliTestApp(Application):
    def compose(self):
        # Define the operators
        printt_args = PrintAppArguments(self, CountCondition(self, 1), name="print-args")
        list_file = ListFilesOp(self, CountCondition(self, 1), name="list-files")
        print_config = ReadConfigOp(self, CountCondition(self, 1), name="print-config")
        list_models = ListModelsOp(self, CountCondition(self, 1), name="list-models")

        # Define the workflow
        self.add_operator(printt_args)
        self.add_operator(list_file)
        self.add_operator(print_config)
        self.add_operator(list_models)


def main():
    app = CliTestApp()
    app.run()


if __name__ == "__main__":
    main()
