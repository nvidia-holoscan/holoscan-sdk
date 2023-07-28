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

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("packager")


class Models:
    """
    Builds model list from a given path where the path could be a model file, a directory that
    contains a model or a directory with multiple models where each model contains within its own
    subdirectory.
    """

    def build(self, models_path: Path) -> Optional[Dict[str, Path]]:
        """Checks if the given path is a file or a directory.

        Args:
            models_path (Path): A user given path that contains one or more models.
        Returns:
            Optional[Dict[str, Path]]: Returns None if no path is given.  Otherwise, returns a
            dictionary where the key is the name of the model and the value contains the path
            to the model.
        """
        if models_path is not None:
            logger.info("Scanning for models in {models_path}...")
            models: Dict[str, Path] = {}
            if models_path.is_file():
                self._configure_model_file(models_path, models)
            elif models_path.is_dir():
                self._configure_model_dir(models_path, models)

            return models
        else:
            return None

    def _configure_model_dir(self, models_path: Path, models: Dict[str, Path]):
        """
        Iterate through the given directory to scan for models.
        If files are found within the directory, we simply assume that all files within the given
        directory contains a model and sets the name of the model to the name of the given directory.

        Any subdirectories found within the given directory are treated as a separate model and the
        name of the subdirectory is set to be the name of the model.

        Args:
            models_path (Path): Path to the model file.
            models (Dict[str, Path]): Where models are added to.
        """
        model_dirs = os.listdir(models_path)

        for model_dir in model_dirs:
            if os.path.isfile(models_path / model_dir):
                model_name = models_path.resolve().stem
                models[model_name] = models_path
                logger.debug(f"Model {model_name}={models_path} added.")
            elif os.path.isdir(models_path / model_dir):
                model_path = models_path / model_dir
                model_name = model_dir
                models[model_name] = model_path
                logger.debug(f"Model {model_name}={model_path} added.")

    def _configure_model_file(self, models_path: Path, models: Dict[str, Path]):
        """
        Adds a new model to 'models' object where the model name is the name of the given file,
        without file extension, and the value is the directory containing the given model file.

        Args:
            models_path (Path): Path to the model file.
            models (Dict[str, Path]): Where the model is added to.
        """
        model_name = models_path.resolve().stem
        models[model_name] = models_path.parent
        logger.debug(f"Model file {model_name}={models[model_name]} added.")
