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

import os
import pathlib

from holoscan.cli.packager.models import Models


class TestModels:
    def test_no_model_file(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: True)
        model_path = None
        models = Models()

        result = models.build(model_path)

        assert not result

    def test_models_file(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: True)
        model_path = pathlib.Path("/my/model/cool_model_file.ext")
        models = Models()

        result = models.build(model_path)

        assert len(result) == 1
        assert model_path.stem in result
        assert result[model_path.stem] == model_path

    def test_models_dir_with_single_model(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: False)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: True)
        monkeypatch.setattr(os.path, "isfile", lambda x: True)
        monkeypatch.setattr(os, "listdir", lambda x: ["file1", "file2", "file3"])
        model_path = pathlib.Path("/my/model/cool_model_dir")
        models = Models()

        result = models.build(model_path)

        assert len(result) == 1
        assert "cool_model_dir" in result
        assert result["cool_model_dir"] == model_path

    def test_models_dir_with_multiple_models(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: False)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: True)
        monkeypatch.setattr(os.path, "isfile", lambda x: False)
        monkeypatch.setattr(os.path, "isdir", lambda x: True)
        monkeypatch.setattr(os, "listdir", lambda x: ["model1", "model2", "model3"])
        model_path = pathlib.Path("/my/model/cool_model_dir")
        models = Models()

        result = models.build(model_path)

        assert len(result) == 3
        assert "model1" in result
        assert "model2" in result
        assert "model3" in result
        assert result["model1"] == (model_path / "model1")
        assert result["model2"] == (model_path / "model2")
        assert result["model3"] == (model_path / "model3")
