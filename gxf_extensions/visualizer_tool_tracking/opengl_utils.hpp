/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_OPENGL_UTILS_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_OPENGL_UTILS_HPP_

#include <glad/glad.h>

#include <string>

void GLAPIENTRY OpenGLDebugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                           GLsizei length, const GLchar* message,
                                           const void* userParam);

bool createGLSLShader(GLenum shader_type, GLuint& shader, const char* shader_src);

bool createGLSLShaderFromFile(GLenum shader_type, GLuint& shader,
                              const std::string& shader_filename);

bool linkGLSLProgram(const GLuint vertex_shader, const GLuint fragment_shader, GLuint& program);
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_OPENGL_UTILS_HPP_
