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

#pragma once

#include <memory>
#include <cstdint>

namespace clara::holoviz
{

class Vulkan;

/**
 * The base class for all layers.
 */
class Layer
{
public:
    /**
     * Layer types
     */
    enum class Type
    {
        Image,      ///< image layer
        Geometry,   ///< geometry layer
        ImGui       ///< ImBui layer
    };

    /**
     * Construct a new Layer object.
     *
     * @param type  layer type
     */
    Layer(Type type);
    Layer() = delete;

    /**
     * Destroy the Layer object.
     */
    virtual ~Layer();

    /**
     * @returns  the layer type
     */
    Type GetType() const;

    /**
     * @returns  the layer priority
     */
    int32_t GetPriority() const;

    /**
     * Set the layer priority.
     *
     * @param priority  new layer priority
     */
    void SetPriority(int32_t priority);

    /**
     * @returns  the layer opacity
     */
    float GetOpacity() const;

    /**
     * Set the layer opacity.
     *
     * @param opacity   new layer opacity
     */
    virtual void SetOpacity(float opacity);

    /**
     * Checks if a layer can be reused (properties have to match).
     *
     * @param other layer which is to be checked for re-usability
     */
    virtual bool CanBeReused(Layer &other) const;

    /**
     * Render the layer.
     *
     * @param vulkan    vulkan instance to use for drawing
     */
    virtual void Render(Vulkan *vulkan) = 0;

protected:
    const Type type_;   ///< layer type

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::holoviz
