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

#ifndef HOLOSCAN_VIZ_LAYERS_GEOMETRY_LAYER_HPP
#define HOLOSCAN_VIZ_LAYERS_GEOMETRY_LAYER_HPP

#include <cstdint>
#include <memory>

#include "layer.hpp"

#include "../holoviz/primitive_topology.hpp"

namespace holoscan::viz {

/**
 * Layer specialication for geometry rendering.
 */
class GeometryLayer : public Layer {
 public:
    /**
     * Construct a new GeometryLayer object.
     */
    GeometryLayer();

    /**
     * Destroy the GeometryLayer object.
     */
    ~GeometryLayer();

    /**
     * Set the color for following geometry.
     *
     * @param r,g,b,a RGBA color. Default (1.0, 1.0, 1.0, 1.0).
     */
    void color(float r, float g, float b, float a);

    /**
     * Set the line width for geometry made of lines.
     *
     * @param width line width in pixels. Default 1.0.
     */
    void line_width(float width);

    /**
     * Set the point size for geometry made of points.
     *
     * @param size  point size in pixels. Default 1.0.
     */
    void point_size(float size);

    /**
     * Draw a geometric primitive.
     *
     * @param topology          primitive topology
     * @param primitive_count   primitive count
     * @param data_size         size of the data array in floats
     * @param data              pointer to data, the format and size of the array 
     *                          depends on the primitive count and topology
     */
    void primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
                                                                       const float *data);

    /**
     * Draw text.
     *
     * @param x     x coordinate
     * @param y     y coordinate
     * @param size  font size
     * @param text  text to draw
     */
    void text(float x, float y, float size, const char *text);

    /// holoscan::viz::Layer virtual members
    ///@{
    bool can_be_reused(Layer &other) const override;
    void end(Vulkan *vulkan) override;
    void render(Vulkan *vulkan) override;
    ///@}

 private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_LAYERS_GEOMETRY_LAYER_HPP */
