// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "backends/rhi_geometry_shader.hpp"

namespace holoviz::backends::detail {

// Vertex-buffer-less geometry: every vertex is synthesized from SV_VertexID/SV_InstanceID
// plus the primitive parameters in a storage buffer — the exact shape proven by the
// monospace text shader. Per-frame upload is therefore only the caller's own primitive
// floats (see frame_data.hpp for the packing), and the pool is sized exactly from the
// layer's declared max_primitives.
//
// Coordinates are normalized [0,1] with origin top-left; Vulkan clip space is y-down, so
// clip = p * 2 - 1 with no flip. Every topology lowers to triangles so
// GeometryBatch::line_width is honored uniformly (the RHI exposes no line-width state, and
// Vulkan wide lines are an optional feature): lines, strips, and crosses expand each
// segment to a square-capped quad; rectangle and oval outlines expand to instanced
// triangle rings. Square caps extend segment ends by the half-width, which covers strip
// joints at any angle; with translucent colors the joint overlap double-blends — the
// accepted tradeoff for overlay geometry (same class as the oval's non-constant width on
// eccentric ellipses). The half-width converts pixels to normalized units per axis via
// viewport_px.
const char* const kRhiGeometrySlangSource = R"(
StructuredBuffer<float> geometryParams;

struct PushConstants {
    float4 color;
    float2 viewport_px;
    float  line_width_px;
    uint   base_float_offset;  // this batch's first float within the bound slot slice
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// System values only: reflection skips SV_ fields, so the pipeline has zero
// vertex attributes and no vertex buffer is ever bound.
struct VertexInput {
    uint vertex_id : SV_VertexID;
    uint instance_id : SV_InstanceID;
};

struct VertexOutput {
    float4 position : SV_POSITION;
};

float fetch(uint float_index) {
    return geometryParams[pc.base_float_offset + float_index];
}

float2 fetch2(uint float_index) {
    return float2(fetch(float_index), fetch(float_index + 1u));
}

float4 to_clip(float2 p_norm) {
    return float4(p_norm * 2.0 - 1.0, 0.0, 1.0);
}

static const uint kQuadCorner[6] = {0u, 1u, 2u, 0u, 2u, 3u};

// Expand the segment p0->p1 (normalized endpoints) to one corner of its square-capped
// quad of line_width_px: work in pixel space so the width is isotropic, offset the
// endpoints by the half-width both along the segment (the caps, which also cover strip
// joints) and across it. Quad corners 0..3 = {p0-t+n, p1+t+n, p1+t-n, p0-t-n}; a
// degenerate (zero-length) segment collapses to a zero-area quad and draws nothing.
float4 expand_segment(float2 p0_norm, float2 p1_norm, uint corner) {
    float2 p0 = p0_norm * pc.viewport_px;
    float2 p1 = p1_norm * pc.viewport_px;
    float2 d = p1 - p0;
    float len = length(d);
    if (len == 0.0) {
        return to_clip(p0_norm);
    }
    float2 t = (d / len) * (0.5 * pc.line_width_px);
    float2 n = float2(-t.y, t.x);
    float2 along = (corner == 0u || corner == 3u) ? p0 - t : p1 + t;
    float2 across = (corner < 2u) ? n : -n;
    return to_clip((along + across) / pc.viewport_px);
}

// kLineList: x0,y0,x1,y1 per segment; 6 vertices x 1 instance per segment, expanded to a
// square-capped quad of line_width_px.
[shader("vertex")]
VertexOutput vertex_line_list(VertexInput input) {
    uint corner = kQuadCorner[input.vertex_id];
    float2 p0 = fetch2(input.instance_id * 4u);
    float2 p1 = fetch2(input.instance_id * 4u + 2u);
    VertexOutput output;
    output.position = expand_segment(p0, p1, corner);
    return output;
}

// kLineStrip: x,y per point; instance k expands the segment between consecutive points
// k and k+1 (6 vertices per instance). Square caps make adjacent quads overlap at the
// shared point, covering the joint at any angle.
[shader("vertex")]
VertexOutput vertex_line_strip(VertexInput input) {
    uint corner = kQuadCorner[input.vertex_id];
    float2 p0 = fetch2(input.instance_id * 2u);
    float2 p1 = fetch2(input.instance_id * 2u + 2u);
    VertexOutput output;
    output.position = expand_segment(p0, p1, corner);
    return output;
}

// kCrossList: cx,cy,size per cross; 12 vertices x 1 instance per cross = the horizontal
// and vertical arm segments (each spanning size/2 from the center), expanded like lines.
[shader("vertex")]
VertexOutput vertex_cross(VertexInput input) {
    uint quad = input.vertex_id / 6u;
    uint corner = kQuadCorner[input.vertex_id - quad * 6u];
    float2 center = fetch2(input.instance_id * 3u);
    float half_size = fetch(input.instance_id * 3u + 2u) * 0.5;
    float2 arm = (quad == 0u) ? float2(half_size, 0.0) : float2(0.0, half_size);
    VertexOutput output;
    output.position = expand_segment(center - arm, center + arm, corner);
    return output;
}

// kRectangleList: x0,y0,x1,y1 per rect (any two opposite corners); 24 vertices = 4 mitered
// side quads forming a closed outline ring of line_width_px. Side k's quad runs
// {outer_k, outer_k+1, inner_k+1, inner_k} over the perimeter corners
// 0=(lo.x,lo.y) 1=(hi.x,lo.y) 2=(hi.x,hi.y) 3=(lo.x,hi.y); outer/inner offset the corner
// diagonally outward/inward by the half-width, so adjacent quads meet exactly at the miter.
[shader("vertex")]
VertexOutput vertex_rect(VertexInput input) {
    float2 r0 = fetch2(input.instance_id * 4u);
    float2 r1 = fetch2(input.instance_id * 4u + 2u);
    float2 lo = min(r0, r1);
    float2 hi = max(r0, r1);

    uint side = input.vertex_id / 6u;
    uint corner = kQuadCorner[input.vertex_id - side * 6u];
    uint perimeter = (corner == 1u || corner == 2u) ? ((side + 1u) & 3u) : side;

    float2 c = float2(perimeter == 1u || perimeter == 2u ? hi.x : lo.x,
                      perimeter >= 2u ? hi.y : lo.y);
    float2 outward = float2(perimeter == 1u || perimeter == 2u ? 1.0 : -1.0,
                            perimeter >= 2u ? 1.0 : -1.0);
    float2 half_w = 0.5 * pc.line_width_px / pc.viewport_px;
    bool outer = corner < 2u;

    VertexOutput output;
    output.position = to_clip(c + (outer ? outward : -outward) * half_w);
    return output;
}

static const uint kOvalSegments = 32u;

// kOvalList: cx,cy,rx,ry per oval; 32 segments x 6 vertices = a closed outline ring. The
// half-width expands/shrinks the radii per axis — not perfectly constant-width for very
// eccentric ellipses, but exact for circles and standard practice for overlay outlines.
[shader("vertex")]
VertexOutput vertex_oval(VertexInput input) {
    float2 center = fetch2(input.instance_id * 4u);
    float2 radii = fetch2(input.instance_id * 4u + 2u);

    uint seg = input.vertex_id / 6u;
    uint corner = kQuadCorner[input.vertex_id - seg * 6u];
    uint point = (seg + ((corner == 1u || corner == 2u) ? 1u : 0u)) % kOvalSegments;

    float angle = (float(point) / float(kOvalSegments)) * 6.283185307179586;
    float2 dir = float2(cos(angle), sin(angle));
    float2 half_w = 0.5 * pc.line_width_px / pc.viewport_px;
    bool outer = corner < 2u;
    float2 r = outer ? radii + half_w : max(radii - half_w, float2(0.0, 0.0));

    VertexOutput output;
    output.position = to_clip(center + dir * r);
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    return pc.color;
}
)";

}  // namespace holoviz::backends::detail
