# Pose Tree Examples

This folder contains examples demonstrating the **Pose Tree** feature in Holoscan SDK, which provides a temporal pose tree to store and query relative coordinate system transformations over time.

## Overview

The Pose Tree feature is designed for applications that need to manage the dynamic spatial relationships between different components or coordinate systems. It allows operators to:
- Define a hierarchy of coordinate frames.
- Update the transformation (pose) between frames at any given time.
- Query the relative pose between any two frames in the tree, even if they are not directly connected.
- Retrieve poses at intermediate timestamps via interpolation.
- Access pose information consistently across an application fragment using the `PoseTreeManager` service.

## Examples

### [Pose Tree Basic](./pose_tree_basic)
- **C++**: [`pose_tree_basic.cpp`](./pose_tree_basic/cpp/pose_tree_basic.cpp)
- **Python**: [`pose_tree_basic.py`](./pose_tree_basic/python/pose_tree_basic.py)

This example simulates the orbital mechanics of the Sun, Earth, and Moon. It's a comprehensive demonstration of how to set up the `PoseTreeManager`, update poses in one operator, and query them from another.

## Key Concepts Demonstrated

- **`PoseTreeManager`**: A `FragmentService` for creating and sharing a `PoseTree` instance.
- **Pose Tree API**: Using `create_frame`, `create_edges`, `set`, and `get` to manage and query poses.
- **Shared State**: Accessing a common resource (`PoseTree`) from multiple operators.
- **3D Math Types**: Using `Pose3d` and `SO3d` for transformations.

## Getting Started

Start with the [Pose Tree Basic](./pose_tree_basic) example to understand the fundamental concepts and see the API in action. The README inside that directory contains detailed build and run instructions.

## Note

The Pose Tree feature is marked as **experimental** in Holoscan SDK v3.4. The API may change in future releases.
