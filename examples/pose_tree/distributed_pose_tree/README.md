# Distributed Pose Tree Example

This example demonstrates how to use the `PoseTree` feature in distributed applications built with Holoscan SDK. It simulates the orbital mechanics of the Sun, Earth, and Moon to showcase how to manage and query dynamic coordinate frame transformations over time.

## Overview

The application shows how to:
1.  Create a `PoseTreeManager` as a shared resource to manage a `PoseTree`.
2.  Share the `PoseTree` between multiple operators (`OrbitSetterOp` and `TransformPrinterOp`).
3.  Use one operator (`OrbitSetterOp`) in a source fragment to update poses in the tree based on a simulated clock.
4.  Use another operator (`TransformPrinterOp`) in a destination fragment to query the transformations between different coordinate frames at specific times.
5.  The simulation runs for 365 days, printing the poses at each daily step.

## C++ API

The C++ application consists of three main components:
1.  `OrbitSetterOp`: At each step, it calculates the new positions and orientations of the Earth and Moon and updates their poses in the shared `PoseTree`. It uses a simulated clock that advances by one day per tick.
2.  `TransformPrinterOp`: At each step, it queries the `PoseTree` for the transformations between `sun`, `earth`, and `moon` frames and logs them to the console.
3.  `PoseTreeOrbitApp`: The main application class that sets up the `PoseTreeManager` resource and connects the two operators.

The example demonstrates how pose tree information can be transmitted between fragments of a distributed application.
The example also demonstrates how to define a custom `fmt::formatter` for `holoscan::Pose3d` to enable easy printing with `HOLOSCAN_LOG_INFO`.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/pose_tree/distributed_pose_tree/cpp/distributed_pose_tree
```

## Python API

The Python implementation mirrors the C++ version, demonstrating the same concepts using the Python API. It includes:
1.  `OrbitSetterOp`: A Python operator that updates poses in the `PoseTree`. It uses `numpy` and `holoscan.pose_tree.SO3` to calculate transformations.
2.  `TransformPrinterOp`: A Python operator that queries and prints the poses.
3.  `PoseTreeOrbitApp`: The Python application that configures the `PoseTreeManager` and the workflow.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/pose_tree/distributed_pose_tree/python/distributed_pose_tree.py
```

## Key API Features Demonstrated

- `holoscan::PoseTreeManager`: Used as a `FragmentService` to provide a shared `PoseTree` instance to multiple operators.
- `PoseTree` API:
    - `create_frame()`: To define new coordinate frames (e.g., "sun", "earth", "moon").
    - `create_edges()`: To define the relationship between frames.
    - `set()`: To update the transformation (pose) of a frame relative to another at a specific timestamp.
    - `get()`: To query the transformation between any two frames at a given timestamp.
- `holoscan::Pose3d` and `holoscan::SO3d` (C++) / `holoscan.pose_tree.Pose3` and `holoscan.pose_tree.SO3` (Python): For representing 3D poses and rotations.

## Distributed Configuration

**Important**: The `PoseTree` configuration is identical for both single fragment and distributed applications. There are no special configuration requirements when using pose trees across multiple fragments.

The `PoseTreeManager` is registered as an application-level service using `register_service()`, which automatically makes the shared `PoseTree` available to all operators across all fragments within the application. This seamless sharing is handled behind the scenes by the Holoscan SDK.

### Configuration comparison:

**Single Fragment (pose_tree_basic):**
```cpp
auto pose_tree_manager = make_resource<PoseTreeManager>(
    "pose_tree_manager",
    from_config("pose_tree_config"),
    Arg("number_frames", static_cast<int32_t>(64)));
register_service(pose_tree_manager);
```

**Distributed (multiple fragments):**
```cpp
auto pose_tree_manager = make_resource<PoseTreeManager>(
    "pose_tree_manager",
    from_config("pose_tree_config"),
    Arg("number_frames", static_cast<int32_t>(64)));
register_service(pose_tree_manager);
```

Both configurations are identical. Operators in different fragments access the same shared pose tree instance through the service interface, ensuring data consistency across the entire distributed application.

### Time Synchronization in Distributed Applications

A key difference between the basic and distributed examples is how simulation time is coordinated:

**Basic Example (Single Fragment):**
Both operators maintain independent simulation clocks that advance in lockstep:
- `OrbitSetterOp`: `sim_time_ += kDay;`
- `TransformPrinterOp`: `sim_time_ += kDay;`

**Distributed Example (Multiple Fragments):**
The simulation time is coordinated between fragments through data flow:
- `OrbitSetterOp` (Fragment1): Calculates time and emits it via `op_output.emit(now, "sim_time");`
- `TransformPrinterOp` (Fragment2): Receives time via `auto sim_time = op_input.receive<double>("sim_time");`

This approach ensures that both fragments operate with the same simulation time, which is crucial for querying pose transformations at consistent timestamps across the distributed application. The fragments are connected using:

```cpp
add_flow(fragment1, fragment2, {{"orbit_setter_op", "transform_printer_op"}});
```

This connection allows the `sim_time` port to transmit timing information from the source fragment to the destination fragment, maintaining temporal coherence across the distributed system.

## Expected Output

The application will run for 365 steps (simulated days). For each day, it will print the poses for the Sun->Earth, Earth->Moon, and Sun->Moon transformations. The output will look similar to this:

C++:
```text
[day 1] Sun → Earth : T=(1.000,0.017,0.000) Q=(0.000,0.000,0.713,0.701)
[day 1] Earth → Moon : T=(0.002,0.001,0.000) Q=(0.000,0.000,0.115,0.993)
[day 1] Sun → Moon : T=(0.999,0.020,0.000) Q=(0.000,0.000,0.789,0.615)
...
[day 365] Sun → Earth : T=(1.000,-0.004,0.000) Q=(0.000,0.000,-0.706,-0.709)
[day 365] Earth → Moon : T=(-0.002,0.002,0.000) Q=(-0.000,0.000,-0.905,-0.425)
[day 365] Sun → Moon : T=(0.998,-0.006,0.000) Q=(0.000,0.000,0.941,-0.337)
```

Python:
```
[day    1] Sun→Earth : T=[0.99985204 0.01720158 0.        ], Q=SO3(quaternion=       0        0 0.713163 0.700999)
[day    1] Earth→Moon: T=[0.00249519 0.00058419 0.        ], Q=SO3(quaternion=       0        0 0.114739 0.993396)
[day    1] Sun→Moon  : T=[0.99922501 0.01968635 0.        ], Q=SO3(quaternion=       0        0 0.788885 0.614541)
...
[day  365] Sun→Earth : T=[ 0.99999075 -0.00430059  0.        ], Q=SO3(quaternion=        0         0 -0.705585 -0.708626)
[day  365] Earth→Moon: T=[-0.00163568  0.00197276  0.        ], Q=SO3(quaternion=       -0         0 -0.905062  -0.42528)
[day  365] Sun→Moon  : T=[ 0.99801097 -0.00592778  0.        ], Q=SO3(quaternion=        0         0  0.941421 -0.337234)
```

The application will terminate after completing the 365-day simulation.

## Note

The Pose Tree feature is marked as **experimental** in Holoscan SDK v3.4. The API may change in future releases.
