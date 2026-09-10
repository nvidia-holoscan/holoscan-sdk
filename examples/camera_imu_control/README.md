# baseline camera/IMU/fusion/control graph

> **User guide:** [Camera + IMU Control](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/camera-imu-control) —
> this hero example is the complete camera, IMU, fusion, and control reference path.

This graph-definition executable demonstrates an embedded fusion graph, a
bounded host `Tensor` producer, an IMU external import, bracketed-window
selection, and a control-result external export. The external endpoint names
are stable and explicit:

- `baseline.camera_imu.imu` — `uint32_t` samples published by `sensor_daemon`;
- `baseline.camera_imu.control` — `uint32_t` results consumed by
  `result_recorder`.

Run the bounded reference path directly:

```sh
./camera_imu_control
```

The executable starts one real `RunSession` plus an independent public
`ChannelRuntime` client. The client publishes captured IMU samples and consumes
one control result through the same catalog contract as the standalone
companions. The publish helper treats a successful result as delivery. Within
its deadline, it retries only the proven pre-submit `kNotReady` and
`kBackpressured` outcomes. `kPublicationPartial` and
`kPublicationIndeterminate` are terminal; every other error is returned for
code-specific handling. The consumer ignores camera-only `SafeDefault` results
and completes only after the value proves contribution from one or more known
positive IMU samples. The executable then stops and joins deterministically.
This in-process client is an executable smoke path, not a replacement for the
separate-process sensor-daemon and recorder examples.

Generate the complete four-file deployment bundle without starting lifecycle,
CUDA, Channel services, or helper threads:

```sh
cmake --build build --target camera_imu_control_graph_artifacts
```

or invoke the standardized cold mode directly:

```sh
./camera_imu_control --emit-artifacts \
  --artifact-root ./holoscan-artifacts --deployment-id developer
```

Pass the exact emitted `endpoint-catalog.json` path to both standalone clients;
they never search for a global "latest" catalog. The executable deliberately
keeps artifact publication separate from normal application execution so build
tooling cannot start lifecycle, sensors, or transport as a side effect.
Rebuilding the explicit artifact target preserves the timestamps of unchanged
canonical artifacts.
