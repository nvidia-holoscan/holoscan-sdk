# Standalone external clients

> **User guide:** [External Clients](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/external-clients) —
> these companion programs demonstrate standalone Channel clients outside a graph.

These programs are ordinary non-Operator processes. Each receives the exact
`endpoint-catalog.json` path produced by the graph-definition executable, then
attaches by the explicit deployment-stable endpoint name. Neither program
includes HoloMQ or iceoryx2 headers.

The sensor daemon publishes `uint32_t` IMU samples with caller-owned capture
coordinates. The recorder consumes the exported control result with
`wait_and_receive_sample()`, which blocks on the endpoint's event lane until a
sample arrives or its absolute deadline passes rather than polling, and prints
public `SampleMetadata` plus the normalized catalog digest. The publisher side
has no event-wake equivalent, so the daemon still retries against a deadline. Producer-epoch lifecycle evidence remains Runtime trace data;
it is not invented as a `SampleMetadata` field. The daemon propagates
connection-refresh failures, considers a sample delivered only after a
successful publication result, and retries only pre-submit `kNotReady` and
`kBackpressured` outcomes within a finite deadline. Partial and indeterminate
publications are terminal and are never retried. It reports finite-stream
completion only after the ordered EOS close succeeds. They are intentionally
small acceptance companions, not a general device adapter or Recorder product.

Both clients require a separately running deployed graph that owns the catalog
endpoints. `camera_imu_control --emit-artifacts` writes the deployment
bundle but does not start the graph or its Channel services. The SDK does not ship
an author-facing graph-only launcher for that manual three-process recipe; use
the no-argument camera/IMU reference executable for an end-to-end smoke test.
