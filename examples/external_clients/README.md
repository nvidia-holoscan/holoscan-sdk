# Standalone external clients

> **User guide:** [External Clients](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/external-clients) —
> these companion programs demonstrate standalone Channel clients outside a graph.

The C++ and Python variants are ordinary non-Operator processes. Each receives the exact
`endpoint-catalog.json` path produced by the graph-definition executable, then
attaches by the explicit deployment-stable endpoint name. Neither program
includes HoloMQ or iceoryx2 headers.

The sensor daemon publishes `uint32_t` IMU samples with caller-owned capture
coordinates. The recorder uses `wait_and_receive_sample()` to block on the
endpoint's event lane rather than polling. By default it consumes exported
control results until it observes remote `kClosed`; a positive optional
`max-samples` argument gives intentionally unbounded graphs a bounded successful
client exit without claiming EOS. It prints public `SampleMetadata` plus the
normalized catalog digest and then closes its local subscriber endpoint. The
publisher side has no event-wake equivalent, so the daemon still retries
against a deadline. Producer-epoch lifecycle evidence remains Runtime trace
data; it is not invented as a `SampleMetadata` field. The daemon propagates
connection-refresh failures, considers a sample delivered only after a
successful publication result, and retries only pre-submit `kNotReady` and
`kBackpressured` outcomes within a finite deadline. Partial and indeterminate
publications are terminal and are never retried. It reports finite-stream
completion only after the ordered EOS close succeeds. In a finite input-driven
graph, that EOS follows accepted data through graph outputs and permits natural
completion after retained work drains. They are intentionally small acceptance
companions, not a general device adapter or Recorder product.

Use the Python variants with the same command-line shape:

```sh
python3 sensor_daemon/python/sensor_daemon.py \
  <catalog.json> <endpoint> [count] [period-ms] [first-value]
python3 recorder/python/recorder.py \
  <catalog.json> <endpoint> [max-samples] [--async]
```

The optional Python recorder mode consumes through
`hs.channel.aio.Subscriber` and verifies that the catalog readiness descriptor
does not create a Python helper thread.

Both languages require a separately running deployed graph that owns the catalog
endpoints. `camera_imu_control --emit-artifacts` writes the deployment
bundle but does not start the graph or its Channel services. The SDK does not ship
an author-facing graph-only launcher for that manual three-process recipe; use
the no-argument camera/IMU reference executable for an end-to-end smoke test.
When a deployment does provide the hero graph process, use
`result_recorder <catalog.json> baseline.camera_imu.control 1`; the independent
camera clock keeps that export live after the finite IMU producer closes.

For a finite self-contained acceptance run, `python/run_clients.py` launches
both Python clients against `external_client_relay` or the matching Python
relay graph. The CTests exercise both graph-language variants, plus the async
recorder against the C++ graph, and verify data-before-EOS ordering across
three independent processes.
