# Create an application using a round-robin broadcast / gather pattern

Consider an application with a serial pipeline where a specific operator (or sequence of operators) dominates the compute time. In that case it may be beneficial to launch multiple copies of this bottleneck operator in parallel, in a round-robin fashion so that multiple frames can be processed in parallel. As a concrete example, suppose we had the following simple serial pipeline:

```text
Tx --> SlowOp --> Rx
```

where:

1. Tx (`PingTxOp`): transmitter set to emit 100 messages (containing an integer) at a rate of 60 Hz
2. `SlowOp`: some arbitrary operation that operates at a rate substantially slower than the transmitter's frame rate. For this application we use just a sleep operation so that this operator cannot tick faster than 15 Hz.
3. Rx (`PingRxOp`): Receiver that just prints the integer that was received.

In this case, the `SlowOp` would bottleneck the operation to run at 15 Hz even though the transmitter could produce data at a rate of 60 Hz. A potential solution (assuming that `SlowOp` is not already using all CPU threads or GPU resources internally), is to allow staggered launch of multiple copies of `SlowOp` in parallel so that multiple frames can be processed in parallel. We then need to gather back the frames (while preserving the original frame order) into a common stream again for the receive operation. This modified application would look like this and is what is implemented in this example:

```text
                               |----> SlowOp1 ---->|
                               |----> SlowOp2 ---->|
Tx --> RoundRobinBroadcastOp --|----> SlowOp3 ---->| --> GatherOneOp --> Rx
                               |----> SlowOp4 ---->|
```

We also use `EventBasedScheduler` rather than the default `GreedyScheduler` so that multiple operators can be launched in parallel. The full set of operators used in this case are:

1. Tx (`PingTxOp`): transmitter set to emit 100 messages (containing an integer) at a rate of 60 Hz
2. `RoundRobinBroadCastOp`: This operator forwards whatever is received on its input ports to one of `num_broadcast` output ports.. As a concrete example, consider the case of `num_broadcast=4` as used in this example. In that case, the first message that is received is sent to `output001`, the second is sent to `output002`, this third to `output003`, the fourth to `output004` and then the fifth message would wrap back around to be sent on `output001` and so on. This dispatch mechanism is often referred to as round-robin dispatch. In this example single `SlowOp` takes approximately 4x longer than the transmit period, we choose to launch four copies of `SlowOp` in parallel so that the app can operate at close to the desired frame rate of 60 Hz.
3. `SlowOp`: some arbitrary operation that operates at a rate substantially slower than the transmitter's frame rate. For this application we use just a sleep operation so that this operator cannot tick faster the 15 Hz.
4. `GatherOneOp`: This operator has `num_gather` input ports and will forward the input from a specific input port to its single output port. We configured this operator to use a similar round-robin check of the input ports so that the order the frames are output by `GatherOneOp` will match the order that they were emitted by `Tx`. The implementation does this by only checking for messages on a specific input port until one has been received. It then increments which input port to check on subsequent compute calls. A different implementation of the `compute` method that checks all ports on each call could be used if it was not important to preserve the original frame order.
5. Rx (`PingRxOp`): Receiver that just prints the integer that was received.

:::{note}
Note that there is another parallel processing example in [examples/multithread](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/multithread). The primary difference there is that there is no `RoundRobinBroadCastOp` or `GatherOneOp`. Instead, each frame output by Tx is sent to ALL parallel branches rather than to each one sequentially. That pattern is more suited to when the branches implement **different** processing streams that each need to be applied to the same frame and could run in parallel. The case in this application is for when there is a single processing stream, but to remove a bottleneck we want to overlap the processing of multiple sequential frames in parallel.
:::

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions (C++)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Set values for `num_broadcast` and `worker_thread_number` as desired in `app_config.yaml`.

Then, run:
```bash
./examples/round_robin_parallel/cpp/round_robin
```

It is recommended to keep `worker_thread_number` of at least 4 to allow up to four `SlowOp` operators to run in parallel. The application has 8 total operators, so 8 threads is a reasonable default.

For the C++ application, the scheduler to be used can be set via the `scheduler` entry in `round_robin.yaml`. It defaults to `event_based` (an event-based multithread scheduler), but can also be set to `greedy` (single threaded). If the greedy scheduler is used, no benefit will be seen from connecting the four `SlowOp` operators in parallel because the scheduler is only allowed to execute one at a time.

### Run instructions (Python)

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the app with the options of your choice. For example, to use 8 worker threads, one would set:

```bash
python3 ./examples/round_robin_parallel/python/round_robin.py --threads 8
```

A minimum of 4 threads should be used in order to allow up to four `SlowOp` operators to run in parallel. The application has 8 total operators, so 8 threads is a reasonable default.

If `--threads 0` is specified, the greedy scheduler will be used instead of this app's default choice of the `EventBasedScheduler`. In that case no benefit will be seen from connecting the four `SlowOp` operators in parallel because the scheduler is only allowed to execute one at a time.
