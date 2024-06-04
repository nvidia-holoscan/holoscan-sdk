# Create an application using the multi-threaded scheduler

These examples demonstrate how to build an application configured to use a multi-threaded scheduler. This application creates a user-controlled number of delay operators that can be run in parallel. If run in a single thread, the application's duration is approximately the sum of the delays of the individual delay operators. When all operators are run in parallel, the duration is roughly equal to the longest duration of any of the individual operators.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/schedulers.html) to learn more about the Multi-threaded scheduler.*

## C++ API

This example shows a simple application using only native operators. There are three operators involved:
  1. a transmitter, set to transmit the integer value 0 on port "out"
  2. a set of delay operators that prints the received value, waits a specified delay, and then increments by a user-specified value. The incremented value is then transmitted.
  3. a receiver that prints the sum of all received values

The user can configure the number of delay operators via the `num_delay_op` parameter in `app_config.yaml`. The first of these will have a delay equal to the value of `delay` in `app_config.yaml`. The jth one of these will have the value `delay + i * delay_step` where `delay_step` is also set in `app_config.yaml`.

The number of workers used by the multi-threaded scheduler is controlled by the `worker_thread_number` parameter in `app_config.yaml`.

Data Flow Tracking is also optionally enabled by changing the `tracking` field in the YAML file. It is set to `false` by default. Other options in the YAML include a `silent` option to suppress verbose output from the operators (`false` by default) and a `count` option that can be used to control how many messages are sent from the transmitter (1 by default).

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Set values for `num_operator`, `delay`, `delay_step` and `worker_thread_number` as desired in `app_config.yaml`.

Then, run:
```bash
./examples/multithread/cpp/multithread
```

For the C++ application, the scheduler to be used can be set via the `scheduler` entry in `ping_async.yaml`. It defaults to `event_based` (an event-based multithread scheduler), but can also be set to either `multi_thread` (polling-based) or `greedy` (single thread).

## Python API

- `multithread.py`: This example demonstrates how to configure and use a multi-threaded scheduler instead of the default single-threaded one. It involves three operators as described for the C++ API example described above. The primary difference is that instead of using a YAML file for the configuration variables, all values are set via the command line. Call the script below with the `--help` option to get a full description of the command line parameters. By default a polling-based multithread scheduler will be used, but if `--event-based` is specified, the event-based multithread scheduler will be used instead.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the app with the options of your choice. For example, to use 8 worker threads to run 32 delay operators with delays ranging linearly from 0.2 to (0.2 + 0.05 * 31), one would set:

```bash
python3 ./examples/multithread/python/multithread.py --threads 8 --num_delay_ops 32 --delay 0.2 --delay_step 0.05 --event-based
```
