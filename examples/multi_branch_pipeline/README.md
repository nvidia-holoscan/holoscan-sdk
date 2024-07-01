# Create an application with multiple processing branches that compute at different rates

These examples demonstrate how to build an application configured such that a common source operator can act as a source to multiple processing pipelines that execute at different rates. By default, operators have a condition on each input port that requires a message to be available before the `compute` method will be called. Similarly, the default condition on an output port is that there is space any connected operators' input ports to receive a message. This defaults make it relatively easy to connect operators and have things execute in the expected order, but there are some scenarios where we may want to override the defaults. For example, consider the geometry of the application used in this example:

```
       increment1--rx1
      /
    tx
      \
       increment2--rx2
```

For the example above, with the default conditions on operator `tx`, it would not execute until both the `increment1` and `increment2` operators are ready to receive a message. This may be okay if both branches should execute at the same rate, but does not support a scenario where one branch runs at a faster rate than another.

This example shows how the default condition on `tx` can be disabled so that it sends a message regardless of whether there is space in each downstream operator's receiver queue. This also necessitates changing the receiver queue policy on `increment1` and `increment2` so that they can reject the incoming message if the queue is already full.

*Visit the [Schedulers section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/components/schedulers.html) to learn more about the schedulers.*

*See the operator creation guides section on input and output ports ([C++](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#specifying-operator-inputs-and-outputs-c) or [Python](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#specifying-operator-inputs-and-outputs-python)) for how to configure the condition on a port.*


## C++ API

This example shows a simple application using only native operators. There are three types of operators involved (see diagram above):
  1. a transmitter (`tx`), that transmits an integer value on port "out".
  2. increment operators (`increment1` and `increment2`) that increment the received value by a given amount and then transmits that new value
  3. receivers (`rx1` and `rx2`) that print their name and received value

The user can select the scheduler to be used by the application, but because there is more than one parallel path, it is recommended to use one of the multi-threaded schedulers in this scenario. The number of workers is controlled by the `worker_thread_number` parameter in `multi_branch_pipeline.yaml`.

The key point of this application is not in the details of the operators involved, but in how their connections are configured so that different branches of the pipeline can execute at different rates. See inline comments in the application code explaining how the output port of `tx` is configured and how the input port of `increment1` and `increment2` are configured.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Set values for `scheduler` and any corresponding parameters such as `worker_thread_number` in `multi_branch_pipeline.yaml`.

Then, run:
```bash
./examples/multi_branch_pipeline/cpp/multi_branch_pipeline
```

For the C++ application, the scheduler to be used can be set via the `scheduler` entry in `multi_branch_pipeline.yaml`. It defaults to `event_based` (an event-based multithread scheduler), but can also be set to either `multi_thread` (polling-based) or `greedy` (single thread).

## Python API

- `multi_branch_pipeline.py`: This example is the same as described for the C++ application above. The primary difference is that instead of using a YAML file for the configuration variables, all values are set via the command line. Call the script below with the `--help` option to get a full description of the command line parameters. By default a polling-based multithread scheduler will be used, but if `--event_based` is specified, the event-based multithread scheduler will be used instead.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the app with the options of your choice. For example

```bash
python3 ./examples/multi_branch_pipeline/python/multi_branch_pipeline.py --threads 5
```
