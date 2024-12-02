(holoscan-flow-tracking)=
# Data Flow Tracking

The Holoscan SDK provides the Data Flow Tracking APIs as a mechanism to profile your application and analyze the fine-grained timing properties and data flow between operators in the graph of a fragment.

Currently, data flow tracking is only supported between the root operators and leaf operators of a graph and in simple cycles in a graph (support for tracking data flow between any pair of operators in a graph is planned for the future).

- A *root operator* is an operator without any predecessor nodes.
- A *leaf operator* (also known as a *sink operator*) is an operator without any successor nodes.

When data flow tracking is enabled, every message is tracked from the root operators to the leaf operators and in cycles. Then, the maximum (worst-case), average, and minimum end-to-end latencies of one or more paths can be retrieved using the Data Flow Tracking APIs.

:::{tip}
- The end-to-end latency between a root operator and a leaf operator is the time taken between the start of a root operator and the end of a leaf operator. Data Flow Tracking enables the support to track the end-to-end latency of every message being passed between a root operator and a leaf operator.
- The reported end-to-end latency for a cyclic path is the time taken between the start of the first operator of a cycle and the time when a message is again received by the first operator of the cycle.
:::

The API also provides the ability to retrieve the number of messages sent from the root operators.

:::{tip}

- The Data Flow Tracking feature is also illustrated in the [flow_tracker](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/flow_tracker)
- Look at the {cpp:class}`C++ <holoscan::DataFlowTracker>` and {py:class}`python <holoscan.core.DataFlowTracker>` API documentation for exhaustive definitions
:::

## Enabling Data Flow Tracking

Before an application ({cpp:class}`C++ <holoscan::Application>`/{py:class}`python <holoscan.core.Application>`) is run with the `run()` method, data flow tracking can be enabled. For single fragment applications, this can be done by calling the `track()` method in {cpp:func}`C++ <holoscan::Fragment::track>` and using the `Tracker` class in {py:class}`python <holoscan.core.Tracker>`.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:emphasize-lines: 2
:name: holoscan-enable-data-flow-tracking-cpp
auto app = holoscan::make_application<MyPingApp>();
auto& tracker = app->track(); // Enable Data Flow Tracking
// Change tracker and application configurations
...
app->run();
```
````
````{tab-item} Python
```{code-block} python
:emphasize-lines: 2
:name: holoscan-enable-data-flow-tracking-python
from holoscan.core import Tracker
...
app = MyPingApp()
with Tracker(app) as tracker:
  # Change tracker and application configurations
  ...
  app.run()
```
````
`````

## Enabling Data Flow Tracking for Distributed Applications

For distributed (multi-fragment) applications, a separate tracker object is used for each Fragment so the API is slightly different than in the single fragment case.


`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:emphasize-lines: 2
:name: holoscan-enable-data-flow-tracking-cpp
auto app = holoscan::make_application<MyPingApp>();
auto trackers = app->track_distributed(); // Enable data flow tracking for a distributed app
// Change tracker and application configurations
...
app->run();
```
Note that instead of a returning a single `DataFlowTracker*` like `track`, the `track_distributed` method returns a `std::unordered_map<std::string, DataFlowTracker*>` where the keys are the names of the fragments.
````
````{tab-item} Python
```{code-block} python
with Tracker(app) as trackers:
    app.run()
```
The `Tracker` context manager detects whether the app is distributed and returns a `dict[str, DataFlowTracker]` as `trackers` in the distributed case. For a single fragment application, the returned value is just a single `DataFlowTracker` object.
````
`````

## Retrieving Data Flow Tracking Results

After an application has been run, data flow tracking results can be accessed by
various methods on the DataFlowTracker ({cpp:class}`C++ <holoscan::DataFlowTracker>`/{py:class}`python <holoscan.core.DataFlowTracker>`) class.

1. `print()` ({cpp:func}`C++ <holoscan::DataFlowTracker::print>`/{py:func}`python <holoscan.core.DataFlowTracker.print>`)
   - Prints all data flow tracking results including end-to-end latencies and the number of
     source messages to the standard output.

2. `get_num_paths()` ({cpp:func}`C++ <holoscan::DataFlowTracker::get_num_paths>`/{py:func}`python <holoscan.core.DataFlowTracker.get_num_paths>`)
   - Returns the number of paths between the root operators and the leaf operators.

3. `get_path_strings()` ({cpp:func}`C++ <holoscan::DataFlowTracker::get_path_strings>`/{py:func}`python <holoscan.core.DataFlowTracker.get_path_strings>`)
   - Returns a vector of strings, where each string represents a path between the root operators
     and the leaf operators. A path is a comma-separated list of operator names.

4. `get_metric()` ({cpp:func}`C++ <holoscan::DataFlowTracker::get_metric>`/{py:func}`python <holoscan.core.DataFlowTracker.get_metric>`)
   - Returns the value of different metrics based on the arguments.
   - `get_metric(std::string pathstring, holoscan::DataFlowMetric metric)` returns the value of a
     metric `metric` for a path `pathstring`. The metric can be one of the following:
        - `holoscan::DataFlowMetric::kMaxE2ELatency`
        ({py:const}`python <holoscan.core.DataFlowMetric.MAX_E2E_LATENCY>`): the maximum end-to-end latency in the path.
        - `holoscan::DataFlowMetric::kAvgE2ELatency` ({py:const}`python <holoscan.core.DataFlowMetric.AVG_E2E_LATENCY>`): the average end-to-end latency in the path.
        - `holoscan::DataFlowMetric::kMinE2ELatency` ({py:const}`python <holoscan.core.DataFlowMetric.MIN_E2E_LATENCY>`): the minimum end-to-end latency in the path.
        - `holoscan::DataFlowMetric::kMaxMessageID` ({py:const}`python <holoscan.core.DataFlowMetric.MAX_MESSAGE_ID>`): the message number or ID which resulted in the
          maximum end-to-end latency.
        - `holoscan::DataFlowMetric::kMinMessageID` ({py:const}`python <holoscan.core.DataFlowMetric.MIN_MESSAGE_ID>`): the message number or ID which resulted in the
          minimum end-to-end latency.
   - `get_metric(holoscan::DataFlowMetric metric = DataFlowMetric::kNumSrcMessages)` returns a map of source operator and its edge, and the number of messages sent from the source operator to the edge.

In the {ref}`above example <holoscan-enable-data-flow-tracking-cpp>`, the data flow tracking results can be printed to the standard output like the
following:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:emphasize-lines: 6
:name: holoscan-enable-data-flow-tracking-results-cpp
auto app = holoscan::make_application<MyPingApp>();
auto& tracker = app->track(); // Enable Data Flow Tracking
// Change application configurations
...
app->run();
tracker.print();
```
````
````{tab-item} Python
```{code-block} python
:emphasize-lines: 6
:name: holoscan-one-operator-workflow-python
from holoscan.core import Tracker
...
app = MyPingApp()
with Tracker(app) as trackers:
  # Change tracker and application configurations
  ...
  app.run()
  tracker.print()
```
````
`````

If this was a distributed application, there would instead be a separate `DataFlowTracker` for each fragment. The overall flow tracking results for all fragments can be printed as in the following:

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:emphasize-lines: 6-10
:name: holoscan-enable-data-flow-tracking-results-cpp
auto app = holoscan::make_application<MyPingApp>();
auto trackers = app->track_distributed(); // Enable data flow tracking for a distributed app
// Change application configurations
...
app->run();
// print the data flow tracking results
for (const auto& [name, tracker] : trackers) {
  std::cout << "Fragment: " << name << std::endl;
  tracker->print();
}
```
````
````{tab-item} Python
```{code-block} python
:emphasize-lines: 6-9
:name: holoscan-one-operator-workflow-python
from holoscan.core import Tracker
...
app = MyPingApp()
with Tracker(app) as trackers:
  # Change tracker and application configurations
  ...
  app.run()
  # print the data flow tracking results
  for fragment_name, tracker in trackers.items():
      print(f"Fragment: {fragment_name}")
      tracker.print()
```
````
`````

## Customizing Data Flow Tracking

Data flow tracking can be customized using a few optional configuration parameters. The `track()` method ({cpp:func}`C++ <holoscan::Fragment::track>`//{py:func}`Python <holoscan.core.Application.track>`) (or `track_distributed` method ({cpp:func}`C++ <holoscan::Application::track_distributed>`/{py:func}`Python <holoscan.core.Application.track_distributed>`)` for distributed apps) can be configured to skip a few messages at the beginning of an application's execution as a *warm-up* period. It is also possible to discard a few messages at the end of an application's run as a *wrap-up* period. Additionally, outlier end-to-end latencies can be ignored by setting a latency threshold value (in ms) which is the minimum latency below which the observed latencies are ignored.
Finally, it is possible to limit the timestamping of messages at all nodes except the root and leaf
operators, so that the overhead of timestamping and sending timestamped messages are reduced. In
this way, end-to-end latencies are still calculated, but pathwise fine-grained data are not stored
for unique pairs of root and leaf operators.

For Python, it is recommended to use the {py:class}`Tracker<holoscan.core.Tracker>` context manager class instead of the `track` or `track_distributed` methods. This class will autodetect if the application is a single fragment or distributed app, using the appropriate method for each.

:::{tip}
For effective benchmarking, it is common practice to include warm-up and cool-down periods by skipping the initial and final messages.
:::

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:caption: Optional parameters to `track()`
Fragment::track(uint64_t num_start_messages_to_skip = kDefaultNumStartMessagesToSkip,
                         uint64_t num_last_messages_to_discard = kDefaultNumLastMessagesToDiscard,
                         int latency_threshold = kDefaultLatencyThreshold,
                         bool is_limited_tracking = false);
```
````
````{tab-item} Python
```{code-block} python
:caption: Optional parameters to `Tracker`
Tracker(num_start_messages_to_skip=num_start_messages_to_skip,
        num_last_messages_to_discard=num_last_messages_to_discard,
        latency_threshold=latency_threshold,
        is_limited_tracking=False)
```
````
`````
The default values of these parameters of `track()` are as follows:
- `kDefaultNumStartMessagesToSkip`: 10
- `kDefaultNumLastMessagesToDiscard`: 10
- `kDefaultLatencyThreshold`: 0 (do not filter out any latency values)
- `is_limited_tracking`: false

These parameters can also be configured using the helper functions:
{cpp:func}`set_skip_starting_messages <holoscan::DataFlowTracker::set_skip_starting_messages>`,
{cpp:func}`set_discard_last_messages <holoscan::DataFlowTracker::set_discard_last_messages>`,
{cpp:func}`set_skip_latencies <holoscan::DataFlowTracker::set_skip_latencies>`,
and {cpp:func}`set_limited_tracking <holoscan::DataFlowTracker::set_limited_tracking>`,

## Logging

The Data Flow Tracking API provides the ability to log every message's graph-traversal information to a file. This enables you to analyze the data flow at a granular level. When logging is enabled, every message's received and sent timestamps at every operator between the root and the leaf operators are logged after a message has been processed at the leaf operator.

The logging is enabled by calling the `enable_logging` method in {cpp:func}`C++ <holoscan::DataFlowTracker::enable_logging>` and by providing the `filename` parameter to `Tracker` in {py:class}`python <holoscan.core.Tracker>`.

`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:emphasize-lines: 3
:name: holoscan-flow-tracking-logging-cpp
auto app = holoscan::make_application<MyPingApp>();
auto& tracker = app->track(); // Enable Data Flow Tracking
tracker.enable_logging("logging_file_name.log");
...
app->run();
```
````
````{tab-item} Python
```{code-block} python
:emphasize-lines: 4
:name: holoscan-flow-tracking-logging-python
from holoscan.core import Tracker
...
app = MyPingApp()
with Tracker(app, filename="logger.log") as tracker:
   ...
   app.run()
```
````
`````

The logger file logs the paths of the messages after a leaf operator has finished its `compute` method. Every path in the logfile includes an array of tuples of the form:

>"(root operator name, message receive timestamp, message publish timestamp) -> ... -> (leaf operator name, message receive timestamp, message publish timestamp)".

This log file can further be analyzed to understand latency distributions, bottlenecks, data flow,
and other characteristics of an application.

## Configuring Clock Synchronization in Multiple Machines for Distributed Application Flow Tracking

For flow tracking in distributed applications that span multiple machines, system administrators
must ensure that the clocks of all machines are synchronized. It is up to the administrator's
preference on how to synchronize the clocks. [Linux PTP](https://linuxptp.sourceforge.net/) is a
popular and commonly used mechanism for clock synchronization.

Install the `linuxptp` package on all machines:

```bash
git clone http://git.code.sf.net/p/linuxptp/code linuxptp
cd linuxptp/
make
sudo make install
```

:::{tip}
The Ubuntu `linuxptp` package can also be used. However, the above repository provides access to
different PTP configurations.
:::

### Check PTP Hardware Timestamping Support

Check if your machine and network interface card supports PTP hardware timestamping:

```bash
$ sudo apt-get update && sudo apt-get install ethtool
$ ethtool -T <interface_name>
```

If the output of the above command is like the one provided below, it means PTP hardware
timestamping may be supported:

```bash
$ ethtool -T eno1
Time stamping parameters for eno1:
Capabilities:
	hardware-transmit     (SOF_TIMESTAMPING_TX_HARDWARE)
	software-transmit     (SOF_TIMESTAMPING_TX_SOFTWARE)
	hardware-receive      (SOF_TIMESTAMPING_RX_HARDWARE)
	software-receive      (SOF_TIMESTAMPING_RX_SOFTWARE)
	software-system-clock (SOF_TIMESTAMPING_SOFTWARE)
	hardware-raw-clock    (SOF_TIMESTAMPING_RAW_HARDWARE)
PTP Hardware Clock: 0
Hardware Transmit Timestamp Modes:
	off                   (HWTSTAMP_TX_OFF)
	on                    (HWTSTAMP_TX_ON)
Hardware Receive Filter Modes:
	none                  (HWTSTAMP_FILTER_NONE)
	all                   (HWTSTAMP_FILTER_ALL)
	ptpv1-l4-sync         (HWTSTAMP_FILTER_PTP_V1_L4_SYNC)
	ptpv1-l4-delay-req    (HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ)
	ptpv2-l4-sync         (HWTSTAMP_FILTER_PTP_V2_L4_SYNC)
	ptpv2-l4-delay-req    (HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ)
	ptpv2-l2-sync         (HWTSTAMP_FILTER_PTP_V2_L2_SYNC)
	ptpv2-l2-delay-req    (HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ)
	ptpv2-event           (HWTSTAMP_FILTER_PTP_V2_EVENT)
	ptpv2-sync            (HWTSTAMP_FILTER_PTP_V2_SYNC)
	ptpv2-delay-req       (HWTSTAMP_FILTER_PTP_V2_DELAY_REQ)
```

However, if the output is the one provided below, it means PTP hardware timestamping is not supported:

```bash
$ ethtool -T eno1
$ ethtool -T eno1
Time stamping parameters for eno1:
Capabilities:
	software-transmit
	software-receive
	software-system-clock
PTP Hardware Clock: none
Hardware Transmit Timestamp Modes: none
Hardware Receive Filter Modes: none
```

### Without PTP Hardware Timestamping Support

Even if PTP hardware timestamping is not supported, it is possible to synchronize the clocks of
different machines using software-based clock synchronization. Here, we show an example of how to synchronize the clocks of two machines using
the [automotive PTP profiles](https://linuxptp.nwtime.org/documentation/). Developers and administrators can use their own profiles.

Select one machine as the clock server and the others as the clients. On the server, run the following command:

```bash
sudo ptp4l -i eno1 -f linuxptp/configs/automotive-master.cfg -m -S
ptp4l[7526757.990]: port 1 (eno1): INITIALIZING to MASTER on INIT_COMPLETE
ptp4l[7526757.991]: port 0 (/var/run/ptp4l): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[7526757.991]: port 0 (/var/run/ptp4lro): INITIALIZING to LISTENING on INIT_COMPLETE
```

On the clients, run the following command:

```bash
$ sudo ptp4l -i eno1 -f linuxptp/configs/automotive-slave.cfg -m -S
ptp4l[7370954.836]: port 1 (eno1): INITIALIZING to SLAVE on INIT_COMPLETE
ptp4l[7370954.836]: port 0 (/var/run/ptp4l): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[7370954.836]: port 0 (/var/run/ptp4lro): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[7370956.785]: rms 5451145770 max 5451387307 freq -32919 +/-   0 delay 72882 +/-   0
ptp4l[7370957.785]: rms 5451209853 max 5451525811 freq -32919 +/-   0 delay 71671 +/-   0
...
... wait until rms value drops in the range of orders of microseconds
ptp4l[7371017.791]: rms 196201 max 324853 freq -13722 +/- 34129 delay 73814 +/-   0
ptp4l[7371018.791]: rms 167568 max 249998 freq  +6509 +/- 30532 delay 73609 +/-   0
ptp4l[7371019.791]: rms 158762 max 216309 freq  -8778 +/- 28459 delay 73060 +/-   0
```

`CLOCK_REALTIME` on both the Linux machines are synchronized to the range of microseconds. Now,
different fragments of a distributed application can be run on these machines, with flow tracking,
end-to-end latency of an application can be measured across these machines.

Eventually, the `ptp4l` commands can be added as system-d services to start automatically on boot.

### With PTP Hardware Timestamping Support

If PTP hardware timestamping is supported, the physical clock of the network interface card can be
synchronized to the system clock, `CLOCK_REALTIME`. This can be done by running the following
commands

```bash
$ sudo ptp4l -i eno1 -f linuxptp/configs/gPTP.cfg --step_threshold=1 -m &
ptp4l[7527677.746]: port 1 (eno1): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[7527677.747]: port 0 (/var/run/ptp4l): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[7527677.747]: port 0 (/var/run/ptp4lro): INITIALIZING to LISTENING on INIT_COMPLETE
ptp4l[7527681.663]: port 1 (eno1): LISTENING to MASTER on ANNOUNCE_RECEIPT_TIMEOUT_EXPIRES
ptp4l[7527681.663]: selected local clock f02f74.fffe.cb3590 as best master
ptp4l[7527681.663]: port 1 (eno1): assuming the grand master role


$ sudo pmc -u -b 0 -t 1 "SET GRANDMASTER_SETTINGS_NP clockClass 248 \
        clockAccuracy 0xfe offsetScaledLogVariance 0xffff \
        currentUtcOffset 37 leap61 0 leap59 0 currentUtcOffsetValid 1 \
        ptpTimescale 1 timeTraceable 1 frequencyTraceable 0 \
        timeSource 0xa0"
sending: SET GRANDMASTER_SETTINGS_NP
ptp4l[7527704.409]: port 1 (eno1): assuming the grand master role
	f02f74.fffe.cb3590-0 seq 0 RESPONSE MANAGEMENT GRANDMASTER_SETTINGS_NP 
		clockClass              248
		clockAccuracy           0xfe
		offsetScaledLogVariance 0xffff
		currentUtcOffset        37
		leap61                  0
		leap59                  0
		currentUtcOffsetValid   1
		ptpTimescale            1
		timeTraceable           1
		frequencyTraceable      0
		timeSource              0xa0


$ sudo phc2sys -s eno1 -c CLOCK_REALTIME --step_threshold=1 --transportSpecific=1 -w -m
phc2sys[7527727.996]: ioctl PTP_SYS_OFFSET_PRECISE: Invalid argument
phc2sys[7527728.997]: CLOCK_REALTIME phc offset   7422791 s0 freq    +628 delay   1394
phc2sys[7527729.997]: CLOCK_REALTIME phc offset   7422778 s1 freq    +615 delay   1474
phc2sys[7527730.997]: CLOCK_REALTIME phc offset       118 s2 freq    +733 delay   1375
phc2sys[7527731.997]: CLOCK_REALTIME phc offset        57 s2 freq    +708 delay   1294
phc2sys[7527732.998]: CLOCK_REALTIME phc offset       -42 s2 freq    +626 delay   1422
phc2sys[7527733.998]: CLOCK_REALTIME phc offset        52 s2 freq    +707 delay   1392
phc2sys[7527734.998]: CLOCK_REALTIME phc offset       -65 s2 freq    +606 delay   1421
phc2sys[7527735.998]: CLOCK_REALTIME phc offset       -48 s2 freq    +603 delay   1453
phc2sys[7527736.999]: CLOCK_REALTIME phc offset        -2 s2 freq    +635 delay   1392
```

From here on, clocks on other machines can also be synchronized to the above server clock.

Further references:

- [Synchronizing Time with Linux PTP](https://tsn.readthedocs.io/timesync.html)
- [Linux PTP Documentation and Configurations](https://linuxptp.nwtime.org/documentation/)
