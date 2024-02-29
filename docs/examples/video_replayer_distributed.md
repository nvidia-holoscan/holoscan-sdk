(video-replayer-distributed-example)=
# Video Replayer (Distributed)

In this example, we extend the previous [video replayer application](./video_replayer.md) into a multi-node [distributed application](../holoscan_create_distributed_app.md). A distributed application is made up of multiple Fragments ({cpp:class}`C++ <holoscan::Fragment>`/{py:class}`Python <holoscan.core.Fragment>`), each of which may run on its own node.

In the distributed case we will:

- create one fragment that loads a video file from disk using **VideoStreamReplayerOp** operator
- create a second fragment that will display the video using the **HolovizOp** operator

These two fragments will be combined into a distributed application such that the display of the video frames could occur on a separate node from the node where the data is read.

:::{note}
The example source code and run instructions can be found in the [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#holoscan-sdk-examples) directory on GitHub, or under `/opt/nvidia/holoscan/examples` in the NGC container and the debian package, alongside their executables.
:::

## Operators and Workflow

Here is the diagram of the operators and workflow used in this example.

```{mermaid}
:align: center
:caption: Workflow to load and display video from a file

%%{init: {"theme": "base", "themeVariables": { "fontSize": "16px"}} }%%

classDiagram
    direction LR

    VideoStreamReplayerOp --|> HolovizOp : output...receivers

    class VideoStreamReplayerOp {
        output(out) Tensor
    }
    class HolovizOp {
        [in]receivers : Tensor
    }
```

This is the same workflow as the [single fragment video replayer](./video_replayer.md),  each operator is assigned to a separate fragment and there is now a network connection between the fragments.


## Defining and Connecting Fragments

Distributed applications define Fragments explicitly to isolate the different units of work that could be distributed to different nodes. In this example:
- We define two classes that inherit from `Fragment`:
  - **Fragment1** contains an instance of **VideoStreamReplayerOp** named "replayer".
  - **Fragment2** contains an instance of **HolovizOp** name "holoviz".
- We create an application, **DistributedVideoReplayerApp**. In its compose method:
  - we call **make_fragment** to initialize both fragments.
  - we then connect the "output" port of "replayer" operator in fragment1 to the "receivers" port of the "holoviz" operator in fragment2 to define the application workflow.
- The operators instantiated in the fragments can still be configured with parameters initialized from the YAML configuration ingested by the application using {cpp:func}`~holoscan::Fragment::from_config` (C++) or {py:func}`~holoscan.core.Fragment.kwargs` (Python).


`````{tab-set}
````{tab-item} C++
```{code-block} cpp
:linenos: true
:emphasize-lines: 2-3, 10-11, 20-21, 30-31, 34
:name: holoscan-one-operator-workflow-cpp

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

class Fragment1 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto replayer = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));
    add_operator(replayer);
  }
};

class Fragment2 : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));
    add_operator(visualizer);
  }
};

class DistributedVideoReplayerApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto fragment1 = make_fragment<Fragment1>("fragment1");
    auto fragment2 = make_fragment<Fragment2>("fragment2");

    // Define the workflow: replayer -> holoviz
    add_flow(fragment1, fragment2, {{"replayer.output", "holoviz.receivers"}});
  }
};

int main(int argc, char** argv) {
  // Get the yaml configuration file
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("video_replayer_distributed.yaml");

  auto app = holoscan::make_application<DistributedVideoReplayerApp>();
  app->config(config_path);
  app->run();

  return 0;
}
```
````
````{tab-item} Python
```{code-block} python
:linenos: true
:emphasize-lines: 4, 20-23, 25, 36, 38, 56-57, 60
:name: holoscan-one-operator-workflow-python

import os

from holoscan.core import Application, Fragment
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class Fragment1(Fragment):
    def __init__(self, app, name):
        super().__init__(app, name)

    def compose(self):
        # Set the video source
        video_path = self._get_input_path()
        logging.info(
            f"Using video from {video_path}"
        )

        # Define the replayer and holoviz operators
        replayer = VideoStreamReplayerOp(
            self, name="replayer", directory=video_path, **self.kwargs("replayer")
        )

        self.add_operator(replayer)

    def _get_input_path(self):
        path = os.environ.get(
            "HOLOSCAN_INPUT_PATH", os.path.join(os.path.dirname(__file__), "data")
        )
        return os.path.join(path, "racerx")


class Fragment2(Fragment):
    def compose(self):
        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        self.add_operator(visualizer)


class DistributedVideoReplayerApp(Application):
    """Example of a distributed application that uses the fragments and operators defined above.

    This application has the following fragments:
    - Fragment1
      - holding VideoStreamReplayerOp
    - Fragment2
      - holding HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):
        # Define the fragments
        fragment1 = Fragment1(self, name="fragment1")
        fragment2 = Fragment2(self, name="fragment2")

        # Define the workflow
        self.add_flow(fragment1, fragment2, {("replayer.output", "holoviz.receivers")})


def main():
    config_file_path = os.path.join(os.path.dirname(__file__), "video_replayer_distributed.yaml")

    logging.info(f"Reading application configuration from {config_file_path}")

    app = DistributedVideoReplayerApp()
    app.config(config_file_path)
    app.run()


if __name__ == "__main__":
    main()
```
````
`````

This particular distributed application only has one operator per fragment, so the operators was added via **`add_operator` ({cpp:func}`C++ <holoscan::Fragment::add_operator>`/{py:func}`Python <holoscan.core.Fragment.add_operator>`)**. In general, each fragment may have multiple operators and connections between operators within a fragment would be made using `add_flow()` ({cpp:func}`C++ <holoscan::Fragment::add_flow>`/{py:func}`Python <holoscan.core.Fragment.add_flow>`) method within the fragment's `compute()` ({cpp:func}`C++ <holoscan::Operator::compute>`/{py:func}`Python <holoscan.core.Operator.compute>`) method.

## Running the Application

Running the application should bring up video playback of the video referenced in the YAML file.

![](../images/video_replayer.png)

:::{note}
Instructions for running the distributed application involve calling the application from the "driver" node as well as from any worker nodes. For details, see the application run instructions in the [examples](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/video_replayer_distributed) directory on GitHub, or under `/opt/nvidia/holoscan/examples/video_replayer_distributed` in the NGC container and the debian package.
:::

:::{tip}
Refer to {ref}`UCX Network Interface Selection <ucx-network-selection>` when running a distributed application across multiple nodes.
:::
