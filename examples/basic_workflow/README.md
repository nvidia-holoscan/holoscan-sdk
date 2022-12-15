# Basic Workflow

Minimal example to demonstrate the use of adding components in a pipeline. The workflow in the example tracks tools in the endoscopy video sample data.

### Data

[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the commands of your choice:

```bash
# C++
./examples/basic_workflow/cpp/basic_workflow

# Python
python3 ./examples/basic_workflow/python/basic_workflow.py
```

> ℹ️ Python apps can run outside those folders if `HOLOSCAN_SAMPLE_DATA_PATH` is set in your environment (automatically done by `./run launch`).