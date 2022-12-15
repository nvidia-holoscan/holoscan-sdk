# Bring Your Own Model - Colonoscopy

This example shows how to use the [Bring Your Own Model](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/clara_holoscan_applications.html#bring-your-own-model-byom-customizing-the-ultrasound-segmentation-application-for-your-model) (BYOM) concept for Holoscan by changing a few properties of the `ultrasound_segmentation` app to run a segmentation of polyps from a colonoscopy video input instead.

### Data

[📦️ (NGC) Sample App Data for AI Colonoscopy Segmentation of Polyps](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_colonoscopy_sample_data)

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run the following:

```bash
# Update the configurations (run again to reverse)
patch -ub -p0 -i examples/bring_your_own_model/python/colonoscopy_segmentation.patch
# Run the application
python3 ./apps/ultrasound_segmentation/python/ultrasound_segmentation.py
```
