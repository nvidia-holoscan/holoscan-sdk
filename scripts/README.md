# Holoscan utility scripts

This folder includes the following scripts:
- [`convert_video_to_gxf_entities.py`](#convert_video_to_gxf_entitiespy)
- [`generate_extension_uuids.py`](#generate_extension_uuidspy)
- [`graph_surgeon.py`](#graph_surgeonpy)

## convert_video_to_gxf_entities.py

Takes in a raw video feed and emits encoded gxf entities for playback with the `stream_playback` codelet.

### Prerequisites

```sh
pip install numpy==1.21.0
```

### Usage

Example usage converting the output of a tool like `ffmpeg` to encoded gxf entities:

```sh
ffmpeg -i endoscopy_1920x1080.avi -pix_fmt rgb24 -f rawvideo pipe:1 | python scripts/convert_video_to_gxf_entities.py --width 1920 --height 1080 --channels 3 --framerate 30
```

## generate_extension_uuids.py

Provides a set of UUIDs to be used by `GXF_EXT_FACTORY_SET_INFO` and `GXF_EXT_FACTORY_ADD` to declare a new GXF extension.

``` sh
python scripts/generate_extension_uuids.py
```

## graph_surgeon.py
When converting a model from PyTorch to ONNX, it is likely that the input of the model is in the form NCHW (batch, channels, height, width), and needs to be converted to NHWC (batch, height, width, channels). This script performs the conversion and generates a modified model.
Note that this script modifies the names of the output by appending `_old` to the input and output.

### Usage

```bash
python3 scripts/graph_surgeon.py input_model.onnx output_model.onnx
```
