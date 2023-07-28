# Holoscan utility scripts

This folder includes the following scripts:
- [`convert_video_to_gxf_entities.py`](#convert_video_to_gxf_entitiespy)
- [`convert_gxf_entities_to_video.py`](#convert_gxf_entities_to_videopy)
- [`generate_extension_uuids.py`](#generate_extension_uuidspy)
- [`graph_surgeon.py`](#graph_surgeonpy)

## convert_video_to_gxf_entities.py

Takes in a raw video feed and emits encoded gxf entities for playback with the `stream_playback` codelet.

### Prerequisites

```sh
pip install numpy~=1.21
```

This script depends on `gxf_entity_codec.py` which is located in the same folder.

### Usage

Example usage converting the output of a tool like `ffmpeg` to encoded gxf entities:

```sh
ffmpeg -i video_1920x1080.avi -pix_fmt rgb24 -f rawvideo pipe:1 | python scripts/convert_video_to_gxf_entities.py --width 1920 --height 1080 --channels 3 --framerate 30 --basename my_video
```

Above command will create two files: `my_video.gxf_entities` and `my_video.gxf_index` from the `video_1920x1080.avi` video file.
Please use `--directory` to specify the directory where the files will be created.

## convert_gxf_entities_to_video.py

Takes in the encoded gxf entities (`.gxf_entities` and `.gxf_index` files) and emit the raw video feed.

### Prerequisites

```sh
pip install numpy~=1.21
```

This script depends on `gxf_entity_codec.py` which is located in the same folder.

### Usage

Example usage reading encoded gxf entities and converting them to a video file:

```sh
python scripts/convert_gxf_entities_to_video.py --base_name my_video | ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i - -f mp4 -vcodec libx264 -pix_fmt yuv420p -r 30 -y converted_video.mp4
```

Above command will read the `my_video.gxf_entities` and `my_video.gxf_index` files and convert them to a video file `converted_video.mp4`.


With the existing endoscopy dataset under `data/endoscopy/video` (which is a 854x480 video with framerate 25fps and 3 channels), we can run the following command to convert the gxf entities to a video file:

```sh
python scripts/convert_gxf_entities_to_video.py --directory data/endoscopy/video --basename surgical_video | ffmpeg -f rawvideo -pix_fmt rgb24 -s 854x480 -r 25 -i - -f mp4 -vcodec libx264 -pix_fmt yuv420p -r 25 -y surgical_video.mp4
```

The output video (`surgical_video.mp4`) can be encoded again into gxf entities with the following command:

```sh
ffmpeg -i surgical_video.mp4 -pix_fmt rgb24 -f rawvideo pipe:1 | python scripts/convert_video_to_gxf_entities.py --width 854 --height 480 --channels 3 --framerate 25

# tensor.gxf_entities and tensor.gxf_index will be created ('tensor' is the default basename)
ls tensor*
# tensor.gxf_entities  tensor.gxf_index
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
