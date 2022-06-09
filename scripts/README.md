# Holoscan utility scripts

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

## download_sample_data.py

Downloads the endoscopy and ultrasound data (AI models, video sources) from NGC, used by the reference applications.

### Prerequisites

NGC CLI tools installed and configured: https://ngc.nvidia.com/setup/installers/cli

### Usage

Default:
```sh
python scripts/download_sample_data.py
```

Help for additional configurations (NGC org, destination folder, data versions)
```sh
python scripts/download_sample_data.py --help
```

## generate_extension_uuids.py

Provides a set of UUIDs to be used by `GXF_EXT_FACTORY_SET_INFO` and `GXF_EXT_FACTORY_ADD` to declare a new GXF extension.

``` sh
python scripts/generate_extension_uuids.py
```
