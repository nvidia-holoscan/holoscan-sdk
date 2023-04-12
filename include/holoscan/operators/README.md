# Operators

These are the operators included as part of the Holoscan SDK:

- **aja_source**: support AJA capture card as source
- **bayer_demosaic**: perform color filter array (CFA) interpolation for 1-channel inputs of 8 or 16-bit unsigned integer and outputs an RGB or RGBA image
- **format_converter**: provides common video or tensor operations in inference pipelines to change datatypes, resize images, reorder channels, and normalize and scale values.
- **holoviz**: handles compositing, blending, and visualization of RGB or RGBA images, masks, geometric primitives, text and depth maps
- **multiai_inference**: perform AI inference
- **multiai_postprocessor**: perform AI postprocessing
- **ping_rx**: "receive" and log an int value
- **ping_tx**: "transmit" an int value
- **segmentation_postprocessor**: generic AI postprocessing operator
- **tensor_rt** *(deprecated)*: perform AI inference with TensorRT
- **video_stream_recorder**: write a video stream output as `.gxf_entities` + `.gxf_index` files on disk
- **video_stream_replayer**: read `.gxf_entities` + `.gxf_index` files on disk as a video stream input
