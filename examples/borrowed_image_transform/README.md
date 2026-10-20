# Borrowed Image transform

This C++ example reads a callback-scoped `SelectedSampleView<ImageT>`, copies only
metadata into a cold-prepared `sensor_io::ImageDescriptorScratch`, edits the ROI,
and publishes fresh Tensor pixels with the source envelope's provenance.

The source emits one host-memory pixel. The sink checks the transformed pixel,
ROI, coordinate-frame name, application frame identity, and trace identity.
Run the `borrowed_image_transform` target or its
`EXAMPLE_CPP_BORROWED_IMAGE_TRANSFORM_TEST` CTest.

Keep views and references inside `compute()`. Scratch must contain no Tensor and
its prepared owners/capacities must be preserved; it is not an owning input loan.
For GPU transforms, submit reads on the Runtime-admitted execution stream.
This example does not claim transport or third-party GPU code is allocation-free.
