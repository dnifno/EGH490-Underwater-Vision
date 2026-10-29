# /cvat_annotations

Stores the CVAT "Segmentation mask 1.1" export for the 11 labelled frames based on the original 1080p BlueROV2 footage from Myora Bay.

```
cvat_annotations/
  ImageSets/Segmentation/default.txt -> list of image file names
  JPEGImages/ -> annotated frames (frame_000946.jpg, ...)
  SegmentationClass/ -> colour-coded semantic segmentation masks (frame_000946.png, ...)
  SegmentationObject/ -> colour-coded individual instances of class masks (frame_000946.png, ...)
  labelmap.txt -> label names and colours
```
