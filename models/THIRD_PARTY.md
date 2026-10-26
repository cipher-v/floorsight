# Third-party model files

| File | What it is | Source URL | Licence | SHA-256 |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet face detector (used by `floorsight/edge/privacy.py` via `cv2.FaceDetectorYN`) | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx | MIT, Copyright (c) 2020 Shiqi Yu - https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_detection_yunet_2026may.onnx` | Same YuNet weights re-exported by opencv_zoo with dynamic input height/width (for OpenCV 5's ONNX Runtime engine, `OPENCV_FORCE_DNN_ENGINE=4`). Not used by default; select with `FLOORSIGHT_YUNET_MODEL`. | https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx | MIT (same directory and LICENSE as above) | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |

Downloaded 2026-10-16. Licence text checked on that date: the `LICENSE` file in
`opencv_zoo/models/face_detection_yunet/` is the MIT licence. Upstream training
repo: https://github.com/ShiqiYu/libfacedetection.train

Both files were checked with OpenCV 5.0.0 (`opencv-python-headless==5.0.0.93`,
default DNN engine): each finds the face in `messi5.jpg` (548x342) and
`lena.jpg` (512x512) at its native size and at 640x360.

Verify after download:

```bash
sha256sum models/face_detection_yunet_2023mar.onnx
```

## Test images (not redistributed)

`tests/edge/test_privacy.py` downloads `messi5.jpg` and `lena.jpg` from
https://github.com/opencv/opencv/tree/4.x/samples/data into `data/test-data/`,
which is gitignored. They are used only as local test input.
