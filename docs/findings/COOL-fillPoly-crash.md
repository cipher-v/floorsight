# COOL `cv2.fillPoly` crash on Graviton4 (found 23 Oct 2026)

**Build.** Cloud Optimized OpenCV (COOL) for AWS Graviton4, AWS Marketplace AMI `ami-033e481a24f94c8cb`, us-east-1:
OpenCV 5.1.0-dev, custom HAL carotene 0.0.1 + KleidiCV 0.7.0, `-O3 -mcpu=neoverse-v2`, numpy 2.5.1. Instance type
c8g.xlarge (Neoverse-V2). Control: `opencv-python-headless==5.0.0.93` (stock wheel) on the same instance.

**Symptom.** `cv2.fillPoly` ends the process with SIGSEGV (exit 139) inside `libopencv_imgproc.so.501` when some
polygon vertices lie a few rows below the image's last row. The stock build fills the in-image part.

**Minimal reproduction** (one fresh process per case; uint8 mask, 251 rows x 321 columns, fill value 1):

| Case | Polygon (x, y) | Stock 5.0.0 | COOL 5.1.0-dev |
|---|---|---|---|
| inside | `[[111,240],[102,245],[92,249],[88,242],[107,236]]` | 167 px | 167 px |
| 1 row below | `[[111,248],[102,251],[92,249],[88,242],[107,236]]` | 244 px | 244 px |
| 4 rows below | `[[111,248],[102,251],[92,254],[88,242],[107,236]]` | 254 px | **SIGSEGV** |

```python
import cv2, numpy as np
cv2.fillPoly(np.zeros((251, 321), np.uint8), [np.array([[111,248],[102,251],[92,254],[88,242],[107,236]], np.int32)], 1)
```

Native backtrace (gdb): `#0 ?? () from libopencv_imgproc.so.501`, `#1 cv::fillPoly(... Point_<int> const** ...)`,
`#2 cv::fillPoly(_InputOutputArray, _InputArray, ...)`. In a replay of 300 real non-convex mask polygons (one process
each), stock filled 300 of 300 and COOL crashed on 1.

**Second difference.** A polygon with a vertex at INT_MIN (what numpy produces when NaN is cast to int32) fills on
stock 5.0.0 but raises on COOL: `cv2.error: cannot create std::vector larger than max_size()`.

**Not yet known.** Whether the cause is in COOL's HAL or build changes, or in OpenCV 5.1.0-dev itself.

**Workaround in Floorsight.** Every polygon is clipped to the image (Sutherland-Hodgman) before `cv2.fillPoly` or
`cv2.fillConvexPoly` (`floorsight/cvsafe.py`, tests in `tests/realclip/test_cvsafe.py`), and non-finite vertices are
dropped. The clipped fill gives the same pixels on both builds.
