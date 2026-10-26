#!/bin/bash
cat /opt/floorsight/OPENCV_SOURCE; cat /etc/floorsight.opencv.env
sudo -u floorsight /opt/floorsight/py - <<'PY'
import cv2, hashlib
bi = cv2.getBuildInformation()
print("cv2", cv2.__version__, cv2.__file__, "build-sha256", hashlib.sha256(bi.encode()).hexdigest()[:16])
for ln in bi.splitlines():
    l = ln.strip()
    if any(k in l for k in ("General configuration", "Version control", "Platform", "Compiler", "C++ flags (Release)", "CPU/HW features", "Baseline", "Dispatched", "requested", "Parallel framework", "HAL", "KleidiCV", "kleidicv", "Video I/O", "FFMPEG", "avcodec", "GStreamer", "DNN", "Inference", "Python 3", "numpy", "Install to", "Timestamp", "Host", "Target")):
        print("  ", l[:160])
print("threads", cv2.getNumThreads(), "optimized", cv2.useOptimized())
PY
sed -n 1,80p /opt/cool/samples/README.md
