#!/bin/bash
for w in /opt/floorsight/perc-stock /opt/floorsight/perc; do
for case in normal big intmin; do
  sudo -u floorsight $w -c "
import cv2, numpy as np, sys
img = np.zeros((500, 800), np.uint8)
pts = {'normal': [[10,10],[200,40],[120,300]], 'big': [[10,10],[2000000,40],[120,300]], 'intmin': [[-2147483648,-2147483648],[200,40],[120,300]]}['$case']
cv2.fillPoly(img, [np.array(pts, np.int32)], 1)
print('$w'.split('/')[-1], cv2.__version__, '$case', 'ok', int(img.sum()))
" 2>&1 | tail -1 || true
  echo "  rc=$? ($w $case)"
done; done
