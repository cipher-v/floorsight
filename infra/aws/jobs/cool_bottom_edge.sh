#!/bin/bash
# Minimal COOL reproduction: fillPoly with vertices below the image's last row, one fresh process per case.
for w in /opt/floorsight/perc-stock /opt/floorsight/perc; do
  for case in inside below_by1 below_by4 saved295; do
    sudo -u floorsight $w -c "
import numpy as np, cv2
pts = {'inside': [[111,240],[102,245],[92,249],[88,242],[107,236]], 'below_by1': [[111,248],[102,251],[92,249],[88,242],[107,236]],
       'below_by4': [[111,248],[102,251],[92,254],[88,242],[107,236]], 'saved295': [[111,248],[102,251],[92,254],[88,242],[107,236],[107,237]]}['$case']
m = np.zeros((251, 321), np.uint8); cv2.fillPoly(m, [np.array(pts, np.int32)], 1); print(int(m.sum()))
" >/tmp/o.txt 2>&1; rc=$?; echo "$(basename $w) $case rc=$rc $(tail -1 /tmp/o.txt | cut -c1-80)"
  done
done
