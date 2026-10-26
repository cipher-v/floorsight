#!/bin/bash
for j in perc-pilot-10 perc-full-01; do
  d=/opt/floorsight/jobs/$j; [ -e "$d/calibration-used.json" ] || sudo -u floorsight cp "$d/floor-calibration-camA.json" "$d/calibration-used.json"; ls -la "$d/calibration-used.json"
done
