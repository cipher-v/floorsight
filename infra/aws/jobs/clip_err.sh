#!/bin/bash
J=/opt/floorsight/jobs/$JOB; grep -vE "FutureWarning|warnings.warn|UserWarning|from scipy|WARN:" $J/${CLIP:-screen-002}.log | tail -${T:-25} | cut -c1-300
[ -n "${POLY:-}" ] && /opt/floorsight/perc -c "import numpy as np; d=np.load('$J/polys/$POLY'); print('POLY', d['shape'].tolist(), d['pts'].tolist())"
