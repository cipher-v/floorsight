#!/bin/bash
cat /opt/floorsight/jobs/$JOB/DONE 2>/dev/null; grep -vE "FutureWarning|warnings.warn|UserWarning|from scipy|WARN:" /opt/floorsight/jobs/$JOB/log.txt | tail -${T:-20} | cut -c1-400
