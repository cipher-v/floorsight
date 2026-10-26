#!/bin/bash
# Which clips of a perception job have a result file, and is the job DONE. Needs: JOB.
J=/opt/floorsight/jobs/$JOB
for c in screen-033 screen-032 clip-dev-004 clip-dev-002 clip-dev-003 screen-002; do [ -f "$J/$c/$c-result.json" ] && printf "%s " "$c"; done; echo
[ -f "$J/DONE" ] && echo "JOBDONE"
systemctl show "fs-$JOB" -p ActiveState --value
