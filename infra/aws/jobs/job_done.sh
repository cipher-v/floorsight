#!/bin/bash
# Is a detached job finished? Prints JOBDONE when its DONE marker exists, else the unit state. Needs: JOB.
J=/opt/floorsight/jobs/$JOB
[ -f "$J/DONE" ] && { echo JOBDONE; exit 0; }
systemctl show "fs-$JOB" -p ActiveState --value
tail -1 "$J/log.txt" 2>/dev/null | cut -c1-200
