#!/bin/bash
# Status of a detached job: unit state, DONE marker, last log lines.
J=/opt/floorsight/jobs/$JOB
systemctl show "fs-$JOB" -p ActiveState -p SubState -p ExecMainStatus 2>/dev/null | tr '\n' ' '; echo
cat "$J/DONE" 2>/dev/null
grep -E "${GREP:-CLIPS_BAD|FAILED|STATUS|perc |stock |Error|error}" "$J/log.txt" 2>/dev/null | tail -${N:-12}
echo "--- tail"; tail -${T:-6} "$J/log.txt" 2>/dev/null | cut -c1-300
df -h / | tail -1
