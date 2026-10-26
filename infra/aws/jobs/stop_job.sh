#!/bin/bash
# Stop a running job unit (its folder and outputs are kept) and record why.
echo "stopped $(date -u +%FT%TZ): ${WHY:-superseded}" | sudo -u floorsight tee "/opt/floorsight/jobs/$JOB/STOPPED-superseded.txt"
systemctl stop "fs-$JOB"; systemctl show "fs-$JOB" -p ActiveState
