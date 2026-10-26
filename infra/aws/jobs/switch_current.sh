#!/bin/bash
# Point /opt/floorsight/current at an already unpacked bundle (no detached job may be running on the old one).
# Needs: REV (e.g. 20261023-r24).
set -euo pipefail
D=/opt/floorsight/app-$REV
[ -d "$D" ] || { echo "no $D"; exit 2; }
RUNNING=$(systemctl list-units 'fs-*' --state=active --no-legend | awk '{print $1}' | tr '\n' ' ')
[ -n "$RUNNING" ] && { echo "refused: active jobs $RUNNING"; exit 3; }
OLD=$(readlink /opt/floorsight/current)
ln -sfn "$D" /opt/floorsight/current
echo "current: $OLD -> $(readlink /opt/floorsight/current)"
