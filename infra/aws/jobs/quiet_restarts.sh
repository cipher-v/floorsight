#!/bin/bash
# Stop needrestart from restarting Floorsight jobs/services after unattended upgrades (list only), and stop the re-running probe.
set -uo pipefail
mkdir -p /etc/needrestart/conf.d
cat > /etc/needrestart/conf.d/50-floorsight.conf <<'NR'
# Floorsight: never auto-restart services (a restart would cut long perception/app jobs); list them instead.
$nrconf{restart} = 'l';
$nrconf{override_rc}{qr(^fs-)} = 0;
$nrconf{override_rc}{qr(^floorsight)} = 0;
NR
systemctl stop fs-probe-02 2>/dev/null; systemctl reset-failed 'fs-*' 2>/dev/null
pgrep -fa unattended-upgrade | head -2
journalctl --no-pager --since "-40 min" | grep -E "needrestart|Restarting services" | tail -3
echo "unattended-upgrades state: $(systemctl is-active unattended-upgrades)"
