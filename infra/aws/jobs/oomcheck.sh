#!/bin/bash
journalctl -k --no-pager --since "-30 min" | grep -iE "oom|killed process" | tail -5
journalctl -u fs-probe-02 --no-pager | tail -8 | cut -c1-200
systemctl list-units 'fs-*' --all --no-pager | head
