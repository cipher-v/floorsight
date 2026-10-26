#!/bin/bash
ps -eo pid,etimes,pcpu,rss,args | grep -i "[d]etector_probe" | cut -c1-160
P=$(ps -eo pid,args | grep -i "[d]etector_probe" | grep -v bash | awk '{print $1}' | head -1); echo "pid=$P"
[ -x /opt/floorsight/venv-perc/bin/py-spy ] || /opt/floorsight/venv-perc/bin/pip install -q py-spy >/dev/null 2>&1
[ -n "$P" ] && /opt/floorsight/venv-perc/bin/py-spy dump --pid $P 2>&1 | head -25
