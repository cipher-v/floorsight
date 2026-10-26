#!/bin/bash
J=/opt/floorsight/jobs/$JOB; cat $J/STARTED; cat $J/restarts.txt 2>/dev/null; date -u +%FT%TZ
ps -eo pid,etimes,pcpu,rss,args | grep -i "[d]etector_probe\|[r]eal_clip_perception" | cut -c1-120
P=$(ps -eo pid,args | grep -iE "[d]etector_probe|[r]eal_clip_perception" | grep -v bash | awk '{print $1}' | head -1)
[ -n "$P" ] && /opt/floorsight/venv-perc/bin/py-spy dump --pid $P 2>&1 | sed -n 5,14p
