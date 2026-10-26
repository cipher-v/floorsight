#!/bin/bash
J=/opt/floorsight/jobs/$JOB; cat $J/summary.txt 2>/dev/null | cut -c1-300; echo ---; tail -${T:-8} $J/${CLIP:-screen-002}.log 2>/dev/null | cut -c1-300; cat $J/restarts.txt 2>/dev/null
ps -eo etimes,pcpu,rss,args | grep -i "[r]eal_clip_perception" | cut -c1-90
