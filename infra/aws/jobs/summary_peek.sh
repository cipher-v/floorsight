#!/bin/bash
grep -E "exit=|CROSSCHECK|FAILCLOSED" /opt/floorsight/jobs/$JOB/summary.txt 2>/dev/null | cut -c1-200
ps -eo etimes,args | grep -i "[r]eal_clip_perception" | grep -o "etimes\|--name [a-z0-9-]*\|^ *[0-9]*" | tr '\n' ' '; echo
