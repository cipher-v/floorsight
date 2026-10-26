#!/bin/bash
J=/opt/floorsight/jobs/$JOB; cat $J/DONE 2>/dev/null; grep -E "perception exit|replay|example" $J/log.txt 2>/dev/null | cut -c1-400
grep -A4 "Fatal Python" $J/screen-002.log 2>/dev/null | head -6; ls $J/polys 2>/dev/null | wc -l
ps -eo etimes,pcpu,args | grep -i "[r]eal_clip_perception" | cut -c1-60
