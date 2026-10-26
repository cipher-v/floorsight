#!/bin/bash
top -bn1 | head -15; free -m | head -2
ps -eo pid,etimes,pcpu,rss,args --sort=-pcpu | head -5 | cut -c1-200
