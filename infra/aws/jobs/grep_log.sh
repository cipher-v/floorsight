#!/bin/bash
grep -n -B2 -A14 "Fatal Python error" /opt/floorsight/jobs/$JOB/log.txt | head -40 | cut -c1-220
grep -nE "passed|failed|error" /opt/floorsight/jobs/$JOB/log.txt | tail -5
