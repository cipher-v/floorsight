#!/bin/bash
wc -l /opt/floorsight/jobs/$JOB/log.txt; head -60 /opt/floorsight/jobs/$JOB/log.txt | cut -c1-220
