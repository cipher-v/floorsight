#!/bin/bash
aws s3 cp --region us-east-1 "/opt/floorsight/jobs/$JOB/$REL" "s3://$BUCKET/jobs/$JOB/partial/$(basename $REL)" --only-show-errors && echo pushed
