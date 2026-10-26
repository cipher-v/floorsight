#!/bin/bash
# copy finished clips' result JSONs of a running job to S3 (results only)
for c in $CLIPS; do f=/opt/floorsight/jobs/$JOB/$c/$c-result.json; [ -e "$f" ] && aws s3 cp --region us-east-1 "$f" "s3://$BUCKET/jobs/$JOB/partial/$c-result.json" --only-show-errors && echo "pushed $c"; done
