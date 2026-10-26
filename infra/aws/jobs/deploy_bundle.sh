#!/bin/bash
# Unpack a code bundle from S3 into its own dated folder /opt/floorsight/app-<REV> (earlier copies are kept)
# and point /opt/floorsight/current at it. Needs: BUCKET, REV (e.g. 20261023-r2), SHA (expected sha256).
set -euo pipefail
D=/opt/floorsight/app-$REV
[ -e "$D" ] && { echo "$D exists; bundles are never unpacked twice"; exit 2; }
aws s3 cp --region us-east-1 "s3://$BUCKET/code/floorsight-$REV.tgz" "/tmp/floorsight-$REV.tgz" --only-show-errors
GOT=$(sha256sum "/tmp/floorsight-$REV.tgz" | cut -d' ' -f1)
[ "$GOT" = "$SHA" ] || { echo "sha256 mismatch: got $GOT want $SHA"; exit 3; }
mkdir -p "$D" && tar -xzf "/tmp/floorsight-$REV.tgz" -C "$D" && chown -R floorsight: "$D"
[ -z "${NOLINK:-}" ] && ln -sfn "$D" /opt/floorsight/current
echo "deployed $REV sha256=$GOT -> $(readlink /opt/floorsight/current) files=$(find "$D" -type f | wc -l)"
