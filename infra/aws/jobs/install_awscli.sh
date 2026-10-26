#!/bin/bash
# AWS CLI v2 from the official zip into /usr/local (the snap build cannot run for users whose home is outside /home).
set -euo pipefail
cd /tmp && curl -fsSL -o awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" && unzip -q -o awscliv2.zip && ./aws/install --update >/dev/null
hash -r; ls -la /usr/local/bin/aws; /usr/local/bin/aws --version
sudo -u floorsight /usr/local/bin/aws sts get-caller-identity --query Arn --output text
