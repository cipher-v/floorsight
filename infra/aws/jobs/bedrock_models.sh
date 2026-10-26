#!/bin/bash
# Which Claude models does this account's Bedrock Messages-API endpoint serve? One 16-token call per model.
cd /opt/floorsight/current
sudo -u floorsight env AWS_REGION=us-east-1 /opt/floorsight/py - <<'PY'
from anthropic import AnthropicBedrockMantle
c = AnthropicBedrockMantle(aws_region="us-east-1")
for m in ["anthropic.claude-opus-5-5", "anthropic.claude-sonnet-5-5", "anthropic.claude-opus-5", "anthropic.claude-haiku-5-5", "anthropic.claude-sonnet-5"]:
    try:
        r = c.messages.create(model=m, max_tokens=16, messages=[{"role": "user", "content": "Reply: ok"}])
        print("OK", m, r.model, r.usage.input_tokens, r.usage.output_tokens)
    except Exception as e:
        print("NO", m, type(e).__name__, str(e)[:160])
PY
