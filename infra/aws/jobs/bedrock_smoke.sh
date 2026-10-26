#!/bin/bash
# One tiny Claude call through the Bedrock Messages-API (Mantle) endpoint with the instance role.
cd /opt/floorsight/app
sudo -u floorsight env AWS_REGION=us-east-1 /opt/floorsight/py - <<'PY'
import json, time
from floorsight.llm.bedrock import make_bedrock_client, MODEL_ID
c = make_bedrock_client("us-east-1")
t = time.time()
r = c.beta.messages.create(model=MODEL_ID, max_tokens=64, output_config={"effort": "low"},
                           messages=[{"role": "user", "content": "Reply with exactly: floorsight bedrock ok"}])
print(json.dumps({"model": r.model, "stop": r.stop_reason, "text": next((b.text for b in r.content if b.type == "text"), None),
                  "usage": {"in": r.usage.input_tokens, "out": r.usage.output_tokens}, "s": round(time.time() - t, 2), "request_id": getattr(r, "_request_id", None)}))
PY
