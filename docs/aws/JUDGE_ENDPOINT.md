# Running judge demo

URL: https://dz0itcb127dm1.cloudfront.net/

Verified on 27 October 2026: authenticated state returns `mode=live`,
`vision=real`, `agent=real`, `source=simlive`, `loop=true`; `/readyz` returns
`ready=true`. A second simulation session was observed. The app uses OpenCV
5.1.0-dev from `/opt/cool/python_3.12/site-packages/cv2` on the existing Graviton
instance. This is a generated-camera demo, not live factory footage.

The owner has the judge token for Devpost's private testing instructions. No
token or origin secret is committed. CloudFront supplies HTTPS, forwards the
websocket and disables response caching. Its secret header guards the origin;
the application checks a separate judge token. Only CloudFront origin addresses
can reach port 8000. The origin leg uses HTTP, matching the existing Terraform
design; it is not end-to-end TLS.

The 40-second scenario repeats at half speed. Floor review decisions are shared
between visitors. External SNS delivery is disabled in this judge instance;
alerts remain in the dashboard. S3, DynamoDB and CloudWatch adapters are enabled
and reported no errors during verification. The local planner makes no Bedrock
calls. Prior cloud resources and evidence were preserved.

## Reproduce on an existing COOL instance

`infra/aws/publish_judge.py` creates a dedicated security group, Elastic IP and
CloudFront distribution without replacing the instance. Its `--state` file is
private and contains secrets. Preserve it for retries and resource cleanup.
Use this helper OR the main Terraform stack's `public_endpoint` resources. Do
not apply both to the same endpoint without importing/reconciling resources.

```bash
python infra/aws/publish_judge.py --instance YOUR_INSTANCE_ID --state PRIVATE_PATH/endpoint.json
```

Package the submitted source into a separate `/opt/floorsight/judge-REV` directory
over SSM, preserving earlier deployments. Copy the private endpoint JSON to a
root-only file. Then run:

```bash
CODE_DIR=/opt/floorsight/judge-REV JUDGE_CONFIG=/root/endpoint.json bash /opt/floorsight/judge-REV/infra/aws/jobs/activate_judge.sh
```

The service is `floorsight-judge`; its data directory is
`/opt/floorsight/judge-runtime`. Restart it with systemd if needed. Stop the
instance after judging when compute is no longer needed; retained cloud
resources still incur charges. Nothing is automatically deleted.
