# AWS deployment code

Start with the [deployment guide](../docs/aws/DEPLOY.md). The public simulator can
be deployed without the private footage used in the recorded development runs.

| File or folder | Purpose |
|---|---|
| [terraform/main.tf](terraform/main.tf) | EC2 Graviton, networking, private S3, DynamoDB, SNS, CloudWatch, IAM and optional public access |
| [terraform/variables.tf](terraform/variables.tf) | Region, COOL AMI, instance, application and retention settings |
| [terraform/terraform.tfvars.example](terraform/terraform.tfvars.example) | Starting configuration; copy to the ignored `terraform.tfvars` |
| [terraform/versions.tf](terraform/versions.tf) and provider lock | Terraform/provider requirements and resolved provider versions |
| [terraform/user_data.sh.tftpl](terraform/user_data.sh.tftpl) | Install and start the app; import COOL; enable AWS adapters |
| [terraform/outputs.tf](terraform/outputs.tf) | Instance/resource details and private SSM access commands |
| [aws/pack_bundle.py](aws/pack_bundle.py), [deploy job](aws/jobs/deploy_bundle.sh) | Package and install an application bundle on an existing instance |
| [aws/ssm_run.py](aws/ssm_run.py), [aws/jobs/](aws/jobs/) | Recorded-run, benchmark and verification helpers |

Use an exact submitted commit for `app_git_ref`. Supply the COOL Marketplace AMI
for your own region and subscription; the example deliberately contains no
account-specific AMI or credentials. The default planner is local and access is
private through SSM. The guide explains optional public access and Bedrock.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Fill in your AMI, region and exact submitted commit.
terraform init
terraform fmt -check
terraform validate
terraform plan -out floorsight.tfplan
# Review the resource and billing changes, then:
terraform apply floorsight.tfplan
```

Formatting and validation passed on 27 October 2026. This verifies the configuration,
not a fresh deployment in every AWS account. Historical cloud evidence is in
[the app receipt](../docs/receipts/app-run-09-summary-20261024.json). Do not commit
state files, plans, tokens or private footage. Do not replace an evidence-bearing
instance just to refresh code; follow the existing-instance steps in the guide.
