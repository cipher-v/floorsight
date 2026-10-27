"""Add a token-protected CloudFront endpoint to an existing project instance.

Keeps the instance, disks and evidence. The private state file is required for
idempotent retries; it contains access secrets and must never be committed.
This is an alternative to the main Terraform stack's public_endpoint resources,
not an additional layer to apply over an already public Terraform deployment.
"""
import argparse
import json
import secrets
import uuid
from pathlib import Path

import boto3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--project", default="anisha-robotics-floorsight")
    a = p.parse_args()
    ec2 = boto3.client("ec2", region_name=a.region)
    cf = boto3.client("cloudfront", region_name=a.region)
    instance = ec2.describe_instances(InstanceIds=[a.instance])["Reservations"][0]["Instances"][0]
    if dict((t["Key"], t["Value"]) for t in instance.get("Tags", [])).get("Project") != a.project:
        raise SystemExit("Instance Project tag does not match --project")
    state = json.loads(a.state.read_text()) if a.state.exists() else {
        "instance": a.instance, "region": a.region, "caller": str(uuid.uuid4()),
        "judge_token": secrets.token_urlsafe(24), "origin_secret": secrets.token_urlsafe(32),
    }
    if (state["instance"], state["region"]) != (a.instance, a.region):
        raise SystemExit("State belongs to a different instance or region")
    def save():
        a.state.parent.mkdir(parents=True, exist_ok=True)
        a.state.write_text(json.dumps(state, indent=2), encoding="utf-8")
        a.state.chmod(0o600)
    save()
    tags = [{"Key": "Project", "Value": a.project}, {"Key": "Name", "Value": "floorsight-judge-endpoint"}]
    if "security_group" not in state:
        state["security_group"] = ec2.create_security_group(
            GroupName="floorsight-judge-" + state["caller"], Description="Judge app: CloudFront origins only",
            VpcId=instance["VpcId"], TagSpecifications=[{"ResourceType": "security-group", "Tags": tags}])["GroupId"]
        save()
    prefix = ec2.describe_managed_prefix_lists(Filters=[{"Name": "prefix-list-name", "Values": ["com.amazonaws.global.cloudfront.origin-facing"]}])["PrefixLists"][0]["PrefixListId"]
    rules = ec2.describe_security_groups(GroupIds=[state["security_group"]])["SecurityGroups"][0]["IpPermissions"]
    if not rules:
        ec2.authorize_security_group_ingress(GroupId=state["security_group"], IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000,
            "PrefixListIds": [{"PrefixListId": prefix, "Description": "CloudFront origins"}]}])
    groups = [g["GroupId"] for g in instance["SecurityGroups"]]
    if state["security_group"] not in groups:
        ec2.modify_instance_attribute(InstanceId=a.instance, Groups=groups + [state["security_group"]])
    if "allocation" not in state:
        if ec2.describe_addresses(Filters=[{"Name": "instance-id", "Values": [a.instance]}])["Addresses"]:
            raise SystemExit("Instance already has an Elastic IP; inspect before changing its endpoint")
        ip = ec2.allocate_address(Domain="vpc", TagSpecifications=[{"ResourceType": "elastic-ip", "Tags": tags}])
        state.update(allocation=ip["AllocationId"], ip=ip["PublicIp"])
        save()
    address = ec2.describe_addresses(AllocationIds=[state["allocation"]])["Addresses"][0]
    if address.get("InstanceId") not in (None, a.instance):
        raise SystemExit("Saved Elastic IP is attached to another instance")
    if address.get("InstanceId") != a.instance:
        ec2.associate_address(InstanceId=a.instance, AllocationId=state["allocation"], AllowReassociation=False)
    if "distribution" not in state:
        instance = ec2.describe_instances(InstanceIds=[a.instance])["Reservations"][0]["Instances"][0]
        origin = instance["PublicDnsName"]
        if not origin:
            raise SystemExit("Instance has no public DNS name; enable VPC DNS hostnames and retry")
        config = {
            "CallerReference": state["caller"], "Comment": "Floorsight protected judge simulator", "Enabled": True,
            "PriceClass": "PriceClass_100", "HttpVersion": "http2", "IsIPV6Enabled": True,
            "Origins": {"Quantity": 1, "Items": [{"Id": "judge-app", "DomainName": origin,
                "CustomHeaders": {"Quantity": 1, "Items": [{"HeaderName": "X-Floorsight-Origin", "HeaderValue": state["origin_secret"]}]},
                "CustomOriginConfig": {"HTTPPort": 8000, "HTTPSPort": 443, "OriginProtocolPolicy": "http-only",
                    "OriginSslProtocols": {"Quantity": 1, "Items": ["TLSv1.2"]}, "OriginReadTimeout": 60, "OriginKeepaliveTimeout": 60}}]},
            "DefaultCacheBehavior": {"TargetOriginId": "judge-app", "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {"Quantity": 7, "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
                    "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}},
                "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
                "OriginRequestPolicyId": "b689b0a8-53d0-40ab-baf2-68738e2966ac", "Compress": True},
            "ViewerCertificate": {"CloudFrontDefaultCertificate": True},
            "Restrictions": {"GeoRestriction": {"RestrictionType": "none", "Quantity": 0}},
            "CustomErrorResponses": {"Quantity": 1, "Items": [{"ErrorCode": 503, "ErrorCachingMinTTL": 0}]},
        }
        result = cf.create_distribution_with_tags(DistributionConfigWithTags={"DistributionConfig": config, "Tags": {"Items": tags}})["Distribution"]
        state.update(distribution=result["Id"], domain=result["DomainName"])
        save()
    status = cf.get_distribution(Id=state["distribution"])["Distribution"]["Status"]
    print(json.dumps({"url": "https://" + state["domain"], "status": status, "private_state": str(a.state)}))


if __name__ == "__main__":
    main()
