#!/usr/bin/env python3
"""Build the serving image and push it to ECR, using AWS CodeBuild.

    python3 scripts/build_image.py                    # region from config.yaml
    python3 scripts/build_image.py --write-config     # also write the URI to config.local.yaml
    python3 scripts/build_image.py --status <id>      # report on an earlier build

WHY CODEBUILD AND NOT `docker build`
    The base image is ~9 GB. Building locally pulls that down to your machine and pushes the result
    back up, so the same bytes cross your connection twice for no reason - CodeBuild keeps Docker Hub
    -> build -> ECR entirely inside AWS.

    Three other things fall out of it:
      * Docker is not a prerequisite for using this project at all.
      * No cross-architecture risk. A local build on an Apple Silicon machine is an emulated
        linux/amd64 build; if the platform flag is ever dropped the image fails on the instance with
        an exec format error, which surfaces only at task start. CodeBuild runs x86_64 natively.
      * The same path works in CI later without Docker-in-Docker.

Everything it creates - the ECR repository, the build bucket, the IAM role, the CodeBuild project - is
created if missing and their policies brought up to date, so re-running is safe and costs nothing extra.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPO_NAME = "gpu-llm-serving"
PROJECT_NAME = "gpu-llm-serving-build"
ROLE_NAME = "GpuLlmServingCodeBuildRole"
TAG = "vllm-0.29.0"   # keep in step with the base image version in container/Dockerfile

# Runs inside CodeBuild. $ECR, $REPO and $IMAGE_TAG come from the project's environment, and
# $AWS_REGION and $AWS_ACCOUNT_ID are provided by CodeBuild itself.
BUILDSPEC = """version: 0.2
phases:
  pre_build:
    commands:
      - echo "logging in to $ECR"
      - aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR
  build:
    commands:
      # linux/amd64 is the platform the GPU instances run. CodeBuild is x86_64 natively, so this is a
      # native build rather than an emulated one.
      - docker build --platform linux/amd64 -t $ECR/$REPO:$IMAGE_TAG .
      # Prove the entrypoint is present and executable before pushing. A missing or non-executable
      # entrypoint produces a task that starts and immediately dies, diagnosable only from ECS logs.
      - docker run --rm --entrypoint /bin/sh $ECR/$REPO:$IMAGE_TAG -c "test -x /usr/local/bin/serve"
  post_build:
    commands:
      - docker push $ECR/$REPO:$IMAGE_TAG
      - echo "pushed $ECR/$REPO:$IMAGE_TAG"
"""

def trust_policy(account: str, partition: str) -> dict:
    """CodeBuild may assume the role only for this account's project of this name (confused-deputy
    conditions), not for any CodeBuild project anywhere."""
    return {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "codebuild.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account},
                # Any region: the role is account-wide and one account builds in more than one region
                # (trying another region for capacity is a documented step). Scoping it to the region
                # of the last run broke the previous region's project at start_build.
                "ArnLike": {"aws:SourceArn": f"arn:{partition}:codebuild:*:{account}:project/{PROJECT_NAME}"},
            },
        }],
    }

# Untagged images are the layers left behind when a tag is moved to a rebuilt image. Nothing reads
# them, and they are billed, so a sample should not quietly accumulate them.
LIFECYCLE = {
    "rules": [{
        "rulePriority": 1,
        "description": "Expire untagged images after 7 days",
        "selection": {"tagStatus": "untagged", "countType": "sinceImagePushed",
                      "countUnit": "days", "countNumber": 7},
        "action": {"type": "expire"},
    }]
}


def config_path() -> str:
    """Honours $CONFIG, the same as infra/app.py. Without it, pointing CONFIG at another config file
    deployed one region while this script pushed to the region in config.yaml."""
    return os.environ.get("CONFIG", os.path.join(ROOT, "config.yaml"))


def local_config_path() -> str:
    """config.local.yaml next to whichever config is in use, the same rule as infra/app.py."""
    return os.path.join(os.path.dirname(os.path.abspath(config_path())), "config.local.yaml")


def config_region() -> str:
    """Region from config.yaml (and config.local.yaml), so it cannot drift from the deployment.

    Defaulting to a hardcoded region is a real trap rather than a convenience: omitting --region
    would push the image into a region the stack never reads from, and it would then be pulled
    cross-region on every task start.
    """
    import yaml
    region = ""
    for path in (config_path(), local_config_path()):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            region = ((yaml.safe_load(fh) or {}).get("region") or region)
    # No $AWS_REGION fallback, the same as app.py: that variable is commonly set to something
    # unrelated, and the build and the deploy must agree on the region.
    return region


def ensure_repo(ecr) -> None:
    try:
        ecr.describe_repositories(repositoryNames=[REPO_NAME])
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(
            repositoryName=REPO_NAME,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
        )
        print(f"  created ECR repository {REPO_NAME}")
    # Applied every run: idempotent, and it also repairs a repository created before this policy
    # existed.
    ecr.put_lifecycle_policy(repositoryName=REPO_NAME,
                             lifecyclePolicyText=json.dumps(LIFECYCLE))


def ensure_bucket(s3, bucket: str, region: str, account: str) -> None:
    """A private, dedicated bucket for the build context, created if absent.

    Dedicated rather than the CDK bootstrap bucket, so this script works before `cdk bootstrap` has
    ever run - the build has to happen first anyway, since the stack will not synthesise without an
    image.
    """
    try:
        # ExpectedBucketOwner: the name is predictable, so a bucket someone else created under it
        # must fail here rather than receive our build context.
        s3.head_bucket(Bucket=bucket, ExpectedBucketOwner=account)
        return
    except Exception as e:                                    # noqa: BLE001 - inspected below
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code not in ("404", "NoSuchBucket", "403", "AccessDenied"):
            raise
        if code in ("403", "AccessDenied"):
            sys.exit(f"bucket {bucket} exists but is not accessible with these credentials.")

    kwargs = {"Bucket": bucket}
    # us-east-1 is the one region that rejects an explicit LocationConstraint.
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={"BlockPublicAcls": True, "IgnorePublicAcls": True,
                                        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={"Rules": [
            {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
    print(f"  created build bucket {bucket}")


def ensure_role(iam, account: str, partition: str) -> str:
    """Create the role if missing, and ALWAYS reconcile its inline policy.

    Reconciling on every run rather than returning early for an existing role: a corrected policy in
    this file must reach a role created by an earlier version, or the failure is an AccessDenied at
    DOWNLOAD_SOURCE that looks nothing like a stale policy. `put_role_policy` is idempotent.
    """
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "Logs", "Effect": "Allow",
             "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
             "Resource": f"arn:{partition}:logs:*:{account}:log-group:/aws/codebuild/{PROJECT_NAME}*"},
            # GetAuthorizationToken takes no resource; everything else is scoped to this one repository,
            # so the build role cannot overwrite any other image in the account.
            {"Sid": "EcrLogin", "Effect": "Allow",
             "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
            {"Sid": "PushToEcr", "Effect": "Allow",
             "Action": ["ecr:BatchCheckLayerAvailability", "ecr:CompleteLayerUpload",
                        "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart",
                        "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
             "Resource": f"arn:{partition}:ecr:*:{account}:repository/{REPO_NAME}"},
            {"Sid": "ReadBuildContext", "Effect": "Allow",
             "Action": ["s3:GetObject", "s3:GetObjectVersion"],
             "Resource": f"arn:{partition}:s3:::{REPO_NAME}-build-{account}-*/build/*"},
        ],
    }
    created = False
    trust = json.dumps(trust_policy(account, partition))
    try:
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
        iam.update_assume_role_policy(RoleName=ROLE_NAME, PolicyDocument=trust)   # a role from an older run
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(
            RoleName=ROLE_NAME, AssumeRolePolicyDocument=trust,
            Description="CodeBuild: build the GPU serving image and push it to ECR",
        )["Role"]["Arn"]
        created = True
        print(f"  created IAM role {ROLE_NAME}")

    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="BuildAndPush",
                        PolicyDocument=json.dumps(policy))
    if created:
        # A brand-new role is not immediately usable by CodeBuild; without this the first build fails
        # with "not authorized to perform sts:AssumeRole".
        print("  waiting 15s for IAM propagation")
        time.sleep(15)
    return arn


def upload_context(s3, bucket: str, account: str) -> str:
    """Zip container/ plus the generated buildspec, and put it in S3."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("buildspec.yml", BUILDSPEC)
        for fn in ("Dockerfile", "serve"):
            path = os.path.join(ROOT, "container", fn)
            if not os.path.exists(path):
                sys.exit(f"missing container/{fn}")
            # Explicit mode so `serve` is executable inside the build, independent of local
            # permissions - a zip preserves what it is given, and a non-executable entrypoint
            # produces a container that starts and dies.
            info = zipfile.ZipInfo(fn)
            info.external_attr = 0o755 << 16
            with open(path, "rb") as fh:
                z.writestr(info, fh.read())
    key = "build/gpu-llm-serving-src.zip"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ExpectedBucketOwner=account)
    print(f"  uploaded build context -> s3://{bucket}/{key}")
    return key


def ensure_project(cb, role_arn: str, bucket: str, src_key: str, ecr: str) -> None:
    env = {
        "type": "LINUX_CONTAINER",
        # LARGE: the base image is ~9 GB and layers need somewhere to go.
        "computeType": "BUILD_GENERAL1_LARGE",
        "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
        # Docker-in-Docker needs privileged mode. This is why the build runs here rather than in
        # something lighter.
        "privilegedMode": True,
        "environmentVariables": [
            {"name": "ECR", "value": ecr},
            {"name": "REPO", "value": REPO_NAME},
        ],
    }
    source = {"type": "S3", "location": f"{bucket}/{src_key}"}
    try:
        cb.create_project(
            name=PROJECT_NAME, source=source, environment=env,
            artifacts={"type": "NO_ARTIFACTS"}, serviceRole=role_arn,
            timeoutInMinutes=60,
            description="Builds the GPU LLM serving image and pushes it to ECR",
        )
        print(f"  created CodeBuild project {PROJECT_NAME}")
    except cb.exceptions.ResourceAlreadyExistsException:
        # Converged rather than left alone, for the same reason as the role policy: the buildspec and
        # environment in this file must reach a project created by an earlier version.
        cb.update_project(name=PROJECT_NAME, source=source, environment=env,
                          serviceRole=role_arn, timeoutInMinutes=60)
        print(f"  updated CodeBuild project {PROJECT_NAME}")


def report(build: dict) -> None:
    print(f"  status: {build['buildStatus']}   phase: {build.get('currentPhase')}")
    for p in build.get("phases", []):
        if p.get("phaseStatus"):
            secs = p.get("durationInSeconds", 0)
            print(f"    {p['phaseType']:<20} {p['phaseStatus']:<12} {secs:>4}s")
            for ctx in p.get("contexts", []):
                if ctx.get("message"):
                    print(f"      {ctx['message'][:200]}")


# CodeBuild's terminal statuses do not mean the same thing, and printing one message for all of them
# sends a reader looking in the wrong place. FAULT and TIMED_OUT are CodeBuild's problem; FAILED is
# usually yours.
STATUS_HINT = {
    "FAILED": "the build ran and a command failed - the phase and message above say which.",
    "FAULT": "an internal CodeBuild fault, not a problem with your input. Retrying usually works.",
    "TIMED_OUT": "the build exceeded the project timeout. A very large model can legitimately need "
                 "longer, or the download stalled.",
    "STOPPED": "the build was stopped, either by hand or by `aws codebuild stop-build`.",
}

def wait_for(cb, build_id: str, region: str) -> bool:
    """Poll until the build finishes, printing each phase as it completes."""
    seen: set[str] = set()
    # The interrupt handler wraps the WHOLE loop body, sleep included. Wrapping only the API call left
    # the guidance below reachable solely if Ctrl-C landed inside a sub-second request, so in practice
    # the user got a bare "interrupted." and a build that kept running, and billing, invisibly.
    try:
        while True:
            build = cb.batch_get_builds(ids=[build_id])["builds"][0]
            for p in build.get("phases", []):
                key = p["phaseType"]
                if p.get("phaseStatus") and key not in seen:
                    seen.add(key)
                    print(f"    {key:<20} {p['phaseStatus']:<12} "
                          f"{p.get('durationInSeconds', 0):>4}s")
            if build.get("buildComplete"):
                ok = build["buildStatus"] == "SUCCEEDED"
                logs = build.get("logs", {})
                if not ok:
                    status = build["buildStatus"]
                    print(f"\nBuild {status}: {STATUS_HINT.get(status, '')}")
                    # The phases were printed as they completed; only their failure context is new.
                    for p in build.get("phases", []):
                        for ctx in p.get("contexts", []):
                            if ctx.get("message"):
                                print(f"    {p['phaseType']}: {ctx['message'][:200]}")
                    if logs.get("deepLink"):
                        print(f"\n  Full log: {logs['deepLink']}")
                    print(f"  Or: aws logs tail /aws/codebuild/{PROJECT_NAME} --since 30m "
                          f"--region {region}")
                return ok
            time.sleep(15)
    except KeyboardInterrupt:
        # The build keeps running - and keeps billing - after Ctrl-C, so say how to follow it or
        # stop it rather than leaving one going invisibly.
        print(f"\n\nStopped watching, but the build is STILL RUNNING.\n"
              f"  follow: python3 scripts/build_image.py --status {build_id}\n"
              f"  cancel: aws codebuild stop-build --id {build_id} --region {region}",
              file=sys.stderr)
        raise


def write_image_uri(uri: str) -> None:
    """Write `image: <uri>` into config.local.yaml, replacing any existing image line."""
    # config.local.yaml, never config.yaml. The URI contains the account id, and config.yaml is
    # tracked - a test fails if an account id appears in it, precisely so this cannot leak.
    path = local_config_path()
    header = ("# Local overrides, deep-merged over config.yaml. Gitignored, so this is where\n"
              "# account-specific values belong.\n")
    # A line-level edit rather than a YAML round-trip, so anything else already in this file -
    # including the API key and any comments - survives being rewritten.
    if os.path.exists(path):
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if not ln.startswith("image:")]
    else:
        lines = header.rstrip("\n").split("\n")
    lines.append(f"image: {uri}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote image into {os.path.basename(path)} (gitignored)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None, help="defaults to `region` in config.yaml")
    ap.add_argument("--write-config", action="store_true",
                    help="write the image URI into config.local.yaml (gitignored)")
    ap.add_argument("--status", default=None, help="report on an existing build id and exit")
    a = ap.parse_args()

    region = a.region or config_region()
    if not region:
        sys.exit("no region: set `region` in config.yaml, or pass --region")

    ident = boto3.client("sts", region_name=region).get_caller_identity()
    account = ident["Account"]
    # From the caller's own ARN, so the policies below are correct in aws-us-gov and
    # aws-cn too - the stack already uses self.partition for the same reason.
    partition = ident["Arn"].split(":")[1]
    # The DNS suffix follows the PARTITION, which is derived just above. Hardcoding amazonaws.com
    # meant this function could not emit a China URI even though the stack's image regex accepts one.
    dns = "amazonaws.com.cn" if partition == "aws-cn" else "amazonaws.com"
    ecr_host = f"{account}.dkr.ecr.{region}.{dns}"
    uri = f"{ecr_host}/{REPO_NAME}:{TAG}"
    bucket = f"{REPO_NAME}-build-{account}-{region}"

    cb = boto3.client("codebuild", region_name=region)
    if a.status:
        builds = cb.batch_get_builds(ids=[a.status])["builds"]
        if not builds:
            sys.exit(f"no build found with id {a.status} (build history ages out, and the id must "
                     f"include the project name).")
        report(builds[0])
        ok = builds[0].get("buildStatus") == "SUCCEEDED"
        if ok and a.write_config:
            write_image_uri(uri)
        # Non-zero on a failed build, so this is usable in a script rather than always succeeding.
        return 0 if ok else 1

    print(f"Building {uri}\n  region: {region}   (nothing large crosses your connection)")
    ensure_repo(boto3.client("ecr", region_name=region))
    s3 = boto3.client("s3", region_name=region)
    ensure_bucket(s3, bucket, region, account)
    # region_name on the IAM client too. IAM is global, but without it boto3 resolves the COMMERCIAL
    # endpoint, which defeats the partition derived above for aws-us-gov and aws-cn.
    role_arn = ensure_role(boto3.client("iam", region_name=region), account, partition)
    src_key = upload_context(s3, bucket, account)
    ensure_project(cb, role_arn, bucket, src_key, ecr_host)

    build_id = cb.start_build(
        projectName=PROJECT_NAME,
        environmentVariablesOverride=[{"name": "IMAGE_TAG", "value": TAG}],
    )["build"]["id"]
    print(f"\nstarted build {build_id}")
    if a.write_config:
        # Written NOW, not after the build: the URI is known in advance, and an interrupted wait must
        # not leave config.local.yaml pointing at nothing. If the build fails, the deploy fails on an
        # image pull, which is loud.
        write_image_uri(uri)
    print(f"  if this shell is interrupted: python3 scripts/build_image.py --status {build_id}"
          + (" --write-config" if a.write_config else ""))
    print("  phases (the base image is ~9 GB, so expect roughly 10-15 minutes):")
    if not wait_for(cb, build_id, region):
        return 1

    print(f"\nImage: {uri}")
    if not a.write_config:
        print("\nAdd it to config.local.yaml (gitignored, merged over config.yaml):\n"
              f"  image: {uri}\n"
              "Or export it for one deploy:\n"
              f"  export SERVING_IMAGE={uri}")
    # The tag is FIXED, so rebuilding pushes a new image to the same URI. The task definition's image
    # string is then unchanged, CloudFormation sees no diff, and `cdk deploy` does nothing - the old
    # image keeps serving and the change appears to have been ignored. Forcing a new deployment is what
    # makes ECS pull the tag again.
    print("\nAlready deployed? The tag is unchanged, so `cdk deploy` will see no difference and the\n"
          "running tasks keep the OLD image. Force ECS to pull it again:\n"
          f"  CLUSTER=$(aws cloudformation describe-stacks --stack-name GpuLlmServing --region {region} \\\n"
          "    --query 'Stacks[0].Outputs[?OutputKey==`ClusterName`].OutputValue' --output text)\n"
          f"  SERVICE=$(aws ecs list-services --cluster \"$CLUSTER\" --region {region} \\\n"
          "    --query 'serviceArns[0]' --output text | awk -F/ '{print $NF}')\n"
          f"  aws ecs update-service --cluster \"$CLUSTER\" --service \"$SERVICE\" --region {region} \\\n"
          "    --force-new-deployment")
    return 0


def _run(entry=None) -> int:
    """Turn an AWS API failure into one actionable line instead of a botocore traceback.

    The most common failure by far is an expired SSO session, and unhandled it
    surfaced as `ClientError: An error occurred (ExpiredToken)` with a stack trace at whichever API
    call happened to come first - which tells the reader nothing about what to do.
    """
    try:
        return (entry or main)()
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except (ClientError, BotoCoreError) as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        msg = getattr(e, "response", {}).get("Error", {}).get("Message", str(e))
        print(f"\nAWS error: {code} - {msg}", file=sys.stderr)
        if code in ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId",
                    "UnrecognizedClientException", "AccessDenied", "AccessDeniedException",
                    "CredentialsError", "NoCredentialsError"):
            print("  Check your credentials (and that they are for the right account), then retry.",
                  file=sys.stderr)
        else:
            print("  Check that `region` in your config is correct and that these credentials can\n"
                  "  reach it.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_run())
