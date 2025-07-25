import boto3
import os
from botocore.exceptions import BotoCoreError, ClientError
from .config import settings
import logging

logger = logging.getLogger(__name__)


def is_running_on_aws():
    """Check if running in AWS environment like ECS, EC2, or Lambda"""
    return (
        os.environ.get("ECS_CONTAINER_METADATA_URI_V4") or
        os.environ.get("AWS_EXECUTION_ENV") or
        os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
    )


def get_bedrock_client_with_sts():
    try:
        # If running on AWS (Fargate, EC2, Lambda), use default credentials (e.g., task role)
        if is_running_on_aws():
            session = boto3.Session()
            caller = session.client("sts").get_caller_identity()
            #print(f"[INFO] Using AWS environment credentials for: {caller['Arn']}")

            return boto3.client(
                "bedrock-runtime",
                region_name=settings.BEDROCK_REGION
            )

        # If running locally, use configured profile and assume role
        print("[INFO] Detected local environment. Assuming role...")

        profile_name = getattr(settings, "AWS_PROFILE", None)
        session = boto3.Session(profile_name=profile_name)

        sts = session.client("sts")
        caller = sts.get_caller_identity()
        #print(f"[INFO] Using AWS_PROFILE: {profile_name}, identity: {caller['Arn']}")

        response = sts.assume_role(
            RoleArn=settings.ASSUME_ROLE_ARN,
            RoleSessionName="LangGraphBedrockSession"
        )

        credentials = response["Credentials"]

        return boto3.client(
            "bedrock-runtime",
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"]
        )

    except ClientError as e:
        logger.error(f"[LangGraph Error] {str(e)}", exc_info=True)
        raise RuntimeError(f"[ERROR] AWS ClientError: {e}")
    except BotoCoreError as e:
        logger.error(f"[LangGraph Error] {str(e)}", exc_info=True)
        raise RuntimeError(f"[ERROR] BotoCoreError: {e}")
