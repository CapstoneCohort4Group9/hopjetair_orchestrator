import boto3
from botocore.exceptions import BotoCoreError, ClientError
from .config import settings


def get_bedrock_client_with_sts():
    try:
        # Priority: Use AWS_PROFILE if defined, otherwise default profile or env
        profile_name = settings.AWS_PROFILE if hasattr(settings, 'AWS_PROFILE') and settings.AWS_PROFILE else None

        session = boto3.Session(profile_name=profile_name)

        # Optional: validate session identity
        caller = session.client("sts").get_caller_identity()
        print(f"Using credentials for: {caller['Arn']}")

        # Step 1: STS AssumeRole using the session
        sts = session.client("sts")

        response = sts.assume_role(
            RoleArn=settings.ASSUME_ROLE_ARN,
            RoleSessionName="LangGraphBedrockSession"
        )

        credentials = response['Credentials']

        # Step 2: Create a Bedrock Runtime client with temporary session credentials
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=settings.BEDROCK_REGION,
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken']
        )

        return bedrock

    except ClientError as e:
        raise RuntimeError(f"AWS ClientError: {e}")
    except BotoCoreError as e:
        raise RuntimeError(f"STS AssumeRole failed: {e}")
