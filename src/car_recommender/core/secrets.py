"""AWS Secrets Manager lookup, used only in the `DEPLOY_ENV=aws` path.

Local development never touches this module — Settings reads
ANTHROPIC_API_KEY / API_PASSWORD from the environment or a .env file by
default. In production (DEPLOY_ENV=aws), Settings.model_post_init calls
get_secret() for whichever of those two values wasn't already supplied via
the environment, pulling them from Secrets Manager instead.
"""

from functools import lru_cache

import boto3


@lru_cache(maxsize=8)
def get_secret(secret_id: str, region: str) -> str:
    """Fetch a secret string from AWS Secrets Manager. Cached per (id, region)
    so a cold-started process only pays the API call once per secret."""
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_id)
    return response["SecretString"]
