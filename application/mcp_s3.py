import logging
import sys

from typing import List, Optional, Any
import asyncio
import os
from mcp.types import Resource
import aioboto3

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-s3")

session = aioboto3.Session()

def _get_configured_buckets() -> List[str]:
    """
    Get configured bucket names from environment variables.
    Format in .env file:
    S3_BUCKETS=bucket1,bucket2,bucket3
    or
    S3_BUCKET_1=bucket1
    S3_BUCKET_2=bucket2
    see env.example ############
    """
    # Try comma-separated list first
    bucket_list = os.getenv('S3_BUCKETS')
    if bucket_list:
        return [b.strip() for b in bucket_list.split(',')]

    buckets = []
    i = 1
    while True:
        bucket = os.getenv(f'S3_BUCKET_{i}')
        if not bucket:
            break
        buckets.append(bucket.strip())
        i += 1

    return buckets            

configured_buckets = _get_configured_buckets()

def is_text_file(key: str) -> bool:
    """Determine if a file is text-based by its extension"""
    text_extensions = {
        '.txt', '.log', '.json', '.xml', '.yml', '.yaml', '.md',
        '.csv', '.ini', '.conf', '.py', '.js', '.html', '.css',
        '.sh', '.bash', '.cfg', '.properties'
    }
    return any(key.lower().endswith(ext) for ext in text_extensions)

async def list_buckets(
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List S3 buckets using async client with pagination
    Parameters:
        max_buckets: the number of buckets 
        region: the region of aws infrastructure, e.g. us-west-2
    """
    async with session.client('s3', region_name=region) as s3:
        
        # Default behavior if no buckets configured
        response = await s3.list_buckets()
        logger.info(f"buckets: {response['Buckets']}")

        buckets = response.get('Buckets', [])

        result = "" 
        for bucket in buckets:
            logger.info(f"bucket: {bucket['Name']}")
            result += f"Name: {bucket['Name']}, CreationDate: {bucket['CreationDate']}\n"

        logger.info(f"bucket info: {result}")
        
        return result        

async def list_objects(
    bucket_name: str, 
    prefix: Optional[str] = "", 
    max_keys: Optional[int] = 1000,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List objects in a specific bucket using async client with pagination
    Parameters:
        bucket_name: Name of the S3 bucket
        prefix: Object prefix for filtering
        max_keys: Maximum number of keys to return,
        region: the name of aws region
    """
    async with session.client('s3', region_name=region) as s3:
        response = await s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=int(max_keys)
        )
        logger.info(f"objects: {response.get('Contents')}")

        result = ""
        for obj in response.get('Contents', []):
            result += f"Key: {obj['Key']}, Size: {obj['Size']}, LastModified: {obj['LastModified']}\n"
        
        return result
    