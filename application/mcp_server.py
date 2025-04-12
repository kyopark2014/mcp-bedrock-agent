import json
import boto3
import traceback
import logging
import sys
import mcp_log as log
import mcp_cost as cost
import mcp_rag as rag
import mcp_s3 as storage

from typing import Dict, Optional, Any
from langchain_experimental.tools import PythonAstREPLTool
from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-server")

try:
    mcp = FastMCP(
        name = "tools",
        instructions=(
            "You are a helpful assistant for the user's question and provide the answer."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# RAG
######################################
@mcp.tool()
def search(keyword: str) -> str:
    "search keyword"

    return rag.retrieve_knowledge_base(keyword)

######################################
# Code Interpreter
######################################
repl = PythonAstREPLTool()

@mcp.tool()
def repl_coder(code):
    """
    Use this to execute python code and do math. 
    If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user.
    code: the Python code was written in English
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    if result is None:
        result = "It didn't return anything."

    return result

######################################
# AWS Cost
######################################
@mcp.tool()
def aws_cost_loader(days: int=30, region: str='us-west-2') -> list:
    """
    load aws cost data
    days: the number of days looking for cost data
    region: the name of aws region
    return: cost data during days
    """

    return cost.get_cost_analysis(days=days, region=region)

######################################
# AWS Logs
######################################
@mcp.tool()
async def list_groups(
    prefix: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """List available CloudWatch log groups."""
    return await log.list_groups(prefix=prefix, region=region)

@mcp.tool()
async def get_logs(
    logGroupName: str,
    logStreamName: Optional[str] = None,
    startTime: Optional[str] = None,
    endTime: Optional[str] = None,
    filterPattern: Optional[str] = None,
    region: Optional[str] = 'us-west-2'
) -> str:
    """Get CloudWatch logs from a specific log group and stream."""

    return await log.get_logs(
        logGroupName=logGroupName,
        logStreamName=logStreamName,
        startTime=startTime,
        endTime=endTime,
        filterPattern=filterPattern,
        region=region
    )

from typing import List, Optional
from mcp.types import Resource

######################################
# AWS S3
######################################
@mcp.tool()
async def list_buckets(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List S3 buckets using async client with pagination
    """
    return await storage.list_buckets(start_after, max_buckets, region)
    
@mcp.tool()  
async def list_objects(
    bucket_name: str, 
    prefix: Optional[str] = "", 
    max_keys: Optional[int] = 1000,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List objects in a specific bucket using async client with pagination
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Object prefix for filtering
        max_keys: Maximum number of keys to return,
        region: Name of the aws region
    """
    return await storage.list_objects(bucket_name, prefix, max_keys, region)

@mcp.tool()    
async def list_resources(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[Resource]:
    """
    List S3 buckets and their contents as resources with pagination
    Args:
        start_after: Start listing after this bucket name
    """
    return await storage.list_resources(start_after, max_buckets, region)

######################################
# AWS Logs
######################################

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


