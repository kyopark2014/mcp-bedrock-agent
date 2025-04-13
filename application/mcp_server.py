import logging
import sys
import json
import datetime
import mcp_log as log
import mcp_cost as cost
import mcp_rag as rag
import mcp_s3 as storage

from mcp import StdioServerParameters
from InlineAgent.tools import MCPStdio
from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent
# from InlineAgent.knowledge_base import KnowledgeBase

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("playwright")

def load_mcp_server_parameters(mcp_config):
    mcp_json = json.loads(mcp_config)
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")

    command = ""
    args = []
    env = ""
    if mcpServers is not None:
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

            break

    if env:
        return StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
    else:
        return StdioServerParameters(
            command=command,
            args=args
        )

def load_multiple_mcp_server_parameters(mcp_config):
    logger.info(f"mcp_config: {mcp_config}")

    mcp_json = json.loads(mcp_config)
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")
  
    server_info = {}
    if mcpServers is not None:
        command = ""
        args = []
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

                server_info[server] = {
                    "command": command,
                    "args": args,
                    "env": env,
                    "transport": "stdio"
                }
            else:
                server_info[server] = {
                    "command": command,
                    "args": args,
                    "transport": "stdio"
                }
    logger.info(f"server_info: {server_info}")

    return server_info


def get_time() -> dict:
    """Get the current time"""
    now = datetime.datetime.now()

    logger.info(f"Current time: {now}")

    #return {"time": now.strftime("%H:%M:%S")}
    return now.strftime("%H:%M:%S")

# restaurant_kb = KnowledgeBase(
#     name="restaurant-kb",
#     description="Use this knowledgebase to get information about restaurants menu.",
#     additional_props={
#         "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 5}}
#     },
# )

import uuid
session_id = f"session-{str(uuid.uuid4())}"
async def run(text, mcp_config, st):
    server_params = load_mcp_server_parameters(mcp_config)

    tool_client = await MCPStdio.create(server_params=server_params)

    try:
        # Define an action group
        tool_action_group = ActionGroup(
            name="ToolActionGroup",
            description="retrieve information using tools",
            mcp_clients=[tool_client],
        )

        time_group = ActionGroup(
            name="TimeService",
            description="Provides time-related information",
            tools=[get_time],
        )

        cost_group = ActionGroup(
            name="costService",
            description="tools for AWS",
            tools=[cost.get_cost_analysis],
        )

        list_log_group = ActionGroup(
            name="ListLogService",
            description="earn the list of log groups on AWS",
            tools=[log.list_groups],
        )

        get_log_group = ActionGroup(
            name="GetLogService",
            description="get the log of the group on AWS",
            tools=[log.get_logs],
        )

        list_bucket_group = ActionGroup(
            name="ListBucketService",
            description="list the buckets on AWS",
            tools=[storage.list_buckets],
        )

        list_object_group = ActionGroup(
            name="ListObjectService",
            description="list the objects in the bucket",
            tools=[storage.list_objects],
        )
        
        # Invoke agent
        result = await InlineAgent(
            foundation_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            instruction="""You are a friendly assistant that is responsible for resolving user queries. """,
            agent_name="tool_agent",
            #action_groups=[tool_action_group, time_group, aws_group],
            action_groups=[
                tool_action_group, 
                time_group, 
                cost_group, 
                list_log_group, 
                get_log_group, 
                list_bucket_group, 
                list_object_group
            ],
        ).invoke(
            input_text=text,
            enable_trace = True,
            add_citation=True,
            session_id=session_id
        )

        logger.info(f"result: {result}")
        st.markdown(result)

        image_url = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result,
            "images": image_url if image_url else []
        })

    finally:
        await tool_client.cleanup()
                
    return result, ""
