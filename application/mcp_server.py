import logging
import sys
import json
import datetime

from mcp import StdioServerParameters
from InlineAgent.tools import MCPStdio
from InlineAgent.action_group import ActionGroup
from InlineAgent.agent import InlineAgent

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

async def run(text, mcp_config, st):
    server_params = load_mcp_server_parameters(mcp_config)

    cost_client = await MCPStdio.create(server_params=server_params)

    try:
        # Define an action group
        cost_action_group = ActionGroup(
            name="CostActionGroup",
            description="retrieve cost analysis data collection",
            mcp_clients=[cost_client],
        )

        time_group = ActionGroup(
            name="TimeService",
            description="Provides time-related information",
            tools=[get_time],
        )

        # Invoke agent
        result = await InlineAgent(
            foundation_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            instruction="""You are a friendly assistant that is responsible for resolving user queries. """,
            agent_name="cost_agent",
            action_groups=[cost_action_group, time_group],
        ).invoke(
            input_text=text
        )

        logger.info(f"result: {result}")
        st.markdown(result)

    finally:
        await cost_client.cleanup()
                
    return result, ""
