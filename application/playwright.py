import logging
import sys

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

server_params = StdioServerParameters(
    command="npx",
    args=["@playwright/mcp@latest"],
)

async def playwright_agent(text:str, st):
    summary_mcp_client = await MCPStdio.create(server_params=server_params)

    try:
        # Step 3: Define an action group
        summary_action_group = ActionGroup(
            name="SummaryActionGroup",
            description="Summary the given link.",
            mcp_clients=[summary_mcp_client],
        )

        result = await InlineAgent(
            foundation_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            instruction="""You are a friendly assistant that is responsible for resolving user queries. """,
            agent_name="summary_agent",
            action_groups=[summary_action_group],
        ).invoke(
            input_text="https://community.aws/content/2v8AETAkyvPp9RVKC4YChncaEbs/running-mcp-based-agents-clients-servers-on-aws 의 내용을 요약해주세요."
        )

        logger.info(f"result: {result}")
        st.markdown(result)
    
    finally:
        await summary_mcp_client.cleanup()

import asyncio
def run_playwright_agent(text, st):
    asyncio.run(playwright_agent(text, st))
