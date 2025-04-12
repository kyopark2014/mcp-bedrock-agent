from InlineAgent.agent import InlineAgent
from InlineAgent.action_group import ActionGroup

import asyncio
import datetime
import sys
import logging

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("time")

def get_time() -> dict:
    """Get the current time"""
    now = datetime.datetime.now()

    logger.info(f"Current time: {now}")

    #return {"time": now.strftime("%H:%M:%S")}
    return now.strftime("%H:%M:%S")

time_group = ActionGroup(
    name="TimeService",
    description="Provides time-related information",
    tools=[get_time],
)

agent = InlineAgent(
    foundation_model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    instruction="You are a helpful assistant that can tell the time.",
    action_groups=[time_group],
    agent_name="TimeAgent",
)

asyncio.run(agent.invoke(input_text="time?"))
