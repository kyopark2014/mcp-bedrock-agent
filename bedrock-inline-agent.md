# Bedrock Inline Agent

## Multi-agent 

[Bedrock Inline Agent](https://awslabs.github.io/multi-agent-orchestrator/agents/built-in/bedrock-inline-agent/)와 같이 multi agent를 구현할 수 있습니다. 

먼저 python package를 설치합니다.

```text
pip install "multi-agent-orchestrator[aws]"
```

이후 아래와 같이 agent를 정의할 수 있습니다. 

```pyhton
from multi_agent_orchestrator.agents import BedrockInlineAgent, BedrockInlineAgentOptions

action_groups = [
  {
    "actionGroupName": "OrderManagement",
    "description": "Handles order-related operations like status checks and updates"
  },
  {
    "actionGroupName": "InventoryLookup",
    "description": "Checks product availability and stock levels"
  }
]

knowledge_bases = [
  {
    "knowledgeBaseId": "KB001",
    "description": "Product catalog and specifications"
  }
]

agent = BedrockInlineAgent(BedrockInlineAgentOptions(
    name='Inline Agent Creator for Agents for Amazon Bedrock',
    description='Specialized in creating Agent to solve customer request dynamically. You are provided with a list of Action groups and Knowledge bases which can help you in answering customer request',
    action_groups_list=action_groups,
    knowledge_bases=knowledge_bases,
    region="us-east-1",
    LOG_AGENT_DEBUG_TRACE=True,
    inference_config={
        'maxTokens': 500,
        'temperature': 0.5,
        'topP': 0.9
    }
))
```


[Amazon Bedrock Agents with Return Control SDK](https://github.com/mikegc-aws/Amazon-Bedrock-Inline-Agents-with-Return-Control)

