# MCP를 이용해 Bedrock Agent 이용하기

Amazon의 Bedrock agent는 완전관리형 Agent 호스팅 서비스로 인프라 관리에 대한 부담없이 편리하게 agent를 이용할 수 있습니다. Bedrock agent에서 MCP를 이용하기 위해서는 [아래 그림](https://github.com/awslabs/amazon-bedrock-agent-samples/tree/main/src/InlineAgent)과 같이 InlineAgent SDK를 이용합니다. 아래와 같이 MCP는 애플리케이션과 함께 있으면서 tools, resources에 대한 접근을 위한 표준 인터페이스를 제공합니다. 

<img src="https://github.com/user-attachments/assets/3641a558-87af-4060-ad25-15fa9b8227aa" width="600">

## InlineAgent SDK의 준비

[InlineAgent](https://github.com/awslabs/amazon-bedrock-agent-samples/tree/main/src/InlineAgent#setup) SDK는 [Amazon Web Services - Labs](https://github.com/awslabs)에서 오픈소스로 제공합니다. 

### 사용 준비

아래와 같이 mcp-bedrock-agent을 다운로드 합니다. 

```text
git clone https://github.com/kyopark2014/mcp-bedrock-agent
```

아래처럼 빌드합니다.

```text
cd mcp-bedrock-agent/InlineAgent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

이제 아래처럼 동작을 테스트 할 수 있습니다. 

```text
cd ../../examples
python hello_world.py
```

### Inline Agent SDK 업데이트

[InlineAgent](https://github.com/awslabs/amazon-bedrock-agent-samples/tree/main/src/InlineAgent#setup)을 참조하여 아래와 같이 InlineAgent를 업데이트 할 수 있습니다. 먼저 github을 다운로드합니다.

```python
git clone https://github.com/awslabs/amazon-bedrock-agent-samples.git
```

"amazon-bedrock-agent-samples/src"에서 "InlineAgent"을 복사해서 다운로드한 "mcp-bedrock-agent"에 아래처럼 복사합니다.

![image](https://github.com/user-attachments/assets/c28c27cc-f87a-4b7d-8630-238e2ea08922)

### 실행하기

Inline Agent에 필요한 패키지는 아래와 같습니다.

```python
pip install opentelemetry-api openinference-instrumentation-langchain opentelemetry-exporter-otlp
```

필요한 패키지를 설치합니다.

```python
pip install pandas aioboto3 langchain_experimental
```

## MCP 활용하기

MCP 설정은 아래와 같은 json 포맷을 활용합니다. 아래는 mcp_config의 예입니다.

```java
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": [
        "@playwright/mcp@latest"
      ]
    }
  }
}
```

아래와 같이 mcp_config로부터 mcp client를 생성합니다. 

```python
from mcp import StdioServerParameters
mcpServers = mcp_json.get("mcpServers")

mcp_clients = []
for server in mcpServers:
    config = mcpServers.get(server)

    if "command" in config:
        command = config["command"]
    if "args" in config:
        args = config["args"]
    if "env" in config:
        env = config["env"]
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
    else:
        server_params = StdioServerParameters(
            command=command,
            args=args
        )
    tool_client = await MCPStdio.create(server_params=server_params)
    mcp_clients.append(tool_client)
```

이제 아래와 같이 action group을 생성합니다. 

```python
from InlineAgent.action_group import ActionGroup
tool_action_group = ActionGroup(
    name="ToolActionGroup",
    description="retrieve information using tools",
    mcp_clients=mcp_clients,
)
```

이때 추가로 action group을 지정할 수 있습니다. 아래와 같이 bucket 리스트를 조회하는 list_buckets를 함수로 정의한 후에 action group의 tools에 등록합니다. InlineAgent SDK에서는 tool의 리턴값으로 string을 사용하고 있으므로 list_buckets에서는 string을 리턴합니다. 또한 list_buckets의 doc string에는 tool의 설명과 함께 "Parameters"로 입력값을 정의하여야 합니다. Bedrock agent에서는 tool의 입력 파라미터가 5개 이내로 제한됩니다.

```python
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
        response = await s3.list_buckets()
        buckets = response.get('Buckets', [])
        result = "" 
        for bucket in buckets:
            result += f"Name: {bucket['Name']}, CreationDate: {bucket['CreationDate']}\n"
        return result        

list_bucket_group = ActionGroup(
    name="ListBucketService",
    description="list the buckets on AWS",
    tools=[list_buckets],
)
```

이제 정의한 action group들을 아래와 같이 InlineAgent에 등록하여 사용할 수 있습니다. 입력은 input_text을 이용하고, enable_trace로 상세한 동작을 확인할 수 있습니다. session_id을 이용하면 메모리를 활용하여 이전 conversation을 응답에 활용할 수 있습니다.  

```python
from InlineAgent.agent import InlineAgent

result = await InlineAgent(
    foundation_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    instruction="""You are a friendly assistant that is responsible for resolving user queries. """,
    agent_name="tool_agent",
    action_groups=[
        tool_action_group, 
        cost_group, 
        list_log_group, 
        get_log_group, 
        list_bucket_group, 
        list_object_group
    ],
).invoke(
    input_text=text,
    enable_trace = True,
    session_id=session_id
)
```

## 실행 결과

여기서 사용한 config.json의 playwright을 이용하면 아래와 같이 특정 url의 내용을 요약할 수 있습니다. 

<img src="https://github.com/user-attachments/assets/90e04264-7b1f-4ff3-9aae-35b35b66af3f" width="700">

"S3의 bucket들의 사용 현황을 분석해주세요."와 같이 입력하면 아래와 같이 현재 S3의 현황을 정리해서 볼 수 있습니다.

<img src="https://github.com/user-attachments/assets/c913e9fe-f6f6-46a9-b96b-36c6beea7340" width="700">


![image](https://github.com/user-attachments/assets/ad0774e4-b423-44ba-bd94-6baaa73062dc)

"지난 3개월의 AWS 리소스 사용 내역을 분석해주세요."라고 입력하면 AWS의 사용량에 대한 분석 결과를 알수 있습니다.

<img src="https://github.com/user-attachments/assets/6d134d3c-fc94-427d-8fba-ebbe482e3d58" width="700">




## Reference

[Running MCP-Based Agents (Clients & Servers) on AWS](https://community.aws/content/2v8AETAkyvPp9RVKC4YChncaEbs/running-mcp-based-agents-clients-servers-on-aws)

[boto3-invoke_inline_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_inline_agent.html)

[Amazon Bedrock Agents with Return Control SDK](https://github.com/mikegc-aws/Amazon-Bedrock-Inline-Agents-with-Return-Control)

[Bedrock Inline Agent](https://awslabs.github.io/multi-agent-orchestrator/agents/built-in/bedrock-inline-agent/)
