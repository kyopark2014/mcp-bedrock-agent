# MCP를 이용해 Bedrock Agent 이용하기

## Inline Agent 준비

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
pip install pandas aioboto3
```



## Reference

[Running MCP-Based Agents (Clients & Servers) on AWS](https://community.aws/content/2v8AETAkyvPp9RVKC4YChncaEbs/running-mcp-based-agents-clients-servers-on-aws)

[boto3-invoke_inline_agent](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_inline_agent.html)

[Amazon Bedrock Agents with Return Control SDK](https://github.com/mikegc-aws/Amazon-Bedrock-Inline-Agents-with-Return-Control)

[Bedrock Inline Agent](https://awslabs.github.io/multi-agent-orchestrator/agents/built-in/bedrock-inline-agent/)
