# Client that invokes the Crew via AgentCore
# Save as client_invoke.py. This is what your external application runs

# uv pip install boto3 python-dotenv
import os, json, uuid
from dotenv import load_dotenv
import boto3

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
AGENTCORE_RUNTIME_ARN = os.getenv("AGENTCORE_RUNTIME_ARN")  # e.g., arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime
PROMPT = os.getenv("PROMPT", "Top 5 AI safety developments in 2025 and why they matter")

# 1) Create the AgentCore client
agentcore = boto3.client("bedrock-agentcore", region_name=REGION)

# 2) Prepare a session id so conversation context sticks across calls
session_id = os.getenv("RUNTIME_SESSION_ID") or str(uuid.uuid4())

# 3) Payload your runtime expects (see runtime_app.py entrypoint)
payload = json.dumps({"prompt": PROMPT}).encode()

# 4) Invoke AgentCore Runtime (streaming by default)
resp = agentcore.invoke_agent_runtime(
    agentRuntimeArn=AGENTCORE_RUNTIME_ARN,
    runtimeSessionId=session_id,
    payload=payload,
    # You can also set contentType / accept if you need a specific format
    # contentType="application/json",
    # accept="text/event-stream",
)

print(f"Status: {resp.get('statusCode')}, contentType: {resp.get('contentType')}")
content_type = resp.get("contentType", "")

if "text/event-stream" in content_type:
    # Handle SSE
    chunks = []
    for line in resp["response"].iter_lines(chunk_size=10):
        if not line:
            continue
        s = line.decode("utf-8")
        if s.startswith("data: "):
            data = s[6:]
            print(data)  # stream to console
            chunks.append(data)
    print("\n--- Complete (SSE) ---")
    print("\n".join(chunks))
elif content_type == "application/json":
    # Non-streaming fallback
    buf = []
    for chunk in resp.get("response", []):
        buf.append(chunk.decode("utf-8"))
    combined = "".join(buf)
    print("\n--- Complete (JSON) ---")
    print(json.loads(combined))
else:
    # Raw fallback
    print(resp)
