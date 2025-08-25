# Runtime app (deployed on AgentCore)
# Save as runtime_app.py (this is the container entrypoint AgentCore invokes)

# uv pip install "bedrock-agentcore" "crewai[tools]" "langchain-aws" "python-dotenv"

import os, uuid
from dotenv import load_dotenv

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from crewai import Agent, Task, Crew, Process
from crewai_tools.aws.bedrock.agents.invoke_agent_tool import BedrockInvokeAgentTool
from langchain_aws import ChatBedrock  # Bedrock-hosted model for the manager

load_dotenv()
app = BedrockAgentCoreApp()

# --- Your existing Bedrock Agents (collaborators) ---
RESEARCHER_AGENT_ID    = os.getenv("RESEARCHER_AGENT_ID")
RESEARCHER_AGENT_ALIAS = os.getenv("RESEARCHER_AGENT_ALIAS")
WRITER_AGENT_ID        = os.getenv("WRITER_AGENT_ID")
WRITER_AGENT_ALIAS     = os.getenv("WRITER_AGENT_ALIAS")
AWS_REGION             = os.getenv("AWS_REGION", "us-east-1")

# --- Manager model hosted on Bedrock (use any Bedrock chat model id) ---
MANAGER_MODEL_ID = os.getenv("MANAGER_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
manager_llm = ChatBedrock(model_id=MANAGER_MODEL_ID, region=AWS_REGION, model_kwargs={"temperature": 0.2})

def make_mission(prompt: str) -> Task:
    return Task(
        description=(
            f"Create a sourced blog post on: {prompt}\n"
            "First research high-confidence facts with citations; then write a draft."
        ),
        expected_output=(
            "JSON with keys: notes[], sources[], draft. "
            "notes[] = 6–10 bullets with inline source ids; "
            "sources[] = objects {id,title,url}; "
            "draft = ~600–700 words with citations."
        ),
        criteria=("≥6 notes; ≥4 distinct credible sources; no unsupported claims; "
                  "draft within 600–700 words; citations present and deduped.")
    )

@app.entrypoint
def crew_entrypoint(payload: dict, context):
    """
    AgentCore Runtime entrypoint.
    Your client will call InvokeAgentRuntime with a JSON payload that includes 'prompt'.
    """
    prompt = payload.get("prompt", "AI safety updates in 2025")

    # 1) Let AgentCore drive sessioning: reuse context.session_id across downstream calls
    session_id = getattr(context, "session_id", None) or str(uuid.uuid4())

    # 2) Wrap Bedrock Agents as CrewAI tools (InvokeAgent under the hood)
    research_tool = BedrockInvokeAgentTool(
        agent_id=RESEARCHER_AGENT_ID,
        agent_alias_id=RESEARCHER_AGENT_ALIAS,
        session_id=session_id,
        enable_trace=False,
        end_session=False,
        description="Invoke Bedrock Researcher for fact-finding with citations."
    )
    write_tool = BedrockInvokeAgentTool(
        agent_id=WRITER_AGENT_ID,
        agent_alias_id=WRITER_AGENT_ALIAS,
        session_id=session_id,
        enable_trace=False,
        end_session=True,  # close after final step
        description="Invoke Bedrock Writer to produce a polished draft."
    )

    # 3) CrewAI agents (thin wrappers that primarily use the tools)
    researcher = Agent(
        role="Researcher (Bedrock)",
        goal="Gather recent, credible facts with citations.",
        tools=[research_tool],
        allow_delegation=False,
    )
    writer = Agent(
        role="Writer (Bedrock)",
        goal="Turn vetted notes into a polished, well-structured draft.",
        tools=[write_tool],
        allow_delegation=False,
    )

    # 4) Manager agent (planner/router with simple ReAct-style rules)
    manager = Agent(
        role="Orchestration Manager",
        goal=("Plan → route to Researcher → verify sources → route to Writer → verify output; "
              "stop when acceptance criteria are met."),
        backstory=("Methodical lead who follows a Reason→Act→Check loop; "
                   "requires citations and never fabricates data/URLs."),
        llm=manager_llm,
        allow_delegation=True,
        verbose=True,
    )

    mission = make_mission(prompt)

    crew = Crew(
        agents=[manager, researcher, writer],
        tasks=[mission],
        process=Process.hierarchical,     # dynamic on-the-fly planning/routing
        manager_agent=manager,            # use our custom manager
        verbose=True,
        max_iterations=8,
    )

    result = crew.kickoff()
    # Return JSON-serializable content to AgentCore (the client will parse this)
    return {"result": getattr(result, "raw", str(result))}

if __name__ == "__main__":
    app.run()
