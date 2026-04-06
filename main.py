# ─── LangGraph ReAct Agent — Entry Point ─────────────────────────────────────
import asyncio
import os
import sys

import nest_asyncio
nest_asyncio.apply()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from graph import AgentState, REACT_SYSTEM, build_graph

# ─── LLM ──────────────────────────────────────────────────────────────────────
# Set GOOGLE_API_KEY in your environment (or replace with your preferred LLM).
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    temperature=0,
)

# ─── MCP Server Config ────────────────────────────────────────────────────────
# math + search run as local subprocesses; weather must already be running on 8000
TOOLS_DIR = os.path.join(os.path.dirname(__file__), "Tools")

MCP_CONFIG = {
    "math": {
        "command": sys.executable,
        "args": [os.path.join(TOOLS_DIR, "math_server.py")],
        "transport": "stdio",
    },
    "search": {
        "command": sys.executable,
        "args": [os.path.join(TOOLS_DIR, "search_server.py")],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    },
}


async def run_agent(query: str) -> str:
    """Initialise MCP tools, build the LangGraph, and run the ReAct loop."""
    # ── Load tools from MCP servers ───────────────────────────────────────────
    mcp = MultiServerMCPClient(MCP_CONFIG)

    tools: list = []
    for server in ["math", "search", "weather"]:
        try:
            server_tools = await mcp.get_tools(server_name=server)
            tools.extend(server_tools)
            print(f"  Loaded {len(server_tools)} tool(s) from '{server}'")
        except Exception as exc:
            print(f"  WARNING: could not load '{server}' tools — {exc}")

    tools_map = {t.name: t for t in tools}
    print(f"  Total tools available: {list(tools_map.keys())}\n")

    # ── Bind tools to LLM and compile graph ───────────────────────────────────
    llm_with_tools = llm.bind_tools(tools)
    graph = build_graph(llm_with_tools, tools_map)

    # ── Initial state ─────────────────────────────────────────────────────────
    initial_state: AgentState = {
        "input": query,
        "messages": [
            SystemMessage(content=REACT_SYSTEM),
            HumanMessage(content=query),
        ],
        "agent_scratchpad": "",
        "final_answer": "",
        "steps": [],
    }

    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    # ── Run the graph ─────────────────────────────────────────────────────────
    result = await graph.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_answer"])
    print("\nSteps taken:", len(result["steps"]))
    for i, step in enumerate(result["steps"], 1):
        print(f"  Step {i}: [{step['action']}] → {str(step['observation'])[:80]}")

    return result["final_answer"]


# ─── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = (
        "What is the weather in Lahore and who is the current Prime Minister "
        "of Pakistan? Now get the age of PM and tell us will this weather "
        "suits PM health."
    )
    asyncio.run(run_agent(query))

