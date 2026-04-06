# ─── LangGraph ReAct Workflow ────────────────────────────────────────────────
import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ─── 1. State ─────────────────────────────────────────────────────────────────

REACT_SYSTEM = """You are a ReAct agent. Strictly follow this loop:
Thought → Action (tool call) → Observation → Thought → ...

RULES:
1. ALWAYS use a tool for factual information — never answer from memory.
2. For multi-part questions, make one tool call per fact.
3. ALWAYS use calculator for any arithmetic — never compute in your head.
4. Only give Final Answer AFTER all required tool calls are complete."""


class AgentState(TypedDict):
    input: str
    # Annotated with operator.add so each node's returned list gets appended
    messages: Annotated[list[BaseMessage], operator.add]
    agent_scratchpad: str          # running Thought/Action/Observation text
    final_answer: str              # populated once the LLM stops calling tools
    steps: list[dict[str, Any]]    # optional history of actions + observations


# ─── 2. Graph Builder ─────────────────────────────────────────────────────────

def build_graph(llm_with_tools, tools_map: dict):
    """
    Construct and compile the LangGraph ReAct workflow.

    Flow:  START → react_node → (action?) → tool_node → react_node → …
                                         ↘ (final answer) → END
    """

    # ── ReAct Node ────────────────────────────────────────────────────────────
    def react_node(state: AgentState) -> dict:
        """Call the LLM; produce either a tool-call action or a final answer."""
        response = llm_with_tools.invoke(state["messages"])

        scratchpad = state.get("agent_scratchpad", "")

        if response.tool_calls:
            for tc in response.tool_calls:
                scratchpad += (
                    f"\nThought: I need to call '{tc['name']}' "
                    f"with args {tc['args']}"
                )
                print(f"  [react_node] Action → [{tc['name']}]  args: {tc['args']}")
        else:
            scratchpad += f"\nFinal Answer: {response.content}"
            print(f"  [react_node] Final Answer: {response.content[:120]}")

        return {
            "messages": [response],
            "agent_scratchpad": scratchpad,
            # Only set final_answer when there are no pending tool calls
            "final_answer": (
                response.content
                if not response.tool_calls
                else state.get("final_answer", "")
            ),
        }

    # ── Tool Node ─────────────────────────────────────────────────────────────
    async def tool_node(state: AgentState) -> dict:
        """Execute every tool call in the latest AIMessage and return observations."""
        last_msg = state["messages"][-1]   # AIMessage with .tool_calls populated
        tool_messages: list[BaseMessage] = []
        steps = list(state.get("steps", []))
        scratchpad = state.get("agent_scratchpad", "")

        for tc in last_msg.tool_calls:
            print(f"  [tool_node]  Executing [{tc['name']}] …")
            result = await tools_map[tc["name"]].ainvoke(tc["args"])
            observation = str(result)
            print(f"  [tool_node]  Observation: {observation[:200]}")

            tool_messages.append(
                ToolMessage(content=observation, tool_call_id=tc["id"])
            )
            scratchpad += f"\nObservation: {observation}"
            steps.append(
                {"action": tc["name"], "args": tc["args"], "observation": observation}
            )

        return {
            "messages": tool_messages,
            "agent_scratchpad": scratchpad,
            "steps": steps,
        }

    # ── Conditional Router ────────────────────────────────────────────────────
    def should_continue(state: AgentState) -> str:
        """If the last LLM message contains tool calls, route to tool_node; otherwise end."""
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "tool_node"
        return END

    # ── Assemble Graph ────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("react_node", react_node)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "react_node")
    graph.add_conditional_edges(
        "react_node",
        should_continue,
        {"tool_node": "tool_node", END: END},
    )
    graph.add_edge("tool_node", "react_node")

    return graph.compile()

