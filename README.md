[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LWzSlHLS)
# Quiz: Convert ReAct Agent to LangGraph 🦜🕸️

## Objective
Convert a standard working **ReAct agent** (implemented in LangChain) into a **LangGraph workflow**. Your implementation must preserve the iterative reasoning and tool-usage behavior inherent to the ReAct framework.

---

## 🛠 Provided Resources
* **Existing ReAct agent code** (LangChain-based).
* **Tool implementations** (functional and ready for use).

---

## 📋 Requirements

### 1. Define State
Create a state structure (TypedDict or Pydantic) to represent the workflow. Your state must include:
* `input`: The original user query.
* `agent_scratchpad`: Stores intermediate reasoning (Thoughts, Actions, Observations).
* `final_answer`: The final response delivered to the user.
* `steps`: (Optional) A list to track history of actions and observations.

### 2. ReAct Node (Reasoning + Action)
Implement a node that:
1.  Takes the current state.
2.  Calls the LLM using **ReAct-style prompting**.
3.  Produces either an **Action** (tool name + arguments) or a **Final Answer**.
4.  Updates the state accordingly.

### 3. Tool Execution Node
Implement a node that:
1.  Executes the tool selected by the ReAct node.
2.  Passes the correct arguments to the tool.
3.  Stores the **Observation** (result) back in the state.
4.  Updates the scratchpad to prepare for the next reasoning step.

### 4. Graph Flow
Construct a LangGraph workflow that follows this logic:

> **START** $\rightarrow$ `react_node` $\rightarrow$ **Conditional Edge**
> * If **Action** $\rightarrow$ `tool_node` $\rightarrow$ `react_node`
> * If **Final Answer** $\rightarrow$ **END**

**The graph must:**
* Support iterative reasoning loops.
* Continue execution until a terminal state (Final Answer) is reached.

### 5. Conditional Routing
Implement the router logic to determine the next step based on the model's output:
- `is_action` $\rightarrow$ Route to `tool_node`.
- `is_final` $\rightarrow$ Route to `END`.

---

## 🧪 Test Case
Your implementation should successfully process complex, multi-step queries such as:

> *"What is the weather in Lahore and who is the current Prime Minister of Pakistan? Now get the age of PM and tell us will this weather suits PM health."*

---

## ⚠️ Constraints
* **No Hardcoding:** Do not hardcode outputs; the logic must be dynamic.
* **Reasoning Integrity:** Maintain the "Thought $\rightarrow$ Action $\rightarrow$ Observation" flow.
* **Scalability:** The agent must be capable of calling tools multiple times in a single run.
* **State Management:** Ensure proper state updates to prevent infinite loops or data loss between iterations.

---

## 🚀 Submission
Push your solution to this repository using the following structure:
```text
.
├── main.py          # Entry point for execution
├── graph.py         # LangGraph definition (optional)
└── README.md        # Project documentation
```

---

## ✅ My Solution

### Implementation Overview

| File | Description |
|------|-------------|
| [graph.py](graph.py) | `AgentState` TypedDict, `build_graph()` with `react_node`, `tool_node`, conditional routing |
| [main.py](main.py) | MCP client setup, LLM init, graph invocation, test query |
| [Tools/search_server.py](Tools/search_server.py) | Tavily web/news search MCP server (stdio) |
| [Tools/math_server.py](Tools/math_server.py) | Calculator MCP server (stdio) |
| [Tools/weather_server.py](Tools/weather_server.py) | Open-Meteo weather MCP server (HTTP on port 8000) |

### Graph Flow
```
START → react_node → [tool calls?] → tool_node → react_node → ... → END
                   ↘ [final answer] → END
```

- **`react_node`** — async, calls LLM via `ainvoke`, records thoughts/actions in scratchpad
- **`tool_node`** — async, executes MCP tools, appends `ToolMessage` observations
- **Conditional router** — checks `last_message.tool_calls`; routes to `tool_node` or `END`

### LLM & Tools Used
- **LLM:** Groq `llama-3.3-70b-versatile`
- **Tools:** `get_current_weather`, `get_weather_forecast`, `search_web`, `search_news`, `calculator`, `add`, `subtract`, `multiply`, `divide`, `power`, `square_root`

---

## ▶️ How to Run

### Prerequisites
```bash
pip install langgraph langchain-groq langchain-mcp-adapters tavily-python langchain-core nest_asyncio
```

### Step 1 — Start the Weather Server (Terminal 1)
```bash
cd Tools
python weather_server.py
```
Keep this terminal open. The server runs on `http://localhost:8000`.

### Step 2 — Run the Agent (Terminal 2)
```bash
# Set API keys
$env:GROQ_API_KEY   = "your_groq_api_key"
$env:TAVILY_API_KEY = "your_tavily_api_key"

python main.py
```

### Sample Output
```
Loaded 7 tool(s) from 'math'
Loaded 2 tool(s) from 'search'
Loaded 2 tool(s) from 'weather'
Total tools available: ['add', 'subtract', ..., 'search_web', 'get_current_weather', ...]

============================================================
Query: What is the weather in Lahore and who is the current Prime Minister of Pakistan?...
============================================================
  [react_node] Action → [get_current_weather]  args: {'city': 'Lahore'}
  [react_node] Action → [search_web]  args: {'query': 'current Prime Minister of Pakistan'}
  [react_node] Action → [search_web]  args: {'query': 'age of current Prime Minister of Pakistan'}
  [tool_node]  Executing [get_current_weather] …
  [tool_node]  Executing [search_web] …
  [tool_node]  Executing [search_web] …
  [react_node] Final Answer: ...

============================================================
FINAL ANSWER:
============================================================
The current Prime Minister of Pakistan is Shehbaz Sharif (age 74).
Weather in Lahore: [conditions]. Given his age, extreme heat or humidity
may affect his health — moderate weather is advisable.

Steps taken: 3
```

### Notes
- Math and search servers are launched **automatically** as subprocesses by `main.py`
- Only the weather server needs to be started manually
- API keys are read from environment variables — never hardcoded
