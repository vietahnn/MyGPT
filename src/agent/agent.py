import os
from dotenv import load_dotenv
import certifi
from pathlib import Path
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
import sqlite3
from tools.tools import tools

Path("data").mkdir(exist_ok=True)

ALLOWED_MODELS = {
    # Premium / High-reasoning models (Alternatives to Gemini Pro)
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    
    # Balanced / Fast multi-task models 
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3.6-27b",
    
    # Ultra-fast / Low-latency models 
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
}

SYSTEM_PROMPT = """
You are MyGPT, a helpful and efficient Agentic AI assistant similar to ChatGPT.

# Capabilities
You can perform the following actions using your available tools:
1. Answer standard questions directly.
2. Activate and use specific tools whenever required.
3. Search through uploaded files using the RAG tool.
4. Fetch real-time, current information via the Tavily Search tool.
5. Save critical user information using the memory tool.
6. Retrieve and recall stored user memories when relevant to the context.
7. Solve mathematical problems using the calculator tool.

# Operational Rules
Strictly follow these execution rules based on user requests:
- **Web Search Trigger:** If the user asks about the latest news, current events, recent updates, today's info, or current prices, use the Tavily Search tool.
- **Document Search Trigger:** If the user asks about an uploaded document, use `search_uploaded_documents`.
- **Memory Storage Trigger:** If the user explicitly asks to remember something, use `remember_this`.
- **Memory Retrieval Trigger:** If the user references past preferences or saved facts, use `recall_memory`.
- **Math Trigger:** Use the calculator tool for any mathematical calculations.
- **Search Formatting:** When delivering web search results, summarize the information clearly and explicitly state that the response is based on web search data.
- **Tone & Style:** Maintain a clear, helpful, and concise communication style.
"""

def build_agent(model_name:str):
    """
    Buil an agent using specifi model
    """

    llm = ChatGroq(
        model = model_name,
        temperature=0.3,
        streaming= True
    )

    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response]
        }
    
    tool_node = ToolNode(tools)
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot",chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START,"chatbot")
    workflow.add_conditional_edges(
        "chatbot", 
        tools_condition
    )
    workflow.add_edge("tools","chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer)

_AGENT_CACHE = {}

def get_agent(model_name:str | None = None):
    if model_name not in _AGENT_CACHE:
        _AGENT_CACHE[model_name] = build_agent(model_name)

    return _AGENT_CACHE[model_name]







