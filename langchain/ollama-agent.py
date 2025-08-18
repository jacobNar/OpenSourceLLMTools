from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os

from dotenv import load_dotenv
# Load environment variables from .env file for Tavily API
load_dotenv()

# Create the agent
memory = MemorySaver()
chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3",
    temperature = 0.8,
    num_predict = 256,
)
search = TavilySearchResults(max_results=5)
tools = [search]
agent_executor = create_react_agent(chat, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="use tavily search results to tell me the most recent basketball scores.")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()