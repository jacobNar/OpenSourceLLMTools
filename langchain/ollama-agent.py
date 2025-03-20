from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,
)
import os

from dotenv import load_dotenv
# Load environment variables from .env file for Tavily API
load_dotenv()

# Create the agent
memory = MemorySaver()
chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
    temperature = 0.8,
    num_predict = 256,
)

search = TavilySearchResults(max_results=5)
tools = [search]
print(tools)
agent_executor = create_react_agent(chat, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="go to ticketmaster.com, in the search bar type luke combs, then hit enter. Tell me the first 3 events you see and give me links.")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()