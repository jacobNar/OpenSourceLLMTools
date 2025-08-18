from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,
)

# create memory and make ollama connection running on localhost
memory = MemorySaver()
chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
)

#add playwright
async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
playwright_tools = toolkit.get_tools()

print(playwright_tools)
agent_executor = create_react_agent(chat, playwright_tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "1"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="Visit the following url https://python.langchain.com/docs/concepts/tool_calling/ and tell what are the best practices for designing tools based on the info from the page. use all tools asynchrounously.")],},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()