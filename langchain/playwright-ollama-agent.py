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
    temperature = 0.8,
    num_predict = 256,
)

#add playwright
async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
playwright_tools = toolkit.get_tools()

print(playwright_tools)
agent_executor = create_react_agent(chat, playwright_tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="you are a internet rag app. I will give you urls, you will visit them, grab the text, and use it as context to answer my questions. visit https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.url_playwright.PlaywrightURLLoader.html and tell me what type of the headless paramater is for the class constructor of PlaywrightURLLoader.")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()