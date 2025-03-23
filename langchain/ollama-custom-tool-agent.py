from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Create the agent
memory = MemorySaver()
chat = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
    temperature = 0.8,
    num_predict = 256,
)

@tool
def get_webpage_text(webpage_url: str) -> str:
    """grabs the text from a webpage using puppeteer."""
    import asyncio
    from playwright.async_api import async_playwright

    async def get_text():
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                executable_path="/usr/bin/chromium-browser",
                headless=True
            )
            page = await browser.new_page()
            
            # Navigate to the URL
            await page.goto(webpage_url, wait_until="networkidle")
            
            # Get text content of main element
            main_content = await page.locator("main").text_content()
            
            # If no main element found, get body text
            if not main_content:
                main_content = await page.locator("body").text_content()
            
            await browser.close()
            return main_content

    # Run the async function
    return asyncio.run(get_text())

tools = [get_webpage_text]
print(tools)
agent_executor = create_react_agent(chat, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="go to https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/ and tell me the 3 main components of a custom tool. if the information is not available, tell me the information was not found on the web page. ")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()