from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import asyncio
# from playwright.async_api import async_playwright, Browser, Page

# class PlaywrightManager:
#     _instance = None
    
#     def __init__(self):
#         self.playwright = None
#         self.browser = None
    
#     @classmethod
#     async def get_instance(cls):
#         if cls._instance is None:
#             cls._instance = cls()
#             cls._instance.playwright = await async_playwright().start()
#             cls._instance.browser = await cls._instance.playwright.chromium.launch(
#                 executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
#                 headless=False
#             )
#             return cls._instance
#         else:
#             return cls._instance

#     async def get_page(self) -> Page:
#         return await self.browser.new_page()

@tool
def get_webpage_text(webpage_url: str) -> str:
    """grabs the text from a webpage using puppeteer."""
    import asyncio
    from playwright.async_api import async_playwright

    async def get_text():
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
                headless=False
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



# Use the agent


async def stream_message(agent_executor, inputs):
    chunks = []
    config = {"configurable": {"thread_id": "abc123"}}
    async for chunk in agent_executor.astream(
        {"messages": [HumanMessage(content=inputs)]}
        , config
    ):
        chunks.append(chunk)
        print("------")
        print(chunk)


async def main ():
    # Create the agent
    memory = MemorySaver()
    chat = ChatOllama(
        base_url = "http://localhost:11434/",
        model = "llama3.2",
        temperature = 0.8,
        num_predict = 1024,
    )
    tools = [get_webpage_text]
    print(tools)
    agent_executor = create_react_agent(chat, tools, checkpointer=memory)
    try:
        inputs = "go to https://python.langchain.com/docs/how_to/custom_tools/. Decide the best way to build a suite of custom tools for playwright. the toolkit should use the same chrome instance for one task even across multiple tool calls."
        task = asyncio.create_task(
            stream_message(agent_executor, inputs)
        )
        await asyncio.wait_for(task, timeout=60)
    except asyncio.TimeoutError:
        print("Task Cancelled.")
    # finally:
    #     # Clean up playwright resources
    #     manager = await PlaywrightManager.get_instance()
    #     await manager.browser.close()
    #     await manager.playwright.stop()

if __name__ == "__main__":
    asyncio.run(main())