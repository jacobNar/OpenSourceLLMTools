from langchain_ollama import ChatOllama
from browser_use import Agent, BrowserConfig, Browser
from dotenv import load_dotenv
load_dotenv()
import os
import asyncio

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

config = BrowserConfig(
    chrome_instance_path="C:/Program Files/Google/Chrome/Application/chrome.exe"
)

browser = Browser(config=config)

llm = ChatOllama(
    base_url = "http://localhost:11434/",
    model = "llama3.2",
)

async def main():
    agent = Agent(
        browser=browser,
        task="open the browser, immediately open a new tab, and go to https://amazon.com. Once there and the page is loaded search for 2tb nvme ssd hard drives and find me the cheapest one.",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())