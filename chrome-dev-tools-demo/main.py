import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from operator import getitem
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


async def main():
    client = MultiServerMCPClient(
        {
            "chrome-devtools": {
                "transport": "stdio",  # Local subprocess communication
                "command": "npx",
                # Absolute path to your math_server.py file
                "args": [
                    "chrome-devtools-mcp@latest",
                    "--headless=false",
                ]
            }
        }
    )
    tools = await client.get_tools()
    desired_tool_names = [
        "new_page",
        "navigate_page"
        "take_snapshot"
        "click",
        "fill_form",
        "fill",
        "wait_for"
    ]

    tools = [
        tool for tool in tools
        if tool.name in desired_tool_names
    ]

    print(tools)

    chat = ChatOllama(
        base_url="http://localhost:11434/",
        model="qwen3:latest",
        temperature=0.8,
        num_predict=40000,  # Increased for 128k context window
    ).bind_tools(tools)

    agent = create_agent(
        chat,
        tools,
        system_prompt="your are an expert at using tools to browse the web. Use the tools effictively to achieve the user's goals. When browsing always use new page to start. Always read the page before trying to click or fill anything as you need to know the element uids."
    )

    # Stream the agent steps as they happen
    print("Streaming agent steps:")
    async for step in agent.astream({
        "messages": [HumanMessage(
            content="use your tools to 1. visit https://promo.united.com/offers/packmoremiles. 2. fill out the form using xkc61520 as the mileageplus number and any number for last 4 of credit card. 3. submit the form. 4. wait for the website to finish loading. 5. verify the form submission by reading the text on the page.")]
    }):
        print(step)

        print('\n---------------------------------\n')

if __name__ == "__main__":
    asyncio.run(main())
