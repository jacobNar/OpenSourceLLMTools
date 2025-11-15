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
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import AnyMessage
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.tools import load_mcp_tools
# The 'chat' object is your ChatOllama instance bound with tools


class AgentState(TypedDict):
    # A list of messages (HumanMessage, AIMessage, ToolMessage, etc.)
    # The 'add_messages' operator appends new messages to the list
    messages: Annotated[List[AnyMessage], add_messages]


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    # Get the last message to continue the conversation
    result = chat.invoke(state['messages'])
    print("result from model")
    print(result)
    return {"messages": [result]}


async def call_tool(state: AgentState, config: RunnableConfig) -> dict:
    ai_message = state['messages'][-1]
    tool_calls = ai_message.tool_calls

    single_tool_call = [tool_calls[0]]
    tool_messages = await tool_executor.ainvoke(single_tool_call, config=config)
    print("result from tool")
    print(tool_messages)
    return tool_messages


def should_continue(state: AgentState) -> str:
    # Check the last message from the LLM
    last_message = state['messages'][-1]

    # If the LLM returned a tool_call, go to the tool node
    if last_message.tool_calls:
        # Note: We must check for tool_calls because we are forcing single iteration
        return "continue"
    else:
        # Otherwise, the LLM returned a final answer, so we end
        return "end"


async def main():
    global tools, chat, agent, tool_executor
    client = MultiServerMCPClient(
        {
            "chrome-devtools": {
                "transport": "stdio",  # Local subprocess communication
                "command": "npx",
                # Absolute path to your math_server.py file
                "args": [
                    "chrome-devtools-mcp@latest",
                    "--headless=false"
                ]
            }
        }
    )
    async with client.session("chrome-devtools") as session:
        tools = await load_mcp_tools(session)
        tool_executor = ToolNode(tools)
        # print(tools)
        print(len(tools), "tools loaded.")

        chat = ChatOllama(
            base_url="http://localhost:11434/",
            model="qwen3:latest",
            temperature=0.8,
            num_predict=40000,  # Increased for 128k context window
        ).bind_tools(tools)

        agent = create_agent(
            chat,
            tools,
            system_prompt="your are an expert at using tools to browse the web. after each tool call replan and make sure that the users requests are fully satisfied.",
        )
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tool", call_tool)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tool",
                "end": END
            }
        )

        workflow.add_edge('tool', 'agent')

        app = workflow.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "3"}}

        print("Streaming agent steps:")
        async for step in app.astream(
            {"messages": [HumanMessage(
                content="use your tools to visit https://promo.united.com/offers/packmoremiles. Once there fill out the form with the account xkc1520 and the card number 1234 and submit it.")]
             }, config=config):
            print(step)
            print('\n---------------------------------\n')

if __name__ == "__main__":
    asyncio.run(main())
