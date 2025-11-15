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
    tool_executor = ToolNode(tools)
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
    global tools, chat, agent
    client = MultiServerMCPClient(
        {
            "chrome-devtools": {
                "transport": "stdio",  # Local subprocess communication
                "command": "npx",
                # Absolute path to your math_server.py file
                "args": [
                    "chrome-devtools-mcp@latest",
                    "--headless=false"
                    "--isolated=true"
                ]
            }
        }
    )
    tools = await client.get_tools()
    # desired_tool_names = [
    #     "new_page",
    #     "navigate_page"
    #     "take_snapshot"
    #     "click",
    #     "fill_form",
    #     "fill",
    #     "wait_for"
    # ]

    # tools = [
    #     tool for tool in tools
    #     if tool.name in desired_tool_names
    # ]

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
        system_prompt="your are an expert at using tools to browse the web. Keep the browser tab open."
    )

    # # Stream the agent steps as they happen
    # print("Streaming agent steps:")
    # async for step in agent.astream({
    #     "messages": [HumanMessage(
    #         content="use your tools to 1. visit https://promo.united.com/offers/packmoremiles. 2. fill out the form using xkc61520 as the mileageplus number and any number for last 4 of credit card. 3. submit the form. 4. wait for the website to finish loading. 5. verify the form submission by reading the text on the page.")]
    # }):
    #     print(step)

    #     print('\n---------------------------------\n')
    workflow = StateGraph(AgentState)

    # 2. Add Nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tool", call_tool)

    # 3. Set Entry Point
    workflow.set_entry_point("agent")

    # 4. Define Edges (The Flow)

    # From Agent node, use the conditional router to decide the next step
    workflow.add_conditional_edges(
        "agent",  # Source node
        should_continue,  # Router function
        {
            "continue": "tool",  # If tool call exists, go to tool execution
            "end": END           # If final answer, end the graph
        }
    )

    # From Tool node, ALWAYS go back to the Agent for the next decision/thought
    workflow.add_edge('tool', 'agent')

    # 5. Compile the app
    app = workflow.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "1"}}
    # --- Invocation ---
    print("Streaming agent steps:")
    async for step in app.astream(
        {"messages": [HumanMessage(
            content="use your tools to visit https://promo.united.com/offers/packmoremiles. Once there fill out the form with the account xkc1520 and the card number 1234. hit submit. Then tell me what the page says after submission.")]
         }, config=config):
        print(step)
        print('\n---------------------------------\n')

if __name__ == "__main__":
    asyncio.run(main())
