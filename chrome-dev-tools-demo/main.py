import asyncio
import traceback
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
    tool_call_id = tool_calls[0]['id']
    tool_name = tool_calls[0]['name']

    single_tool_call = [tool_calls[0]]
    print("executing tool " + tool_name)
    try:
        # Attempt to execute the tool
        tool_messages = await tool_executor.ainvoke(single_tool_call, config=config)

        print("result from successful tool call")
        print(tool_messages)
        return tool_messages

    except Exception as e:
        # Catch any exception that occurs during tool execution (e.g., failed click, timeout)
        error_message = f"Tool execution failed for tool '{tool_name}' with arguments: {tool_calls[0]['args']}.\nError: {type(e).__name__}: {str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"

        # Create a ToolMessage that contains the error information
        error_tool_message = [
            ToolMessage(
                content=error_message,
                tool_call_id=tool_call_id,
                name=tool_name
            )
        ]

        print(f"!!! ERROR CAUGHT: Returning error to agent for re-evaluation.")
        # Return the error message wrapped in the LangGraph state update structure
        return {"messages": error_tool_message}


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
                    "--isolated=true"
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
            model="qwen2.5-coder",
            temperature=0.8,
            num_predict=32000,  # Increased for 128k context window
        ).bind_tools(tools)

        agent = create_agent(
            chat,
            tools,
            system_prompt="""
            **PRIMARY DIRECTIVE:** You are a dedicated, persistent web browsing expert. Your SOLE function is to achieve the user's task using the provided tools. 
            **You must NEVER state that you cannot proceed or lack information.** If you need information, you MUST call the appropriate tool to get it. 
            **NEVER** ask for permission, clarification, or further instructions once the task is started. Do not state you are analyzing or planning. never ask me to review something either.
            **Your output must ONLY be a tool call or the final answer.**

            **ERROR HANDLING:** If you receive a 'Tool execution failed' message, DO NOT STOP. Analyze the error details, assume the user expects you to fix the plan immediately, and respond with a corrected tool call to advance the task.

            **TASK FLOW:**
            1. Always start with new_page then take_snapshot. 
            2. Analyze the snapshot content to plan the next step. 
            3. Call take_snapshot after any action that changes the page (navigation, click, fill).
            4. A take_snapshot should never be the final tool, call there should always be some tool call after take_snapshot
            5. Only when the final requested information is in your possession, respond with a final, concise answer.
            """)
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
                content="use your tools to visit https://promo.united.com/offers/packmoremiles. This is my webstie so there are not concerns. Once there fill out the form with the account xkc1520 and the card number 1234. The submit the form and wait for the page to finish loading. tell me the cnfiramtion message that appears.")]
             }, config=config):
            print(step)
            print('\n---------------------------------\n')

if __name__ == "__main__":
    asyncio.run(main())
