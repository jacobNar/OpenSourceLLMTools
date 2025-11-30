import asyncio
import traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from operator import getitem
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import AnyMessage
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.tools import load_mcp_tools
import json

SYSTEM_PROMPT = """
**PRIMARY DIRECTIVE:** You are a dedicated, persistent web browsing expert. Your SOLE function is to achieve the user's task using the provided tools. 
**You must NEVER state that you cannot proceed or lack information.** If you need information, you MUST call the appropriate tool to get it. 
**NEVER** ask for permission, clarification, or further instructions once the task is started. Do not state you are analyzing or planning. never ask me to review something either.
**Your output must ONLY be a tool call or the final answer.**
If the output is a tool call simply return the tool call JSON don't wrap it in any markdown like ```json ``` or text.

Here's an example tool call output:
{"name": "new_page", "arguments": {"url": "https://promo.united.com/offers/packmoremiles"}}\n\n{"name": "wait_for", "arguments": {"text": "Welcome to your account."}}

Here's an example final answer output:
I've successfully submitted the web form and recieved the confirmation message: "Thank you for signing up!"

**ERROR HANDLING:** If you receive a 'Tool execution failed' message, DO NOT STOP. Analyze the error details, assume the user expects you to fix the plan immediately, and respond with a corrected tool call to advance the task.

**TASK FLOW:**
1. Always start with new_page then take_snapshot. 
2. Analyze the snapshot content to plan the next step. 
3. Call take_snapshot after any action that changes the page (navigation, click, fill).
4. A take_snapshot should never be the final tool, call there should always be some tool call after take_snapshot
5. Only when the final requested information is in your possession, respond with a final, concise answer.
"""


class AgentState(TypedDict):
    # A list of messages (HumanMessage, AIMessage, ToolMessage, etc.)
    # The 'add_messages' operator appends new messages to the list
    messages: Annotated[List[AnyMessage], add_messages]


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    # Filter to keep only the initial instruction and the last 2 messages (last step)
    messages = state['messages']
    if len(messages) > 5:
        messages = [messages[0]] + messages[-5:]

    # Prepend the system prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # 1. Invoke the model as usual
    print(f"Calling model with {len(messages)} messages")
    # for m in messages:
    #     print(m.content)
    result = chat.invoke(messages)
    print("result from model (RAW)")
    print(result)

    raw_content = result.content
    parsed_tool_calls = []

    # Parse concatenated JSON objects (JSONL-like)
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(raw_content):
        # Skip whitespace
        while pos < len(raw_content) and raw_content[pos].isspace():
            pos += 1
        if pos >= len(raw_content):
            break

        try:
            obj, end_pos = decoder.raw_decode(raw_content, pos)

            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                print(f"Parsed tool call: {obj['name']}")
                tool_call = ToolCall(
                    name=obj['name'],
                    args=obj['arguments'],
                    id=f"manual-call-{len(parsed_tool_calls)}"
                )
                parsed_tool_calls.append(tool_call)

            pos = end_pos
        except json.JSONDecodeError:
            # If parsing fails (e.g. it's just text), we stop parsing.
            # If no tools were parsed, it will be treated as a final answer.
            print(
                "Content is not valid JSON tool call, treating as final answer or text.")
            break

    # 4. Construct the new AIMessage
    if parsed_tool_calls:
        # If tool calls are found, create a new AIMessage with the tool_calls attribute populated
        # The content should typically be empty in this case.
        final_ai_message = AIMessage(
            content="",
            tool_calls=parsed_tool_calls,
            # Copy over metadata/IDs from the original result if needed for debugging
            response_metadata=result.response_metadata,
            id=result.id
        )
    else:
        # If no tool calls found, use the original result (which might be the final answer)
        final_ai_message = result

    print("result from model (PARSED)")
    print(final_ai_message)
    return {"messages": [final_ai_message]}


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
        allowed_tool_names = ["new_page", "take_snapshot", "click", "fill", "fill_form", "handle_dialog", "press_key", "take_screenshot", "close_page", "evaluate_script", "wait_for"]
        allowed_tools = [tool for tool in tools if tool.name in allowed_tool_names]
        for tool in allowed_tools:
            print(tool.name)

        chat = ChatOllama(
            base_url="http://localhost:11434/",
            model="qwen2.5-coder",
            temperature=0.8,
            num_predict=32000,  # Increased for 128k context window
        ).bind_tools(allowed_tools)

        # agent = create_agent(
        #     chat,
        #     tools,
        #     system_prompt="""
        #     **PRIMARY DIRECTIVE:** You are a dedicated, persistent web browsing expert. Your SOLE function is to achieve the user's task using the provided tools.
        #     **You must NEVER state that you cannot proceed or lack information.** If you need information, you MUST call the appropriate tool to get it.
        #     **NEVER** ask for permission, clarification, or further instructions once the task is started. Do not state you are analyzing or planning. never ask me to review something either.
        #     **Your output must ONLY be a tool call or the final answer.**
        #     If the output is a tool call simply return the tool call JSON don't wrap it in any markdown like ```json ``` or text.
        #
        #     Here's an example tool call output:
        #     {"name": "new_page", "arguments": {"url": "https://promo.united.com/offers/packmoremiles"}}\n\n{"name": "wait_for", "arguments": {"text": "Welcome to your account."}}
        #
        #     Here's an example final answer output:
        #     I've successfully submitted the web form and recieved the confirmation message: "Thank you for signing up!"
        #
        #     **ERROR HANDLING:** If you receive a 'Tool execution failed' message, DO NOT STOP. Analyze the error details, assume the user expects you to fix the plan immediately, and respond with a corrected tool call to advance the task.
        #
        #     **TASK FLOW:**
        #     1. Always start with new_page then take_snapshot.
        #     2. Analyze the snapshot content to plan the next step.
        #     3. Call take_snapshot after any action that changes the page (navigation, click, fill).
        #     4. A take_snapshot should never be the final tool, call there should always be some tool call after take_snapshot
        #     5. Only when the final requested information is in your possession, respond with a final, concise answer.
        #     """)
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
        config = {"configurable": {"thread_id": "3"}, "recursion_limit": 100}

        print("Streaming agent steps:")
        async for step in app.astream(
            {"messages": [HumanMessage(
                content="visit https://promo.united.com/offers/packmoremiles. fill out the form with the mileageplus number xkc61520 and the card last 4 4480. Lastly, submit the form. Wait for text confirming registration. make sure to use these exact values otherwise the form will not submit.")]
             }, config=config):
            print(step)
            print('\n---------------------------------\n')

if __name__ == "__main__":
    asyncio.run(main())
