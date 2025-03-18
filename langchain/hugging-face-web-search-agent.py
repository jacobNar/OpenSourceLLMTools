# Import relevant functionality 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import torch
import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# from huggingface_hub import login
# login() 

from dotenv import load_dotenv
# Load environment variables from .env file for Tavily API
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Create the agent
memory = MemorySaver()
# llm = HuggingFacePipeline.from_model_id(
#     model_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     pipeline_kwargs=dict(
#         max_new_tokens=1024,
#         do_sample=False,
#         repetition_penalty=1.03
#     ),
# )

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model = ChatHuggingFace(llm=llm, verbose=True)
search = TavilySearchResults(max_results=5)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="i live in northern illinois. Can you tell me the latest news in my area?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()