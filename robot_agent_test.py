#!/usr/bin/env python3.11 
# -*- coding: utf-8 -*-

# LangChain imports:
from langchain.embeddings.base import Embeddings
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings

# LangGraph imports:
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.graph import add_messages
from langgraph.types import Command

# LangMem imports:
from langmem import create_manage_memory_tool, create_search_memory_tool
from langmem import create_multi_prompt_optimizer

# Various imports:
from typing import Literal
from typing_extensions import TypedDict, Literal, Annotated
from IPython.display import Image, display
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import os


# Local imports:
from tools import find_object, count_objects, search_for_specific_person, find_item_with_characteristic, get_person_gesture, get_all_items, speak, listen, question_and_answer, answer_question, go_to_location, go_back, follow_person, stop_robot, ask_for_object, give_object
from prompts import agent_system_prompt_memory, evaluator_system_prompt, evaluator_user_prompt
from utils import format_few_shot_examples

_ = load_dotenv()

example_task = "Grab a bottle, and bring it to the living room"
# Local embeddings
"""
# Local embeddings
from sentence_transformers import SentenceTransformer
from typing import List
class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
    
embed_model = LocalEmbeddings(model_name="all-MiniLM-L6-v2")

"""

embed_model = AzureOpenAIEmbeddings(api_key=os.getenv("AZURE_OPENAI_API_KEY"),azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

store = InMemoryStore(
    index={"embed": embed_model}
)


llm = AzureChatOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"),azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))


class Router(BaseModel):
    """Analyze the given task and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["feasible", "unfeasible"] = Field(
        description="The classification of a given task: 'feasible' for tasks the robot can perform, "
        "'unfeasible' for tasks the robot cannot perform."
    )
    

llm_router = llm.with_structured_output(Router)

class State(TypedDict):
    task_input: dict
    messages: Annotated[list, add_messages]
    

def evaluator_router(state: State, config, store) -> Command[
    Literal["task_agent", "__end__"]
]:
    task = state['task_input']['task']

    namespace = (
        "robot_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(
        namespace, 
        query=str({"task": state['task_input']})
    ) 
    examples=format_few_shot_examples(examples)

    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    
    system_prompt = evaluator_system_prompt.format(
        examples=examples
    )
    user_prompt = evaluator_user_prompt.format(
        task=task, 
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "feasible":
        print("✅ Classification: FEASIBLE - This task is feasible")
        goto = "task_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Perform the task {state['task_input']}",
                }
            ]
        }
    elif result.classification == "unfeasible":
        print("🚫 Classification: UNFEASIBLE - This task is unfeasible")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "task_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "task_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)

tools = [
    find_object, 
    count_objects,
    search_for_specific_person,
    find_item_with_characteristic,
    get_person_gesture,
    get_all_items,
    speak,
    listen,
    question_and_answer,
    answer_question,
    go_to_location,
    go_back,
    follow_person,
    stop_robot,
    ask_for_object,
    give_object
]

prompt_instructions = {
    "task_instructions": "Use these tools when appropriate to fulfill the given task."
}

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["task_instructions"]
            )
        }
    ] + state['messages']

task_agent = create_react_agent(
    "azure_openai:"+os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    tools=tools,
    prompt=create_prompt,
    # Use this to ensure the store is passed to the agent 
    store=store
)


robot_agent = StateGraph(State)
robot_agent = robot_agent.add_node(evaluator_router)
robot_agent = robot_agent.add_node("task_agent", task_agent)
robot_agent = robot_agent.add_edge(START, "evaluator_router")
robot_agent = robot_agent.compile(store=store)

# Save the image to a file instead of displaying it
graph_png = task_agent.get_graph(xray=True).draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(graph_png)
    

task_input = {
    "task": "Go to the kitchen and find a bottle",
}

config = {"configurable": {"langgraph_user_id": "lance"}}

print("Task input:", task_input)
response = robot_agent.invoke(
    {"task_input": task_input},
    config=config
)

for m in response["messages"]:
    m.pretty_print()



