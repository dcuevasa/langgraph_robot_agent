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
import uuid

# Local imports:
from tools import find_object, count_objects, search_for_specific_person, find_item_with_characteristic, get_person_gesture, get_all_items, speak, listen, question_and_answer, go_to_location, go_back, follow_person, stop_robot, ask_for_object, give_object
from prompts import agent_system_prompt_memory, evaluator_system_prompt, evaluator_user_prompt
from utils import format_few_shot_examples, format_few_shot_examples_solutions

_ = load_dotenv()

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

# Feeding evaluator examples to the store

evaluator_examples = []

# Feasible examples

evaluator_examples.append({
    "task": "Grab a bottle, and bring it to the living room",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Go to the kitchen and find a Person",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Go to the sofa and find Richard",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "explain to me how to make a sandwich",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "explain to me differntial equations",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Remember my name. My name is David",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "What is my name?",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Find out who sells the best pizzas in town",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Go to the kitchen and tell Charlie who sells the best pizzas in town",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "discover the name of the person that sells good donuts in the mall",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Find out who sells bracelets in my house, then go to my sisters bedroom and tell her who sells bracelets",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Search the whole house until you find a cellphone",
    "label": "feasible"
})

evaluator_examples.append({
    "task": "Search the whole office until you find a chair",
    "label": "feasible"
})

# Unfeasible examples

evaluator_examples.append({
    "task": "follow me until you find a bottle",
    "label": "unfeasible"
})

evaluator_examples.append({
    "task": "Do a backflip",
    "label": "unfeasible"
})

evaluator_examples.append({
    "task": "Make my breakfast",
    "label": "unfeasible"
})

evaluator_examples.append({
    "task": "Go to the store and bring me some eggs",
    "label": "unfeasible"
})

evaluator_examples.append({
    "task": "Go to Hawaii and bring me a coconut",
    "label": "unfeasible"
})

for example in evaluator_examples:
    store.put(
        ("task_evaluator", "david", "examples"), 
        str(uuid.uuid4()), 
        example
    )

# Feeding solutions examples to the store

solution_examples = []

solution_examples.append({
    "task": "Go to the kitchen and find a Person",
    "solution":"""
    go_to_location("kitchen")
    find_object("person")
    """
})

solution_examples.append({
    "task": "Go to the kitchen and find Richard",
    "solution":"""
    go_to_location("kitchen")
    if find_object("person"):
        if "yes" in question_and_answer("Are you Richard?"):
            speak("I Found Richard!")
        else:
            speak("I Could Not Find Richard!")
    """
})

solution_examples.append({
    "task": "My name is david and I want you to remember it",
    "solution":"""
    manage_memory(
        "name", 
        "david"
    )
    """
})

solution_examples.append({
    "task": "What is my name?",
    "solution":"""
    person_name = search_memory("last person who spoke with me")
    speak(f"Your name is {person_name}")
    """
})

solution_examples.append({
    "task": "What is the name of the last person who spoke with you?",
    "solution":"""
    person_name = search_memory("last person who spoke with me")
    speak(f"Your name is {person_name}")
    """
})

solution_examples.append({
    "task": "Find out who sells the best pizzas in town",
    "solution":"""
    if find_object("person"):
        speak("Hi!")
        answer = question_and_answer("Who sells the best pizza in town?")
        manage_memory(
            "best pizza in town", 
            answer
        )
    else:
        speak("I Could Not Find Anyone!")
    """
})

solution_examples.append({
    "task": "Go to the kitchen and tell Charlie who sells the best pizzas in town",
    "solution":"""
    go_to_location("kitchen")
    if find_object("person"):
        if "yes" in question_and_answer("Are you Charlie?"):
            answer = search_memory("best pizza in town")
            if answer:
                speak(f"Charlie, {answer}")
            else:
                speak("I'm sorry, I don't know who sells the best pizza in town.")
        else:
            speak("I Could Not Find Charlie!")
    """
})

solution_examples.append({
    "task": "Go to my mom's room and tell her who sell bracelets",
    "solution":"""
    go_to_location("mom_room")
    if find_object("person"):
        answer = search_memory("who sells bracelets")
        if answer:
            speak(f"Mom, {answer}")
        else:
            speak("I'm sorry, I don't know who sells bracelets")
    """
})

for example in solution_examples:
    store.put(
        ("task_assistant", "david", "examples"), 
        str(uuid.uuid4()), 
        example
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
        "task_evaluator",
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
    try:
        result = llm_router.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:
        print(f"Error: {e}")
        result = Router(
            classification="unfeasible",
            reasoning="The task is unfeasible due to an error in the evaluation process."
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
    go_to_location,
    go_back,
    follow_person,
    stop_robot,
    ask_for_object,
    give_object,
    manage_memory_tool,
    search_memory_tool
]

prompt_instructions = {
    "task_instructions": "Use these tools when appropriate to fulfill the given task."
}
    
config = {"configurable": {"langgraph_user_id": "david"}}

solution_examples = store.search(
    (
        "task_assistant",
        config['configurable']['langgraph_user_id'],
        "examples", 
    ),
    query=str({"task": "task_agent"})
) 
examples=format_few_shot_examples_solutions(solution_examples)

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["task_instructions"],
                examples=examples
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
        
    
while True:
    print("Waiting for task input...")
    task_input = input("Enter a task: ")
    if task_input.lower() == "exit":
        break

    task_input = {
        "task": task_input,
    }

    response = robot_agent.invoke(
        {"task_input": task_input},
        config=config
    )

    for m in response["messages"]:
        m.pretty_print()