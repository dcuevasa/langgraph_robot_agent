from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START
from langchain_openai import AzureChatOpenAI
from typing import Union
from langgraph.prebuilt import create_react_agent
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from typing import Literal
from langgraph.graph import END
from pydantic import BaseModel, Field
from langgraph.store.memory import InMemoryStore
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import uuid
from utils import format_few_shot_examples, format_few_shot_examples_solutions
import asyncio

from tools import find_object, count_objects, search_for_specific_person, find_item_with_characteristic, get_person_gesture, get_all_items, speak, listen, question_and_answer, go_to_location, go_back, follow_person, stop_robot, ask_for_object, give_object
from prompts import agent_system_prompt_memory, evaluator_system_prompt, evaluator_user_prompt

_ = load_dotenv()

embed_model = AzureOpenAIEmbeddings(api_key=os.getenv("AZURE_OPENAI_API_KEY"),azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"))

store = InMemoryStore(
    index={"embed": embed_model}
)

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
    find_object("person")
    if "yes" in question_and_answer("Are you Richard?"):
        speak("I Found Richard!")
    else:
        speak("I Could Not Find Richard!")
    """
})


for example in solution_examples:
    store.put(
        ("task_assistant", "david", "examples"), 
        str(uuid.uuid4()), 
        example
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
    give_object
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

# Choose the LLM that will drive the agent
llm = AzureChatOpenAI(model="azure_openai:"+os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"))
prompt = create_prompt
agent_executor = create_react_agent(llm,tools=tools, prompt=prompt)



class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )




from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct solution. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | AzureChatOpenAI(
    model="azure_openai:"+os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"), temperature=0
).with_structured_output(Plan)



class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | AzureChatOpenAI(
    model="azure_openai:"+os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"), temperature=0
).with_structured_output(Act)





async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"






workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(store=store)

graph_png = app.get_graph(xray=True).draw_mermaid_png()
with open('graph_cot.png', 'wb') as f:
    f.write(graph_png)


"""

config = {"configurable": {"langgraph_user_id": "david"}}
    
while True:
    print("Waiting for task input...")
    task_input = input("Enter a task: ")
    if task_input.lower() == "exit":
        break

    task_input = {
        "input": task_input,
    }

    response = app.invoke(
        {"task_input": task_input},
        config=config
    )

    for m in response["messages"]:
        m.pretty_print()
"""



async def main():
    config = {"recursion_limit": 10}
    inputs = {"input": "go to the kitchen and bring me a bottle"}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":
    asyncio.run(main())