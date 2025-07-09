import os
import json
import ast
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv



load_dotenv()
llm_api_key = os.getenv("ANTHROPIC_API_KEY")
api_max_tokens = 64000                      # Maximum ammount

""" PATHS VARIABLES """
path_to_initPrompt_1 = r"..\Prompts\gen_test_cases\initial_prompt_1.txt"
path_to_initPrompt_2 = r"..\Prompts\gen_test_cases\initial_prompt_2.txt"
path_to_reflectionPrompt = r"..\Prompts\gen_test_cases\reflection_prompt.txt"
path_to_finalPrompt = r"..\Prompts\gen_test_cases\final_prompt.txt"



class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature = 0.3,
    max_tokens = api_max_tokens,
    api_key = llm_api_key
)

def generate_test_cases_from_requirements(list_of_requirements: list, context: str) -> list:
    """
    Params: 
    dict: list of requirements
    context: the project background of these requirements

    Output:
    dict: list of test cases

    """


    """
    1. Get the prompts
        Prompt CONTENT are written to satisfy Athropic model.
    """
    with open(path_to_initPrompt_1,"r") as f:
        initPrompt_1 = f.read()

    with open(path_to_initPrompt_2,"r") as f:
        initPrompt_2 = f.read()
        
        
    content_prompt_initial = [
                    {
                        "type": "text",
                        "text": initPrompt_1
                    },
                    {
                        "type": "text",
                        "text": str(list_of_requirements)
                    },
                    {
                        "type": "text",
                        "text": context
                    },
                    {
                        "type": "text",
                        "text": initPrompt_2
                    }
                ]

    with open(path_to_reflectionPrompt,"r") as f:
        reflectionPrompt = f.read()

        content_prompt_reflection = [
                    {
                        "type": "text",
                        "text": reflectionPrompt
                    }
                ]
        
    with open(path_to_finalPrompt,"r") as f:
        finalPrompt = f.read()
    content_prompt_final = [
                    {
                        "type": "text",
                        "text": finalPrompt
                    }
                ]


    """
    2. Define writing Steps
    """
    def write_initial_draft(state: AgentState):
        model_response = llm.invoke([HumanMessage(content=content_prompt_initial)])
        
        return {"messages": [HumanMessage(content=content_prompt_initial), model_response]}
        
    def write_reflection(state: AgentState):
        prompt_messages = state['messages'] + [HumanMessage(content=content_prompt_reflection)] 
        model_response = llm.invoke(prompt_messages)

        return{"messages": [HumanMessage(content=content_prompt_reflection), model_response]}

    def write_final(state: AgentState):
        prompt_messages = state['messages'] + [HumanMessage(content=content_prompt_final)] 
        ## The AIMessage [ is to tell Claude to only create list
        model_response = llm.invoke(prompt_messages)

        return{"messages":[HumanMessage(content=content_prompt_final), model_response]}

    """
    3. Build graph
    """
    builder = StateGraph(AgentState)
    builder.add_node(write_initial_draft, "write_initial_draft")
    builder.add_node(write_reflection, "write_reflection")
    builder.add_node(write_final, "write_final")

    builder.add_edge(START, "write_initial_draft")
    builder.add_edge("write_initial_draft","write_reflection")
    builder.add_edge("write_reflection","write_final")
    builder.add_edge("write_final",END)

    graph = builder.compile()


    """
    4. Invoke
    """
    langchainConfig_messages = graph.invoke({"messages": []})


    """
    5. Output Parsing
    """
    output_testCase_string = langchainConfig_messages['messages'][-1].content
    print(output_testCase_string)

    output_testCase_list = []
    try:
        output_testCase_list = json.loads(output_testCase_string)
    except json.JSONDecodeError as e:
        print(f"AI failed to generate appropriate JSON error: {e}")
        print(repr(output_testCase_string))
        output_testCase_list = None


    return output_testCase_list






