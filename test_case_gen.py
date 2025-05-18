import os
import json
import base64
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv



load_dotenv()
llm_api_key = os.getenv("ANTHROPIC_API_KEY")
api_max_tokens = 300

""" PATHS VARIABLES """
path_to_initPrompt_1 = r"Prompts\initial_prompt_1.txt"
path_to_pdf = "ReqView-Example_Software_Requirements_Specification_SRS_Document.pdf"
path_to_initPrompt_2 = r"Prompts\initial_prompt_2.txt"
path_to_reflectionPrompt = r"Prompts\reflection_prompt.txt"
path_to_finalPrompt = r"Prompts\final_prompt.txt"
path_to_output = "output_test_matrix.json"




class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature = 0.3,
    max_tokens = api_max_tokens,
    api_key = llm_api_key
)

def gen_requirements_pdf_to_test_case(pdf_in_base64_encoded_string: str):
    """
    Params: 
    str: pdf encode into base64 encoded string

    Output:
    Json: test cases

    """


    """
    1. Get the prompts
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
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_in_base64_encoded_string
                        },
                        "cache_control": {"type": "ephemeral"}
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
        prompt_messages = state['messages'] + [HumanMessage(content=content_prompt_final)] +  [AIMessage(content="{")]
        ## The AIMessage { is to tell Claude to only create JSON
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
    for message in langchainConfig_messages['messages']:
        print(message)

    """
    5. Output Parsing
    """
    output_testCase_JSON_string = langchainConfig_messages['messages'][-1].content
    output_testCase_JSON = json.loads(output_testCase_JSON_string)

    def remove_newlines(obj):
        if isinstance(obj, dict):
            return {k: remove_newlines(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [remove_newlines(elem) for elem in obj]
        elif isinstance(obj, str):
            return obj.replace('\\n', ' ')
        else:
            return obj

    output_testCase_JSON = remove_newlines(output_testCase_JSON)

    with open(path_to_output, "w") as file:
        json.dump(output_testCase_JSON, file, indent=4)

    return output_testCase_JSON






with open(path_to_pdf, "rb") as f:
    pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

output = gen_requirements_pdf_to_test_case(pdf_data)
print(output)