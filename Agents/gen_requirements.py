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
path_to_initPrompt_1 = r"..\Prompts\gen_requirements\initial_prompt_1.txt"
path_to_initPrompt_2 = r"..\Prompts\gen_requirements\initial_prompt_2.txt"
path_to_reflectionArewrite_Prompt = r"..\Prompts\gen_requirements\reflection_and_rewrite_prompt.txt"
path_to_summaryPrompt = r"..\Prompts\gen_requirements\context_summary.txt"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature = 0.3,
    max_tokens = api_max_tokens,
    api_key = llm_api_key
)

def generate_requirement_from_doc(pdf_in_base64_encoded_string: str) -> tuple[list,str]:
    """
    Params: 
    str: Base64 encoded string representation of the pdf

    Output:
    list: list of requirements
    str: Context summary of the document

    """


    """
    1. Load the prompts
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

    with open(path_to_reflectionArewrite_Prompt,"r") as f:
        reflection_and_rewrite_Prompt = f.read()
        content_prompt_reflection_and_rewrite = [
                    {
                        "type": "text",
                        "text": reflection_and_rewrite_Prompt
                    }
                ]
    with open(path_to_summaryPrompt,"r") as f:
        summary_Prompt = f.read()
        content_prompt_summary = [
                    {
                        "type": "text",
                        "text": summary_Prompt
                    }
                ]        

    """
    2. Define writing Steps
    """
    def write_initial_draft(state: AgentState):
        model_response = llm.invoke([HumanMessage(content=content_prompt_initial)])
        
        return {"messages": [HumanMessage(content=content_prompt_initial), model_response]}
        
    def write_reflection_and_final(state: AgentState):
        prompt_messages = state['messages'] + [HumanMessage(content=content_prompt_reflection_and_rewrite)]
        model_response = llm.invoke(prompt_messages)

        return{"messages": [HumanMessage(content=content_prompt_reflection_and_rewrite), model_response]}

    def write_summary(state: AgentState):
        prompt_messages = state['messages'] + [HumanMessage(content=content_prompt_summary)] + [AIMessage(content="[")]
        model_response = llm.invoke(prompt_messages)

        return{"messages": [HumanMessage(content=content_prompt_summary), model_response]}

    """
    3. Build graph
    """
    builder = StateGraph(AgentState)
    builder.add_node(write_initial_draft, "write_initial_draft")
    builder.add_node(write_reflection_and_final, "write_reflection_and_final")
    builder.add_node(write_summary, "write_summary")
    builder.add_edge(START, "write_initial_draft")
    builder.add_edge("write_initial_draft","write_reflection_and_final")
    builder.add_edge("write_reflection_and_final","write_summary")
    builder.add_edge("write_summary", END)

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
    
    output_requirements_string = langchainConfig_messages['messages'][-3].content
    output_summary_string = langchainConfig_messages['messages'][-1].content

    try:
        output_requirements_list = ast.literal_eval("["+output_requirements_string)
    except json.JSONDecodeError as e:
        print(f"AI failed to generate requirements in appropriate format error: {e}")

    return output_requirements_list, output_summary_string






