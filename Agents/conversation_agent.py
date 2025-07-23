from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId


from gen_test_cases import generate_test_cases_from_requirements
from gen_requirements import generate_requirements_from_doc 
from find_requirements_by_description import requirement_info_from_description
load_dotenv()
llm_api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens = None,
    api_key = llm_api_key,
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    project_name: str
    context: str
    requirements: list
    testCases: list


@tool
def generate_requirements_from_document_pdf_tool(
    x: Annotated[str, "the PATH to the pdf document from which the tool will generate requirements"],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Extracts a brief summary and a list of requirements from a project document.

    **Inputs:**
    - `x` (str): The PATH to the pdf document
    
    **Behavior:**
    - Decodes the PDF document from input path into base64 string.
    - Analyzes the document content to extract project requirements.
    - Generates a summary of the document.
    - This will overwrite everything in state["requirements"]. Refer to other tools if the user wants addition.

    **Outputs:**
    - Updates `state["context"]` with the extracted summary.
    - Updates `state["requirements"]` with a structured list of requirements.
    - Adds a success tool message in `state["messages"]`.

    **User Interaction**
    - Don't mention anything about the path of the pdf. Always mention ONLY about the pdf itself. For example: Don't say "Please provide the path of the pdf.". Say "Please provide the pdf"
    """
    requirements, context = generate_requirements_from_doc(x)
    return Command(update={
        "context": context,
        "requirements": requirements,
        "messages": [
            ToolMessage(
                "Successfully created requirements from the document",
                tool_call_id=tool_call_id
            )
        ]
    })
    
@tool
def generate_testCases_fromRequirements_tool(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Generates a list of test cases from provided requirements.

    **Inputs:**
    - `state["requirements"]` (list[dict]): A list of software or system requirements.
    - `state["context"]` (str): A brief summary to provide additional context.

    **Behavior:**
    - If `requirements` or `context` are missing, returns an error message.
    - Processes requirements to generate structured test cases.
    - Returns a list of test cases in `state["testCases"]`.
    - This will overwrite everything in state["testCases"]. Refer to other tools if the user wants addition.

    **Outputs:**
    - Updates `state["testCases"]` with generated test cases.
    - Adds a success message to `state["messages"]`.
    """
    requirements = state.get('requirements')
    context = state.get('context')
    if not requirements or not context:
        return Command(update={
            "messages": [
            ToolMessage(
                "Requirements or Context missing",
                tool_call_id=tool_call_id
            )
        ]
    })
    testCases_list = generate_test_cases_from_requirements(requirements, context)
    return Command(update={
        "testCases": testCases_list,
        "messages": [
            ToolMessage(
                "Successfully created test cases from the requirements",
                tool_call_id=tool_call_id
            )
        ]
    })

@tool
def requirement_info_from_description_tool(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    description: Annotated[str,"The descrption that the tool based on to find full information"]
) -> Command:
    """
    Inquiry requirements that meets the description and get the full information of those requirements (ID, description,priority , dependency)"
    
    **Inputs:**
    - description: The description of the requirement that the tool will try to find and parse information"

    **Behavior:**
    - Looks through all the requirements in the state['requirements']
    - Parse full infomation of the requirements that aligns with the given description into ToolMessage()
    - If no requirement is found, report back 

    **Outputs:**
    - Adds a Tool message to `state["messages"]`. If successful, the tool message is a list containing dictionaries of the aligned requirements, else the message is a empty list.
    """
    if state.get('requirements') is None:
        return Command(update={{"messages": [ToolMessage(content="There are not yet requirement in the workspace")]}})
    tool_answer_string = requirement_info_from_description(description, state['requirements'])
    return Command(update={{"messages": [ToolMessage(content=tool_answer_string, tool_call_id=tool_call_id)]}})


@tool
def change_requirement_info(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    req_id: Annotated[str,"The ID of the requirement that needs change"],
    attribute: Annotated[str,'The attribute of the requirement that needs change'],
    value: Annotated[str,'The value that the attribute of a requirement will change into']
) -> Command:
    """
    Change the information of a requirement, or technically change the one attribute value from a requirement

    **Input**
    - req_id: The ID of the requirement. The tool use this to find the targeted requirement
    - attribute: The attribute inside the requirement dictionary that needs change. The tool will check if the attribute is one of the keys
    - value: 'The value that the attribute of a requirement will change into'

    **Behavior**
    - The tool finds that requirement using its ID by searching the dictionary in the list that has that ID.
    - The tool parse the values from the parameters into a conventional way to change value in Python list.
    - The tool will check if that requirement/attribute exisits. If not return an error ToolMessage

    **Output**
    - Updates state['requirements'] with the modified requirement
    - Add a successful ToolMessage

    **Note**
    - Requires exisiting requirement to utilize this tool
    - Can use other tools to locate the ID if the user is using natural language to describe the requirement. Then use that tool result to parse onto this tool
    """
    if state.get('requirements') is None:
        return Command(update={{"messages": [ToolMessage(content="There are not yet requirements in the workspace")]}})
    
    import copy
    requirements_copy = copy.deepcopy(state['requirements'])
    if any(requirement.get("ID") == req_id for requirement in requirements_copy):
        for req in requirements_copy:
            if req['ID'] == req_id:
                req[attribute] = value
                break
        return Command(update={
        "requirements": requirements_copy,
        "messages": [ToolMessage(
                "Successfully modify the requirement",
                tool_call_id=tool_call_id
                )]
            })
    else:
        return Command(update={"messages":[ToolMessage(content="Invalid key value")]})  




tools = [generate_testCases_fromRequirements_tool,
         generate_requirements_from_document_pdf_tool]
llm_with_tools = llm.bind_tools(tools)
tools_node = ToolNode(tools)

def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tools_node)

builder.add_edge(START,"call_model")
builder.add_conditional_edges(
    "call_model",
    tools_condition,
)
builder.add_edge("tools","call_model")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

def call_conversation_model(state: AgentState, userInputContent: str, path_to_file: Optional[str] = None) -> AgentState:
    if path_to_file is not None:
        userInputContent += " The pdf path: "+ path_to_file
    state['messages'].append(HumanMessage(content=userInputContent))
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke(state,config)

    newState = graph.get_state(config).values
    return newState