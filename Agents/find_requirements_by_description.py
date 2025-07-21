from dotenv import load_dotenv
import os

from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv



load_dotenv()
llm_api_key = os.getenv("ANTHROPIC_API_KEY")
api_max_tokens = 64000                      # Maximum ammount

llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature = 0.3,
    max_tokens = api_max_tokens,
    api_key = llm_api_key
)

def requirement_info_from_description(description:str, requirements_list:list) -> list:
    system_message = """
You are provided with a `description` string that outlines a specific QA need or objective in the context of the Software Development Life Cycle (SDLC). You also receive a list of `requirements`, where each requirement is a dictionary with attributes such as "test id", "priority".

Your task is to:
1. Analyze the `description` and infer its intended QA function or goal.
2. Scan the list of requirement dictionaries.
3. Identify which requirements are semantically aligned with the description.
4. Return a list containing the full dictionaries of matching requirements.

Example:
description = "Looking for requirements related to test automation during the integration phase"
requirements = [
  {"id": "R1", "type": "manual testing", "objective": "exploratory testing", "phase": "system testing", "tool": "none", "priority": "medium", "owner": "QA team"},
  {"id": "R2", "type": "automated testing", "objective": "regression coverage", "phase": "integration testing", "tool": "Selenium", "priority": "high", "owner": "QA automation lead"},
  {"id": "R3", "type": "code review", "objective": "quality gate", "phase": "development", "tool": "SonarQube", "priority": "low", "owner": "Dev lead"}
]

Expected output:
[
  {"id": "R2", "type": "automated testing", "objective": "regression coverage", "phase": "integration testing", "tool": "Selenium", "priority": "high", "owner": "QA automation lead"}
    
    """ 
    message = system_message + "here is the descripition: "+ description +"Here is the requirement list: "+str(list)
    response = llm.invoke(message)
    print(response)
    return response