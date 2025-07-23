from dotenv import load_dotenv
import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv



load_dotenv()
llm_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens = None,
    api_key = llm_api_key,
)

def requirement_info_from_description(description:str, requirements_list:list) -> list:
    system_message = """
    You are provided with a `description` string that outlines a specific QA need or objective in the context of the Software Development Life Cycle (SDLC). You also receive a list of `requirements`, where each requirement is a dictionary with attributes such as "test id", "priority".

    Your task is to:
    1. Analyze the `description` and infer its intended QA function or goal.
    2. Scan the list of requirement dictionaries.
    3. Identify which requirements are semantically aligned with the description.
    4. Copy the dictionary of that requirement and add it onto a Python list
    4. Return a list containing the full dictionaries of matching requirements. ONLY return the list and nothing else

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
    ]    
    """ 
    messages = [SystemMessage(content=system_message)] 
    messages.append(HumanMessage(content="here is the description: "+ description +" Here is the requirement list: " + str(requirements_list)))
    response = llm.invoke(messages)
    return response.content


