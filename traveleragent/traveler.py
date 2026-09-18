from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import base_url
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_client = OpenAIChatCompletionClient(
    model = "openai/gpt-oss-120b",
    base_url = "https://api.groq.com/openai/v1",
    api_key = os.getenv("GROQ_API_KEY")
)

planner_agent = AssistantAgent(
    name = "Travel_Planner",
    description="A travel planner agent that helps users plan their trips.",
    model_client = model_client
)

research_agent = AssistantAgent(
    name = "Researcher",
    description="A researcher agent that helps users find information and answer questions.",
    model_client = model_client,
    system_message="You are a researcher agent. Your task is to help users find information and answer questions by conducting research and providing relevant data.",
)

team = RoundRobinGroupChat(
    participants=[planner_agent,research_agent],
    system_message="You are a travel planning team. Your task is to help users plan their trips by providing information about destinations, itineraries, and travel tips.",
    termination_condition=TextMentionTermination("TERMINATE")
)



async def run_traveleragent(query):
    result = team.run(task = query)
    return result





