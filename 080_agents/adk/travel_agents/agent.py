#%% packages
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from dotenv import load_dotenv, find_dotenv
from . import prompt
from .team_agents.city_selector_agent import city_selector_agent
from .team_agents.event_planning_agent import event_planning_agent

load_dotenv(find_dotenv())
    
#%%
root_agent = Agent(
    name="travel_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to provide information on travel planning."
    ),
    instruction=(
        "You are a helpful agent who can provide information on travel planning."
    ),
    tools=[
        AgentTool(agent=city_selector_agent),
        AgentTool(agent=event_planning_agent),
    ],
)