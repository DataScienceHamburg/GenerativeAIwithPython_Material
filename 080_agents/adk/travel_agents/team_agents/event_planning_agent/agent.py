"""Event_planning_agent for finding events using search tools."""

from google.adk import Agent
from google.adk.tools import google_search

from . import prompt

MODEL = "gemini-2.5-pro"


event_planning_agent = Agent(
    model=MODEL,
    name="event_planning_agent",
    instruction=prompt.EVENT_PLANNING_PROMPT,
    output_key="event_planning",
    tools=[google_search],
)