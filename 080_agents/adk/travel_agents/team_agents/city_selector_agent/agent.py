"""City_selector_agent for selecting the best city for the user based on their preferences and location."""

from google.adk import Agent

from . import prompt

MODEL = "gemini-2.5-pro"

city_selector_agent = Agent(
    model=MODEL,
    name="city_selector_agent",
    instruction=prompt.CITY_SELECTOR_PROMPT,
)