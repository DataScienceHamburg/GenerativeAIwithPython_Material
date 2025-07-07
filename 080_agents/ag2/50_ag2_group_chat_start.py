#%% packages
import os
from autogen import ConversableAgent, LLMConfig
# TODO: import the necessary packages for group chat functionality
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% agents
user = ConversableAgent(name="user", human_input_mode="ALWAYS")


llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    moderator_agent = ConversableAgent(
        name="Moderator",
        system_message="You are a moderator and you are responsible for the conversation. Route the conversation to the right agent.",
    )


    visionary_agent = ConversableAgent(
        name="Visionary",
        system_message="You are a visionary and want your ideas to be implemented. You are focused on big ideas, creative solutions, and out-of-box thinking. They propose novel approaches and aren't afraid of ambitious concepts. Your personality is imaginative, bold, perhaps a bit abstract, future-oriented.",
    )
    technology_agent = ConversableAgent(
        name="TechnologyAgent",
        system_message="You are a technology agent and you are responsible for the technology. You bring the discussion back to feasibility, resources, timelines, and practical steps. You are grounded, detail-oriented, realistic, perhaps a little bureaucratic (in a good way).",
    )

    financial_agent = ConversableAgent(
        name="FinancialAgent",
        system_message="You are a financial agent and you are responsible for the financials. You bring the discussion back to feasibility, resources, timelines, and practical steps. You are grounded, detail-oriented, realistic, perhaps a little bureaucratic (in a good way).",
    )

#%% let group manager dynmically handle the conversation
# TODO: set up the pattern for the group chat

#%% start the conversation
# TODO: start the group chat


