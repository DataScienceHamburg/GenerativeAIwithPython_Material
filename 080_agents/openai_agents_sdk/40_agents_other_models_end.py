#%% packages
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(usecwd=True))

# source: https://openai.github.io/openai-agents-python/models/
# %%
agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    model="litellm/anthropic/claude-3-5-sonnet-20240620",
)

result = Runner.run_sync(agent, "what are differential equations?")
print(result.final_output)
# %%
