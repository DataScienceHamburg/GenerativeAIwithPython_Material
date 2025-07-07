#%% packages
from agents import Agent, Runner
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(usecwd=True))

# source: https://openai.github.io/openai-agents-python/models/
# %%
agent = Agent(
    name="Butler",
    instructions="You are a helpful AI agent.",
      
)

result = Runner.run_sync(agent, "what is MCP, A2A, and ACP?")
print(result.final_output)
# %%
