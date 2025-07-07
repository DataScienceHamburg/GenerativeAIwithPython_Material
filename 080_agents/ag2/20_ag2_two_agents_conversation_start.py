
#%% packages
from autogen import ConversableAgent
from dotenv import load_dotenv, find_dotenv
import os
#%% load the environment variables
load_dotenv(find_dotenv(usecwd=True))

#%% LLM config
llm_config = {"config_list": [
    {"model": "gpt-4o-mini", 
     "api_key": os.environ.get("OPENAI_API_KEY")}]}

#%% set up the agent: Jack, the flat earther

#%% set up the agent: Alice, the scientist

# %% start the conversation

# %% check the result
 
