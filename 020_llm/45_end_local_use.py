#%% packages
from langchain_ollama import ChatOllama

#%% pull model in terminal
# !ollama pull qwen3:4b

# %% model setup
model = ChatOllama(model="qwen3:4b", temperature=0.5, extract_reasoning=False)

# %% invoke model
model.invoke("What is your name?")

# %%
