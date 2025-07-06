#%% packages
from langchain_ollama import ChatOllama
#%% pull model in terminal
# !ollama pull qwen3:4b

# %% model setup
model = ChatOllama(model="qwen3:4b", temperature=0.2, extract_reasoning=True)
# %% invoke model
res = model.invoke("What is Ollama?")
# %%
res.model_dump()
# %%
