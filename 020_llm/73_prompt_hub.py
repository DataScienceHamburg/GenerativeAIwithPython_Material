#%% packages
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv('.env')
from pprint import pprint

#%% fetch prompt
prompt = hub.pull("hardkothari/prompt-maker")

#%% get input variables
prompt.input_variables

# %% model
model = ChatOpenAI(model="gpt-4o-mini", 
                   temperature=0)

# %% chain
chain = prompt | model | StrOutputParser()

# %% invoke chain
lazy_prompt = "A mysterious fog rolls into a quiet coastal town."
task = "Write a short story (around 500 words) exploring the impact of the fog on the town and its inhabitants. Consider what might be hidden within the fog."
improved_prompt = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})
# %%
print(improved_prompt)

# %% run model with improved prompt
res = model.invoke(improved_prompt)
print(res.content)

# %% original prompt
res = model.invoke(f"{lazy_prompt} {task}")
print(res.content)
# %%
