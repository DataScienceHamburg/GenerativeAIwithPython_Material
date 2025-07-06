#%% packages
from langchain import hub
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv('.env')


#%% fetch prompt

#%% get input variables


# %% model
model = ChatOpenAI(model="gpt-4o-mini", 
                   temperature=0)

# %% chain

# %% invoke chain
lazy_prompt = "A mysterious fog rolls into a quiet coastal town."
task = "Write a short story (around 500 words) exploring the impact of the fog on the town and its inhabitants. Consider what might be hidden within the fog."

# %%

# %% run model with improved prompt

# %% original prompt
res = model.invoke(f"{lazy_prompt} {task}")
print(res.content)
# %%
