#%% packages
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
# Model overview: https://console.groq.com/docs/models
MODEL_NAME = 'gpt-4o-mini'
model = ChatOpenAI(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('OPENAI_API_KEY'))

# %% Run the model
res = model.invoke("What is a Generative AI?")
# %% find out what is in the result
res.model_dump()
# %% only print content
print(res.content)
# %%
