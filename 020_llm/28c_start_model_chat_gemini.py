#%% packages
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# TODO: add package import

# %%
# Model overview: https://ai.google.dev/models/gemini
MODEL_NAME = 'gemini-1.5-flash'
# TODO: add the model

# %% Run the model
res = model.invoke("What is a Generative AI?")
# %% find out what is in the result
res.model_dump()
# %% only print content
print(res.content)
# %%
