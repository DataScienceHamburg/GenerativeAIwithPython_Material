#%% packages
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %%
# Model overview: https://ai.google.dev/models/gemini
MODEL_NAME = 'gemini-1.5-flash'
model = ChatGoogleGenerativeAI(model=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('GOOGLE_API_KEY'))

# %% Run the model
res = model.invoke("What is a Generative AI?")
# %% find out what is in the result
res.model_dump()
# %% only print content
print(res.content)
# %%
