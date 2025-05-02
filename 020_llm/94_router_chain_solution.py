#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utils.math import cosine_similarity
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))


# %% Model and Embeddings Setup
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

#%% Pydantic Model
class PatientCriticality(BaseModel):
    criticality: str = Field(description="The criticality of the patient")
    reason: str = Field(description="The reason for the criticality of the patient")

#%%
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You work in an emergency admission. You are given a patient description and you need to decide on the criticality of the patient. Return a JSON object with the criticality of the patient and the reason for the criticality."),
    ("human", "{topic}")
])
chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=PatientCriticality)

# %%
# chain.invoke({"topic": "I have a bleeding nose?"})
chain.invoke({"topic": "My wife is pregnant and she is in labor."})