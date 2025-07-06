#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
load_dotenv(find_dotenv(usecwd=True))
#%% set up prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates English into another language. You only return the translation, nothing else."),
    ("user", "Translate this sentence: <<'{input}'>> into <<'{target_language}'>>"),
])

#%% model
model = ChatGroq(
    model_name="llama3-8b-8192"
)

# %% chain 
chain = prompt_template | model | StrOutputParser()

#%% invoke chain
res = chain.invoke({"input": "I love programming.", "target_language": "German"})
# %%
pprint(res)

# %%
