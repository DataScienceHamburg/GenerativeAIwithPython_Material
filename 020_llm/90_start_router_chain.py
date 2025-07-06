# %% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% Model and Embeddings Setup
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# TODO: add embedding model

#%% Prompt Templates
template_cooking = "Provide a recipe based on the following ingredients: {user_input}, state that you are a culinary agent"
template_travel = "Suggest a travel destination and itinerary based on the user's interest: {user_input}, state that you are a travel agent"
template_coding = "Provide a code snippet or explanation for the following coding question: {user_input}, state that you are a coding agent"


# %% Cooking-Chain
prompt_cooking = ChatPromptTemplate.from_messages([
    ("system", template_cooking),
    ("human", "{user_input}")
])
chain_cooking = prompt_cooking | model | StrOutputParser()

# %% Travel-Chain
prompt_travel = ChatPromptTemplate.from_messages([
    ("system", template_travel),
    ("human", "{user_input}")
])
chain_travel = prompt_travel | model | StrOutputParser()

#%%
# Coding-Chain
prompt_coding = ChatPromptTemplate.from_messages([
    ("system", template_coding),
    ("human", "{user_input}")
])
chain_coding = prompt_coding | model | StrOutputParser()

#%% combine all chains
chains = [chain_cooking, chain_travel, chain_coding]

# %% TODO: Create Prompt Embeddings

#%%
print(len(chain_embeddings))

# %% Prompt Router
def my_prompt_router(input: str):
    # TODO: embed the user input
    
    # TODO: calculate similarity
    
    # TODO: get the index of the most similar prompt
    # return the corresponding chain
    return "return the right chain"


#%% Testing the Router
query = "What can I make with chicken and vegetables?"
# query = "I'd like to visit a warm beach in Europe."
# query = "How do I write a for loop in Python?"
chain = my_prompt_router(query)
print(chain.invoke(query))

# %%