#%% packages
from langchain_openai import OpenAIEmbeddings
# TODO: package import
import sys
sys.path.append('..')
from helper_funs import load_langchain_docs_from_json

#%% load the data
docs = load_langchain_docs_from_json("../Sustainable_Development_Goals_Report_2023_split.json")
len(docs)
#%% embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#%% TODO: create vector store


#%% TODO: make the vector store persistent


#%% TODO: query the vector store


#%% TODO: better use retriever


# %%
