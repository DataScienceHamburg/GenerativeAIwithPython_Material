#%% packages
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import sys
sys.path.append('..')
from helper_funs import load_langchain_docs_from_json

#%% load the data
docs = load_langchain_docs_from_json("../howto-logging-split.json")
len(docs)

#%%
#%% embedding
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


#%% create vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="./chroma_db"
)


#%% TODO: query the vector store


#%% TODO: better use retriever

