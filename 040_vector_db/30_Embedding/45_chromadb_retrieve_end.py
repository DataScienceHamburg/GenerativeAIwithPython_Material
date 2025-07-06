#%% packages
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import sys
sys.path.append('..')
from helper_funs import load_langchain_docs_from_json

#%% load the data
docs = load_langchain_docs_from_json("../howto-logging-split.json")
len(docs)

#%% embedding
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


#%% create vector store
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model
)


#%% query the vector store
query = "How can I log to a file?"
vector_store.similarity_search(query)
#%%
#%% better: retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)
retriever.invoke(query)

