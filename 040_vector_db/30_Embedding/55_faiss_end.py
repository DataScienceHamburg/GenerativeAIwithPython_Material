#%% packages
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import sys
sys.path.append('..')
from helper_funs import load_langchain_docs_from_json

#%% load the data
docs = load_langchain_docs_from_json("../Sustainable_Development_Goals_Report_2023_split.json")
len(docs)
#%% embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#%% create vector store
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

#%% make the vector store persistent
vector_store.save_local("./faiss_db")

#%% query the vector store
query = "How to perform logging?"
vector_store.similarity_search(query)

#%% better: retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)
retriever.invoke(query)

