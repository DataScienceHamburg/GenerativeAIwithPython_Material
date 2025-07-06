#%% packages
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))


#%% embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#%% load vector store
vector_store = Chroma(
    persist_directory="../040_vector_db/30_Embedding/chroma_db",
    embedding_function=embeddings
)

#%% TODO: get number of documents in the vector store


#%% TODO: RAG function

# %% TODO: test RAG function

# %% TODO: show the context

# %% TODO: show the response

