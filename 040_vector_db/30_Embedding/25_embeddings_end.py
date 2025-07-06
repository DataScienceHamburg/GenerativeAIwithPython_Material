#%% packages

from langchain_openai import OpenAIEmbeddings
import sys
sys.path.append('..')
from helper_funs import load_langchain_docs_from_json

#%% load the data
docs = load_langchain_docs_from_json("../howto-logging-split.json")
len(docs)

#%%
#%% embedding
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

#%%
docs_contents = [doc.page_content for doc in docs]
docs_embed = embeddings_model.embed_documents(texts=docs_contents)
#%% number of embeddings
len(docs_embed)

#%% size of the embeddings
len(docs_embed[0])


