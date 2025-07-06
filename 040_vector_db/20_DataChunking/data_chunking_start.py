#%% packages
import json
import sys
sys.path.append('..')
from helper_funs import clean_text, load_langchain_docs_from_json, save_langchain_docs_to_json
# TODO: add imports for splitting
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

#%% load the data
docs = load_langchain_docs_from_json("../howto-logging.json")
docs

#%%
for i in range(len(docs)):
    processed_page_content = clean_text(docs[i].page_content)
    docs[i].page_content = processed_page_content
# %% chunking size 
# 1 token = 4 characters
# 2000 characters = 500 tokens
# 1000 characters = 250 tokens
# 200 characters = 50 tokens

#%% chunking overlap
# 10 % of the chunk size:
# 1000 characters = 100 tokens overlap

#%% TODO: Fixed Character Splitter instance
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
# %% TODO: split the documents
docs_split = text_splitter.split_documents(docs)
len(docs_split)
#%% TODO: visualise the chunk sizes

# %% save the documents
# %%