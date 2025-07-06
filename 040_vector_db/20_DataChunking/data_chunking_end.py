#%% packages
import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import seaborn as sns
import sys
sys.path.append('..')
from helper_funs import clean_text, load_langchain_docs_from_json, save_langchain_docs_to_json
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
# 2000 characters = 200 tokens overlap


#%% 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
# %% split the documents
docs_split = text_splitter.split_documents(docs)
len(docs_split)

#%% visualise the chunk sizes
chunk_sizes = [len(chunk.page_content) for chunk in docs_split]
sns.histplot(chunk_sizes)
# %% save the documents
save_langchain_docs_to_json(docs_split, "howto-logging-split.json")
# %%