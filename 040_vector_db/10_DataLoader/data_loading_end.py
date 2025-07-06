#%% (1) Packages
import os
import requests
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import zipfile
import sys

sys.path.append('..')
from helper_funs import save_langchain_docs_to_json
#%% (2) File Handling
# Get the current working directory
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)

URL_PYTHON_DOCUMENTATION = "https://docs.python.org/3/archives/python-3.13-docs-pdf-a4.zip"
file_name = URL_PYTHON_DOCUMENTATION.split('/')[-1]
file_path_local = os.path.join(current_dir, file_name)

if not os.path.exists(file_path_local):
    # Download the file
    response = requests.get(URL_PYTHON_DOCUMENTATION)
    with open(file_path_local, 'wb') as f:
        f.write(response.content)

#%% unzip the file
if not os.path.exists(os.path.join(current_dir, "docs-pdf")):
    with zipfile.ZipFile(file_path_local, 'r') as zip_ref:
        zip_ref.extractall(current_dir)



#%% load the file
file_path_local_logging = os.path.join(current_dir, "docs-pdf", "howto-logging.pdf")
loader = PyMuPDF4LLMLoader(file_path_local_logging)
docs = loader.load()
docs

#%% number of documents
len(docs)

#%% first document
docs[0].model_dump()

#%% first document content
docs[10].page_content

#%% length of the first document
len(docs[10].page_content)

#%% save the documents as a json file

save_langchain_docs_to_json(docs, "../howto-logging.json")
