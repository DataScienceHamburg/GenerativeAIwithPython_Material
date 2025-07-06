#%% (1) Packages
import os
import requests
import zipfile
# TODO: 
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


#%% TODO:load the file


#%% TODO: check the number of documents


#%% TODO: first document

#%% TODO: first document content

#%% TODO: length of the first document

#%% TODO: save the documents as a json file

