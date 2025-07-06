#%% packages
import re
import unicodedata
import json
from langchain_core.documents import Document

#%% save the documents to a json file
def save_langchain_docs_to_json(documents: list[Document], filename: str):
    """
    Saves a list of LangChain Document objects to a JSON file.

    Args:
        documents: A list of LangChain Document objects.
        filename: The name of the file to save the documents to (e.g., "my_docs.json").
    """
    data_to_save = []
    for doc in documents:
        data_to_save.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved {len(documents)} documents to {filename}")
    except IOError as e:
        print(f"Error saving documents to file: {e}")

#%% load the documents from a json file
def load_langchain_docs_from_json(filename: str) -> list[Document]:
    """
    Loads LangChain Document objects from a JSON file.

    Args:
        filename: The name of the JSON file to load documents from.

    Returns:
        A list of LangChain Document objects.
    """
    loaded_documents = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                doc = Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {})
                )
                loaded_documents.append(doc)
        print(f"Successfully loaded {len(loaded_documents)} documents from {filename}")
        return loaded_documents
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
        return []
    except IOError as e:
        print(f"Error loading documents from file: {e}")
        return []

#%% data cleaning
def clean_text(text: str) -> str:
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and digits (keep alphanumeric and spaces)
    # Adjust regex based on what you want to keep/remove
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# %%
