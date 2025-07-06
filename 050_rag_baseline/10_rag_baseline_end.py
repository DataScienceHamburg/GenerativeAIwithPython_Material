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

#%% get number of documents in the vector store
len(vector_store.get()["ids"])

#%% RAG function
def rag(query:str, n_results:int=5, search_type:str="similarity"):
    """
    RAG function to retrieve the most relevant information from the vector store
    and return the response
    Args:
        query (str): The user's question
        n_results (int): The number of results to return
        search_type (str): The type of search to perform
    Returns: 
        docs_content (list): The content of the retrieved documents
        model_response (str): The response from the model
        prompt (str): The complete prompt used to generate the response
    """
    retriever = vector_store.as_retriever(search_type=search_type,
    search_kwargs={"k": n_results})
    docs = retriever.invoke(query)
    docs_content = [doc.page_content for doc in docs]
    context_information = ';'.join([f'{doc.page_content}' for doc in docs])
    messages = [
        ("system", "You are a helpful assistant. You will be shown the user's question, and the relevant information. Answer the user's question using ONLY this information. Say 'I don't know' if you don't know the answer."),
        ("user", "Question: {query}. \n Information: {joined_information}.")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model | StrOutputParser()
    res = chain.invoke({"query": query, "joined_information": context_information})
    # return also the complete prompt
    return docs_content, res, prompt.invoke({"query": query, "joined_information": context_information})

# %%
context, response, prompt = rag(query="How do I log to a file?", n_results=10)

# %% show the context
for doc in context:
    print(doc)
    print("-"*100)

# %% show the response
print(response)
# %%
