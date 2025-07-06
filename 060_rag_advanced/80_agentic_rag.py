#%% packages
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent import ReActAgent

import nest_asyncio
nest_asyncio.apply()
# %% function tools
def count_words(text: str) -> int:
    """Count the words of the input text."""
    return len(text.split())

def count_letters_in_text(text: str, letter: str) -> int:
    """Count the letters of the input text."""
    # count the letters of the input text
    letter_count = text.lower().count(letter.lower())
    return letter_count

count_words_tool = FunctionTool.from_defaults(fn=count_words)
count_letters_in_text_tool = FunctionTool.from_defaults(fn=count_letters_in_text)

count_letters_in_text_tool.name = "count_letters_in_text"
count_letters_in_text_tool.description = "Count the letters of the input text."

#%% Test
llm = OpenAI(model="gpt-4o-mini")
# query = "How many words are in this text: 'The quick brown fox jumps over the lazy dog'?"
query = "How many words are in the text 'Hello world'?"
response = llm.predict_and_call(
    [count_words_tool, count_letters_in_text_tool], 
    query, 
    verbose=True
)
print(str(response))

# %% data loading
documents = SimpleDirectoryReader(input_files=["data/corrective_rag.pdf"]).load_data()

# %% data chunking
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
# %% set parameters for LLM
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

#%% index creation
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

#%% query engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

#%% set up tools
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to the paper on the topic of corrective RAG."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the paper on the topic of corrective RAG."
    ),
)

#%% Create an agent that can use both query engine tools and function tools
agent = ReActAgent.from_tools(
    [summary_tool, vector_tool, count_words_tool, count_letters_in_text_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True
)

#%% Test the agent with a query that requires both document knowledge and function tools
query = "What is the result of a low-quality retriever in corrective RAG?"
response = agent.query(query)
print(str(response))

#%% Test with a query that uses the letter counting function
query = "How many times does the letter 'a' appear in the summary of the document?"
response = agent.query(query)
print(str(response))
# %%
