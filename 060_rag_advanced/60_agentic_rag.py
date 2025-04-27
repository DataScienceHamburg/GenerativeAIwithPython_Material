#%% packages
from llama_index.core.tools import FunctionTool

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import nest_asyncio
nest_asyncio.apply()
# %% excourse on function tools
# def add(x: int, y: int) -> int:
#     """Adds two integers together."""
#     return x + y

# def mystery(x: int, y: int) -> int: 
#     """Mystery function that operates on top of two numbers."""
#     return (x + y) * (x + y)


# add_tool = FunctionTool.from_defaults(fn=add)
# mystery_tool = FunctionTool.from_defaults(fn=mystery)
# # %%
# llm = OpenAI(model="gpt-4o-mini")
# response = llm.predict_and_call(
#     [add_tool, mystery_tool], 
#     "Tell me the output of the mystery function on 2 and 9", 
#     verbose=True
# )
# print(str(response))

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
        "Useful for summarization questions related to the paper"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the paper."
    ),
)
#%%
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)
#%%
# query = "What is the summary of the document?"
query = "What is the result of a low-quality retriever"
response = query_engine.query(query)
print(str(response))
# %%
