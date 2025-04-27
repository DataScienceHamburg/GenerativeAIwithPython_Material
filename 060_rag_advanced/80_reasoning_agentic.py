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
from llama_index.core.agent import AgentRunner, FunctionCallingAgentWorker
import nest_asyncio
nest_asyncio.apply()

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
#%% agent worker
agent_worker = FunctionCallingAgentWorker.from_tools(
    [summary_tool, vector_tool],
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True
)

agent = AgentRunner(agent_worker)

#%%
query = "What is the result of a low-quality retriever?"
task = agent.chat(query)

#%%
query = "Who killed batmans parents?"
task = agent.chat(query)

# %% history of previous queries
agent.chat_history
# %%
agent.get_completed_tasks()


#%% rerun a query (fetches from cache)
query = "What is the result of a low-quality retriever?"
task = agent.chat(query)
