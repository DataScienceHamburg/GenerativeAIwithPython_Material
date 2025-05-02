#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
#%% set up prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a storyteller. You can write stories about characters in certain situations."),
    ("user", "Character: {character}; Situation: {situation}")
])

    
# %%
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
chain = prompt | model | StrOutputParser()
# %%
input_variables = {"character": "Bert", "situation": "He finds out that he is in a computer simulation."}
res = chain.invoke(input_variables)
# %%
print(res)


#%% calculate the cost
cost_input_tokens = 2.5 # $/1M tokens
cost_output_tokens = 10 # $/1M tokens

# %%
usage_metadata = res.model_dump()['usage_metadata']
used_input_tokens = usage_metadata['input_tokens']
used_output_tokens = usage_metadata['output_tokens']
# %%
cost = used_input_tokens * cost_input_tokens / 1e6 + used_output_tokens * cost_output_tokens / 1e6
cost # $
# %%