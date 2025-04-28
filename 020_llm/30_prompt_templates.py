#%% packages
from langchain_core.prompts import ChatPromptTemplate

#%% set up prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates English into another language."),
    ("user", "Translate this sentence: '{input}' into {target_language}"),
])

#%% invoke prompt template
prompt.invoke({"input": "I love programming.", "target_language": "German"})

# %%
