#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv(usecwd=True))


#%% structured output

#%% set up prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates a source language into other languages. No babbling! Make sure your output is a valid JSON object with the following fields: source_text, target_text, source_language, target_language, tonality [range of 1 to 5, with 1 being most negative and 5 being most positive]."),
    ("user", "Translate this sentence: '{input}' into {target_language}"),
])

# %% model
model = ChatOpenAI(model="gpt-4o-mini", 
                   temperature=0)

# %% chain
chain = prompt | model 

# %% invoke chain

# %% input: a sentence
# output: {'tonality': ['warm', 'aggressive', 'depressed']}