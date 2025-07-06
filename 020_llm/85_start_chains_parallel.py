#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

#%% Model Instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#%% Prepare Prompts
# example: style variations (friendly, polite) vs. (savage, angry)
system_prompt = "You are a storyteller. You write a short story given a setting and a genre."


story_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{setting} {genre}")
])

#%% Prepare Chains
story_chain = story_prompt | llm | StrOutputParser()

#%% Prepare Input Variables
input_variables_1 = {"setting": "A bustling cyberpunk metropolis on a rainy night", 
                     "genre": "Film Noir/Mystery"}
input_variables_2 = {"setting": "A serene, ancient forest untouched by time", 
                     "genre": "Fantasy"}


# %% Runnable Parallel

# %% Invoke
