#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
map_chain = RunnableParallel(
    story_1=lambda x: story_chain.invoke(input_variables_1),
    story_2=lambda x: story_chain.invoke(input_variables_2)
)

# %% Invoke
result = map_chain.invoke({})
# %% Print
from pprint import pprint
pprint(result)
# %%
result['story_1']
# %%
result['story_2']
# %%

