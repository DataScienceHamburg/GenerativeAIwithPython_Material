#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
load_dotenv('.env')
from pydantic import BaseModel, Field

#%% structured output
class TranslationResponse(BaseModel):
    source_text: str = Field(description="the source text")
    target_text: str = Field(description="the translated text")
    source_language: str =  Field(description="the source language of the input text")
    target_language: str =  Field(description="the target language of the input text")
    tonality: int = Field(description="the tonality of the input text")

#%% set up prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates a source language into other languages. No babbling! Make sure your output is a valid JSON object with the following fields: source_text, target_text, source_language, target_language, tonality [range of 1 to 5, with 1 being most negative and 5 being most positive]."),
    ("user", "Translate this sentence: '{input}' into {target_language}"),
])

# %% model
model = ChatOpenAI(model="gpt-4o-mini", 
                   temperature=0)

# %% chain
chain = prompt | model | JsonOutputParser(pydantic_object=TranslationResponse)

# %% invoke chain
res = chain.invoke({"input": "I kind oflove programming.", "target_language": "German"})
res
# %% input: a sentence
# output: {'tonality': ['warm', 'aggressive', 'depressed']}