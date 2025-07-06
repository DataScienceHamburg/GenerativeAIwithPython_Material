#%% packages

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import numpy as np
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser
import os

#%% --- 1. Standard RAG Components ---
# Mock Knowledge Base (replace with your actual data and indexing)
documents = [
    "Harnessing the kinetic energy of the wind, turbines stand as sentinels across landscapes, converting a natural force into clean electricity.",
    "As wind technology advances, larger rotor diameters and taller towers are enabling greater energy capture, even in areas with lower average wind speeds.",
    "Sunlight, an abundant resource, is directly transformed into electrical power through photovoltaic panels, blanketing rooftops and sprawling across solar farms.",
    "The efficiency of photovoltaic cells continues to improve, with new materials and designs pushing the boundaries of solar energy conversion.",
    "Beyond wind and solar, geothermal energy taps into the Earth's internal heat, providing a stable and consistent source of power.",
    "Utilizing the temperature difference between surface and deep water, ocean thermal energy conversion presents a vast, yet largely untapped, renewable resource.",
    "Hydropower, a mature renewable technology, leverages the power of flowing water to generate electricity, often with the added benefit of water management.",
    "The integration of diverse renewable energy sources into smart grids is crucial for ensuring a reliable and resilient energy supply.",
    "Energy storage solutions, such as advanced batteries, are becoming increasingly vital for smoothing out the intermittent nature of some renewable energy sources like wind and solar.",
    "Innovations in renewable energy are not only decarbonizing the power sector but also creating new economic opportunities and fostering energy independence."
]



#%% Embeddings Function and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a vector store directly with documents
vector_store = FAISS.from_texts(
    documents,
    embeddings
)


#%% test the index
query = "Which  energy sources are renewable?"
results = vector_store.similarity_search(
    query,
    k=6,
    fetch_k=20,
    lambda_mult=0.2  # Diversity parameter: 0 = max diversity, 1 = max relevance
)
results

#%% Generate Drafts and Rationale
def generate_draft_with_rationale(query, relevant_documents, num_drafts=3):
    drafts_with_rationales = []
    drafter_model = ChatGroq(model="gemma2-9b-it") # Using a smaller, faster model for drafting

    chunk_indices = list(range(len(relevant_documents)))

    for i in range(num_drafts):
        # chooose randomly two documents from the relevant documents
        # create two indices
        selected_indices = np.random.choice(chunk_indices, size=2, replace=False)
        # remove the indices from the chunk_indices
        chunk_indices = [i for i in chunk_indices if i not in selected_indices]
        # create the context
        context = "\n".join([relevant_documents[i].page_content for i in selected_indices])
        # Generate a draft answer based on the context
        draft_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant generating potential answers to a question based on the provided context. Only answer based on the context, do not make up information. Directly answer the question."),
            ("user", f"Context: {context}\n\nQuestion: {query}\n\nGenerate a possible answer:")
        ])
        draft_chain = draft_prompt_template | drafter_model | StrOutputParser()
        draft = draft_chain.invoke({})

        # Generate a rationale for the draft answer
        rationale_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant explaining the reasoning behind a generated answer based on the context."),
            ("user", f"Context: {context}\n\nQuestion: {query}\n\nAnswer: {draft}\n\nExplain your reasoning:")
        ])
        rationale_chain = rationale_prompt_template | drafter_model | StrOutputParser()
        rationale = rationale_chain.invoke({})

        drafts_with_rationales.append({"draft": draft, "rationale": rationale})
    return drafts_with_rationales


#%% Test the generate_draft_with_rationale function
query = "Which energy sources are renewable, and what are the benefits?"
generate_draft_with_rationale(query=query, relevant_documents=results[:6], num_drafts=3)

#%% Evaluate Drafts
class Evaluation(BaseModel):
    justification: str
    score: int

def evaluate_draft(query, draft_with_rationale):
    evaluator_model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)        
    context = "\n".join([doc for doc in draft_with_rationale['rationale']])

    evaluation_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a critical AI assistant evaluating the quality and relevance of an answer based on the question, the provided context, and its rationale. Provide a score from 1 to 10, use 1 digit after the decimal point, followed by a brief justification. Return the score and justification in a json format."),
        ("user", f"Context: {context}\n\nQuestion: {query}\n\nAnswer: {draft_with_rationale['draft']}\nRationale: {draft_with_rationale['rationale']}\nScore (1-10) and Justification:")
    ])
    evaluation_chain = evaluation_prompt_template | evaluator_model | JsonOutputParser(pydantic_object=Evaluation)
    evaluation_output = evaluation_chain.invoke({})

    return evaluation_output

#%% Speculative RAG
def speculative_rag(query, vector_store):
    # 1. Retrieve relevant documents
    relevant_documents = vector_store.similarity_search(query, k=6,
    fetch_k=20,
    lambda_mult=0.2  # Diversity parameter: 0 = max diversity, 1 = max relevance
    )
    
    # 2. Generate multiple draft answers with rationales based on the context
    drafts_with_rationales = generate_draft_with_rationale(query=query, relevant_documents=relevant_documents, num_drafts=3)
    print("\nGenerated Drafts with Rationales (based on context):")
    for i, d in enumerate(drafts_with_rationales):
        print(f"Draft {i+1}: {d['draft']}")
        print(f"Rationale {i+1}: {d['rationale']}")
        print("-" * 20)

    # 3. Evaluate each draft based on the query, context, and rationale
    scored_drafts = []
    for d in drafts_with_rationales:
        evaluation = evaluate_draft(query, d)
        scored_drafts.append({"draft": d['draft'], "evaluation": evaluation})
        print(f"Score: {evaluation['score']}, Justification: {evaluation}")
        print("-" * 20)

    # 4. Select the draft with the highest score
    best_draft = max(scored_drafts, key=lambda x: x['evaluation']['score'])
    print(f"\nSelected Best Draft (Score: {best_draft['evaluation']['score']}):")
    return best_draft['draft']

# %% --- Example Usage ---
query = "Which energy sources are renewable, and what are the benefits?"
final_answer = speculative_rag(query, vector_store)
# %%
print(f"\nFinal Answer: {final_answer}")

# %%
