#%% packages
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import pandas as pd
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from pydantic import BaseModel, Field

# %%
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
# data source: https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k
#%% import flickr8k dataset
df = pd.read_csv("data/flickr_texts/Flickr8k.token.txt", sep="\t", header=None, names=["image", "caption"])

#%% function for creating embeddings
def create_embeddings(text, image):
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    # return the embeddings
    return outputs.image_embeds, outputs.text_embeds

#%% create FAISS indices for both image and text embeddings
image_index = faiss.IndexFlatL2(512)  # CLIP embeddings are 512-dimensional
text_index = faiss.IndexFlatL2(512)

#%% function to add embeddings to FAISS indices
def add_to_faiss(image_embeddings, text_embeddings):
    image_index.add(image_embeddings.detach().numpy())
    text_index.add(text_embeddings.detach().numpy())

#%% function to query by text
def query_by_text(text, k=5):
    # Create text embedding for the query
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)
    
    # Search in text index
    distances, indices = text_index.search(text_embeddings.detach().numpy(), k)
    return distances[0], indices[0]

#%% function to query by image
def query_by_image(image, k=5):
    # Create image embedding for the query
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embeddings = clip_model.get_image_features(**inputs)
    
    # Search in image index
    distances, indices = image_index.search(image_embeddings.detach().numpy(), k)
    return distances[0], indices[0]

#%% populate FAISS indices with initial data
if not os.path.exists("image_index.faiss") or not os.path.exists("text_index.faiss"):
    for i in range(100):
        try:    
            current_image = df.iloc[i]["image"]
            current_text = df.iloc[i]["caption"]
            image = Image.open(os.path.join("data/flickr_images", current_image[:-2]))
            image_embeddings, text_embeddings = create_embeddings(current_text, image)
            add_to_faiss(image_embeddings, text_embeddings)
        except Exception as e:
            print(f"Error at {i}, image: {current_image}, error: {str(e)}")
    faiss.write_index(image_index, "image_index.faiss")
    faiss.write_index(text_index, "text_index.faiss")
#%% save the FAISS indices

#%% load the FAISS indices
image_index = faiss.read_index("image_index.faiss")
text_index = faiss.read_index("text_index.faiss")

#%% Example usage
# Query by text
# text_query = "a dog running in the park"
# distances, indices = query_by_text(text_query)
# print("Text query results:", indices)
# # get the text captions for the indices
# captions = df.iloc[indices]["caption"]
# print("Captions:", captions)

# # show the most likely image
# most_likely_image = df["image"].iloc[indices[0]]
# image = Image.open(os.path.join("data/flickr_images", most_likely_image[:-2]))
# image.show()


#%% multimodal RAG
def multimodal_rag(user_query):
    
    # search in text index
    _, indices = query_by_text(user_query)
    
    # get the text captions for the indices
    caption = df.iloc[indices]["caption"]
    print(caption)
    
    # get the most likely image
    most_likely_image = df["image"].iloc[indices[0]]
    image = os.path.join("data/flickr_images", most_likely_image[:-2])
    return image, ", ".join(caption)

#%% test multimodal RAG
# image, caption = multimodal_rag("a dog running in the park")
# print(image)
# print(caption)

#%% check if user wants to get text or image
class UserWantsTextOrImage(BaseModel):
    text: bool = Field(description="Whether the user wants text or image")


# test
# chain.invoke({"user_query": "show me a text of a dog running in the park"})

#%%
st.title("Multimodal RAG")
# add chatbot interface
user_query = st.chat_input("Enter a question")
if user_query:
    st.chat_message("user").markdown(user_query)
    image, caption = multimodal_rag(user_query)
    # check if user wants text or image
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that checks whether the user query refers to a text or image response. return a json with the key 'text' and the value being a boolean. if the user wants a text response, the value should be true, otherwise it should be false. default to true."),
        ("user", "{user_query}"),
    ])
    model = ChatGroq(model_name="llama3-8b-8192", temperature=0.2)
    chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=UserWantsTextOrImage)
    text_bool = chain.invoke({"user_query": user_query})
    
    # show text or image
    if text_bool['text']:
        prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You will receive several descriptions of scenes. Please return which scenes are most relevant to the user query."),
        ("user", "{scene_description}"),
        ])
        model = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        chain = prompt_template | model | StrOutputParser()
        scene_description = chain.invoke({"scene_description": caption})

        st.chat_message("assistant").markdown(scene_description)
    else:
        st.chat_message("assistant").markdown("Here is the image you requested:")
        print(image)
        st.image(image)
