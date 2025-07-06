#%% packages
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import numpy as np
import seaborn as sns

#%% (2) Load the model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", task_type="SEMANTIC_SIMILARITY")
# %% Movie quotes
sentences = [
    "What's that? It's blue light. What does it do? It turns blue.", # Rambo 2
    "That's what she said.", # The Office
    "I'm sorry, Dave. I'm afraid I can't do that.", # 2001: A Space Odyssey
    "I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion. I watched C-beams glitter in the dark near the Tannhäuser Gate. All those moments will be lost in time, like tears in rain.",  # Blade Runner
    "Is it strange to want to be with people?", # Ex Machina
    "This is your last chance. After this, there is no turning back. You take the blue pill - the story ends, you wake up in your bed and believe whatever you want to believe. You take the red pill - you stay in Wonderland, and I show you how deep the rabbit hole goes.", # The Matrix
    "Do. Or do not. There is no try.", # Star Wars
    "Roads? Where we're going, we don't need roads.", # Back to the Future
    "The line must be drawn here! This far, no further!" # Star Trek: The Next Generation
    "I'm the king of the world!", # Titanic
    "You can't handle the truth!", # A Few Good Men
    "Keep your friends close, but your enemies closer.", # The Godfather: Part II
    "I'm going to make him an offer he can't refuse.", # The Godfather
]
# %% (4) Get the embeddings
sentence_embeddings = embeddings.embed_documents(sentences)

# %% (5) Calculate linear correlation matrix for embeddings
sentence_embeddings_corr = np.corrcoef(sentence_embeddings)
# show annotation with one digit
sentences_labels = [sentence[:30] for sentence in sentences]
sns.heatmap(sentence_embeddings_corr, annot=True,
            fmt=".2f",
            xticklabels=sentences_labels, 
            yticklabels=sentences_labels,
            # only show the upper triangle
            mask=np.triu(np.ones(sentence_embeddings_corr.shape))
            )
# %%
