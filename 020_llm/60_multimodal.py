#%%
from groq import Groq
import base64
#%% Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#%% Encode an image
image_path = "../../data/BertFighterJet.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

#%% 
client = Groq()
result = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="llama-3.2-11b-vision-preview",
)
#%% show result
print(result.choices[0].message.content)
# %%
