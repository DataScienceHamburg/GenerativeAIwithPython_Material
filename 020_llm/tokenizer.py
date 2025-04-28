#%% packages
from transformers import AutoTokenizer

#%%
 
def show_tokens(sentence: str, tokenizer_name: str):
    """ Show the tokens each separated by a different color """
    # Using ANSI color codes for background colors
    colors = [
        '\033[41m',  # Red background
        '\033[42m',  # Green background 
        '\033[43m',  # Yellow background
        '\033[44m',  # Blue background
        '\033[45m',  # Magenta background
        '\033[46m'   # Cyan background
    ]
    reset = '\033[0m'  # Reset color code

    # Load the tokenizer and tokenize the input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids

    # Extract vocabulary length  
    print(f"Vocab length: {len(tokenizer)}")

    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            colors[idx % len(colors)] +
            tokenizer.decode(t) +
            reset,
            end=' '
        )
#%%
text = """
Ich heiße Bert.
🎵 鸟
12.0*50=600
"""
 
show_tokens(text, "bert-base-cased")
#%% 
# show_tokens(text, "gpt2")
 
# show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")

