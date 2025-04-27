#%% packages
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from datasets import load_dataset
import torch
# %% load dataset
dataset = load_dataset("squad", split="train[:10]")
# %%
len(dataset)
# %%
# Initialize the smaller model (RAG Drafter)
drafter_model_name = "distilbert-base-uncased-distilled-squad"
drafter_model = AutoModelForQuestionAnswering.from_pretrained(drafter_model_name)
drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_model_name)

# Initialize the larger model (RAG Verifier)
verifier_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
verifier_model = AutoModelForQuestionAnswering.from_pretrained(verifier_model_name)
verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_name)

# Set up pipelines
drafter_pipeline = pipeline("question-answering", model=drafter_model, tokenizer=drafter_tokenizer)
verifier_pipeline = pipeline("question-answering", model=verifier_model, tokenizer=verifier_tokenizer)
# %%
def generate_drafts(question, context, num_drafts=3):
   drafts = []
   for _ in range(num_drafts):
       draft = drafter_pipeline(question=question, context=context)
       drafts.append(draft)
   return drafts
# %%
def verify_drafts(question, context, drafts):
    best_draft = None
    highest_score = -float('inf')

    inputs = verifier_tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs['offset_mapping'][0]
    input_ids = inputs['input_ids'].to(verifier_model.device)

    for draft in drafts:
        start_char = draft['start']
        end_char = draft['end']

        start_index = None
        end_index = None

        for idx, (start, end) in enumerate(offset_mapping):
            if start_index is None and start_char >= start and start_char < end:
                start_index = idx
            if end_index is None and end_char > start and end_char <= end:
                end_index = idx
            if start_index is not None and end_index is not None:
                break

        if start_index is None or end_index is None or start_index >= input_ids.size(1) or end_index >= input_ids.size(1):
            print(f"Draft skipped: Out of bounds or no matching tokens. Start: {start_char}, End: {end_char}, Start Index: {start_index}, End Index: {end_index}, Context Length: {len(context)}")
            continue

        with torch.no_grad():
            outputs = verifier_model(input_ids=input_ids)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            score = start_logits[0, start_index].item() + end_logits[0, end_index].item()

        if score > highest_score:
            highest_score = score
            best_draft = draft

    if best_draft is None:
        print("No valid draft found after verification.")

    return best_draft
# %%
correct = 0
total = 3  # Evaluate on 10 samples for simplicity
highest_score = 0
for i in range(total):
    sample = dataset[i]
    question = sample['question']
    context = sample['context']
    drafts = generate_drafts(question, context)
    print(f"Drafts: {drafts}")
    best_answer = verify_drafts(question, context, drafts)

    print(f"Q: {question}")

    if best_answer is not None:
        print(f"Ground truth: {sample['answers']['text'][0]}")
        print(f"A: {best_answer['answer']} (Score: {highest_score:.4f})\n")
        if best_answer['answer'].lower() in sample['answers']['text'][0].lower():
            correct += 1
    else:
        print("No valid draft found.\n")

accuracy = correct / total * 100
print(f"Accuracy: {accuracy}%")
# %%
# %%
