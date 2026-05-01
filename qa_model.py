import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "google/flan-t5-small"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
model.eval()


def generate_answer(question, scene_context):
    context_text = "\n".join(scene_context)

    prompt = f"""
You are a visual reasoning assistant.

You are given a list of detected objects in a scene.

Rules:
- Answer ONLY using the provided context
- Be concise and natural
- Do not repeat descriptions


Scene context:
{context_text}

Question: {question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()