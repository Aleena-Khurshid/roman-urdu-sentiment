"""Gradio demo for Hugging Face Spaces.

Upload to your Space root: this app.py, spaces/requirements.txt (as
requirements.txt) and src/preprocess.py (as preprocess.py).
Then set MODEL_ID below to your Hub model.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text

MODEL_ID = os.getenv("MODEL_ID", "YOUR_HF_USERNAME/roman-urdu-sentiment")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()


@torch.no_grad()
def classify(text: str) -> dict:
    batch = tokenizer(clean_text(text), truncation=True, max_length=128,
                      return_tensors="pt")
    probs = torch.softmax(model(**batch).logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Roman Urdu message"),
    outputs=gr.Label(num_top_classes=3, label="Sentiment"),
    title="Roman Urdu Sentiment Classifier",
    description="Fine-tuned multilingual transformer for Roman Urdu customer messages.",
    examples=[
        ["mera balance bina wajah kat gaya, bohat bura experience"],
        ["delivery time pe aa gayi, shukriya"],
        ["package ka rate kya hai?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
