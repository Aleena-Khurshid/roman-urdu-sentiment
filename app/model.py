"""Loads the fine-tuned model once and runs predictions on CPU."""

import os
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import clean_text

# Either a local folder (models/roman-urdu-sentiment) or a Hugging Face Hub id
# like "your-username/roman-urdu-sentiment". Set it with an environment variable.
MODEL_ID = os.getenv("MODEL_ID", "models/roman-urdu-sentiment")
MAX_LENGTH = 128


class SentimentModel:
    def __init__(self, model_id: str = MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()                         # turn off dropout for inference
        self.id2label = self.model.config.id2label

    @torch.no_grad()                              # no gradients needed = faster, less RAM
    def predict(self, texts: list[str]) -> list[dict]:
        cleaned = [clean_text(t) for t in texts]
        batch = self.tokenizer(cleaned, padding=True, truncation=True,
                               max_length=MAX_LENGTH, return_tensors="pt")
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)     # raw scores -> probabilities

        results = []
        for text, p in zip(texts, probs):
            best = int(p.argmax())
            results.append({
                "text": text,
                "label": self.id2label[best],
                "confidence": round(float(p[best]), 4),
                "scores": {self.id2label[i]: round(float(v), 4)
                           for i, v in enumerate(p)},
            })
        return results


@lru_cache(maxsize=1)                              # load only once, reuse afterwards
def get_model() -> SentimentModel:
    return SentimentModel()
