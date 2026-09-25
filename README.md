# Roman Urdu Sentiment Classifier

Classifies Roman Urdu customer messages (for example *"mera balance bina wajah kat gaya"*) as **negative**, **neutral** or **positive**, so support teams at telecom and fintech companies can prioritise angry customers automatically.

**Live demo:** `<your Hugging Face Space link>`  
**Model on Hugging Face Hub:** `<your model link>`

!\[Demo screenshot](assets/demo.png)

## Why this problem

Millions of Pakistani users write in Roman Urdu: Urdu in English letters, with no fixed spelling (`bohat`, `bahut`, `bht`). Most sentiment models are trained on English and fail here.v## Results (held-out test set)


## Results (held-out test set)

| Model | Accuracy | Macro F1 | Training time |

|---|---|---|---|

| TF-IDF (word + char n-grams) + Logistic Regression | \*\*0.678\*\* | \*\*0.673\*\* | `\~x s` (CPU) |

| DistilBERT-multilingual, fine-tuned | 0.644 | 0.628 | `\~x min` (Colab T4) |


\*\*Key finding:\*\* the classical baseline outperformed the fine-tuned transformer.
---

## Likely reasons: multilingual BERT's subword vocabulary was learned mostly from

## Urdu script, not Roman Urdu, so non-standard spellings ("nhi", "kha") get split

## into weak subwords, while character n-grams capture spelling variants directly.

## Label noise in the dataset also limits how much a larger model can gain.

The baseline is also \~100x smaller and much faster on CPU.

---

!\[Confusion matrix](assets/transformer\_confusion.png)

**Why macro F1?** It averages F1 across the three classes equally, so a model cannot look good by only predicting the largest class.

## Approach

1. **Data:** Roman Urdu Data Set (Sharf, 2017, UCI, CC BY 4.0), about 20k labelled sentences.
2. **Cleaning:** lowercase, remove URLs and mentions, squash stretched letters (`bohtttt → bohtt`), fix label typos, and drop sentences that appear with conflicting labels.
3. **Split:** 80/10/10 stratified train/validation/test. The test set is used once, at the end.
4. **Baseline:** TF-IDF with word n-grams and character n-grams (character n-grams handle spelling variants) + Logistic Regression.
5. **Transformer:** fine-tuned `distilbert-base-multilingual-cased` for 3 epochs, keeping the epoch with best validation macro F1.
6. **Serving:** FastAPI with input validation, model loaded once at startup, CPU inference.

## Error analysis

<!-- Write 3–4 honest sentences from YOUR notebook output (section 8). E.g. which classes get confused most, and examples of likely label noise. -->

## Project structure

```
roman-urdu-sentiment/
├── notebooks/train\_colab.ipynb   # data, baseline, fine-tuning, evaluation (run in Colab)
├── src/preprocess.py             # text cleaning shared by training and serving
├── app/
│   ├── main.py                   # FastAPI routes
│   ├── model.py                  # loads model, runs prediction
│   └── schemas.py                # request/response validation
├── spaces/                       # Gradio demo for Hugging Face Spaces
├── tests/                        # pytest tests (fake model, run in seconds)
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate                 # Windows  (Linux/Mac: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Use the model from Hugging Face Hub (or put the downloaded folder in models/roman-urdu-sentiment)
set MODEL\_ID=your-username/roman-urdu-sentiment     # Windows CMD
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \\
     -d "{\\"text\\": \\"yar net bohat slow hai\\"}"
```

## Tests

```bash
pytest -q
```

## Limitations

* Labels in the source dataset are noisy; some "mistakes" are actually questionable labels.
* The data comes from general reviews and social media, not real telecom support tickets.
* Sarcasm and mixed Urdu-script/Roman text are not handled well.

## Future work

* Try `xlm-roberta-base` and compare.
* Docker image and cloud deployment.
* Add an "urgency" label for complaint triage.

## Acknowledgements

Dataset: Z. Sharf, *Roman Urdu Data Set*, UCI Machine Learning Repository, https://doi.org/10.24432/C58325 (CC BY 4.0).

