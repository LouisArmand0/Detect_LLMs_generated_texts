import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def get_log_prob(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    # outputs.loss is mean negative log-likelihood
    return -outputs.loss.item() * inputs["input_ids"].shape[1]  # total log-prob


def perturb_text(text, mask_tokenizer, mask_model, device, mask_ratio=0.15, n_perturbations=20):
    """Randomly mask spans and infill with T5."""
    words = text.split()
    perturbed = []

    for _ in range(n_perturbations):
        masked_words = words.copy()
        n_masks = max(1, int(len(words) * mask_ratio))
        mask_indices = np.random.choice(len(words), n_masks, replace=False)

        sentinel = 0
        for idx in sorted(mask_indices):
            masked_words[idx] = f"<extra_id_{sentinel}>"
            sentinel += 1

        masked_text = " ".join(masked_words)
        inputs = mask_tokenizer(masked_text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            output = mask_model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=True,
                top_p=0.96,
                temperature=1.0
            )

        filled = mask_tokenizer.decode(output[0], skip_special_tokens=True)
        perturbed.append(filled)

    return perturbed


def detectgpt_score(text, score_model, score_tokenizer, mask_model, mask_tokenizer, device, n_perturbations=5):
    original_log_prob = get_log_prob(text, score_model, score_tokenizer, device)
    perturbations = perturb_text(text, mask_tokenizer, mask_model, device, n_perturbations=n_perturbations)
    perturbed_log_probs = [get_log_prob(p, score_model, score_tokenizer, device) for p in perturbations]
    return original_log_prob - np.mean(perturbed_log_probs)  # high → LLM-generated

data = pd.read_csv('src/data/clean_data/kaggle_dataset.csv')

# Keep only N texts per class
N = 200

df = data.groupby("generated").sample(n=N, random_state=42)
texts = df["text"].tolist()
labels = df["generated"].tolist()



# --- Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # use Apple Silicon GPU

# Scoring model: lightweight GPT2 base (117M vs 1.5B for xl)
score_model_name = "gpt2"
score_tokenizer = GPT2TokenizerFast.from_pretrained(score_model_name)
score_model = GPT2LMHeadModel.from_pretrained(score_model_name).to(device)
score_model.eval()

# Perturbation model: t5-small (60M vs 770M for large)
mask_model_name = "t5-small"
mask_tokenizer = T5Tokenizer.from_pretrained(mask_model_name)
mask_model = T5ForConditionalGeneration.from_pretrained(mask_model_name).to(device)
mask_model.eval()

# DetectGPT scores (normalized to [0,1] for fair comparison)
detectgpt_scores = [detectgpt_score(t, score_model, score_tokenizer, mask_model, mask_tokenizer, device) for t in texts]
detectgpt_probs = MinMaxScaler().fit_transform(
    np.array(detectgpt_scores).reshape(-1, 1)
).flatten()
preds = [1 if probs > 0.5 else 0 for probs in detectgpt_probs]

# Evaluate both
print(f"DetectGPT Accuracy: {accuracy_score(labels, preds):.4f}")
print(f"DetectGPT AUROC: {roc_auc_score(labels, preds):.4f}")
