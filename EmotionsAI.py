'''from pyplutchik import plutchik
import matplotlib.pyplot as plt

emotions = {
    "joy": 1,
    
    "fear": 0.2,
    "surprise": 0.3,
    "sadness": 0.7,
    "disgust": 0.1,
    "anger": 0.8,
    
}

plutchik(emotions)
plt.show()  # <-- this line makes the window pop up!
'''
'''import pandas as pd

# Choose your language code (e.g., 'arq', 'hin', 'eng' — only 10 have intensities)
lang = 'eng'

# File paths
categories_base = "hf://datasets/brighter-dataset/BRIGHTER-emotion-categories/"
intensities_base = "hf://datasets/brighter-dataset/BRIGHTER-emotion-intensities/"

splits = {
    'train': f"{lang}/train-00000-of-00001.parquet",
    'dev': f"{lang}/dev-00000-of-00001.parquet",
    'test': f"{lang}/test-00000-of-00001.parquet"
}

# Load both category and intensity splits
cat_df = pd.read_parquet(categories_base + splits["train"])
int_df = pd.read_parquet(intensities_base + splits["train"])

# Peek at both to check structure
print("Categories sample:\n", cat_df.head())
print("Intensities sample:\n", int_df.head())'''

# Required installations:
# pip install openai pandas scikit-learn scipy pyarrow fsspec huggingface_hub

import os
import openai
import pandas as pd
import json
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

# Set your OpenAI API key
openai.api_key = "sk-..."  # Replace with your GPT-4 key

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Few-shot examples with CoT-style reasoning
FEW_SHOT_EXAMPLES = """
Example 1:
Text: "I finally got the job I’ve been dreaming about!"
Reasoning: The speaker expresses excitement and satisfaction. This suggests joy and surprise.
Emotions: ["joy", "surprise"]
Intensity: {"joy": 3, "surprise": 2}

Example 2:
Text: "She betrayed me after everything we went through."
Reasoning: Betrayal often evokes anger and sadness.
Emotions: ["anger", "sadness"]
Intensity: {"anger": 2, "sadness": 3}

Example 3:
Text: "The room was silent. I could feel my heart pounding faster with each second."
Reasoning: This situation evokes tension and uncertainty, associated with fear.
Emotions: ["fear"]
Intensity: {"fear": 2}
"""

# Prompt with few-shot + step-by-step
PROMPT_TEMPLATE = FEW_SHOT_EXAMPLES + """

Now analyze the following:
Text: "{text}"
Think step by step and explain your reasoning. Then output your final answer in this format:
Emotions: ["emotion1", "emotion2", ...]
Intensity: {"emotion1": x, "emotion2": x, ...}
"""
'''
# Parse model response
def parse_response(response_text):
    try:
        lines = response_text.strip().splitlines()
        emotions = json.loads(lines[-2].split(":", 1)[1].strip())
        intensity = json.loads(lines[-1].split(":", 1)[1].strip())
        return emotions, intensity
    except Exception as e:
        return None, None

# Run multiple prompts and aggregate (self-consistency)
def self_consistent_prediction(text, num_samples=5, model="gpt-4"):
    results = []
    for _ in range(num_samples):
        prompt = PROMPT_TEMPLATE.format(text=text)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            content = response['choices'][0]['message']['content']
            emotions, intensity = parse_response(content)
            if emotions and intensity:
                results.append((emotions, intensity))
        except Exception as e:
            print(f"OpenAI error: {e}")
    return aggregate_predictions(results)

# Aggregate responses using majority vote (labels) and mean (intensity)
def aggregate_predictions(results):
    emotion_votes = Counter()
    intensity_scores = defaultdict(list)

    for emotions, intensity in results:
        for e in emotions:
            emotion_votes[e] += 1
            intensity_scores[e].append(intensity.get(e, 0))

    total = len(results)
    final_emotions = [e for e, count in emotion_votes.items() if count >= total / 2]
    final_intensity = {e: round(sum(intensity_scores[e]) / len(intensity_scores[e]), 2)
                       for e in final_emotions}
    return final_emotions, final_intensity

# Load BRIGHTER data from Hugging Face
def load_brighter_data(lang='arq', split='test'):
    cat_path = f"hf://datasets/brighter-dataset/BRIGHTER-emotion-categories/{lang}/{split}-00000-of-00001.parquet"
    int_path = f"hf://datasets/brighter-dataset/BRIGHTER-emotion-intensities/{lang}/{split}-00000-of-00001.parquet"
    cat_df = pd.read_parquet(cat_path)
    int_df = pd.read_parquet(int_path)
    merged = pd.merge(cat_df, int_df, on="text", suffixes=("_cat", "_int"))
    return merged

# Convert label list to binary vector
def labels_to_vec(label_list):
    return [1 if e in label_list else 0 for e in EMOTIONS]

# Evaluation
def evaluate(predictions, golds, pred_intensities, gold_intensities):
    pred_labels = [labels_to_vec(p) for p in predictions]
    gold_labels = [labels_to_vec(g) for g in golds]

    # Multi-label F1
    f1 = f1_score(gold_labels, pred_labels, average="macro")

    # Pearson per emotion
    pearson_scores = {}
    for e in EMOTIONS:
        gold_vals = [gi.get(e, 0) for gi in gold_intensities]
        pred_vals = [pi.get(e, 0) for pi in pred_intensities]
        if any(gold_vals):
            r, _ = pearsonr(gold_vals, pred_vals)
            pearson_scores[e] = round(r, 3)
        else:
            pearson_scores[e] = "N/A"

    return f1, pearson_scores

# Main pipeline
if __name__ == "__main__":
    lang = "arq"  # any of the 10 BRIGHTER languages with intensity labels
    df = load_brighter_data(lang=lang, split="test")

    sample_size = 10  # Small batch for quick testing — increase this
    predictions = []
    golds = []
    pred_intensities = []
    gold_intensities = []

    for _, row in df.sample(sample_size, random_state=42).iterrows():
        text = row["text"]
        gold_labels = row["labels_cat"]
        gold_scores = row["labels_int"]

        pred_labels, pred_scores = self_consistent_prediction(text, num_samples=5)

        predictions.append(pred_labels)
        pred_intensities.append(pred_scores)
        golds.append(gold_labels)
        gold_intensities.append(gold_scores)

        print(f"\n📝 Text: {text}")
        print(f"✅ Gold: {gold_labels} — {gold_scores}")
        print(f"🤖 Pred: {pred_labels} — {pred_scores}")

    # Final evaluation
    f1_macro, pearson = evaluate(predictions, golds, pred_intensities, gold_intensities)
    print("\n📊 F1 (macro):", round(f1_macro, 4))
    print("📈 Pearson per emotion:", pearson)'''
