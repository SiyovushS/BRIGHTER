#!/usr/bin/env python
import os
import csv
import json
import argparse
import re
from typing import List
import wandb
import random
from openai import AzureOpenAI
import pathlib
import socket
from collections import defaultdict
wandb.init(mode="disabled")
import glob
import time

from sklearn.metrics import f1_score
from scipy.stats import pearsonr

# Import vLLM components




# Azure OpenAI Setup
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"  # Set this to your Azure deployment name
AZURE_OPENAI_VERSION = "2025-01-01-preview"


###########################################################
# GLOBAL SETTINGS
###########################################################
ALL_LANGUAGES = [
    "xho", "zul", "afr", "amh", "arq", "ary", "chn", "deu", "eng", "esp", "hau", "hin",
    "ibo", "ind", "jav", "kin", "mar", "orm", "pcm", "ptbr", "ptmz", "rus", "ron", "som", "sun",
    "swa", "swe", "tat", "tir", "ukr", "vmw", "yor"
]

FULL_ABLATION_LANGUAGES = [
    "eng",    # English
    # "ptbr",   # Portuguese-BR
    # "deu",    # German
]

NATIVE_PROMPT_ABLATION_LANGUAGES = [
    "eng",    # English
    "ptbr",   # Portuguese-BR
    "vmw",    # Emakhuwa
    "deu",    # German
    "arq",    # Algerian Arabic
    "chn",    # Chinese
    "ukr",    # Ukrainian
    "hin",    # Hindi
    "mar",    # Marathi
    "hau",    # Hausa
    "zul",    # Zulu
    "swe",    # Swedish
]

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Default "main" config
main_config = {
    "variant": "v2",
    "n_shot": 0,
    "top_k": 1
}

# Ablation settings (if used)
few_shot_counts = [0, 1, 2, 4, 8, 16]
top_k_list = [1, 2, 4, 8]

# Where test CSVs live
TEST_DIRS = {
    "binary":    "./track_a/test",
    "intensity": "./track_b/test"
}

LANG_NATIVE_PROMPTS = {
    "eng": (
        "Evaluate whether the author of the following text conveys the emotion {{EMOTION}}.\n"
        "Think step by step before you answer. Finish your response with 'Therefore, my answer is ' "
        "followed by 'yes' or 'no'."
    ),
    "ptbr": (
        "Avalie se o autor do texto a seguir transmite a emoção {{EMOTION}}.\n"
        "Pense passo a passo antes de responder. Termine sua resposta com "
        "'Portanto, minha resposta é ' seguida por 'yes' ou 'no'."
    ),
    "vmw": (
        "Muthokorerye akhala wira ole olempe yoolepa ela, owiiriha atthu wummwo {{EMOTION}}.\n"
        "Muupuwelele vakhaani-vakhaani muhinatthi waakhula nikoho. Mmalihe waakhula wanyu ni masu ala "
        "\"Nto waakhula waka ori \"ottharelanaka ni\" ayo\" wala \"nnakhala nnaari."
    ),
    "arq": (
        "يرجى منك أن تقيِّم إن كان مؤلف النص التالي يشعر ب{{EMOTION}}.\n"
        "فكِّر خطوة بخطوة قبل الإجابة. أنهي إجابتك بـ \"لذلك فإن إجابتي هي \" متبوعة بـ \"yes\" أو \"no\"."
    ),
    "chn": (
        "请评估以下文本的作者是否表达了情感{{EMOTION}}。\n"
        "回答前请一步步思考，并以以下内容结束："
        "“因此，我的答案是”后接“yes”或“no”."
    ),
    "ukr": (
        "Оціни, чи передає автор наступного тексту емоцію {{EMOTION}}.\n"
        "Думай крок за кроком, перш ніж відповідати. Закінчи відповідь словами «Отже, моя відповідь» з наступним "
        "«yes» або «no»."
    ),
    "hin": (
        "मूल्यांकन करें कि क्या निम्नलिखित पाठ का लेखक {{EMOTION}} भावना को व्यक्त करता है।\n"
        "उत्तर देने से पहले चरण दर चरण सोचें। अपना उत्तर \"इसलिए, मेरा उत्तर \" के बाद \"yes\" या \"no\" लिखें."
    ),
    "mar": (
        "खालील मजकुराचा लेखक {{EMOTION}} भावना व्यक्त करतो का याचे मूल्यांकन करा.\n"
        "उत्तर देण्यापूर्वी टप्प्याटप्प्याने विचार करा. तुमचे उत्तर लिहा \"तर, माझे उत्तर\" आणि नंतर \"yes\" किंवा \"no\" लिहा."
    ),
    "deu": (
        "Beurteile, ob der Autor des folgenden Textes die Emotion {{EMOTION}} vermittelt.\n"
        "Denk Schritt für Schritt, bevor du antwortest. Beende deine Aussage mit "
        "\"Die finale Antwort ist \" gefolgt von \"yes\" oder \"no\"."
    ),
    "hau": (
        "ƙimanta shin ko marubucin wannan rubutun yana nufin motsin ran {{EMOTION}}. "
        "Yi nazari daki da daki kafin ka bada answer. Kammala amsar da zaka bani da "
        "\"Saboda haka, amsa ta ita ce \" biye da \"yes\" ko \"no\"."
    ),
    "zul": (
        "Hlola ukuthi umbhali wombhalo olandelayo uyawudlulisela yini umzwelo {{EMOTION}}."
    ),
    "swe": (
        "Bedöm om författaren till följande text förmedlar känslan {{EMOTION}}.\n"
        "Tänk steg för steg innan du svarar. Avsluta ditt svar med "
        "\"Därför är mitt svar\" följt av \"yes\" eller \"no\"."
    )
}

# New TASK_CONFIGS dict without stored few-shot examples
TASK_CONFIGS = {
    "binary": {
        "prompt_variants": {
            "v1": (
                "Evaluate whether the following text conveys the emotion of {{EMOTION}}.\n"
                "Think step by step before you answer. Finish your response with 'Answer:' followed by 'yes' or 'no'."
            ),
            "v2": (
                "Analyze the text below for the presence of {{EMOTION}}.\n"
                "Explain your reasoning briefly and conclude with 'Answer:' followed by either 'yes' or 'no'."
            ),
            "v3": (
                "Examine the following text to determine whether {{EMOTION}} is present.\n"
                "Provide a concise explanation for your assessment and end with 'Answer:' followed by either 'yes' or 'no'."
            ),
        },
    },
    "intensity": {
        "prompt_variants": {
            "v1": (
                "Determine the intensity (0: none, 1: low, 2: medium, 3: high) of {{EMOTION}} in the text.\n"
                "Provide reasoning and end with 'Answer:' followed by the intensity score (0..3)."
            ),
            "v2": (
                "Assess the level of {{EMOTION}} in the text below (0=none 1=low 2=medium 3=high).\n"
                "Summarize your reasoning and conclude with 'Answer:' followed by the correct number."
            ),
            "v3": (
                "Evaluate the intensity of {{EMOTION}} in the following text on a scale from 0 (none) to 3 (high).\n"
                "Provide a brief explanation for your assessment and conclude with 'Answer:' followed by the rating."
            ),
        },
    }
}


###########################################################
# DATA LOADER
###########################################################
def load_test_data_multicolumn(filepath: str, task: str) -> List[dict]:
    data_expanded = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Warning: No headers found in {filepath}")
                return data_expanded
            available_emotions = [emo for emo in EMOTIONS if emo in reader.fieldnames]
            for row in reader:
                text_val = row.get("text", "").strip()
                if not text_val:
                    continue  # Skip empty texts
                for emo in available_emotions:
                    val_str = row.get(emo, "").strip()
                    try:
                        numeric_val = int(float(val_str)) if val_str else 0
                    except ValueError:
                        numeric_val = 0
                    if task == "binary":
                        label_val = 1 if numeric_val == 1 else 0
                    else:
                        label_val = max(0, min(3, numeric_val))
                    data_expanded.append({
                        "text": text_val,
                        "emotion": emo,
                        "label": label_val
                    })
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
    return data_expanded

###########################################################
# PROMPT, GENERATION, PARSING
###########################################################
def construct_prompt(prompt_template: str,
                     few_shot_examples: List[dict],
                     input_text: str,
                     emotion: str,
                     task: str) -> str:
    """
    Formats few-shot examples more clearly before inserting them into the prompt.
    """
    
    out = ""

    out += "### Task ###\n"
    
    desc = prompt_template.replace("{{EMOTION}}", emotion)
    out += desc

    if few_shot_examples:
        out += "\n\n### Examples ###\n"
        for i, ex in enumerate(few_shot_examples, 1):
            if task == "intensity":
                out += f"Example {i}:\nInput: {ex['input']}\nAnswer: {ex['label']}\n\n"
            else:
                out += f"Example {i}:\nInput: {ex['input']}\nAnswer: {'yes' if ex['label'] == 1 else 'no'}\n\n"

    out += "\n### Your Turn ###\n"
    out += "Input: " + input_text + "\n"

    return out

def robust_parse_binary(generated_text: str) -> int:
    """
    Extracts the answer from the model output for binary classification.
    Looks for the exact phrase 'Answer: yes' or 'Answer: no' and returns 1 or 0.
    """
    match = re.search(r"Answer:\s*(yes|no)", generated_text, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == "yes" else 0
    print(f"Warning: Could not parse binary answer from: {generated_text}")
    return None


def robust_parse_intensity(generated_text: str) -> int:
    """
    Extracts the answer from the model output for intensity classification.
    Looks for 'Answer: 0', 'Answer: 1', 'Answer: 2', or 'Answer: 3' exactly.
    """
    match = re.search(r"Answer:\s*([0-3])", generated_text)
    if match:
        return int(match.group(1))
    print(f"Warning: Could not parse intensity answer from: {generated_text}")
    return None

def parse_output(generated_text: str, task: str) -> int:
    if task == "binary":
        return robust_parse_binary(generated_text)
    else:
        return robust_parse_intensity(generated_text)

###########################################################
# EVALUATION
###########################################################
def evaluate_model_on_test_set(
    engine,
    test_data: List[dict],
    prompt_template: str,
    task: str,
    top_k: int,
    n_shot: int,
    model_name: str,
    out_json: str
) -> dict:
    # ------------------------------------------------------
    # 1) Sample few-shot examples from test_data
    # ------------------------------------------------------
    # Update this section in evaluate_model_on_test_set function:
    if n_shot > 0:
        # Group data by emotion AND label
        emotion_label_samples = defaultdict(lambda: defaultdict(list))
        for d in test_data:
            emotion_label_samples[d["emotion"]][d["label"]].append(d["text"])

        if task == "binary":
            # For each sample, get examples for its specific emotion
            few_shot_examples_by_emotion = {}
            for emotion in set(d["emotion"] for d in test_data):
                if n_shot >= 2 and (n_shot % 2 == 0):
                    # Even distribution between label=0 and label=1 for this emotion
                    half = n_shot // 2
                    examples = []
                    for label in [0, 1]:
                        samples = emotion_label_samples[emotion][label]
                        if len(samples) >= half:
                            chosen = random.sample(samples, half)
                        else:
                            # If not enough samples, take all available and supplement from other label
                            chosen = samples
                            remaining = half - len(chosen)
                            other_label = 1 if label == 0 else 0
                            if len(emotion_label_samples[emotion][other_label]) >= remaining:
                                chosen.extend(random.sample(emotion_label_samples[emotion][other_label], remaining))
                        for txt in chosen:
                            examples.append({
                                "input": txt,
                                "label": label
                            })
                    few_shot_examples_by_emotion[emotion] = examples
                else:
                    # For odd n_shot or <2, still try to maintain balance
                    examples = []
                    all_samples = [(txt, label) 
                                 for label in [0, 1] 
                                 for txt in emotion_label_samples[emotion][label]]
                    random.shuffle(all_samples)
                    used = set()
                    for txt, label in all_samples:
                        if txt not in used and len(examples) < n_shot:
                            examples.append({
                                "input": txt,
                                "label": label
                            })
                            used.add(txt)
                    few_shot_examples_by_emotion[emotion] = examples

        else:  # task == "intensity"
            few_shot_examples_by_emotion = {}
            for emotion in set(d["emotion"] for d in test_data):
                if n_shot >= 4 and (n_shot % 4 == 0):
                    # Even distribution among 0..3 for this emotion
                    portion = n_shot // 4
                    examples = []
                    for label in range(4):
                        samples = emotion_label_samples[emotion][label]
                        if len(samples) >= portion:
                            chosen = random.sample(samples, portion)
                        else:
                            # If not enough samples for this intensity, take what we have
                            chosen = samples
                            # Could add logic here to supplement from nearby intensities
                        for txt in chosen:
                            examples.append({
                                "input": txt,
                                "label": label
                            })
                    few_shot_examples_by_emotion[emotion] = examples
                else:
                    # For n_shot not divisible by 4, try to maintain rough balance
                    examples = []
                    all_samples = [(txt, label) 
                                 for label in range(4) 
                                 for txt in emotion_label_samples[emotion][label]]
                    random.shuffle(all_samples)
                    used = set()
                    for txt, label in all_samples:
                        if txt not in used and len(examples) < n_shot:
                            examples.append({
                                "input": txt,
                                "label": label
                            })
                            used.add(txt)
                    few_shot_examples_by_emotion[emotion] = examples

    else:
        # 0-shot case
        few_shot_examples_by_emotion = {emotion: [] for emotion in set(d["emotion"] for d in test_data)}

    # Update the prompt construction to use emotion-specific examples
    prompts = [
        construct_prompt(
            prompt_template, 
            few_shot_examples_by_emotion[sample["emotion"]], 
            sample["text"], 
            sample["emotion"],
            task
        )
        for sample in test_data
    ]

    print(f"  Running evaluation for {len(prompts)} samples ...")
    if len(prompts) > 0:
        print(f"  Example prompt:\n{prompts[0]}\n---")
    if model_name == "openai/gpt-4":
        from openai import AzureOpenAI

        print("Using Azure GPT-4.1...")

        flagged_prompts = []

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2025-01-01-preview",
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/deployments/gpt-4.1"
        )
        generation_results = []
        parsed_outputs = []
        raw_outputs = []
        MAX_RETRIES = 5
        for i, prompt in enumerate(prompts):
            output = None
            parsed = None
            text = ""
            attempt = 1

            while attempt <= MAX_RETRIES:
                try:
                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [{"type": "text", "text": prompt}]}
                        ],
                        max_tokens=80
                    )
                    text = (response.choices[0].message.content or "")
                    parsed = parse_output(text, task)
                    if parsed is not None:
                        output = text.strip()
                        break

                    print(f"[RETRY] Prompt #{i} gave unparseable output: {repr(text)} — retrying…")
                    time.sleep(1)  # avoid hammering
                except Exception as e:
                    err = str(e)

                    if "content management policy" in err or "ResponsibleAIPolicyViolation" in err:
                        print(f"[FLAGGED] Prompt #{i} violated policy: {err}")
                        flagged_prompts.append({"index": i, "prompt": prompt, "error": err})
                        break

                    if isinstance(e, (socket.gaierror, ConnectionError)) or "Temporary failure in name resolution" in err:
                        print(f"[OFFLINE] Prompt #{i} failed due to no internet. Waiting and retrying...")
                        time.sleep(10)
                        continue

                    print(f"[ERROR] Prompt #{i} error: {err} — retrying (attempt {attempt}/{MAX_RETRIES})")
                    attempt += 1

            if output is None:
                print(f"[SKIPPED] Prompt #{i} failed after {MAX_RETRIES} attempts.")
                raw_outputs.append("")
                parsed_outputs.append(None)
                flagged_prompts.append({
                    "index": i,
                    "prompt": prompt,
                    "reason": f"Max retries reached with unparseable output after {MAX_RETRIES} attempts.",
                    "last_output": text.strip() if isinstance(text, str) else "Unknown"
                })
                continue

            print(f"\n--- Prompt #{i} ---")
            print(f"Prompt:\n{prompt}\nOutput:\n{output}\n")
            raw_outputs.append(output)
            parsed_outputs.append(parsed)
    elif model_name == "google/gemini-2F":
        import requests

        print("Using Google gemini-2F...")

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
        GEMINI_API_ENDPOINT = os.getenv("GEMINI_API_ENDPOINT")

        headers = {
            "Content-Type": "application/json"
        }
        BASE_BACKOFF = 5
        MAX_BACKOFF= 300
        flagged_prompts = []
        raw_outputs = []
        parsed_outputs = []
        MAX_RETRIES = 5

        for i, prompt in enumerate(prompts):
            output = None
            parsed = None

            for attempt in range(1, MAX_RETRIES + 1):
                payload = {
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user"
                    }]
                }
                try:
                    response = requests.post(
                        GEMINI_API_ENDPOINT,
                        headers=headers,
                        json=payload
                    )
                    print(f"[DEBUG] Prompt #{i} Attempt {attempt} → HTTP {response.status_code}")
                    print(f"[DEBUG] Full response body:\n{response.text}\n")
                    response.raise_for_status()

                    text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                    print(f"[DEBUG] Raw model text for prompt #{i}:\n{text!r}\n")
                    parsed = parse_output(text, task)
                    if parsed is not None:
                        output = text
                        break
                    print(f"[RETRY] Prompt #{i} unparseable: {repr(text)} — retrying…")

                except requests.exceptions.HTTPError as e:
                    code = response.status_code
                    # Retry on rate-limit or service-unavailable
                    if code in (429, 503):
                        backoff = min(BASE_BACKOFF * 2**(attempt - 1), MAX_BACKOFF)
                        sleep_time = random.uniform(0, backoff)
                        print(f"[{code}] attempt {attempt}/{MAX_RETRIES}, sleeping {sleep_time:.1f}s…")
                        time.sleep(sleep_time)
                        continue
                    # Policy violations: stop retrying this prompt
                    if "content policy" in str(e).lower():
                        flagged_prompts.append({
                            "index": i,
                            "prompt": prompt,
                            "error": str(e)
                        })
                        break
                    # Other HTTP errors: treat as fatal
                    print(f"[ERROR] HTTP {code} for prompt #{i}: {e}")
                    break

                except (requests.exceptions.ConnectionError, socket.gaierror) as e:
                    # Transient network error: small constant backoff
                    print(f"[OFFLINE] {e}, sleeping {BASE_BACKOFF}s…")
                    time.sleep(BASE_BACKOFF)
                    continue

            # After the retry loop
            if output is None:
                print(f"[SKIPPED] Prompt #{i} failed after {MAX_RETRIES} attempts.")
                flagged_prompts.append({
                    "index": i,
                    "prompt": prompt,
                    "reason": f"Max retries reached after {MAX_RETRIES} attempts"
                })
                raw_outputs.append("")
                parsed_outputs.append(None)
            else:
                raw_outputs.append(output)
                parsed_outputs.append(parsed)
    # ------------------------------------------------------
    # 3) Generate predictions with vLLM
    # ------------------------------------------------------
    else:
        max_tokens = 80
        if "DeepSeek" in model_name:
            print("Using DeepSeek model, increasing max_tokens from 80 to 1024")
            max_tokens = 1024
        if top_k <= 1:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=0.95,
                n=top_k
            )
        else:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                n=top_k
            )
        generation_results = engine.generate(prompts, sampling_params)

    # ------------------------------------------------------
    # 4) Collect + parse predictions, compute metrics
    # ------------------------------------------------------
    emotion2refs = defaultdict(list)
    emotion2preds = defaultdict(list)

    for i, (sample, prompt) in enumerate(zip(test_data, prompts)):
        gold_label = sample["label"]
        e = sample["emotion"]
        
        if i >= len(parsed_outputs):  # safety check
            print(f"[WARNING] Missing parsed output for prompt #{i}, skipping.")
            continue

        parsed = parsed_outputs[i]

        if parsed is None:
            print(f"[SKIPPED] Prompt #{i} had unparseable output.")
            continue

        emotion2refs[e].append(int(gold_label))
        emotion2preds[e].append(int(parsed))

        

    if task == "binary":
        # F1 computation
        f1_per_emotion = {}
        for emo in EMOTIONS:
            if emo in emotion2refs:
                refs = emotion2refs[emo]
                preds = emotion2preds[emo]
                # Print a few examples of prediction and reference
                if len(refs) > 0:
                    print(f"  {emo} refs: {refs[:3]} vs. preds: {preds[:3]}")
                else:
                    print("No refs for", emo)

                f1_val = f1_score(refs, preds, average="binary", zero_division=0)
                f1_per_emotion[emo] = f1_val
        macro_f1 = (
            sum(f1_per_emotion.values()) / len(f1_per_emotion)
            if len(f1_per_emotion) > 0 else 0.0
        )
        if flagged_prompts:
            out_path = pathlib.Path(out_json)
            flagged_filename = out_path.with_name(out_path.stem + "_flagged.json")
            os.makedirs(os.path.dirname(flagged_filename), exist_ok=True)
            with open(flagged_filename, "w", encoding="utf-8") as f:
                json.dump(flagged_prompts, f, indent=2)

        out_path = pathlib.Path(out_json)
        csv_path = out_path.with_name(out_path.stem + "_predictions.csv")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_index", "prompt", "output", "parsed_label", "gold_label", "emotion"])
            for i, (sample, prompt) in enumerate(zip(test_data, prompts)):
                output = raw_outputs[i] if i < len(raw_outputs) else ""
                parsed = parsed_outputs[i] if i < len(parsed_outputs) else ""
                label = sample["label"]
                emotion = sample["emotion"]
                writer.writerow([i, prompt, output, parsed, label, emotion])
        return {"f1_per_emotion": f1_per_emotion, "macro_f1": macro_f1}
    else:
        # Pearson computation
        pearson_per_emotion = {}
        for emo in EMOTIONS:
            if emo in emotion2refs:
                refs = emotion2refs[emo]
                preds = emotion2preds[emo]
                # Print a few examples of prediction and reference
                if len(refs) > 0:
                    print(f"  {emo} refs: {refs[:3]} vs. preds: {preds[:3]}")
                else:
                    print("No refs for", emo)
                
                r_val = pearsonr(refs, preds)[0]
                pearson_per_emotion[emo] = r_val
        avg_pearson = (
            sum(pearson_per_emotion.values()) / len(pearson_per_emotion)
            if len(pearson_per_emotion) > 0 else 0.0
        )
        if flagged_prompts:
            out_path = pathlib.Path(out_json)
            flagged_filename = out_path.with_name(out_path.stem + "_flagged.json")
            os.makedirs(os.path.dirname(flagged_filename), exist_ok=True)
            with open(flagged_filename, "w", encoding="utf-8") as f:
                json.dump(flagged_prompts, f, indent=2)
        out_path = pathlib.Path(out_json)
        csv_path = out_path.with_name(out_path.stem + "_predictions.csv")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_index", "prompt", "output", "parsed_label", "gold_label", "emotion"])
            for i, (sample, prompt) in enumerate(zip(test_data, prompts)):
                output = raw_outputs[i] if i < len(raw_outputs) else ""
                parsed = parsed_outputs[i] if i < len(parsed_outputs) else ""
                label = sample["label"]
                emotion = sample["emotion"]
                writer.writerow([i, prompt, output, parsed, label, emotion])
        return {"pearson_per_emotion": pearson_per_emotion, "avg_pearson": avg_pearson}
                               
def evaluate_ablation(engine,
                      test_data: List[dict],
                      task: str,
                      prompt_variants: dict,
                      shot_counts: List[int],
                      topk_list: List[int],
                      main_variant: str,
                      main_n_shot: int,
                      main_top_k: int,
                      model_name: str) -> dict:
    """
    Runs ablations over:
      1) Prompt variants
      2) Few-shot example counts
      3) top_k sampling

    Also does optional English-v1 vs. native-v1 if relevant.
    Returns a dict with 'prompt_variant', 'few_shot', 'top_k', etc.
    """
    results = {}

    # 1. Prompt variants
    print("=== Ablation: Prompt Variants ===")
    variant_results = {}
    for variant, tmpl in prompt_variants.items():
        scores_dict = evaluate_model_on_test_set(
            engine, test_data, tmpl, task, main_top_k, n_shot=main_n_shot, model_name=model_name,out_json=f"llm_track_ab_results/tmp_{model_name.replace('/', '_')}_{task}_prompt_{variant}.json"
        )
        variant_results[variant] = scores_dict

        if task == "binary":
            main_score = scores_dict["macro_f1"]
            print(f"  Prompt variant '{variant}': macro-F1 = {main_score:.4f}")
        else:
            main_score = scores_dict["avg_pearson"]
            print(f"  Prompt variant '{variant}': avg-Pearson = {main_score:.4f}")

    results['prompt_variant'] = variant_results

    # 2. Few-shot examples
    print("=== Ablation: Few-shot Examples ===")
    few_shot_results = {}
    main_prompt = prompt_variants[main_variant]
    for n_shot in shot_counts:
        scores_dict = evaluate_model_on_test_set(
            engine, test_data, main_prompt, task, main_top_k, n_shot=n_shot, model_name=model_name,out_json=f"llm_track_ab_results/tmp_{model_name.replace('/', '_')}_{task}_fewshot_{n_shot}.json"
        )
        few_shot_results[n_shot] = scores_dict

        if task == "binary":
            main_score = scores_dict["macro_f1"]
            print(f"  n_shot = {n_shot}: macro-F1 = {main_score:.4f}")
        else:
            main_score = scores_dict["avg_pearson"]
            print(f"  n_shot = {n_shot}: avg-Pearson = {main_score:.4f}")

    results['few_shot'] = few_shot_results

    # 3. top_k
    print("=== Ablation: Top_k Values ===")
    topk_results = {}
    for k in topk_list:
        scores_dict = evaluate_model_on_test_set(
            engine, test_data, main_prompt, task, k, n_shot=main_n_shot, model_name=model_name, out_json=f"llm_track_ab_results/tmp_{model_name.replace('/', '_')}_{task}_topk_{k}.json"
        )
        topk_results[k] = scores_dict

        if task == "binary":
            main_score = scores_dict["macro_f1"]
            print(f"  top_k = {k}: macro-F1 = {main_score:.4f}")
        else:
            main_score = scores_dict["avg_pearson"]
            print(f"  top_k = {k}: avg-Pearson = {main_score:.4f}")

    results['top_k'] = topk_results

    return results

###########################################################
# MAIN: Single task + single language
###########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM inference with vLLM for a single task and language."
    )
    parser.add_argument("--n_shot", type=int, default=None,
                    help="Number of few-shot examples to use (overrides default config).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Hugging Face model ID (e.g., 'bigscience/bloom')")
    parser.add_argument("--task", type=str, required=True,
                        choices=["binary", "intensity"],
                        help="Which task to run: 'binary' or 'intensity'")
    parser.add_argument("--output_file", type=str, default=None,
                        help="JSON file with final results.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallel.")
    # The following arguments are not fully exploited by vLLM in this code,
    # but included for future expansions / placeholders:
    parser.add_argument("--bnb_quant_type", type=str, default="nf4",
                        choices=["fp4", "nf4"],
                        help="4-bit quantization type (not fully used here).")
    parser.add_argument("--use_double_quant", action="store_true",
                        help="Enable double quantization if supported by vLLM.")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Compute dtype (ignored in vLLM example).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip running if output file already exists.")

    args = parser.parse_args()

    

    # Load model
    model_name = args.model_name
    safe_model = model_name.replace("/", "_")
    print(f"\n>>> Loading LLM: {model_name} ")
    if not args.model_name.startswith("openai/") and not args.model_name.startswith("google/gemini"):
       from vllm import LLM, SamplingParams
       engine = LLM(
           model=model_name,
           tokenizer=model_name,
           tensor_parallel_size=args.tensor_parallel_size
       )
    else:
       engine = None
    print(" LLM engine loaded.\n")

    for lang in ALL_LANGUAGES:
        # Prepare the output file
        if args.output_file is None:
            var_name = main_config["variant"]
            topk_main = main_config["top_k"]
            n_shot_main = args.n_shot if args.n_shot is not None else main_config["n_shot"]

            out_json = (
                f"llm_track_ab_results/results_{safe_model}_{args.task}_{lang}_"
                f"{n_shot_main}shot_{var_name}_topk{topk_main}.json"
            )
        else:
            out_json = args.output_file

        # Check if the output file exists and then skip if it does if the args.skip_existing is set
        if args.skip_existing and os.path.exists(out_json):
            print(f"Output file {out_json} already exists. Skipping.")
            print(f"[DEBUG] skip_existing={args.skip_existing}, checking {out_json} exists? {os.path.exists(out_json)}")
            continue

        # Check if we have a valid test CSV
        test_dir = TEST_DIRS[args.task]
        csv_path = os.path.join(test_dir, f"{lang}.csv")
        if not os.path.isfile(csv_path):
            print(f"Missing test CSV for lang={lang}, task={args.task}: {csv_path}")
            continue

        # Load the data
        data = load_test_data_multicolumn(csv_path, task=args.task)
        if not data:
            raise ValueError(f"No data loaded for {lang} at {csv_path}")

        # Prepare the main prompt template + few shot examples
        config = TASK_CONFIGS[args.task]
        var_name = main_config["variant"]
        topk_main = main_config["top_k"]
        n_shot_main = args.n_shot if args.n_shot is not None else main_config["n_shot"]

        main_prompt = config["prompt_variants"][var_name]

        # Evaluate single-run
        print(f"Running main evaluation for task={args.task}, lang={lang} ...")
        main_res = evaluate_model_on_test_set(
            engine=engine,
            test_data=data,
            prompt_template=main_prompt,
            task=args.task,
            top_k=topk_main,
            n_shot=n_shot_main,
            model_name=model_name,
            out_json=out_json
        )
        # Log the main result
        if args.task == "binary":
            print(f"Main macro-F1 = {main_res['macro_f1']:.4f}")
            print(f"Per-emotion F1s: {main_res['f1_per_emotion']}")
        else:
            print(f"Main avg-Pearson = {main_res['avg_pearson']:.4f}")

        # Possibly run ablations if language is in ablation list
        ablation_res = {}
        if lang in FULL_ABLATION_LANGUAGES:
            print("  ~ Running ablations for this language ~")
            ablation_res = evaluate_ablation(
                engine=engine,
                test_data=data,
                task=args.task,
                prompt_variants=config["prompt_variants"],
                shot_counts=few_shot_counts,
                topk_list=top_k_list,
                main_variant=var_name,
                main_n_shot=n_shot_main,
                main_top_k=topk_main,
                model_name=model_name
            )
        
        if lang in NATIVE_PROMPT_ABLATION_LANGUAGES:
            # Extra ablation: compare English v1 vs. Native v1 for binary tasks
            if args.task == "binary" and lang in LANG_NATIVE_PROMPTS:
                print("  ~ Comparing English v1 vs. Native v1 prompt ~")
                eng_v1_scores = evaluate_model_on_test_set(
                    engine, data,
                    config["prompt_variants"]["v1"],
                    args.task, topk_main, n_shot_main, model_name,
                    out_json=f"llm_track_ab_results/tmp_{safe_model}_{args.task}_{lang}_engv1.json"
                )
                native_v1_scores = evaluate_model_on_test_set(
                    engine, data,
                    native_v1_prompt,
                    args.task, topk_main, n_shot_main, model_name,
                    out_json=f"llm_track_ab_results/tmp_{safe_model}_{args.task}_{lang}_nativev1.json"
                )
                ablation_res["english_v1_vs_native_v1"] = {
                    "f1_english_v1": eng_v1_scores,
                    "f1_native_v1": native_v1_scores
                }
                print(f"     English v1 macro-F1 = {eng_v1_scores['macro_f1']:.4f} "
                    f"vs. Native v1 macro-F1 = {native_v1_scores['macro_f1']:.4f}")

        final_output = {
            "task": args.task,
            "language": lang,
            "model_name": model_name,
            "n_shot": n_shot_main,
            "prompt_variant": var_name,
            "top_k": topk_main,
            "main_result": main_res,
            "ablation": ablation_res
        }

        # Write results
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"\nAll done! Wrote results to {out_json}\n")
        print(f"  → Metadata: prompt_variant={var_name}, n_shot={n_shot_main}, top_k={topk_main}")
        if args.output_file is None and lang == ALL_LANGUAGES[-1]:
            combined_results = []

            for lang_code in ALL_LANGUAGES:
                # Pattern to match all variants of the filename for this language, task, and model
                pattern = f"llm_track_ab_results/results_{safe_model}_{args.task}_{lang_code}_*.json"
                
                # Find all matching files
                matching_files = glob.glob(pattern)
                
                if not matching_files:
                    print(f"No result files found for language {lang_code} with pattern {pattern}")
                    continue
                
                for filepath in matching_files:
                    with open(filepath, "r", encoding="utf-8") as f:
                        combined_results.append(json.load(f))
            final_path = f"llm_track_ab_results/final_bothTasks_{safe_model}.json"
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(combined_results, f, indent=2)
            print(f"[✓] Wrote combined final JSON to: {final_path}")