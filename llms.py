#!/usr/bin/env python
import os
import csv
import json
import argparse
import re
from typing import List, Optional, Tuple, Callable, Union
import wandb
import random
from openai import AzureOpenAI
import pathlib
import socket
from collections import defaultdict
wandb.init(mode="disabled")
import glob
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import sys

class SamplingParams:
    def __init__(self, max_tokens, temperature, top_p, n):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n

class MockLLM:
    def generate(self, prompts: List[str], sampling_params):
        class Result:
            def __init__(self, texts: List[str]):
                self.texts = texts

        dummy_outputs = []
        for prompt in prompts:
            # Detect intensity prompts by the "0: none" or "0=none" snippet
            is_intensity = ("0: none" in prompt) or ("0=none" in prompt)

            # Choose from 0..3 for intensity, yes/no for binary
            choices = ['0', '1', '2', '3'] if is_intensity else ['yes', 'no']

            responses = []
            for _ in range(sampling_params.n):
                responses.append(f"Answer: {random.choice(choices)}")

            dummy_outputs.append(Result(responses))

        return dummy_outputs
USE_MOCK_LLM = True

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
    "top_k": 1,
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
            "v4": (
               "You are an expert analyzer. Read the text and think step by step about whether it conveys {{EMOTION}}.\n"
               "Show your chain of thought, then conclude with 'Answer:' followed by 'yes' or 'no'."
            ),
            "tree_of_thoughts": (
                "You are solving the task of identifying whether the emotion {{EMOTION}} is present in a text.\n"
                "Reason through multiple steps if needed. Each step should bring you closer to the final answer.\n"
                "After thinking it through, answer clearly: 'Answer: yes' or 'no'."
            ),
            "cbp_simple": (
                "Determine whether the emotion {{EMOTION}} is expressed in the text.\n"
                "Conclude with 'Answer:' followed by 'yes' or 'no'."
            ),
            "cbp_medium": (
                "Evaluate whether the following text conveys the emotion of {{EMOTION}}.\n"
                "Explain your reasoning briefly. Conclude with 'Answer:' followed by 'yes' or 'no'."
            ),
            "cbp_complex": (
                "Carefully read the text and determine if the emotion {{EMOTION}} is expressed.\n"
                "Think step-by-step and show your full reasoning. End with 'Answer:' followed by 'yes' or 'no'."
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
            "v4": (
                "You are an expert in emotional analysis. Carefully read the text and think step by step about how intensely it expresses the emotion '{{EMOTION}}'.\n"
                "Provide a brief reasoning, then conclude with 'Answer:' followed by a number from 0 (not at all) to 3 (very strongly)."
            ),
            "tree_of_thoughts": (
                "You are solving the task of assessing the intensity of {{EMOTION}} in a piece of text.\n"
                "Reason through multiple steps, examining each clue that indicates how strong the emotion is.\n"
                "After thinking it through step by step, conclude with “Answer:” followed by the appropriate intensity score (0, 1, 2, or 3)."
            ),
            "cbp_simple": (
                "Rate the intensity (0 to 3) of the emotion {{EMOTION}} in this text.\n"
                "Conclude with 'Answer:' followed by the number."
            ),
            "cbp_medium": (
                "Evaluate the intensity of {{EMOTION}} in this text from 0 (none) to 3 (high).\n"
                "Give a short explanation, then write 'Answer:' followed by the score."
            ),
            "cbp_complex": (
                "Analyze the text carefully and assess how strongly {{EMOTION}} is conveyed.\n"
                "Think through all clues step-by-step. Finish with 'Answer:' and a number from 0 to 3."
            ),
        }
    }
}    


class AzureEngineWrapper:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def generate(self, prompts, sampling_params):
        class Result:
            def __init__(self, texts):
                self.texts = texts
        results = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                n=sampling_params.n
            )
            texts = [choice.message.content.strip() for choice in response.choices]
            results.append(Result(texts))
        return results





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

def robust_parse_binary(output: str) -> Optional[int]:
    # Only use the LAST "Answer: ..." line
    lines = output.strip().splitlines()
    for line in reversed(lines):
        match = re.match(r"^Answer:\s*(yes|no)$", line.strip(), re.IGNORECASE)
        if match:
            return 1 if match.group(1).lower() == "yes" else 0
    return None


def robust_parse_intensity(output: str) -> Optional[int]:
    lines = output.strip().splitlines()
    for line in reversed(lines):
        match = re.match(r"^Answer:\s*([0-3])$", line.strip())
        if match:
            return int(match.group(1))
    return None

def parse_output(generated_text: str, task: str) -> Optional[int]:
    if task == "binary":
        return robust_parse_binary(generated_text)
    else:
        return robust_parse_intensity(generated_text)

def run_self_refine(prompt: str, task: str, llm, flagged_prompts: list, max_refinements=3, max_tries=3):
    """
    Self-Refine loop with:
    - content moderation skip + flag
    - indefinite network retry
    - max retries for parse failures
    - flagged_prompts: shared list with main pipeline
    """
    try_count = 0
    original_prompt = prompt
    current_output = ""
    sampling_params = SamplingParams(
        max_tokens=80,
        temperature=0.7,
        top_p=0.95,
        n=1
    )
    while try_count < max_tries:
        try:
            # 1. Generate initial response
            response = llm.generate([prompt], sampling_params)[0].texts[0]

            if response is None or not isinstance(response, str) or "Answer:" not in response:
                try_count += 1
                continue

            current_output = response.strip()

            # 2. Critique
            critique_prompt = (
                f"{original_prompt}\n\nYour previous answer was:\n{current_output}\n\n"
                f"Critique your response. What was unclear or incorrect?"
            )
            critique = llm.generate([critique_prompt], sampling_params)[0].texts[0]

            if critique is None or not isinstance(critique, str):
                try_count += 1
                continue

            # 3. Refine
            refine_prompt = (
                f"{original_prompt}\n\nYour previous answer was:\n{current_output}\n"
                f"Critique: {critique}\n\nPlease revise your answer based on the critique:"
            )
            revision = llm.generate([refine_prompt], sampling_params)[0].texts[0]

            if revision is None or not isinstance(revision, str) or "Answer:" not in revision:
                try_count += 1
                continue

            return revision.strip(), None  # Success

        except Exception as e:
            error_msg = str(e).lower()

            # 🔒 Fatal errors → stop the program
            if isinstance(e, (
                openai.error.InvalidRequestError,
                openai.error.AuthenticationError,
                openai.error.PermissionError,
                openai.error.ServiceUnavailableError
            )) or any(kw in error_msg for kw in ["invalid request", "auth", "permission", "unavailable"]):
                print(f"❌ Fatal error in self_refine(): {e}")
                sys.exit(1)

            # 🔁 Retry on network/server/rate errors (don’t increment try count)
            if isinstance(e, (
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError
            )) or any(kw in error_msg for kw in ["connection", "timeout", "rate", "500"]):
                print(f"🔁 Recoverable error in self_refine(): {e}")
                time.sleep(2)
                continue

            # 📛 Content moderation
            if "content policy" in error_msg or "violation" in error_msg:
                flagged_prompts.append({
                    "prompt": prompt,
                    "reason": "content_moderation_violation",
                    "stage": "self_refine"
                })
                return None, "flagged"

            # ⚠️ Other unexpected errors → count as one try
            print(f"⚠️ Error in self_refine(): {e}")
            try_count += 1
            continue

    # 📛 Exceeded retry limit due to repeated unparseable outputs
    flagged_prompts.append({
        "prompt": prompt,
        "reason": f"Unparseable after {max_tries} attempts",
        "stage": "self_refine"
    })
    return None, "flagged"

def run_rasc(
    prompt: str,
    task: str,
    llm,
    max_samples: int = 10,
    min_confidence: float = 0.7,
    max_tries: int = 5,
) -> Tuple[Optional[int], List[str], List[dict]]:
    """
    Reasoning-Aware Self-Consistency (RASC) with retry and content moderation.
    """
    from collections import Counter
    import hashlib
    import openai
    import time

    sampling = SamplingParams(max_tokens=80, temperature=0.7, top_p=0.95, n=1)
    seen = set()
    responses = []
    flagged = []
    attempt = 0

    def reasoning_score(response: str) -> float:
        reasoning_part = response.strip().split("Answer:")[0]
        word_count = len(reasoning_part.split())
        return min(1.0, word_count / 30.0)

    while attempt < max_tries and len(responses) < max_samples:
        attempt += 1
        try:
            result = llm.generate([prompt], sampling)[0]
            text = result.texts[0].strip()

            if not text or "Answer:" not in text:
                continue

            key = hashlib.md5(text.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)

            parsed = parse_output(text, task)
            if parsed is None:
                continue

            score = reasoning_score(text)
            responses.append((parsed, score))

            # Weighted majority vote
            counter = Counter()
            for val, s in responses:
                counter[val] += s
            best_val, best_weight = counter.most_common(1)[0]
            total_weight = sum(counter.values())
            confidence = best_weight / total_weight

            if confidence >= min_confidence:
                return best_val, [r[0] for r in responses], flagged

        except Exception as e:
            err = str(e).lower()
            # Network errors → retry without increment
            if isinstance(e, (
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError,
            )) or any(kw in err for kw in ["timeout", "connection", "rate", "500"]):
                print(f"🔁 Recoverable error in RASC: {e}")
                attempt -= 1
                time.sleep(2)
                continue

            # Fatal errors → exit
            if isinstance(e, (
                openai.error.InvalidRequestError,
                openai.error.AuthenticationError,
                openai.error.PermissionError,
                openai.error.ServiceUnavailableError
            )) or any(kw in err for kw in ["invalid request", "auth", "permission", "unavailable"]):
                print(f"❌ Fatal error in RASC: {e}")
                sys.exit(1)

            # Content moderation
            if "content policy" in err or "violation" in err:
                flagged.append({
                    "prompt": prompt,
                    "reason": "content_moderation_violation",
                    "stage": "rasc"
                })
                return None, [], flagged

            # Other errors → count toward retry limit
            print(f"⚠️ Unknown error in RASC: {e}")
            flagged.append({
                "prompt": prompt,
                "error": str(e),
                "stage": "rasc"
            })

    # Final vote fallback
    if responses:
        counter = Counter()
        for val, s in responses:
            counter[val] += s
        final = counter.most_common(1)[0][0]
        return final, [str(r[0]) for r in responses], flagged

    flagged.append({
        "prompt": prompt,
        "reason": f"Unparseable after {max_tries} attempts",
        "stage": "rasc"
    })
    return None, [], flagged

def run_tree_of_thoughts(
    prompt_base: str,
    emotion: str,
    task: str,
    input_text: str,
    llm,
    max_steps: int = 3,
    beam_width: int = 3,
    max_retries: int = 5,
) -> Tuple[Optional[int], List[dict]]:
    """
    Simplified Tree of Thoughts with retry + moderation handling.
    Returns (final_answer, flagged_prompts).
    """
    state_queue = [""]
    all_flagged = []
    max_thought_retries = 10

    for step in range(max_steps):
        new_states = []
        for state in state_queue:
            full_prompt = prompt_base + state
            prompts = [full_prompt] * beam_width
            sampling = SamplingParams(
                max_tokens=80, temperature=0.7, top_p=0.95, n=beam_width
            )

            
            thought_retry_count = 0
            success = False
            while thought_retry_count < max_thought_retries:
                thought_retry_count += 1
                try:
                    result = llm.generate(prompts, sampling)[0]
                    parseable_thoughts = 0
                    for thought in result.texts:
                        branch = state + thought.strip() + "\n"
                        if parse_output(branch, task) is not None:
                            new_states.append(branch)
                            parseable_thoughts += 1

                    if parseable_thoughts > 0:
                        success = True
                        break  # ✅ got at least one valid child state

                except Exception as e:
                    err = str(e).lower()

                    # 🛑 Fatal errors → print and exit
                    if isinstance(e, (
                        openai.error.InvalidRequestError,
                        openai.error.AuthenticationError,
                        openai.error.PermissionError,
                        openai.error.ServiceUnavailableError
                    )) or any(kw in err for kw in ["invalid request", "auth", "permission", "unavailable"]):
                        print(f"❌ Fatal error in tree_of_thoughts: {e}")
                        sys.exit(1)

                    # 🔁 Retry on recoverable errors
                    if isinstance(e, (
                        openai.error.APIError,
                        openai.error.APIConnectionError,
                        openai.error.RateLimitError
                    )) or any(kw in err for kw in ["connection", "timeout", "rate", "500"]):
                        print(f"🔁 Recoverable error in tree_of_thoughts: {e}")
                        time.sleep(2)
                        thought_retry_count -= 1  # Don’t count this toward max retries
                        continue

                    # 🚫 Content moderation
                    if "policy" in err or "violation" in err:
                        all_flagged.append({
                            "step": step,
                            "state": state,
                            "error": err,
                            "reason": "content policy violation",
                            "prompt": full_prompt
                        })
                        break

                    # ⚠️ Unknown error → count against retry limit
                    print(f"⚠️ Unexpected error in tree_of_thoughts: {e}")
                    time.sleep(2)
                    continue

            # ❌ If all retries failed to produce a parseable thought
            if not success:
                all_flagged.append({
                    "input_text": input_text,
                    "emotion": emotion,
                    "task": task,
                    "step": step,
                    "state": state,
                    "error": "No parseable thoughts after max retries",
                    "prompt": full_prompt
                })

        # prune to top‑beam_width by length
        state_queue = sorted(new_states, key=lambda s: -len(s))[:beam_width]
        if not state_queue:
            break

    # final voting
    answers = [parse_output(s, task) for s in state_queue]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, all_flagged

    if task == "binary":
        final = max(set(answers), key=answers.count)
    else:
        final = round(sum(answers) / len(answers))

    return final, all_flagged
def sample_dataset(csv_path: str, sample_size: int, balanced: bool, balancing_strategy="approximate") -> pd.DataFrame:
    """
    Load the CSV and greedily sample `sample_size` rows so that
    each of the six emotions is covered roughly equally.

    We assign each row a multi-hot vector over emotions,
    then at each step pick the row that best reduces the
    current imbalance vs. the ideal target count per emotion.
    """
    df = pd.read_csv(csv_path)
    all_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotions     = [emo for emo in all_emotions if emo in df.columns]
    if not emotions:
        raise ValueError(f"No emotion columns found in {csv_path}: expected one of {all_emotions}")
    if sample_size > len(df):
        print(f"[WARNING] Requested sample_size={sample_size} larger than dataset size={len(df)}. Reducing sample_size to dataset size.")
        sample_size = len(df)
    if not balanced:
        # Just sample randomly if not balancing
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    if balancing_strategy == "strict":
        # Only keep emotion groups with at least 1 sample
        emotion_groups = {emo: df[df[emo] == 1] for emo in emotions}
        # Filter out empty groups
        emotion_groups = {emo: g for emo, g in emotion_groups.items() if len(g) > 0}
        if not emotion_groups:
            raise ValueError("No emotion groups with samples found for strict balancing.")
        
        min_count = min(len(g) for g in emotion_groups.values())
        per_emo_sample = min(min_count, sample_size // len(emotion_groups))

        sampled_dfs = [g.sample(n=per_emo_sample, random_state=42) for g in emotion_groups.values()]
        sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[INFO] Strictly balanced sample size per emotion (non-empty groups): {per_emo_sample}")
        return sampled
    
    else:
        # compute float target per emotion
        target = {emo: sample_size / len(emotions) for emo in emotions}

        # precompute each row’s emotion vector
        vectors = df[emotions].fillna(0).astype(int).to_numpy()

        chosen_idxs = []
        counts = np.zeros(len(emotions), dtype=float)

        # greedy selection
        for _ in range(min(sample_size, len(df))):
            # for each candidate not yet chosen, compute new counts if picked
            best_idx, best_score = None, float('inf')
            for idx in range(len(df)):
                if idx in chosen_idxs:
                    continue
                new_counts = counts + vectors[idx]
                # squared error to target
                err = sum((new_counts[i] - target[emo])**2 for i, emo in enumerate(emotions))
                if err < best_score:
                    best_score, best_idx = err, idx
            if best_idx is None:
                break
            chosen_idxs.append(best_idx)
            counts += vectors[best_idx]

        sampled = df.iloc[chosen_idxs].reset_index(drop=True)
        print(f"[INFO] Sampled {len(sampled)} rows (target was {sample_size}).")
        for i, emo in enumerate(emotions):
            print(f"[INFO] {emo:8s}: sampled {int(counts[i])} vs. target {target[emo]:.1f}")
        return sampled

def sample_dataset_intensity(csv_path: str, sample_size: int, balanced: bool, balancing_strategy="approximate") -> pd.DataFrame:
    """
    Load the CSV and greedily sample `sample_size` rows so that
    each of the 6 emotions × 4 intensity levels (0–3) is covered roughly equally.

    We build a one‑hot 24‑dim vector per row, then at each step
    pick the row that best reduces the squared‑error to the ideal counts.
    """
    df = pd.read_csv(csv_path)
    all_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    emotions     = [emo for emo in all_emotions if emo in df.columns]
    if not emotions:
        raise ValueError(f"No emotion columns found in {csv_path}: expected one of {all_emotions}")
    levels = [0, 1, 2, 3]
    if not balanced:
            return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    if balancing_strategy == "strict":
        groups = []
        # Collect groups with at least one sample
        for emo in emotions:
            for lvl in levels:
                group = df[df[emo].fillna(0).astype(int).clip(0,3) == lvl]
                if len(group) > 0:
                    groups.append(group)

        if not groups:
            raise ValueError("No emotion-level groups with samples found for strict balancing.")

        min_count = min(len(g) for g in groups)
        per_group_sample = min(min_count, sample_size // len(groups))

        sampled_dfs = [g.sample(n=per_group_sample, random_state=42) for g in groups]
        sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[INFO] Strictly balanced sample size per emotion-level (non-empty groups): {per_group_sample}")
        return sampled
    
    else:
        # target per (emotion, level) category
        target = sample_size / (len(emotions) * len(levels))

        # build a (N, 24) matrix: one-hot for each (i_emotion, level)
        N = len(df)
        vecs = np.zeros((N, len(emotions)*len(levels)), dtype=int)
        for i, emo in enumerate(emotions):
            vals = df[emo].fillna(0).astype(int).clip(0,3).to_numpy()
            for j, lvl in enumerate(levels):
                vecs[:, i*4 + j] = (vals == lvl).astype(int)

        chosen, counts = [], np.zeros(len(emotions)*len(levels), dtype=float)

        for _ in range(min(sample_size, N)):
            best_idx, best_err = None, float("inf")
            for idx in range(N):
                if idx in chosen:
                    continue
                new_counts = counts + vecs[idx]
                err = ((new_counts - target)**2).sum()
                if err < best_err:
                    best_err, best_idx = err, idx

            if best_idx is None:
                break
            chosen.append(best_idx)
            counts += vecs[best_idx]

        sampled = df.iloc[chosen].reset_index(drop=True)
        print(f"[INFO] Sampled {len(sampled)} rows (target was {sample_size}).")
        # report per‑category counts
        for i, emo in enumerate(emotions):
            for j, lvl in enumerate(levels):
                cnt = int(counts[i*4 + j])
                print(f"[INFO] {emo:8s} lvl {lvl}: sampled {cnt} vs. target {target:.1f}")
        return sampled

if os.getenv("TEST_SAMPLER") == "1":
    # pick a real CSV (here for binary’s English test set)
    test_csv = os.path.join(TEST_DIRS["intensity"], "eng.csv")
    for size in [24, 48, 96, 192]:
        print(f"\n=== Testing binary sampler for sample_size={size} ===")
        _ = sample_dataset(csv_path, sample_size=args.sample_size, balanced=args.balanced, balancing_strategy=args.balancing_strategy)
        print(f"\n=== Testing intensity sampler for sample_size={size} ===")
        _ = sample_dataset_intensity(csv_path, sample_size=args.sample_size, balanced=args.balanced, balancing_strategy=args.balancing_strategy)
    df = pd.read_csv("./track_a/test/eng.csv")

# Emotions you're interested in
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    # Loop through each emotion column and show how many times each intensity appears
    for emo in emotions:
        if emo in df.columns:
            print(f"{emo}: {df[emo].value_counts().sort_index().to_dict()}")
        else:
            print(f"{emo}: [COLUMN MISSING]")
    sys.exit(0)

def try_generate_with_retries(
    prompt: str,
    generator_fn: Callable[[str], Union[List[str], "ResultBatch"]],
    task: str,
    max_retries: int,
    flagged_list: List[dict]
) -> Tuple[Optional[any], List[str]]:
    """
    Calls generator_fn(prompt) up to max_retries times, 
    handles network errors (infinite retry), policy errors (flag & stop),
    unparseable outputs (counted toward retries), and on success returns
    (parsed_prediction, raw_texts). On failure returns (None, last_texts).
    """
    attempt = 0
    last_texts: List[str] = []
    while True:
        try:
            # 1) Call the LLM
            result = generator_fn(prompt)
            #   - generator_fn should return either a list of strings (texts)
            #     or a vLLM/Azure-like batch object with `.texts`
            texts = result.texts if hasattr(result, "texts") else result

            # 2) Try parsing
            parsed = [parse_output(t, task) for t in texts]
            valid = [p for p in parsed if p is not None]
            if valid:
                # Success
                if task == "binary":
                    pred = max(set(valid), key=valid.count)
                else:
                    pred = round(sum(valid) / len(valid))
                return pred, texts

            # 3) Unparseable → count against retries
            attempt += 1
            last_texts = texts
            if attempt >= max_retries:
                flagged_list.append({
                    "prompt": prompt,
                    "outputs": texts,
                    "reason": f"Unparseable after {max_retries} attempts"
                })
                return None, texts
            # otherwise loop to retry

        except Exception as e:
            err = str(e).lower()
            if isinstance(e, (
                openai.error.APIError,
                openai.error.APIConnectionError,
                openai.error.RateLimitError
            )) or any(kw in err for kw in ["connection", "timeout", "rate", "500"]):
                print(f"🔁 Retrying due to recoverable error: {e}")
                time.sleep(2)
                continue

            if isinstance(e, (
                openai.error.InvalidRequestError,
                openai.error.AuthenticationError,
                openai.error.PermissionError,
                openai.error.ServiceUnavailableError
            )) or any(kw in err for kw in ["invalid request", "auth", "permission", "403", "unavailable"]):
                print(f"❌ Fatal error: {e}")
                sys.exit(1)

            attempt += 1
            if attempt >= max_retries:
                flagged_list.append({
                    "prompt": prompt,
                    "reason": f"Unhandled error after {max_retries} attempts",
                    "error": err
                })
                return None, []
            # else loop to retry

###########################################################
# EVALUATION
###########################################################
def evaluate_model_on_test_set(
    model_name: str,                         
    llm: Union["AzureEngineWrapper", "MockLLM"],  
    test_data: List[dict],
    prompt_template: str,
    task: str,
    top_k: int,
    n_shot: int,
    reasoning_mode: str,            
    max_steps: int,                 
    beam_width: int,
    out_json: str #Usless do not use
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
    all_raw = []
    all_preds = []
    all_flagged = []
    max_retries = 5

    # Only OpenAI GPT models support these advanced methods
    if model_name.startswith("openai"):
        if reasoning_mode == "default":
            sampling = SamplingParams(
                max_tokens=80,
                temperature=0.0,
                top_p=0.95,
                n=top_k
            )
            # prepare a generator function that returns a batch-like object
            def gen_fn(prompt_text):
                if model_name.startswith("openai/"):
                    return llm.generate([prompt_text], sampling)[0]
                else:
                    raise NotImplementedError(f"Only Azure OpenAI models supported currently. Got model_name={model_name}")

                # Loop through each prompt
            for prompt in prompts:
                pred, raw_texts = try_generate_with_retries(
                    prompt=prompt,
                    generator_fn=gen_fn,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)

        elif reasoning_mode == "rasc":
            for prompt in prompts:
                pred, raw_texts, new_flags = run_rasc(
                    prompt=prompt,
                    task=task,
                    llm=llm,
                    max_samples=top_k,  # reuse top_k as max_samples
                    max_tries=max_retries
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)
                all_flagged.extend(new_flags)

        elif reasoning_mode == "self_consistency":
            sampling_sc = SamplingParams(
                max_tokens=80,
                temperature=0.7,  # encourage diverse outputs
                top_p=0.95,
                n=top_k
            )

            def gen_fn_sc(prompt_text: str):
                if model_name.startswith("openai/"):
                    return llm.generate([prompt_text], sampling_sc)[0]
                else:
                    raise NotImplementedError(f"Only Azure OpenAI models supported currently. Got model_name={model_name}")

            for prompt in prompts:
                pred, raw_texts = try_generate_with_retries(
                    prompt=prompt,
                    generator_fn=gen_fn_sc,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged,
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)

        elif reasoning_mode == "self_refine":
            for prompt in prompts:
                pred, diagnostics = run_self_refine(prompt, task, llm, all_flagged, max_refinements=3, max_tries=max_retries)
                if pred is not None:
                    all_preds.append(parse_output(pred, task)) 
                    all_raw.append([pred])
                else:
                    all_preds.append(None)
                    all_raw.append([])

        elif reasoning_mode == "tree_of_thoughts":
            for prompt, sample in zip(prompts, test_data):
                final, new_flags = run_tree_of_thoughts(
                prompt_base=prompt,
                input_text=sample["text"],
                emotion=sample["emotion"],
                task=task,
                llm=llm,
                max_steps=max_steps,
                beam_width=beam_width,
                )
                all_preds.append(final)
                all_raw.append(None)
                all_flagged.extend(new_flags)
        elif reasoning_mode == "complexity_based":
            cbp_levels = ["cbp_simple", "cbp_medium", "cbp_complex"]
            sampling_cbp = SamplingParams(
                max_tokens=80,
                temperature=0.7,
                top_p=0.95,
                n=1
            )
            for sample, base_prompt in zip(test_data, prompts):
                cbp_preds = []
                cbp_raws = []
                for level in cbp_levels:
                    cbp_template = TASK_CONFIGS[task]["prompt_variants"][level]
                    cbp_prompt = construct_prompt(
                        cbp_template,
                        few_shot_examples_by_emotion[sample["emotion"]],
                        sample["text"],
                        sample["emotion"],
                        task
                    )

                    def cbp_gen_fn(prompt_text):
                        return llm.generate([prompt_text], sampling_cbp)[0]

                    pred, raw_texts = try_generate_with_retries(
                        prompt=cbp_prompt,
                        generator_fn=cbp_gen_fn,
                        task=task,
                        max_retries=max_retries,
                        flagged_list=all_flagged
                    )
                    if pred is not None:
                        cbp_preds.append(pred)
                    cbp_raws.append(raw_texts)

                # Aggregate prediction (majority vote for binary, avg for intensity)
                if cbp_preds:
                    if task == "binary":
                        final = max(set(cbp_preds), key=cbp_preds.count)
                    else:
                        final = round(sum(cbp_preds) / len(cbp_preds))
                else:
                    final = None

                all_preds.append(final)
                all_raw.append(cbp_raws)
        elif reasoning_mode == "plan_and_solve":
            for sample in test_data:
                input_text = sample["text"]
                emotion = sample["emotion"]

                # Step 1: Generate a plan
                plan_prompt = (
                    f"Analyze the following text and create a high-level plan for determining the level of emotion.\n\n"
                    f"Text: {input_text}\n"
                    f"Emotion: {emotion}\n"
                    f"Plan:"
                )

                sampling_plan = SamplingParams(max_tokens=100, temperature=0.7, top_p=0.95, n=1)
                plan_result = llm.generate([plan_prompt], sampling_plan)[0]
                plan = plan_result.texts[0].strip()

                # Step 2: Solve using the plan
                if task == "binary":
                    solve_prompt = (
                        f"Text: {input_text}\n"
                        f"Emotion: {emotion}\n"
                        f"Plan: {plan}\n\n"
                        f"Based on the plan, decide whether the emotion '{emotion}' is expressed in the text. "
                        f"Conclude with 'Answer: yes' or 'no'."
                    )
                else:  # intensity
                    solve_prompt = (
                        f"Text: {input_text}\n"
                        f"Emotion: {emotion}\n"
                        f"Plan: {plan}\n\n"
                        f"Based on the plan, rate the intensity of emotion '{emotion}' in the text on a scale from 0 (none) to 3 (high). "
                        f"Conclude with 'Answer: 0', 'Answer: 1', 'Answer: 2', or 'Answer: 3'."
                    )

                # Use retry wrapper
                pred, raw_texts = try_generate_with_retries(
                    prompt=solve_prompt,
                    generator_fn=lambda p: llm.generate([p], SamplingParams(max_tokens=80, temperature=0.7, top_p=0.95, n=top_k))[0],
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged,
                )

                all_preds.append(pred)
                all_raw.append(raw_texts)
        else:
            raise ValueError(f"Unsupported reasoning_mode: {reasoning_mode}")


    flagged_path = pathlib.Path(out_json).with_name(pathlib.Path(out_json).stem + "_flagged" + pathlib.Path(out_json).suffix)
    if all_flagged:
        os.makedirs(flagged_path.parent, exist_ok=True)
        with open(flagged_path, "w", encoding="utf-8") as fp:
            json.dump(all_flagged, fp, indent=2)

    preds_csv = pathlib.Path(out_json).with_name(
        pathlib.Path(out_json).stem + "_predictions" + pathlib.Path(out_json).suffix
    )
    os.makedirs(preds_csv.parent, exist_ok=True)
    with open(preds_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["prompt_index", "prompt", "raw_outputs", "parsed", "gold", "emotion"])
        for idx, (sample, prompt) in enumerate(zip(test_data, prompts)):
            raw = all_raw[idx] if idx < len(all_raw) else []
            pred = all_preds[idx] if idx < len(all_preds) else None
            writer.writerow([
                idx,
                prompt,
                raw,
                pred,
                sample["label"],
                sample["emotion"]
            ])

    # ----------------------------------------
    # 6) Compute metrics
    # ----------------------------------------
    emotion2refs = defaultdict(list)
    emotion2preds = defaultdict(list)
    for sample, pred in zip(test_data, all_preds):
        if pred is None:
            continue
        emotion2refs[sample["emotion"]].append(sample["label"])
        emotion2preds[sample["emotion"]].append(pred)

    if task == "binary":
        f1_per_emotion = {
            emo: f1_score(refs, emotion2preds[emo], average="binary", zero_division=0)
            for emo, refs in emotion2refs.items()
        }
        macro_f1 = sum(f1_per_emotion.values()) / len(f1_per_emotion) if f1_per_emotion else 0.0
        return {"f1_per_emotion": f1_per_emotion, "macro_f1": macro_f1}

    else:  # intensity
        pearson_per_emotion = {
            emo: (pearsonr(refs, emotion2preds[emo])[0] if len(refs) > 1 else 0.0)
            for emo, refs in emotion2refs.items()
        }
        avg_pearson = sum(pearson_per_emotion.values()) / len(pearson_per_emotion) if pearson_per_emotion else 0.0
        return {"pearson_per_emotion": pearson_per_emotion, "avg_pearson": avg_pearson}
                               
def evaluate_ablation(
    test_data,
    prompt_variants,
    main_prompt,
    main_top_k,
    main_n_shot,
    shot_counts,
    topk_list,
    task,
    model_name,
    language,
    llm,
    reasoning_mode,
    max_steps,
    tot_beam_width,
    balanced,
    balancing_strategy,
):
    results = {}
    #Note when running evaulate model on test set here, we don't actaully give a directory that code can save all the results

    # 1. Prompt variants
    print("=== Ablation: Prompt Variants ===")
    variant_results = {}
    for variant, tmpl in prompt_variants.items():
        if reasoning_mode not in ["default", "self_consistency", "self_refine"]:
            if variant.startswith("cbp_") or variant == "tree_of_thoughts":
                continue
        scores_dict = evaluate_model_on_test_set(
            test_data=test_data,
            prompt_template=tmpl,
            task=task,
            top_k=main_top_k,
            n_shot=main_n_shot,
            model_name=model_name,
            out_json=f"...",
            llm=llm,
            reasoning_mode=reasoning_mode,
            max_steps=max_steps,
            beam_width=tot_beam_width,
        )
        variant_results[variant] = scores_dict

        score = scores_dict["macro_f1"] if task == "binary" else scores_dict["avg_pearson"]
        print(f"  Prompt variant '{variant}': {score:.4f}")

    results['prompt_variant'] = variant_results

    # 2. Few-shot examples
    print("=== Ablation: Few-shot Examples ===")
    few_shot_results = {}
    for n_shot in shot_counts:
        scores_dict = evaluate_model_on_test_set(
            test_data=test_data,
            prompt_template=main_prompt,
            task=task,
            top_k=main_top_k,
            n_shot=n_shot,
            model_name=model_name,
            out_json=f"...",
            llm=llm,
            reasoning_mode=reasoning_mode,
            max_steps=max_steps,
            beam_width=tot_beam_width,
        )
        few_shot_results[n_shot] = scores_dict
        score = scores_dict["macro_f1"] if task == "binary" else scores_dict["avg_pearson"]
        print(f"  n_shot = {n_shot}: {score:.4f}")

    results['few_shot'] = few_shot_results

    # 3. top_k
    if reasoning_mode == "self_consistency":
        print("=== Ablation: Top_k Values (self_consistency only) ===")
        topk_results = {}
        for k in topk_list:
            scores = evaluate_model_on_test_set(
                test_data=test_data,
                prompt_template=main_prompt,
                task=task,
                top_k=k,
                n_shot=main_n_shot,
                model_name=model_name,
                llm=llm,
                reasoning_mode=reasoning_mode,
                max_steps=max_steps,
                beam_width=tot_beam_width,
                out_json=f"..."
            )
            topk_results[k] = scores
            print(f"  top_k = {k}: {scores['macro_f1']:.4f}")
        results['top_k'] = topk_results
    
    sample_sizes = [30, 48, 60, 90, 120, 150, 180, 240]
    print("=== Ablation: Sample Sizes ===")
    sample_size_results = {}
    for size in sample_sizes:
        print(f"\n  → Sampling {size} examples ...")
        if balanced:
            # Compute test CSV path inside function
            csv_path = os.path.join(TEST_DIRS[task], f"{language}.csv")

            if task == "binary":
                sampled_df = sample_dataset(csv_path, sample_size= args.sample_size, balanced=args.balanced, balancing_strategy=args.balancing_strategy)
            else:
                sampled_df = sample_dataset_intensity(csv_path, sample_size=args.sample_size, balanced=args.balanced, balancing_strategy=args.balancing_strategy)

            new_test_data = []
            for row in sampled_df.itertuples(index=False):
                for emo in EMOTIONS:
                    val = getattr(row, emo, 0)
                    label = 1 if val == 1 else 0 if task == "binary" else max(0, min(3, int(val)))
                    new_test_data.append({
                        "text": row.text,
                        "emotion": emo,
                        "label": label
                    })
        else:
            # Just slice test_data (unbalanced)
            new_test_data = test_data[:size]

        scores = evaluate_model_on_test_set(
            test_data=new_test_data,
            prompt_template=main_prompt,
            task=task,
            top_k=main_top_k,
            n_shot=main_n_shot,
            model_name=model_name,
            llm=llm,
            reasoning_mode=reasoning_mode,
            max_steps=max_steps,
            beam_width=tot_beam_width,
            out_json= f"..."
        )
        sample_size_results[size] = scores
        score = scores["macro_f1"] if task == "binary" else scores["avg_pearson"]
        print(f"  sample_size = {size}: {score:.4f}")

    results["sample_size"] = sample_size_results

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
    parser.add_argument("--top_k", type=int, default=1, help="Number of completions to sample for self-consistency or top-k reasoning.")
    parser.add_argument("--use_double_quant", action="store_true",
                        help="Enable double quantization if supported by vLLM.")
    parser.add_argument("--compute_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Compute dtype (ignored in vLLM example).")
    parser.add_argument("--language", type=str, default=None,
                    help="Language code to run on (e.g., 'eng', 'ptbr'). If not set, will run on all languages.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip running if output file already exists.")
    parser.add_argument("--prompt_variant", type=str, default=None,
                    help="Which prompt variant to use (e.g., v1, v2, cot_rich)")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, build and print example prompts for each variant and exit (no model calls)."
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="default",
        choices=["default", "self_consistency", "tree_of_thoughts","self_refine","complexity_based","plan_and_solve","rasc"],
        help="Choose the reasoning strategy: default (1-shot), self_consistency (vote), or tree_of_thoughts (search)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Total number of examples to draw (will attempt to balance equally across emotions).")
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="If set, sample sample_size examples equally across emotions. Otherwise use full dataset.")
    parser.add_argument(
        "--balancing_strategy",
        type=str,
        choices=["approximate", "strict"],
        default="approximate",
        help="Choose how to balance the dataset: 'approximate' for near-equal, 'strict' for exact equal based on the smallest class count."
    )
    parser.add_argument(
        "--tot_steps",
        type=int,
        default=3,
        help="Max steps for tree_of_thoughts"
    )
    parser.add_argument(
        "--tot_beam_width",
        type=int,
        default=3,
        help="Beam width for tree_of_thoughts"
    )
    args = parser.parse_args()

    if args.reasoning_mode not in ["self_consistency", "rasc"]:
        print(f"[DEBUG] For reasoning_mode={args.reasoning_mode}, forcing top_k=1 (was {args.top_k})")
        args.top_k = 1

    if USE_MOCK_LLM:
        print("[DEBUG] Using Mock LLM for offline testing.")
        llm_engine = MockLLM()
    else:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=AZURE_OPENAI_VERSION,
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + f"/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        )
        llm_engine = AzureEngineWrapper(client, AZURE_OPENAI_DEPLOYMENT)

    if args.model_name.startswith("openai/"):
        vllm_engine = None
    else:
        from vllm import LLM, SamplingParams
        vllm_engine = LLM(model=args.model_name,
                          tokenizer=args.model_name,
                          tensor_parallel_size=args.tensor_parallel_size)
    
    # Dry‑run: just construct & print a prompt for each variant, then exit
    if args.dry_run:
        from pprint import pprint
        task_cfg = TASK_CONFIGS[args.task]
        variants = task_cfg["prompt_variants"]
        print(f"\n=== Dry‑run: building prompts for task='{args.task}' ===")
        for name, template in variants.items():
            variant_name = args.prompt_variant or name
            prompt_template = variants[variant_name]
            # no few‑shot examples, dummy text/emotion
            example = construct_prompt(
                prompt_template,
                few_shot_examples=[],
                input_text="This is a TEST sentence to check {{EMOTION}}.",
                emotion="joy",
                task=args.task
            )
            print(f"\n--- variant = {variant_name} ---\n")
            print(example)
        sys.exit(0)

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

    langs_to_run = [args.language] if args.language else ALL_LANGUAGES
    for lang in langs_to_run:
        # Prepare the output file
        if args.output_file is None:
            var_name = args.prompt_variant if args.prompt_variant else main_config["variant"]
            topk_main = main_config["top_k"]            
            n_shot_main = args.n_shot if args.n_shot is not None else main_config["n_shot"]

            out_json = (
                f"llm_track_ab_results/results_{safe_model}_{args.task}_{args.language}_"
                f"{args.sample_size}samples_{args.n_shot}shot_{args.prompt_variant}_topk{args.top_k}_{args.reasoning_mode}.json"
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
        # ——— Load (or sample) the data ———
        if args.balanced and args.sample_size:
            if args.task == "binary":
                sampled_df = sample_dataset(
                    csv_path,
                    args.sample_size,
                    balanced=True,
                    balancing_strategy=args.balancing_strategy  # <-- ADD THIS
                )
            else:  # intensity
                sampled_df = sample_dataset_intensity(
                    csv_path,
                    args.sample_size,
                    balanced=True,
                    balancing_strategy=args.balancing_strategy  # <-- ADD THIS
                )

            data = []
            for row in sampled_df.itertuples(index=False):
                for emo in EMOTIONS:
                    val = getattr(row, emo, 0)
                    if args.task == "binary":
                        label = 1 if val == 1 else 0
                    else:  # intensity
                        label = max(0, min(3, int(val)))
                    data.append({
                        "text": row.text,
                        "emotion": emo,
                        "label": label
                    })


        # Prepare the main prompt template + few shot examples
        config = TASK_CONFIGS[args.task]
        topk_main = main_config["top_k"]
        n_shot_main = args.n_shot if args.n_shot is not None else main_config["n_shot"]
        #var_name = main_config["variant"]
        #main_prompt = config["prompt_variants"][var_name]

        if args.reasoning_mode == "tree_of_thoughts":
            # Use the dedicated tree_of_thoughts prompt variant for main run
            if "tree_of_thoughts" in config["prompt_variants"]:
                var_name = "tree_of_thoughts"
                main_prompt = config["prompt_variants"][var_name]
            else:
                raise ValueError("No Tree of Thoughts prompt variant defined in TASK_CONFIGS for this task.")
        else:
            # Use the prompt variant passed by user or default
            var_name = args.prompt_variant if args.prompt_variant else main_config["variant"]
            main_prompt = config["prompt_variants"][var_name]

        # Evaluate single-run
        print(f"Running main evaluation for task={args.task}, lang={lang} ...")
        main_res = evaluate_model_on_test_set(
            model_name=model_name,
            llm=llm_engine,
            test_data=data,
            prompt_template=main_prompt,
            task=args.task,
            top_k=topk_main,
            n_shot=n_shot_main,
            out_json=out_json,
            reasoning_mode=args.reasoning_mode,     
            max_steps=args.tot_steps,                
            beam_width=args.tot_beam_width
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
            task = args.task
            reasoning_mode = args.reasoning_mode
            model_name = args.model_name
            prompt_variants = TASK_CONFIGS[task]['prompt_variants']
            if args.prompt_variant is not None:
                if args.prompt_variant not in prompt_variants:
                    raise ValueError(f"Prompt variant {args.prompt_variant} not found for task {task}")
                prompt_variants = {args.prompt_variant: prompt_variants[args.prompt_variant]}
            print("  ~ Running ablations for this language ~")
            shot_counts = [0, 1, 2, 4, 6]
            topk_list = [1, 2, 4, 8]

            # Call ablation (fix argument names)
            ablation_res = evaluate_ablation(
                test_data=data,
                prompt_variants=prompt_variants,
                main_prompt=main_prompt,
                main_top_k=topk_main,
                main_n_shot=n_shot_main,
                shot_counts=shot_counts,
                topk_list=topk_list,
                task=args.task,
                model_name=args.model_name,
                language=lang,
                llm=llm_engine,
                reasoning_mode=args.reasoning_mode,
                max_steps=args.tot_steps,
                tot_beam_width=args.tot_beam_width,
                balanced= args.balanced,
                balancing_strategy = args.balancing_strategy
            )
        
#     if lang in NATIVE_PROMPT_ABLATION_LANGUAGES:
#           if args.task == "binary" and lang in LANG_NATIVE_PROMPTS:
#                print("  ~ Comparing English v1 vs. Native v1 prompt ~")
#
                # Required variables
#                model_name = args.model_name
#                task = args.task
#                test_data = data  # Assuming 'data' is defined earlier from sampled dataset
#
#                # Prompt templates
#                eng_v1_prompt = TASK_CONFIGS[task]["prompt_variants"]["v1"]
#                native_v1_prompt = LANG_NATIVE_PROMPTS[lang]
#
#               # Other arguments
#                top_k = args.top_k if hasattr(args, "top_k") else 4
#                n_shot = args.n_shot if hasattr(args, "n_shot") else 4
#                reasoning_mode = args.reasoning_mode
#                max_steps = args.tot_steps if hasattr(args, "tot_steps") else 3
#                beam_width = args.tot_beam_width if hasattr(args, "tot_beam_width") else 3
#
#                # Run evaluation for English v1
#                eng_v1_scores = evaluate_model_on_test_set(
#                    model_name=model_name,
#                    llm=llm_engine,
#                    test_data=test_data,
#                    prompt_template=eng_v1_prompt,
#                    task=task,
#                    top_k=top_k,
#                    n_shot=n_shot,
#                    reasoning_mode=reasoning_mode,
#                    max_steps=max_steps,
#                    beam_width=beam_width,
#                    out_json=f"llm_track_ab_results/tmp_{model_name.replace('/', '_')}_{task}_{lang}_engv1.json"
#                )
#
#                # Run evaluation for Native v1
#                native_v1_scores = evaluate_model_on_test_set(
#                    model_name=model_name,
#                    llm=llm_engine,
#                    test_data=test_data,
#                    prompt_template=native_v1_prompt,
#                    task=task,
#                    top_k=top_k,
#                    n_shot=n_shot,
#                    reasoning_mode=reasoning_mode,
#                    max_steps=max_steps,
#                    beam_width=beam_width,
#                    out_json=f"llm_track_ab_results/tmp_{model_name.replace('/', '_')}_{task}_{lang}_nativev1.json"
#                )
#
#                ablation_res["english_v1_vs_native_v1"] = {
#                    "f1_english_v1": eng_v1_scores,
#                    "f1_native_v1": native_v1_scores
#                }
#
#                print(f"     English v1 macro-F1 = {eng_v1_scores['macro_f1']:.4f} "
#                    f"vs. Native v1 macro-F1 = {native_v1_scores['macro_f1']:.4f}")'''

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
