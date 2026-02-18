#!/usr/bin/env python
import os
import csv
import json
import argparse
import re
from typing import List, Optional, Tuple, Callable, Union
import wandb
import random
import math
from openai import AzureOpenAI
from openai import OpenAIError
import openai
import pathlib
import socket
from hashlib import md5
from collections import defaultdict
import glob
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, cohen_kappa_score
)
from scipy.stats import (pearsonr, spearmanr)
import sys

class SamplingParams:
    def __init__(self, max_tokens, temperature, top_p, n):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n

class MockLLM:
    def __init__(self, reasoning_mode: str = "default", task: str = "binary"):
        self.reasoning_mode = reasoning_mode
        self.task = task

    def query_confidence_bin(self, step_text: str, sampling=None):
        """Simulate confidence response randomly for A-J (0.05 to 0.95)."""
        bin_letter = random.choice(list(confidence_map.keys()))
        score = confidence_map[bin_letter]
        return bin_letter, score

    def set_reasoning_mode(self, mode: str):
        self.reasoning_mode = mode

    def set_task(self, task: str):
        self.task = task

    def generate(self, prompts: List[str], sampling_params):
        class Result:
            def __init__(self, texts: List[str]):
                self.texts = texts

        def random_choice():
            return random.choice(['yes', 'no']) if self.task == "binary" else str(random.randint(0, 3))

        dummy_outputs = []
        for prompt in prompts:
            responses = []
            for _ in range(sampling_params.n):
                answer = random_choice()

                if self.reasoning_mode == "tree_of_thoughts":
                    thought_steps = [
                        "Step 1: Identify emotional cues in the sentence.",
                        "Step 2: Evaluate their intensity and relevance.",
                        "Step 3: Cross-check with known examples.",
                        f"Answer: {answer}"
                    ]
                    responses.append("\n".join(random.sample(thought_steps, k=len(thought_steps))))

                elif self.reasoning_mode == "self_refine":
                    explanation = random.choice([
                        "The expression of emotion is somewhat present.",
                        "It's unclear but possible that the emotion is there.",
                        "The cues suggest a subtle presence of emotion."
                    ])
                    responses.append(f"{explanation}\nAnswer: {answer}")

                elif self.reasoning_mode == "plan_and_solve":
                    plan = random.choice([
                        "- Look for emotional words.\n- Evaluate tone and intensity.\n- Decide.",
                        "- Scan for sentiment.\n- Map to emotion scale.\n- Conclude.",
                        "- Identify triggers.\n- Assess context.\n- Finalize rating."
                    ])
                    responses.append(f"Plan:\n{plan}\nAnswer: {answer}")

                elif self.reasoning_mode == "complexity_based":
                    reasoning = random.choice([
                        "After analyzing the tone and keywords, I conclude:",
                        "The emotional content is evaluated based on intensity markers.",
                        "Considering the phrasing and sentiment, my assessment is:"
                    ])
                    responses.append(f"{reasoning}\nAnswer: {answer}")

                elif self.reasoning_mode == "self_consistency":
                    explanation = random.choice([
                        "There are some signs of emotion.",
                        "The wording reflects emotional content.",
                        "The expression feels neutral with slight emotion."
                    ])
                    responses.append(f"{explanation}\nAnswer: {answer}")

                elif self.reasoning_mode == "rasc":
                    reasoning = random.choice([
                        "The emotional indicators are strong and repetitive.",
                        "Subtle hints of the emotion are scattered in the text.",
                        "The emotion is directly referenced through explicit language."
                    ])
                    responses.append(f"{reasoning}\nAnswer: {answer}")

                else:  # "default"
                    responses.append(f"Answer: {answer}")

            dummy_outputs.append(Result(responses))

        return dummy_outputs
USE_MOCK_LLM = False # Set this to false if wanted to use actuall LLM


class ErrorMockLLM(MockLLM):
    """
    On each call to .generate(), randomly:
      - raise an OpenAIError with a specific code (recoverable/fatal/moderation)
      - return unparseable texts (no 'Answer:')
      - or delegate to MockLLM for a normal response
    """
    def __init__(self, reasoning_mode: str, task: str, error_prob=0.4):
        super().__init__(reasoning_mode=reasoning_mode, task=task)
        self.error_prob = error_prob

        # Define error constructors
        def make_rate_limit_error():
            e = OpenAIError("rate limit exceeded")
            e.code = "rate_limit_exceeded"
            e.http_status = 429
            return e

        def make_invalid_request_error():
            e = OpenAIError("invalid request")
            e.code = "invalid_request_error"
            e.http_status = 400
            return e

        def make_content_filter_error():
            e = OpenAIError("content policy violation")
            e.code = "content_filter"
            e.http_status = 200
            return e

        self.error_constructors = [
            make_rate_limit_error,
            make_invalid_request_error,
            make_content_filter_error,
        ]

    def generate(self, prompts: List[str], sampling_params):
        roll = random.random()
        # 40% chance to simulate an error
        if roll < self.error_prob:
            # pick error type
            err = random.choice(self.error_constructors)()
            raise err

        # 20% chance to return unparseable text
        elif roll < self.error_prob + 0.2:
            class R: pass
            fake_texts = ["This is gibberish", "No valid answer here"]
            return [type("R", (), {"texts": fake_texts})()]

        # otherwise: delegate to normal MockLLM
        return super().generate(prompts, sampling_params)
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

#Default prompts when preforming for all languges 
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

# All the primary prompts used
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

rankcot_docs = [
    # Document 1: Greater Good Science Center - Reading Emotions in Text Messages https://greatergood.berkeley.edu/article/item/six_tips_for_reading_emotions_in_text_messages?utm_source=chatgpt.com
    """
    How do we know what a person is feeling when they don’t tell us? Here are six tips to help you better detect emotions in text messages—or, failing that, prevent yourself from jumping to conclusions based on scant evidence. Keep in mind that texts are a difficult medium for communicating emotion. We have no facial expressions, tone of voice, or conversation to give us more information.

    The words people use often have emotional undertones. Think about some common words, like love, hate, wonderful, hard, work, explore, or kitten. If a text reads, ‘I love this wonderful kitten,’ we can easily conclude that it is expressing positive emotion. But if it reads, ‘This wonderful kitten is hard work,’ what emotion do we think is being conveyed? Exploring the emotional cores of individual words helps anchor our interpretations.
    """,

    # Document 2: Frontiers in Psychology - Mimicking Spoken Pauses in Text Messages https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1410698/full?utm_source=chatgpt.com
    """
    In contrast with face-to-face conversations, text messages lack important extralinguistic cues such as tone of voice and gestures. We ask how texters are able to communicate the same nuanced social and emotional meaning without access to this rich set of multimodal cues.

    The inclusion of a period after a single-word text (e.g., ‘yup.’) can convey abruptness or insincerity. All of these cues—prosodic and nonverbal—can significantly influence meaning. Texters strategically use punctuation, spacing, and ‘textisms’ to stand in for missing vocal and facial signals.
    """,

    # Document 3: WIRED - The Meaning of All Caps https://www.wired.com/story/all-caps-because-internet-gretchen-mcculloch/?utm_source=chatgpt.com
    """
    WHEN YOU WRITE IN ALL CAPS IT SOUNDS LIKE YOU’RE SHOUTING. Using capital letters to indicate strong feeling may be the most famous example of typographical tone of voice.

    A single capped word, on the other hand, is simply EMPHATIC. Examples like ‘NOT’, ‘ALL’, ‘YOU’, and ‘SO’ are often the same kinds of words we stress in spoken conversation (or commercials). All-caps is a typographic way of conveying the cues of louder, faster, or higher-pitched speech.
    """,

    # Document 4: Purdue OWL - Tone, Mood, and Audience https://owl.purdue.edu/owl/general_writing/writing_style/diction/tone_mood_audience.html?utm_source=chatgpt.com
    """
    Tone is the author’s attitude toward the subject. In written English, it is conveyed through word choice (diction) and the details an author includes or omits. To identify tone, look for words that carry strong connotations—positive, negative, or neutral—and consider why the author chose them.

    Sentence structure also shapes tone. Short, clipped sentences often feel abrupt or urgent; long, flowing sentences can feel reflective or lyrical. By mapping patterns of diction and syntax, readers can infer the author’s stance and emotional coloring.
    """,

    # Document 5: Writers.com - What Is Tone in Literature? https://writers.com/what-is-tone-in-literature?utm_source=chatgpt.com
    """
    Tone is the author’s stance toward a story’s events and characters. It emerges when you examine the words the author selects—whether they’re harsh, playful, formal, or colloquial—and how those words make you feel as a reader.

    To detect tone, ask yourself: What details does the narrator emphasize? Are descriptions vivid or restrained? Do word choices carry irony, warmth, or distance? Close‐reading those elements reveals the undercurrent of feeling guiding the narrative.
    """,

    # Document 6: Albert.io Blog - How To Identify Author’s Tone https://www.albert.io/blog/how-to-identify-authors-tone/?utm_source=chatgpt.com
    """
    Start with word choice. Look for exaggerated adjectives ("brilliant," "terrifying") or adverbs ("eagerly," "coldly") that signal an attitude. Ask: Are these words inflating the positive or negative aspects of the subject?

    Next, examine sentence patterns. Rhetorical questions, exclamations, and varied punctuation (dashes, ellipses) all create shifts in pace and emphasis, which in turn mirror shifts in the author’s emotional stance. Track how these devices recur to pinpoint tone.
    """,

    # Document 7: MasterClass - Examples of Tone Words in Writing https://www.masterclass.com/articles/examples-of-tone-words-in-writing?utm_source=chatgpt.com
    """
    Authors convey tone through diction that evokes specific emotions—solemn, satirical, earnest, sarcastic. Recognizing these ‘tone words’ in context helps readers label the underlying attitude.

    To sharpen your sense of tone, practice matching tone words to short excerpts. Notice which adjectives capture the mood and why—for instance, a scene described with ‘drab,’ ‘dreary,’ and ‘monotonous’ feels despondent, whereas ‘vibrant,’ ‘bubbling,’ and ‘radiant’ feels buoyant.
    """,
]

def retrieve_docs(query, k=5):
    query_tokens = set(query.lower().split())
    doc_scores = []

    for doc in rankcot_docs:
        doc_tokens = set(doc.lower().split())
        overlap = len(query_tokens & doc_tokens)
        doc_scores.append((overlap, doc))

    # Sort by score (descending), then return top-k
    ranked_docs = [doc for _, doc in sorted(doc_scores, key=lambda x: -x[0])]
    return ranked_docs[:k]


AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_VERSION = "2024-12-01-preview"

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

def llm_score_reasoning(text: str, llm, sampling, flagged_list) -> float:
    prompt = (
        f"Here is a reasoning path:\n{text}\n\n"
        f"On a scale from 1 to 10, how logically sound and relevant is this reasoning path?\n"
        f"Just return a number from 1 to 10.\nScore:"
    )

    max_tries = 3
    attempt = 0
    while attempt < max_tries:
        try:
            response = llm.generate([prompt], sampling)[0].texts[0]
            match = re.search(r"\b(10|[1-9])\b", response.strip())
            return int(match.group(1)) / 10.0 if match else 0.0
        except OpenAIError as e:
            action = handle_openai_error(e, flagged_list, prompt, stage="rasc_score")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2)
                continue
            if action == "flagged":
                return 0.0
        except Exception as e:
            print(f"⚠️ Unknown error in scoring: {e}")
            flagged_list.append({
                "prompt": prompt,
                "reason": str(e),
                "stage": "rasc_score"
            })
            return 0.0
        attempt += 1

    flagged_list.append({
        "prompt": prompt,
        "reason": "Unparseable after 3 attempts",
        "stage": "rasc_score"
    })
    return 0.0

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

def handle_openai_error(e, flagged_list, prompt, stage):
    code = getattr(e, "code", None)
    status = getattr(e, "http_status", None)
    msg = str(e).lower()

    # Fatal: bad request, auth, permission
    if code in {"invalid_request_error", "authentication_error", "permission_error"} \
       or status in {400, 401, 403}:
        print(f"❌ Fatal error at {stage}: {e}")
        return "fatal"

    # Retry: rate limit or 5xx
    if code == "rate_limit_exceeded" or status in {429, 502, 503, 504}:
        print(f"🔁 Recoverable error at {stage}: {e}")
        return "retry"

    # Flag: content moderation
    if code == "content_filter" or "content policy" in msg or "moderation" in msg:
        flagged_list.append({
            "prompt": prompt,
            "reason": "content_moderation_violation",
            "stage": stage
        })
        return "flagged"

    # Skip/Count as one try
    print(f"⚠️ Unhandled OpenAIError at {stage}: {e}")
    return "skip"

def get_semantic_key(text):
    stripped = text.lower().split("answer:")[0]
    return md5(stripped.encode()).hexdigest()

def run_self_refine(prompt, task, llm, flagged_prompts, max_refinements=3, max_tries=3) -> Tuple[Optional[str], Optional[dict]]:
    try_count = 0
    original = prompt

    while try_count < max_tries:
        try:
            # generate
            out = llm.generate([prompt], SamplingParams(80,0.7,0.95,1))[0].texts[0]
            pre_conf_bin, pre_conf_score = query_confidence_bin(llm, out, SamplingParams(40, 0.0, 1.0, 1))
            if not isinstance(out, str) or "Answer:" not in out:
                try_count += 1
                continue

            # critique
            critique = llm.generate([f"{original}\n\nYour previous answer was:\n{out}\n\nCritique your response."],
                                    SamplingParams(80,0.7,0.95,1))[0].texts[0]
            # refine
            revision = llm.generate([f"{original}\n\nYour previous answer was:\n{out}\nCritique: {critique}\nPlease revise:"],
                                    SamplingParams(80,0.7,0.95,1))[0].texts[0]
            post_conf_bin, post_conf_score = query_confidence_bin(llm, revision, SamplingParams(40, 0.0, 1.0, 1))
            if not isinstance(revision, str) or "Answer:" not in revision:
                try_count += 1
                continue

            return revision.strip(), {
                "pre_conf_bin": pre_conf_bin, "pre_conf_score": pre_conf_score,
                "post_conf_bin": post_conf_bin, "post_conf_score": post_conf_score
            }

        except OpenAIError as e:
            action = handle_openai_error(e, flagged_prompts, prompt, stage="self_refine")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2)
                continue
            if action == "flagged":
                return None, "flagged"
            # skip → count
        except Exception as e:
            print(f"⚠️ Error in self_refine: {e}")

        try_count += 1

    flagged_prompts.append({
        "prompt": prompt,
        "reason": f"Unparseable after {max_tries} attempts",
        "stage": "self_refine"
    })
    return None, "flagged"

def run_rasc(prompt, task, llm, max_samples=10, min_conf=0.7, max_tries=5):
    from collections import Counter

    semantic_seen = set()
    sampling = SamplingParams(80, 0.7, 0.95, 1)
    responses, flagged = [], []
    samples, attempt = 0, 0

    while attempt < max_tries and samples < max_samples:
        try:
            out = llm.generate([prompt], sampling)[0].texts[0].strip()

            if "Answer:" not in out:
                attempt += 1
                continue

            semantic_key = get_semantic_key(out)
            if semantic_key in semantic_seen:
                continue
            semantic_seen.add(semantic_key)

            parsed = parse_output(out, task)
            if parsed is None:
                attempt += 1
                continue

            # Use LLM to score the reasoning path
            s = llm_score_reasoning(out, llm, SamplingParams(40, 0.0, 1.0, 1), flagged)

            responses.append((parsed, s))
            samples += 1

            # Weighted voting
            counter = Counter({v: sum(s_ for v_, s_ in responses if v_ == v) for v, _ in responses})
            best, weight = counter.most_common(1)[0]
            total_weight = sum(counter.values())
            if total_weight > 0 and weight / total_weight >= min_conf:
                return best, [r for r, _ in responses], flagged

        except OpenAIError as e:
            action = handle_openai_error(e, flagged, prompt, stage="rasc")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2)
                continue
            if action == "flagged":
                return None, [], flagged

        except Exception as e:
            print(f"⚠️ Unknown error in RASC: {e}")
            flagged.append({"prompt": prompt, "error": str(e), "stage": "rasc"})
            attempt += 1

        attempt += 1

    # Fallback: use highest weighted answer if any
    if responses:
        avg = Counter({v: sum(s for v_, s in responses if v_ == v) for v, _ in responses})
        return max(avg, key=avg.get), [r for r, _ in responses], flagged

    flagged.append({
        "prompt": prompt,
        "reason": f"Unparseable after {max_tries} attempts",
        "stage": "rasc"
    })
    return None, [], flagged

def run_tree_of_thoughts(
    prompt_base, emotion, task, input_text, llm,
    max_steps=3, beam_width=3, max_retries=5
) -> Tuple[Optional[int], List[dict], List[dict]]:
    state_queue, all_flagged = [""], []
    confidence_trace = []
    for step in range(max_steps):
        new_states = []
        for state in state_queue:
            full = prompt_base + state
            prompts = [full]*beam_width
            sampling = SamplingParams(80,0.7,0.95, beam_width)
            thought_retries = 0

            while thought_retries < max_retries:
                try:
                    batch = llm.generate(prompts, sampling)[0].texts
                    for thought in batch:
                        branch = state + thought.strip() + "\n"
                        if parse_output(branch, task) is not None:
                            new_states.append(branch)
                            conf_bin, conf_score = query_confidence_bin(llm, thought.strip(), sampling)
                            confidence_trace.append({
                                "step": step,
                                "beam": len(new_states),
                                "text": thought.strip(),
                                "conf_bin": conf_bin,
                                "conf_score": conf_score
                            })
                    if new_states:
                        break
                except OpenAIError as e:
                    action = handle_openai_error(e, all_flagged, full, stage="tree_of_thoughts")
                    if action == "fatal":
                        sys.exit(1)
                    if action == "retry":
                        time.sleep(2)
                        continue
                    if action == "flagged":
                        break  # this branch flagged
                    # skip → count
                except Exception as e:
                    print(f"⚠️ Unexpected in ToT: {e}")
                    all_flagged.append({
                        "prompt": full,
                        "error": str(e),
                        "stage": "tree_of_thoughts"
                    })

                thought_retries += 1

            if thought_retries >= max_retries and not new_states:
                all_flagged.append({
                    "prompt": full,
                    "reason": "No parseable thoughts after retries",
                    "stage": "tree_of_thoughts"
                })

        # prune
        state_queue = sorted(new_states, key=len, reverse=True)[:beam_width]
        if not state_queue:
            break

    # final vote
    answers = [parse_output(s, task) for s in state_queue if parse_output(s, task) is not None]
    if not answers:
        return None, all_flagged
    final = (max(set(answers), key=answers.count)
             if task=="binary" else round(sum(answers)/len(answers)))
    return final, confidence_trace, all_flagged
    
def sample_dataset(csv_path: str, sample_size: int, balanced: bool, balancing_strategy="approximate") -> pd.DataFrame:
    """
    Load the CSV and greedily sample `sample_size` rows so that
    each of the six emotions is covered roughly equally.

    We assign each row a multi-hot vector over emotions,
    then at each step pick the row that best reduces the
    current imbalance vs. the ideal target count per emotion.
    """
    df = pd.read_csv(csv_path)
    if not sample_size:
        sample_size = len(df)
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
        if sample_size == len(df) or sample_size == None or sample_size == 0:
            per_emo_sample = min_count  # use full possible balanced set
        else:
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
    if not sample_size:
        sample_size = len(df)
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

        if sample_size == len(df) or sample_size == None or sample_size == 0:
            per_group_sample = min_count
        else:
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
) -> Tuple[Optional[int], List[str]]:
    attempt = 0
    last_texts: List[str] = []

    while True:
        try:
            # 1) Call the LLM
            result = generator_fn(prompt)
            texts = result.texts if hasattr(result, "texts") else result

            # 2) Parse
            parsed = [parse_output(t, task) for t in texts]
            valid = [p for p in parsed if p is not None]
            if valid:
                pred = (max(set(valid), key=valid.count)
                        if task == "binary"
                        else round(sum(valid) / len(valid)))
                return pred, texts

            # 3) Unparseable → count as one try
            attempt += 1
            last_texts = texts
            if attempt >= max_retries:
                flagged_list.append({
                    "prompt": prompt,
                    "outputs": texts,
                    "reason": f"Unparseable after {max_retries} attempts"
                })
                return None, texts

        except OpenAIError as e:
            action = handle_openai_error(e, flagged_list, prompt, stage="try_generate")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2)
                continue      # retry w/o increment
            if action == "flagged":
                return None, []
            # skip → fall through to count as one try

        except Exception as e:
            print(f"⚠️ Unexpected error at try_generate: {e}")
            # fall through to count as one try

        # shared “skip” handling
        attempt += 1
        if attempt >= max_retries:
            flagged_list.append({
                "prompt": prompt,
                "outputs": last_texts,
                "reason": f"Unparseable or skip after {max_retries} attempts"
            })
            return None, last_texts

confidence_map = {
    'A': 0.05, 'B': 0.15, 'C': 0.25, 'D': 0.35, 'E': 0.45,
    'F': 0.55, 'G': 0.65, 'H': 0.75, 'I': 0.85, 'J': 0.95
}

def query_confidence_bin(llm, step_text: str, sampling) -> Tuple[Optional[str], Optional[float]]:
    """
    Ask the model to self-report its confidence, with retry logic.
    Returns a bin letter (A–J) and its mapped score, or (None, None) on failure.
    """
    confidence_prompt = (
        f"Based on your reasoning so far:\n\n"
        f"{step_text.strip()}\n\n"
        "How confident are you that your answer is correct?\n"
        "Please choose one of the following options:\n"
        "A. 0-10%\nB. 10-20%\nC. 20-30%\nD. 30-40%\nE. 40-50%\n"
        "F. 50-60%\nG. 60-70%\nH. 70-80%\nI. 80-90%\nJ. 90-100%\n\n"
        "Confidence:"
    )

    max_tries = 3
    attempts = 0
    flagged = []  # collect any flagged prompts, if desired

    while attempts < max_tries:
        try:
            response = llm.generate([confidence_prompt], sampling)[0]
            text = response if isinstance(response, str) else response.strip()
            match = re.search(r"\b([A-J])\b", text.upper())
            if match:
                letter = match.group(1)
                return letter, confidence_map.get(letter)
            # no parse → retry
            attempts += 1

        except OpenAIError as e:
            action = handle_openai_error(e, flagged, confidence_prompt, stage="query_confidence")
            if action == "retry":
                time.sleep(2 ** attempts)
                attempts += 1
                continue
            if action == "flagged":
                # record in flagged list but don’t crash
                return None, None
            if action == "fatal":
                # let fatal errors bubble
                raise

        except Exception:
            # unexpected issue (parsing, etc.) → count as a failed attempt
            attempts += 1

    # exhausted retries
    return None, None

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
            default_confidence = []
            all_conf_scores = [] 

            sampling = SamplingParams(
                max_tokens=80,
                temperature=0.0, #How much variance
                top_p=0.95, #Nucleus Sampling
                n=1  # number of generations per prompt
            )

            def gen_fn(prompt_text):
                if model_name.startswith("openai/"):
                    return llm.generate([prompt_text], sampling)[0]
                else:
                    raise NotImplementedError(
                        f"Only Azure OpenAI models supported currently. Got model_name={model_name}"
                    )

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

                # self-reported confidence → treat conf_score as P(correct)
                conf_bin, conf_score = None, None
                if raw_texts:
                    last_response = raw_texts[-1]
                    conf_bin, conf_score = query_confidence_bin(llm, last_response, sampling)
                default_confidence.append({"conf_bin": conf_bin, "conf_score": conf_score})
                # for final AUROC/AUPRC, we need a flat list of confidences
                # and define correctness = (pred == gold)
                if conf_score is not None:
                    all_conf_scores.append(conf_score)

            confidence_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, conf_dict, pred) in enumerate(
                zip(test_data, default_confidence, all_preds)
            ):
                bin_label  = conf_dict["conf_bin"] or "error"
                score_val  = conf_dict["conf_score"] if conf_dict["conf_score"] is not None else None
                confidence_table.add_data(
                    idx,
                    prompts[idx],
                    sample["label"],
                    pred,
                    bin_label,
                    score_val,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_default": confidence_table})
            
            y_and_conf = [
                (1 if p == g else 0, conf)
                for p, g, conf in zip(all_preds, [s["label"] for s in test_data], all_conf_scores)
                if conf is not None
            ]

            if y_and_conf:
                y_correct, confs = zip(*y_and_conf)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({
                "auroc": auroc,
                "auprc": auprc
            })
        
        elif reasoning_mode == "rasc":
            all_conf_scores = []
            rasc_confidence = []
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
                conf_bin, conf_score = (None, None)
                if pred is not None and isinstance(raw_texts, list) and len(raw_texts) > 0:
                    last = raw_texts[-1]
                    conf_bin, conf_score = query_confidence_bin(llm, last, SamplingParams(40, 0.0, 1.0, 1))

                rasc_confidence.append({
                    "raw_response": last if pred is not None else "",
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })
                # collect for AUROC/AUPRC as “confidence-as-corrector”
                if conf_score is not None:
                    all_conf_scores.append(conf_score)
            confidence_rasc_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred",
                "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, rasc_confidence)):
                bin_label  = conf_dict["conf_bin"] or "error"
                score_val  = conf_dict["conf_score"] if conf_dict["conf_score"] is not None else None
                if conf_data is None:
                    continue
                confidence_rasc_table.add_data(
                    idx,
                    prompts[idx],
                    sample["label"],
                    pred,
                    conf_data["raw_response"],
                    bin_label,
                    score_val,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_rasc": confidence_rasc_table})
            # log AUROC/AUPRC over confidence-vs-correctness
            y_and_conf = [
                (1 if p == g else 0, conf)
                for p, g, conf in zip(all_preds, [s["label"] for s in test_data], all_conf_scores)
                if conf is not None
            ]

            if y_and_conf:
                y_correct, confs = zip(*y_and_conf)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({
                "auroc": auroc,
                "auprc": auprc
            })
        
        elif reasoning_mode == "rankcot":
            # ── Local storage for confidence only ──
            rankcot_confidence = []
            all_conf_scores = []

            # ── Hyperparameters ──
            retrieval_k = args.top_k
            cot_sampling = SamplingParams(150, 0.7, 0.95, 1)
            refine_sampling= SamplingParams(100, 0.7, 0.95, 1)
            final_sampling = SamplingParams( 80, 0.0, 0.95, 1)
            conf_sampling  = SamplingParams( 40, 0.0, 1.00, 1)

            for sample in test_data:
                query   = sample["text"]
                emotion = sample["emotion"]

                # Step A: Retrieve top-k documents
                docs = retrieve_docs(query)[:retrieval_k]

                # Step B: Generate + optionally refine a CoT per doc
                cots = []
                for doc in docs:
                    cot_prompt = (
                        f"Context Document:\n{doc}\n\n"
                        f"Question: {query}\n"
                        f"Emotion: {emotion}\n"
                        "Think step by step and generate your chain of thought."
                    )
                    cot_pred, cot_raws = try_generate_with_retries(
                        prompt=cot_prompt,
                        generator_fn=lambda p: llm.generate([p], cot_sampling)[0],
                        task=task,
                        max_retries=max_retries,
                        flagged_list=all_flagged
                    )
                    cot = cot_raws[0].strip() if cot_raws else ""

                    # optional self-refine
                    refine_prompt = (
                        f"{cot}\n\nReview your chain of thought above and improve it if needed:"
                    )
                    ref_pred, ref_raws = try_generate_with_retries(
                        prompt=refine_prompt,
                        generator_fn=lambda p: llm.generate([p], refine_sampling)[0],
                        task=task,
                        max_retries=max_retries,
                        flagged_list=all_flagged
                    )
                    refined = ref_raws[0].strip() if ref_raws else cot

                    cots.append(refined)

                # Step C: Rank the CoTs (here: pick longest; swap in your own scorer)
                best_idx = max(range(len(cots)), key=lambda i: len(cots[i].split()))
                best_cot = cots[best_idx]

                # Step D: Generate final answer conditioned on best CoT
                final_prompt = (
                    f"{best_cot}\n\n"
                    "Based on the above reasoning, answer: "
                    "'Answer: yes' or 'no' (binary), "
                    "or 'Answer: 0-3' (intensity)."
                )
                final_pred, final_raws = try_generate_with_retries(
                    prompt=final_prompt,
                    generator_fn=lambda p: llm.generate([p], final_sampling)[0],
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                final_out = final_raws[0].strip() if final_raws else ""
                pred = parse_output(final_out, task)

                # ── Append to global accumulators ──
                all_preds.append(pred)
                all_raw.append([best_cot, final_out])

                # Step E: Self-report confidence on the final output
                conf_bin, conf_score = query_confidence_bin(llm, final_out, conf_sampling)
                rankcot_confidence.append({
                    "best_doc_index": best_idx,
                    "conf_bin":       conf_bin,
                    "conf_score":     conf_score
                })
                if conf_score is not None:
                    all_conf_scores.append(conf_score)

            # ── Log RankCoT confidence only ──
            table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred",
                "best_doc_index", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, diag) in enumerate(
                zip(test_data, all_preds[-len(test_data):], rankcot_confidence)
            ):
                # note: all_preds[-len(test_data):] picks only this mode’s preds
                bin_label  = conf_dict["conf_bin"] or "error"
                score_val  = conf_dict["conf_score"] if conf_dict["conf_score"] is not None else None
                table.add_data(
                    idx,
                    sample["text"],
                    sample["label"],
                    pred,
                    diag["best_doc_index"],
                    bin_label,
                    score_val,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_rankcot": table})
            y_and_conf = [
                (1 if p == g else 0, conf)
                for p, g, conf in zip(all_preds, [s["label"] for s in test_data], all_conf_scores)
                if conf is not None
            ]

            if y_and_conf:
                y_correct, confs = zip(*y_and_conf)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({
                "auroc": auroc,
                "auprc": auprc
            })

        elif reasoning_mode == "self_consistency":
            sampling_sc = SamplingParams(
                max_tokens=80,
                temperature=0.7,  # encourage diverse outputs
                top_p=0.95, #Nucleus Sampling
                n=top_k #number of prompt generation is depended on top_k
            )

            sc_conf_bins_all = []
            sc_conf_scores_all = []
            all_conf_scores = []

            def gen_fn_sc(prompt_text: str):
                if model_name.startswith("openai/"):
                    return llm.generate([prompt_text], sampling_sc)[0]
                else:
                    raise NotImplementedError(f"Only Azure OpenAI models supported currently. Got model_name={model_name}")
            
            sc_conf_bins_all = []
            sc_conf_scores_all = []

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

                chain_bins = []
                chain_scores = []
                for text in raw_texts or []:
                    bin_letter, score = query_confidence_bin(llm, text, sampling_sc)
                    chain_bins.append(bin_letter)
                    chain_scores.append(score)
                    if score is not None:
                        all_conf_scores.append(score)

                sc_conf_bins_all.append(chain_bins)
                sc_conf_scores_all.append(chain_scores)
            confidence_sc_table = wandb.Table(columns=["index", "prompt", "gold", "pred", "conf_bins", "conf_scores", "avg_conf", "emotion"])
            for idx, (sample, bins, scores, pred) in enumerate(zip(test_data, sc_conf_bins_all, sc_conf_scores_all, all_preds)):
                avg_conf = np.mean([s for s in scores if s is not None]) if scores else None
                confidence_sc_table.add_data(
                    idx,
                    prompts[idx],
                    sample["label"],
                    pred,
                    str([b or "error" for b in bins]),  # mark missing bins
                    str([s for s in scores]), 
                    avg_conf,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_self_consistency": confidence_sc_table})
            y_and_conf = [
                (1 if p == g else 0, conf)
                for p, g, conf in zip(all_preds, [s["label"] for s in test_data], all_conf_scores)
                if conf is not None
            ]

            if y_and_conf:
                y_true, confs = zip(*y_and_conf)
                auroc = roc_auc_score(y_true, confs)
                auprc = average_precision_score(y_true, confs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({
                "auroc": auroc,
                "auprc": auprc
            })
        elif reasoning_mode == "self_refine":
            self_refine_conf = []
            all_conf_scores = []
            for prompt in prompts:
                pred, diagnostics = run_self_refine(prompt, task, llm, all_flagged, max_refinements=3, max_tries=max_retries)
                if pred is not None:
                    all_preds.append(parse_output(pred, task))
                    all_raw.append([pred])
                else:
                    all_preds.append(None)
                    all_raw.append([])
                self_refine_conf.append(diagnostics)
                if isinstance(diagnostics, dict):
                    post_score = diagnostics.get("post_conf_score")
                    if post_score is not None:
                        all_conf_scores.append(post_score)
            
            confidence_refine_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred",
                "pre_conf_bin", "pre_conf_score",
                "post_conf_bin", "post_conf_score",
                "emotion"
            ])
            for idx, (sample, pred, diag) in enumerate(zip(test_data, all_preds, self_refine_conf)):
                if not isinstance(diag, dict):
                    # still log a row if you want to capture skips?
                    continue

                pre_bin   = diag.get("pre_conf_bin") or "error"
                pre_score = diag.get("pre_conf_score")       # may be None
                post_bin  = diag.get("post_conf_bin")  or "error"
                post_score= diag.get("post_conf_score")      # may be None

                confidence_refine_table.add_data(
                    idx,
                    prompts[idx],
                    sample["label"],
                    pred,
                    pre_bin,
                    pre_score,
                    post_bin,
                    post_score,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_self_refine": confidence_refine_table})
            y_and_conf = [
                (1 if p == s["label"] else 0, diag["post_conf_score"])
                for p, s, diag in zip(all_preds, test_data, self_refine_conf)
                if isinstance(diag, dict) and diag.get("post_conf_score") is not None
            ]

            if y_and_conf:
                ys, cs = zip(*y_and_conf)
                auroc = roc_auc_score(ys, cs)
                auprc = average_precision_score(ys, cs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({"auroc": auroc, "auprc": auprc})
        
        elif reasoning_mode == "tree_of_thoughts":
            tot_conf_traces = []
            all_conf_scores = []
            for prompt, sample in zip(prompts, test_data):
                final, conf_trace, new_flags = run_tree_of_thoughts(
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
                tot_conf_traces.append(conf_trace)
                if conf_trace:
                    last_score = conf_trace[-1].get("conf_score")
                    if last_score is not None:
                        all_conf_scores.append(last_score)
            confidence_curve_table = wandb.Table(columns=["index", "step", "beam", "text", "conf_bin", "conf_score", "emotion"])
            for idx, (trace, sample) in enumerate(zip(tot_conf_traces, test_data)):
                for item in trace:
                    bin_label = item.get("conf_bin")   or "error"
                    score_val = item.get("conf_score") # may be None
                    confidence_curve_table.add_data(
                        idx,
                        item["step"],
                        item["beam"],
                        item["text"],
                        bin_label,
                        score_val,
                        sample["emotion"]
                    )
            wandb.log({"confidence_curve_table": confidence_curve_table})
            y_and_conf = [
                (1 if p == s["label"] else 0, score)
                for p, s, score in zip(all_preds, test_data, all_conf_scores)
                if score is not None
            ]
            if y_and_conf:
                ys, cs = zip(*y_and_conf)
                auroc = roc_auc_score(ys, cs)
                auprc = average_precision_score(ys, cs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({"auroc": auroc, "auprc": auprc})
        
        elif reasoning_mode == "complexity_based":
            cb_confidence = []
            all_conf_scores = []
            cbp_levels = ["cbp_simple", "cbp_medium", "cbp_complex"]
            sampling_cbp = SamplingParams(
                max_tokens=80,
                temperature=0.7, #How much variance(0 none, 1 alot)
                top_p=0.95, #Nucleus Sampling
                n=1 #number of generations per prompt
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
                    conf_bin, conf_score = (None, None)
                    if pred is not None and isinstance(raw_texts, list) and len(raw_texts) > 0:
                        conf_bin, conf_score = query_confidence_bin(llm, raw_texts[-1], SamplingParams(40, 0.0, 1.0, 1))
                    if conf_score is not None:
                        all_conf_scores.append(conf_score)
                    cb_confidence.append({
                        "reasoning_mode": reasoning_mode,
                        "prompt_variant": level,
                        "conf_bin": conf_bin,
                        "conf_score": conf_score,
                        "raw_response": raw_texts[-1] if raw_texts else ""
                    })
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
            confidence_cb_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response",
                "reasoning_mode", "prompt_variant", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, diag) in enumerate(zip(test_data, all_preds, cb_confidence)):
                if diag is None:
                    continue
                bin_label = diag.get("conf_bin")   or "error"
                score_val = diag.get("conf_score") # may be None
                confidence_cb_table.add_data(
                    idx,
                    prompts[idx],
                    sample["label"],
                    pred,
                    diag["raw_response"],
                    diag["reasoning_mode"],
                    diag["prompt_variant"],
                    bin_label,
                    score_val,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_complexity_based": confidence_cb_table})
            y_and_conf = [
                (1 if p == s["label"] else 0, score)
                for p, s, score in zip(all_preds, test_data, all_conf_scores)
                if score is not None
            ]
            if y_and_conf:
                ys, cs = zip(*y_and_conf)
                auroc = roc_auc_score(ys, cs)
                auprc = average_precision_score(ys, cs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({"auroc": auroc, "auprc": auprc})
        elif reasoning_mode == "plan_and_solve":
            plan_and_solve_conf = []
            all_conf_scores = []
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
                try:
                    plan_result = llm.generate([plan_prompt], sampling_plan)[0]
                    plan = plan_result.texts[0].strip()
                except OpenAIError as e:
                    action = handle_openai_error(e, all_flagged, plan_prompt, stage="plan_and_solve-plan")
                    if action == "fatal":
                        sys.exit(1)
                    if action == "retry":
                        time.sleep(2)
                        # you might want to retry here or skip this sample
                        continue
                    if action == "flagged":
                        # skip to next sample
                        continue
                    # treat as skip/unparseable → skip this sample
                    plan_and_solve_conf.append(None)
                    continue

                # Step 2: Solve using the plan
                if task == "binary":
                    solve_prompt = f"...Conclude with 'Answer: yes' or 'no'."
                else:
                    solve_prompt = f"...Conclude with 'Answer: 0', 'Answer: 1', 'Answer: 2', or 'Answer: 3'."
                pred, raw_texts = try_generate_with_retries(
                    prompt=solve_prompt,
                    generator_fn=lambda p: llm.generate([p], SamplingParams(max_tokens=80, temperature=0.7, top_p=0.95, n=top_k))[0],
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged,
                )
                conf_bin, conf_score = (None, None)
                if pred is not None and isinstance(raw_texts, list) and len(raw_texts) > 0:
                    conf_bin, conf_score = query_confidence_bin(llm, raw_texts[-1], SamplingParams(40, 0.0, 1.0, 1))
                if conf_score is not None:
                    all_conf_scores.append(conf_score)
                plan_and_solve_conf.append({
                    "plan": plan,
                    "solve_response": raw_texts[-1] if raw_texts else "",
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                all_preds.append(pred)
                all_raw.append(raw_texts)
            
            confidence_plan_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred",
                "plan", "solve_text",
                "conf_bin", "conf_score",
                "emotion"
            ])
            for idx, (sample, conf_data, pred) in enumerate(zip(test_data, plan_and_solve_conf, all_preds)):
                if conf_data is None:
                    continue
                bin_label = diag.get("conf_bin")   or "error"
                score_val = diag.get("conf_score") # may be None
                confidence_plan_table.add_data(
                    idx,
                    prompts[idx],
                    sample["label"],
                    pred,
                    conf_data["plan"],
                    conf_data["solve_response"],
                    bin_label,
                    score_val,
                    sample["emotion"]
                )
            wandb.log({"confidence_table_plan_and_solve": confidence_plan_table})
            y_and_conf = [
                (1 if p == s["label"] else 0, score)
                for p, s, score in zip(all_preds, test_data, all_conf_scores)
                if score is not None
            ]
            if y_and_conf:
                ys, cs = zip(*y_and_conf)
                auroc = roc_auc_score(ys, cs)
                auprc = average_precision_score(ys, cs)
            else:
                auroc = float("nan")
                auprc = float("nan")

            wandb.log({"auroc": auroc, "auprc": auprc})
        else:
            raise ValueError(f"Unsupported reasoning_mode: {reasoning_mode}")

    # ----------------------------------------
    # 6) Compute metrics
    # ----------------------------------------
    
    if all_flagged:
        wandb.log({"flagged_prompts_count": len(all_flagged)})
        flagged_table = wandb.Table(columns=["prompt", "reason", "stage"])
        for item in all_flagged:
            flagged_table.add_data(item.get("prompt", ""), item.get("reason", ""), item.get("stage", ""))
        wandb.log({"flagged_prompts": flagged_table})


    preds_table = wandb.Table(columns=["index", "prompt", "raw_outputs", "parsed", "gold", "emotion"])
    for idx, (sample, prompt) in enumerate(zip(test_data, prompts)):
        raw = all_raw[idx] if idx < len(all_raw) else []
        pred = all_preds[idx] if idx < len(all_preds) else None
        preds_table.add_data(idx, prompt, str(raw), pred, sample["label"], sample["emotion"])
    wandb.log({"predictions_table": preds_table})

    emotion2refs = defaultdict(list)
    emotion2preds = defaultdict(list)
    f1_per_emotion = {}
    for emo in emotion2refs:
        try:
            f1 = f1_score(emotion2refs[emo], emotion2preds[emo], zero_division=0)
            f1_per_emotion[emo] = f1
        except:
            f1_per_emotion[emo] = 0.0
    for sample, pred in zip(test_data, all_preds):
        if pred is None:
            continue
        emotion2refs[sample["emotion"]].append(sample["label"])
        emotion2preds[sample["emotion"]].append(pred)

    if task == "binary":
        all_preds_flat = []
        all_labels_flat = []
        all_emotions_flat = []

        for emo in emotion2refs:
            all_preds_flat.extend(emotion2preds[emo])
            all_labels_flat.extend(emotion2refs[emo])
            all_emotions_flat.extend([emo] * len(emotion2refs[emo]))

        f1_macro = f1_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
        precision_macro = precision_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
        recall_macro = recall_score(all_labels_flat, all_preds_flat, average="macro", zero_division=0)
        accuracy = accuracy_score(all_labels_flat, all_preds_flat)

        # TNR = TN / (TN + FP)
        tnrs = []
        emotions = list(set(all_emotions_flat))
        for emo in emotions:
            y_true = [label for label, e in zip(all_labels_flat, all_emotions_flat) if e == emo]
            y_pred = [pred for pred, e in zip(all_preds_flat, all_emotions_flat) if e == emo]
            if len(set(y_true)) < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            tnrs.append(tnr)
        avg_tnr = sum(tnrs) / len(tnrs) if tnrs else 0.0

        # Log to WandB
        wandb.log({
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "accuracy": accuracy,
            "true_negative_rate_avg": avg_tnr
        })

        return {
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "accuracy": accuracy,
            "true_negative_rate_avg": avg_tnr,
            "f1_per_emotion": f1_per_emotion,
            "auroc": auroc,
            "auprc": auprc
        }

    else:  # intensity
        all_labels_flat = []
        all_preds_flat = []
        all_emotions_flat = []

        for emo in emotion2refs:
            all_labels_flat.extend(emotion2refs[emo])
            all_preds_flat.extend(emotion2preds[emo])
            all_emotions_flat.extend([emo] * len(emotion2refs[emo]))

        # Pearson (per emotion + avg)
        pearson_per_emotion = {
            emo: (pearsonr(emotion2refs[emo], emotion2preds[emo])[0]
                if len(emotion2refs[emo]) > 1 else 0.0)
            for emo in emotion2refs
        }
        avg_pearson = np.mean(list(pearson_per_emotion.values()))

        # Spearman
        try:
            spearman_corr = spearmanr(all_labels_flat, all_preds_flat).correlation
        except Exception:
            spearman_corr = 0.0

        # MAE + RMSE
        mse = mean_squared_error(all_labels_flat, all_preds_flat)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(all_labels_flat, all_preds_flat)

        # QWK
        try:
            qwk = cohen_kappa_score(all_labels_flat, all_preds_flat, weights="quadratic")
        except Exception:
            qwk = 0.0

        # Accuracy@1 (exact match)
        acc_1 = np.mean(np.array(all_labels_flat) == np.array(all_preds_flat))

        # Accuracy@1±
        acc_1pm = np.mean(np.abs(np.array(all_labels_flat) - np.array(all_preds_flat)) <= 1)
        # Log to WandB
        wandb.log({
            "pearson_avg": avg_pearson,
            "spearman": spearman_corr,
            "mae": mae,
            "rmse": rmse,
            "qwk": qwk,
            "accuracy_1": acc_1,
            "accuracy_1pm": acc_1pm,
        })

        return {
            "pearson_per_emotion": pearson_per_emotion,
            "avg_pearson": avg_pearson,
            "spearman": spearman_corr,
            "mae": mae,
            "rmse": rmse,
            "qwk": qwk,
            "accuracy_1": acc_1,
            "accuracy_1pm": acc_1pm,
            "auroc": auroc,
            "auprc": auprc
        }
                               
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
    if not args.skip_ablations:
        results = {}
        #Note when running evaulate model on test set here, we don't actaully give a directory that code can save all the results

        # 1. Prompt variants
        print("=== Ablation: Prompt Variants ===")
        variant_results = {}
        for variant, tmpl in prompt_variants.items():
            if reasoning_mode not in ["default", "self_consistency", "self_refine"]:
                if variant.startswith("cbp_") or variant == "tree_of_thoughts":
                    continue

            wandb_run_name = (
                f"ablation_{model_name.replace('/', '-')}_{task}_{language}_{reasoning_mode}_prompt_variant={variant}"
            )
            wandb.init(
                entity="CongAndSiy",
                project="emotion-eval",
                name=wandb_run_name,
                config={
                    "ablation_type": "prompt_variant",
                    "variant": variant,
                    "model": model_name,
                    "task": task,
                    "language": language,
                    "reasoning_mode": reasoning_mode,
                    "top_k": main_top_k,
                    "n_shot": main_n_shot
                },
                reinit=True
            )

            scores_dict = evaluate_model_on_test_set(
                test_data=test_data,
                prompt_template=tmpl,
                task=task,
                top_k=main_top_k,
                n_shot=main_n_shot,
                model_name=model_name,
                out_json="...",  # not used
                llm=llm,
                reasoning_mode=reasoning_mode,
                max_steps=max_steps,
                beam_width=tot_beam_width
            )

            variant_results[variant] = scores_dict
            wandb.log(scores_dict)
            wandb.finish()

            score = (
                scores_dict["f1_macro"] if task == "binary"
                else scores_dict["avg_pearson"]
            )
            print(f"  Prompt variant '{variant}': {score:.4f}")


        results['prompt_variant'] = variant_results

        # 2. Few-shot examples
        print("=== Ablation: Few-shot Examples ===")
        few_shot_results = {}
        for n_shot in shot_counts:
            wandb_run_name = (
                f"ablation_{model_name.replace('/', '-')}_{task}_{language}_{reasoning_mode}_n_shot={n_shot}"
            )
            wandb.init(
                entity="CongAndSiy",
                project="emotion-eval",
                name=wandb_run_name,
                config={
                    "ablation_type": "n_shot",
                    "n_shot": n_shot,
                    "model": model_name,
                    "task": task,
                    "language": language,
                    "reasoning_mode": reasoning_mode,
                    "top_k": main_top_k,
                },
                reinit=True
            )

            scores_dict = evaluate_model_on_test_set(
                test_data=test_data,
                prompt_template=main_prompt,
                task=task,
                top_k=main_top_k,
                n_shot=n_shot,
                model_name=model_name,
                out_json="...",
                llm=llm,
                reasoning_mode=reasoning_mode,
                max_steps=max_steps,
                beam_width=tot_beam_width,
            )

            few_shot_results[n_shot] = scores_dict
            wandb.log(scores_dict)
            wandb.finish()

            score = (
                scores_dict["f1_macro"] if task == "binary"
                else scores_dict["avg_pearson"]
            )
            print(f"  n_shot = {n_shot}: {score:.4f}")

        results['few_shot'] = few_shot_results

        # 3. top_k
        if reasoning_mode == "self_consistency":
            print("=== Ablation: Top_k Values (self_consistency only) ===")
            topk_results = {}
            for k in topk_list:
                wandb_run_name = (
                    f"ablation_{model_name.replace('/', '-')}_{task}_{language}_{reasoning_mode}_top_k={k}"
                )
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=wandb_run_name,
                    config={
                        "ablation_type": "top_k",
                        "top_k": k,
                        "model": model_name,
                        "task": task,
                        "language": language,
                        "reasoning_mode": reasoning_mode,
                        "n_shot": main_n_shot,
                    },
                    reinit=True
                )

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
                    out_json="..."
                )

                topk_results[k] = scores
                wandb.log(scores)
                wandb.finish()

                score = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  top_k = {k}: {score:.4f}")
            results['top_k'] = topk_results
        
        sample_sizes = [30, 48, 60, 90, 120, 150, 180, 240]
        print("=== Ablation: Sample Sizes ===")
        sample_size_results = {}
        for size in sample_sizes:
            print(f"\n  → Sampling {size} examples ...")
            if balanced:
                csv_path = os.path.join(TEST_DIRS[task], f"{language}.csv")
                if task == "binary":
                    sampled_df = sample_dataset(csv_path, sample_size=size, balanced=True, balancing_strategy=balancing_strategy)
                else:
                    sampled_df = sample_dataset_intensity(csv_path, sample_size=size, balanced=True, balancing_strategy=balancing_strategy)

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
                new_test_data = test_data[:size]

            wandb_run_name = (
                f"ablation_{model_name.replace('/', '-')}_{task}_{language}_{reasoning_mode}_sample_size={size}"
            )
            wandb.init(
                entity="CongAndSiy",
                project="emotion-eval",
                name=wandb_run_name,
                config={
                    "ablation_type": "sample_size",
                    "sample_size": size,
                    "model": model_name,
                    "task": task,
                    "language": language,
                    "reasoning_mode": reasoning_mode,
                    "n_shot": main_n_shot,
                    "top_k": main_top_k,
                    "balanced": balanced,
                    "balancing_strategy": balancing_strategy
                },
                reinit=True
            )

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
                out_json="..."
            )

            sample_size_results[size] = scores
            wandb.log(scores)
            wandb.finish()

            score = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
            print(f"  sample_size = {size}: {score:.4f}")

        results["sample_size"] = sample_size_results

        return results
    else:
        print("Skipping all ablation loops (–skip_ablations set).")

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
        choices=["default", "self_consistency", "tree_of_thoughts","self_refine","complexity_based","plan_and_solve","rasc","rankcot"],
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
    parser.add_argument(
        "--skip_ablations",
        action="store_true",
        help="If set, only run the main evaluation and skip all ablation loops.",
    )
    parser.add_argument(
        "--error_test",
        action="store_true",
        help="If set, use ErrorMockLLM to randomly simulate API errors.",
    )
    args = parser.parse_args()
    
    if args.reasoning_mode not in ["self_consistency", "rasc", "rankcot"]:
        print(f"[DEBUG] For reasoning_mode={args.reasoning_mode}, forcing top_k=1 (was {args.top_k})")
        args.top_k = 1

    if USE_MOCK_LLM:
        if args.error_test:
            print("[DEBUG] Using ErrorMockLLM (error testing mode).")
            llm_engine = ErrorMockLLM(reasoning_mode=args.reasoning_mode, task=args.task)
        else:
            print("[DEBUG] Using MockLLM for offline testing.")
            llm_engine = MockLLM(reasoning_mode=args.reasoning_mode, task=args.task)
    else:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=AZURE_OPENAI_VERSION,
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT") + f"/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        )
        llm_engine = AzureEngineWrapper(client, AZURE_OPENAI_DEPLOYMENT)


    prefix = ""
    if isinstance(llm_engine, ErrorMockLLM):
        prefix = "error_"
    elif isinstance(llm_engine, MockLLM):
        prefix = "mock_"

    wandb.init(
        entity="CongAndSiy",
        project="emotion-eval",  # Change if needed
        name=f"{prefix}_main_{args.model_name.replace('/', '-')}_{args.task}_{args.language or 'all'}_{args.reasoning_mode}",
        config={
            "model": args.model_name,
            "task": args.task,
            "language": args.language or "all",
            "n_shot": args.n_shot,
            "top_k": args.top_k,
            "reasoning_mode": args.reasoning_mode,
            "sample_size": args.sample_size,
            "balanced": args.balanced
        }
    )



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
        wandb.define_metric("language")       # Define the custom metric if not already
        wandb.log({"language": lang}) 
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
        elif args.balanced and args.balancing_strategy:
            if args.task == "binary":
                csv_path = os.path.join(TEST_DIRS["binary"], f"{args.language}.csv")
                df = sample_dataset(
                    csv_path=csv_path,
                    sample_size=args.sample_size,
                    balanced=args.balanced,
                    balancing_strategy=args.balancing_strategy
                )
            elif args.task == "intensity":
                csv_path = os.path.join(TEST_DIRS["intensity"], f"{args.language}.csv")
                df = sample_dataset_intensity(
                    csv_path=csv_path,
                    sample_size=args.sample_size,
                    balanced=args.balanced,
                    balancing_strategy=args.balancing_strategy
                )
            data = []
            for row in df.itertuples(index=False):
                for emo in EMOTIONS:
                    label = (1 if getattr(row, emo, 0) == 1
                                else 0 if args.task=="binary"
                                else max(0, min(3, int(getattr(row, emo, 0)))))
                    data.append({
                        "text": row.text,
                        "emotion": emo,
                        "label": label
                    })
        else:
            raise ValueError(f"Unknown task: {args.task}")

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
        #wandb.log(main_res)
        # 1) Debug-print the raw values
        if args.task == "binary":
            f1 = float(main_res.get("f1_macro", 0.0))
            acc = float(main_res.get("accuracy", 0.0))
            auroc = float(main_res.get("auroc", 0.0))
            auprc = float(main_res.get("auprc", 0.0))

            summary_table = wandb.Table(columns=[
                "model", "task", "reasoning_mode", "prompt_variant", "language",
                "f1_macro", "accuracy", "auroc", "auprc"
            ])
            summary_table.add_data(
                args.model_name, args.task, args.reasoning_mode, args.prompt_variant, args.language,
                f1, acc, auroc, auprc
            )
            wandb.log({"summary_metrics": summary_table})

        elif args.task == "intensity":
            pearson = float(main_res.get("avg_pearson", 0.0))
            spearman = float(main_res.get("avg_spearman", 0.0))
            mse = float(main_res.get("mse", 0.0))
            rmse = float(main_res.get("rmse", 0.0))
            mae = float(main_res.get("mae", 0.0))
            auroc = float(main_res.get("auroc", 0.0))
            auprc = float(main_res.get("auprc", 0.0))

            summary_table = wandb.Table(columns=[
                "model", "task", "reasoning_mode", "prompt_variant", "language",
                "avg_pearson", "avg_spearman", "mse", "rmse", "mae", "auroc", "auprc"
            ])
            summary_table.add_data(
                args.model_name, args.task, args.reasoning_mode, args.prompt_variant, args.language,
                pearson, spearman, mse, rmse, mae, auroc, auprc
            )
            wandb.log({"summary_metrics": summary_table})

        wandb.finish()
        
        # Log the main result
        if args.task == "binary":
            print("Returned results:", main_res)
            print(f"Per-emotion F1s: {main_res['f1_per_emotion']}")
        else:
            print(f"Main avg-Pearson = {main_res['avg_pearson']:.4f}")

        # Possibly run ablations if language is in ablation list
        ablation_res = None
        if not args.skip_ablations:
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
        else:
            print("Skipping all ablation loops (–skip_ablations set).")
        
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
        #os.makedirs(os.path.dirname(out_json), exist_ok=True)
        #with open(out_json, "w", encoding="utf-8") as f:
        #    json.dump(final_output, f, indent=4)
        #print(f"\nAll done! Wrote results to {out_json}\n")
        #print(f"  → Metadata: prompt_variant={var_name}, n_shot={n_shot_main}, top_k={topk_main}")
        #if args.output_file is None and lang == ALL_LANGUAGES[-1]:
        #    combined_results = []

        #    for lang_code in ALL_LANGUAGES:
        #        # Pattern to match all variants of the filename for this language, task, and model
        #        pattern = f"llm_track_ab_results/results_{safe_model}_{args.task}_{lang_code}_*.json"
        #        
        #        # Find all matching files
        #        matching_files = glob.glob(pattern)
        #        
        #        if not matching_files:
        #            print(f"No result files found for language {lang_code} with pattern {pattern}")
        #            continue
                
        #        for filepath in matching_files:
        #            with open(filepath, "r", encoding="utf-8") as f:
        #                combined_results.append(json.load(f))
        #    final_path = f"llm_track_ab_results/final_bothTasks_{safe_model}.json"
        #    with open(final_path, "w", encoding="utf-8") as f:
        #        json.dump(combined_results, f, indent=2)
        #    print(f"[✓] Wrote combined final JSON to: {final_path}")
