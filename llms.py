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
from openai import OpenAI
from openai import OpenAIError
import openai
import pathlib
import socket
from collections import Counter
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

class VLLMEngineWrapper:
    """
    Thin adapter around vLLM so it matches our llm.generate(prompts, sampling_params)
    contract and returns .texts just like AzureEngineWrapper/MockLLM.
    """
    def __init__(self, model_name: str, tensor_parallel_size: int = 1, dtype: str = "auto"):
        from vllm import LLM
        self.model_name = model_name
        self.engine = LLM(
            model=model_name,
            tokenizer=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype  # "auto" lets vLLM pick a good default (bf16/fp16)
        )

    def generate(self, prompts, sampling_params):
        # map our SamplingParams -> vLLM.SamplingParams
        from vllm import SamplingParams as VSamplingParams

        vparams = VSamplingParams(
            max_tokens=int(sampling_params.max_tokens or 128),
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            n=int(sampling_params.n or 1),
            stop=None  # rely on your prompt “Answer: …” convention
        )
        outs = self.engine.generate(prompts, vparams)

        class Result:
            def __init__(self, texts):
                self.texts = texts

        results = []
        for req_out in outs:
            # vLLM returns a list of candidate outputs per prompt
            texts = [o.text.strip() for o in (req_out.outputs or [])]
            # Defensive fallback if model produced fewer than n candidates
            if not texts:
                texts = [""]
            results.append(Result(texts))
        return results

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
                "Evaluate whether the following text conveys the emotion {{EMOTION}}.\n"
                "Think step by step before you answer.\n"
                "If the emotion is present, reply with “1”; if not, reply with “0”.\n"
                "Finish your response with exactly “Answer: 1” or “Answer: 0”."
            ),
            "v2": (
                "Analyze the text below for the presence of {{EMOTION}}.\n"
                "Explain your reasoning briefly.\n"
                "If {{EMOTION}} is present, reply with “1”; otherwise, reply with “0”.\n"
                "Finish with exactly “Answer: 1” or “Answer: 0”."
            ),
            "v3": (
                "Examine the following text to determine whether it conveys {{EMOTION}}.\n"
                "Provide a concise explanation for your assessment.\n"
                "Reply with “1” if the emotion is present or “0” if it is not.\n"
                "End your answer with exactly “Answer: 1” or “Answer: 0”."
            ),
            "v4": (
                "You are an expert in emotional analysis. Read the text and think step by step about whether it conveys {{EMOTION}}.\n"
                "Show your chain of thought briefly.\n"
                "Respond with “1” if you detect the emotion, or “0” if you do not.\n"
                "Conclude with exactly “Answer: 1” or “Answer: 0”."
            ),
            "tree_of_thoughts": (
                "You are solving the task of identifying whether the emotion {{EMOTION}} is present in a text.\n"
                "Reason through multiple steps if needed. Each step should bring you closer to the final answer.\n"
                "After thinking it through, return exactly one line:\n"
                "'Answer: 1' (if the emotion is present) or 'Answer: 0' (if not)."
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
    def __init__(self, model_name):
        self.client = OpenAI()
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

            rows = list(reader)  # materialize once

            # keep emotions that actually have any non-empty/non-zero cell
            available_emotions = []
            for emo in EMOTIONS:
                if emo not in (reader.fieldnames or []):
                    continue
                if any((r.get(emo, "").strip() not in ("", "0", "0.0")) for r in rows):
                    available_emotions.append(emo)

            for row in rows:
                text_val = (row.get("text") or "").strip()
                if not text_val:
                    continue
                for emo in available_emotions:
                    val_str = (row.get(emo) or "").strip()
                    try:
                        numeric_val = int(float(val_str)) if val_str else 0
                    except ValueError:
                        numeric_val = 0

                    if task == "binary":
                        # treat *any* positive as presence
                        label_val = 1 if numeric_val > 0 else 0
                    else:
                        # clamp 0..3
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

def llm_score_reasoning(text: str, llm, sampling, flagged_list, prior_texts: Optional[List[str]] = None, alpha: float = 0.5) -> float:
    """
    Ask the LLM for two scores in [0,1]:
      - QUALITY: logical soundness & relevance of this path alone
      - CONSISTENCY: agreement with prior sampled paths (if provided)
    Returns a combined score in [0,1] as alpha*QUALITY + (1-alpha)*CONSISTENCY.

    Backward compatible: if the model doesn't return the new format,
    we fall back to parsing a single 1..10 number and map to [0,1].
    """
    prior_snippet = ""
    if prior_texts:
        # Keep prompt short—include up to 3 previous unique snippets
        uniq = []
        seen = set()
        for p in prior_texts:
            k = (p or "").strip()
            if not k or k in seen:
                continue
            uniq.append(k)
            seen.add(k)
            if len(uniq) >= 3:
                break
        if uniq:
            prior_snippet = "\n\nPrior sampled paths (for consistency reference):\n- " + "\n- ".join(uniq)

    prompt = (
        f"Here is a candidate reasoning path:\n{text}\n"
        f"{prior_snippet}\n\n"
        "Score this path with two numbers in [0,1]:\n"
        "QUALITY: <float between 0 and 1>\n"
        "CONSISTENCY: <float between 0 and 1>\n"
        "Only output two lines exactly in this format.\n"
        "QUALITY: "
    )

    max_tries = 3
    attempt = 0
    while attempt < max_tries:
        try:
            response = llm.generate([prompt], sampling)[0].texts[0]
            # Try new format first
            q_match = re.search(r"QUALITY:\s*([01](?:\.\d+)?)", response, re.IGNORECASE)
            c_match = re.search(r"CONSISTENCY:\s*([01](?:\.\d+)?)", response, re.IGNORECASE)
            if q_match and c_match:
                q = float(q_match.group(1)); c = float(c_match.group(1))
                q = min(max(q, 0.0), 1.0); c = min(max(c, 0.0), 1.0)
                return float(alpha*q + (1.0 - alpha)*c)

            # Fallback: old single 1..10 score → map to [0,1]
            single = re.search(r"\b(10|[1-9])\b", response.strip())
            if single:
                return int(single.group(1)) / 10.0

            # Unparseable → try again
            attempt += 1
            continue

        except OpenAIError as e:
            action = handle_openai_error(e, flagged_list, prompt, stage="rasc_score")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2)
                continue
            if action == "flagged":
                return 0.0
            attempt += 1
        except Exception as e:
            print(f"⚠️ Unknown error in scoring: {e}")
            flagged_list.append({"prompt": prompt, "reason": str(e), "stage": "rasc_score"})
            return 0.0

    flagged_list.append({"prompt": prompt, "reason": "Unparseable after 3 attempts", "stage": "rasc_score"})
    return 0.0

def robust_parse_binary(generated_text: str) -> Optional[int]:
    """
    Parse binary output (0 or 1) from model response.
    Filters out reasoning text that speculates about possible answers.
    Returns None if no clear, final answer is found.
    """
    if not generated_text:
        return None

    # Normalize whitespace and lowercase
    text = generated_text.strip().lower()

    # Remove common "thinking" lead-ins
    thinking_patterns = [
        r"i think it might be",
        r"it could be",
        r"maybe it's",
        r"possibly",
        r"i believe",
        r"i guess"
    ]
    for pat in thinking_patterns:
        text = re.sub(pat, "", text)

    # Look for an explicit "final answer" or answer section
    final_match = re.search(r"(final answer|answer\s*[:\-]?)\s*(\d)", text)
    if final_match:
        val = final_match.group(2)
        if val in ["0", "1"]:
            return int(val)

    # Otherwise, look for the first standalone 0 or 1 at the end
    matches = re.findall(r"\b[01]\b", text)
    if matches:
        # Prefer last occurrence as final decision
        return int(matches[-1])
    yn = re.search(r"\b(answer\s*[:\-]?\s*)?(yes|no)\b", text)
    if yn:
        return 1 if yn.group(2) == "yes" else 0

    return None


def robust_parse_intensity(generated_text: str) -> Optional[int]:
    """
    Parse intensity rating (e.g., 0-4) from model response.
    Filters out speculation and only returns clear, final values.
    """
    if not generated_text:
        return None

    text = generated_text.strip().lower()

    thinking_patterns = [
        r"i think it might be",
        r"it could be",
        r"maybe it's",
        r"possibly",
        r"i believe",
        r"i guess"
    ]
    for pat in thinking_patterns:
        text = re.sub(pat, "", text)

    final_match = re.search(r"(final answer|answer\s*[:\-]?)\s*([0-3])\b", text)
    if final_match:
        val = final_match.group(2)
        if val.isdigit():
            val_int = int(val)
            if 0 <= val_int <= 3:
                return val_int

    matches = re.findall(r"\b[0-3]\b", text)
    if matches:
        return int(matches[-1])

    return None


def parse_output(generated_text: str, task: str) -> Optional[int]:
    """
    Route to appropriate parser based on task type.
    """
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

def run_self_refine(
    prompt: str,
    task: str,
    llm,
    max_refinements: int,
    flagged_prompts: List[dict],
    max_tries: int = 3
) -> Tuple[Optional[str], dict, List[dict]]:
    """
    Returns: (final_text_or_none, meta_dict_with_pre/post_conf, flagged_list)
    meta_dict keys: pre_conf_bin, pre_conf_score, post_conf_bin, post_conf_score
    """
    local_flagged: List[dict] = []
    conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
    gen_sampling  = SamplingParams(220, 0.7, 0.95, 1)

    # 1) initial generation with retries + error routing
    attempt = 0
    out = None
    while attempt < max_tries:
        try:
            out = llm.generate([prompt], gen_sampling)[0].texts[0]
            break
        except OpenAIError as e:
            action = handle_openai_error(e, local_flagged, prompt, stage="self_refine:init")
            if action == "fatal":
                flagged_prompts.extend(local_flagged)
                return None, {}, local_flagged
            if action == "retry":
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            local_flagged.append({"prompt": prompt, "reason": f"initial gen failed: {e}", "stage": "self_refine:init"})
        attempt += 1

    if out is None:
        flagged_prompts.extend(local_flagged)
        return None, {}, local_flagged

    pre_conf_bin, pre_conf_score = (None, None)
    if parse_output(out, task) is not None:
        pre_conf_bin, pre_conf_score = query_confidence_bin(llm, out, conf_sampling)

    # 2) refine loop (critique -> revise) with recoverable retries
    revision = out
    refinements = 0
    best_conf = pre_conf_score if pre_conf_score is not None else 0.0
    last_pred = parse_output(revision, task)
    stable_pred_count = 0
    no_improve_count = 0
    K_min = 2
    CONF_TARGET = 0.85
    CONF_DELTA = 0.05
    PATIENCE = 1
    best_text = revision
    best_conf_bin = pre_conf_bin
    best_conf_score = pre_conf_score

    while refinements < max_refinements:
        try:
            critique = llm.generate(
                [f"{prompt}\n\nYour previous answer was:\n{revision}\n\nCritique your response."],
                gen_sampling
            )[0].texts[0]

            new_text = llm.generate(
                [f"{prompt}\n\nYour previous answer was:\n{revision}\nCritique: {critique}\nPlease revise:"],
                gen_sampling
            )[0].texts[0]

            revision = new_text
            pred_now = parse_output(revision, task)
            conf_bin_now, conf_score_now = query_confidence_bin(llm, revision, conf_sampling) if pred_now is not None else (None, None)

            if pred_now == last_pred:
                stable_pred_count += 1
            else:
                stable_pred_count = 0
            last_pred = pred_now

            if conf_score_now is not None:
                if conf_score_now >= best_conf + CONF_DELTA:
                    best_conf = conf_score_now
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            if conf_score_now is not None and (
                best_conf_score is None or
                conf_score_now >= best_conf + CONF_DELTA or
                conf_score_now > best_conf_score
            ):
                best_conf = conf_score_now
                best_conf_score = conf_score_now
                best_conf_bin = conf_bin_now
                best_text = revision

            # Smart early stop
            if refinements + 1 >= K_min and (
                (conf_score_now is not None and conf_score_now >= CONF_TARGET) or
                no_improve_count >= PATIENCE or
                stable_pred_count >= 2 or
                (pred_now is not None)
            ):
                break
            refinements += 1

        except OpenAIError as e:
            action = handle_openai_error(e, local_flagged, prompt, stage="self_refine:loop")
            if action == "fatal":
                flagged_prompts.extend(local_flagged)
                return None, {}, local_flagged
            if action == "retry":
                time.sleep(2)
                continue
            if action == "flagged":
                flagged_prompts.extend(local_flagged)
                return None, {}, local_flagged
            refinements += 1  # count only non-recoverables
        except Exception as e:
            local_flagged.append({"prompt": prompt, "reason": str(e), "stage": "self_refine:loop"})
            refinements += 1

    revision = best_text
    post_conf_bin = best_conf_bin
    post_conf_score = best_conf_score
    if (post_conf_score is None) and parse_output(revision, task) is not None:
        post_conf_bin, post_conf_score = query_confidence_bin(llm, revision, conf_sampling)

    meta = {
        "pre_conf_bin":  pre_conf_bin,  "pre_conf_score":  pre_conf_score,
        "post_conf_bin": post_conf_bin, "post_conf_score": post_conf_score,
    }
    flagged_prompts.extend(local_flagged)
    return revision.strip() if isinstance(revision, str) else None, meta, local_flagged

def run_rasc(
    prompt: str,
    task: str,
    llm,
    max_samples: int = 10,
    min_conf: float = 0.7,
    alpha: float = 0.5,   
    max_tries: int = 5
) -> Tuple[Optional[int], dict, List[dict]]:
    from collections import Counter

    semantic_seen = set()
    responses = []  # list of tuples: (raw_text, parsed_label, score, semantic_key)
    flagged = []
    samples, attempt = 0, 0
    sampling = SamplingParams(200, 0.7, 0.95, 1)
    conf_sampling = SamplingParams(160, 0.0, 1.0, 1)

    samples_to_stop = None
    winner_label = None
    winner_text = ""

    while attempt < max_tries and samples < max_samples:
        try:
            result_text = llm.generate([prompt], sampling)[0].texts[0].strip()

            if "Answer:" not in result_text:
                attempt += 1
                continue

            semantic_key = get_semantic_key(result_text)
            if semantic_key in semantic_seen:
                attempt += 1
                continue
            semantic_seen.add(semantic_key)

            parsed = parse_output(result_text, task)
            if parsed is None:
                attempt += 1
                continue

            # Prior paths for consistency scoring
            prior_paths = [r for (r, _, _, _) in responses] if responses else None
            score = llm_score_reasoning(result_text, llm, conf_sampling, flagged, prior_texts=prior_paths, alpha=alpha)

            responses.append((result_text, parsed, score, semantic_key))
            samples += 1

            # Weighted voting
            counter = Counter({
                v: sum(s for (_, v_, s, _) in responses if v_ == v)
                for v in {v_ for (_, v_, _, _) in responses}
            })
            best_label, best_weight = counter.most_common(1)[0]
            total_weight = sum(counter.values())

            if total_weight > 0 and (best_weight / total_weight) >= min_conf:
                samples_to_stop = samples
                winner_label = best_label
                # keep the highest-scoring rationale among those with winner_label
                winner_text = max(
                    (r for (r, v_, s, _) in responses if v_ == best_label),
                    key=lambda r: next(s for (rr, vv, s, _) in responses if rr == r and vv == best_label),
                    default=""
                )
                break  # early exit

        except OpenAIError as e:
            action = handle_openai_error(e, flagged, prompt, stage="rasc")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2)
                continue
            if action == "flagged":
                return None, {}, flagged
            attempt += 1
        except Exception as e:
            flagged.append({"prompt": prompt, "error": str(e), "stage": "rasc"})
            attempt += 1

    # Fallback if no early stop
    if winner_label is None:
        if responses:
            counter = Counter({
                v: sum(s for (_, v_, s, _) in responses if v_ == v)
                for v in {v_ for (_, v_, _, _) in responses}
            })
            winner_label = max(counter, key=counter.get)
            samples_to_stop = samples_to_stop or samples
            # highest-scoring rationale for the winning label
            winner_text = max(
                (r for (r, v_, s, _) in responses if v_ == winner_label),
                key=lambda r: next(s for (rr, vv, s, _) in responses if rr == r and vv == winner_label),
                default=""
            )
        else:
            winner_label = None
            winner_text = ""
            flagged.append({"prompt": prompt, "reason": f"Unparseable after {max_tries} attempts", "stage": "rasc"})

    # Average self-reported confidence across parseable responses (optional; keep as before)
    conf_scores = []
    conf_bin = None
    if responses:
        for raw, parsed, _score, _ in responses:
            if parsed is not None:
                bin_label, cscore = query_confidence_bin(llm, raw, conf_sampling)
                if cscore is not None:
                    conf_scores.append(cscore)
                if conf_bin is None:
                    conf_bin = bin_label
    avg_conf = np.mean(conf_scores) if conf_scores else None

    meta = {
        "conf_bin": conf_bin,
        "conf_score": avg_conf,
        "n_samples": len(responses),
        "samples_to_stop": samples_to_stop,
        "winning_rationale": winner_text
    }
    return winner_label, meta, flagged

_value_sampling = SamplingParams(max_tokens=160, temperature=0.0, top_p=1.0, n=1)

def _parse_score(text: str) -> Optional[float]:
    """
    Parse 'SCORE: x.y' from the model output and clamp to [0,1].
    """
    if not text:
        return None
    import re
    m = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if not m:
        return None
    try:
        v = float(m.group(1))
        if v < 0: v = 0.0
        if v > 1: v = 1.0
        return v
    except Exception:
        return None

def evaluate_state(llm, prompt_prefix: str, state_text: str, value_trials: int, max_retries: int, flagged_list: List[dict]) -> Tuple[Optional[float], str]:
    """
    Ask the LM to self-evaluate a partial branch. Returns (score, raw_eval_text).
    """
    eval_prompt = (
        f"{prompt_prefix}{state_text}\n\n"
        "You are evaluating whether the above partial reasoning is promising.\n"
        "On a 0 to 1 scale, where 1 is 'very promising' and 0 is 'impossible', "
        "give a single line 'SCORE: <float>'. No other text."
    )

    best_score, raw_text = None, ""
    trials = max(1, value_trials)
    for _ in range(trials):
        score_i, raw_i = None, ""
        def _gen(p):
            return llm.generate([p], _value_sampling)[0]
        parsed, raw_texts = try_generate_with_retries(
            prompt=eval_prompt,
            generator_fn=_gen,
            task="binary",   # parsing is custom; 'task' unused for value prompts
            max_retries=max_retries,
            flagged_list=flagged_list
        )
        # raw_texts is the generation object; be defensive:
        try:
            raw_i = (raw_texts.texts[0] if raw_texts and raw_texts.texts else "") or ""
        except Exception:
            raw_i = ""
        score_i = _parse_score(raw_i)
        if score_i is not None and (best_score is None or score_i > best_score):
            best_score, raw_text = score_i, raw_i
    return best_score, raw_text

def _bin_from_score(s: Optional[float]) -> Optional[str]:
    if s is None:
        return None
    if s >= 0.75:
        return "sure"
    if s >= 0.5:
        return "maybe"
    return "impossible"

def run_tree_of_thoughts(
    prompt_base,
    emotion,
    task,
    input_text,
    llm,
    beam_width,
    max_steps=3,
    max_retries=5,
    search="bfs",
    vth=0.5,
    value_trials=1
) -> Tuple[Optional[int], List[dict], List[dict]]:
    """
    Tree of Thoughts (paper-style):
      * BFS (top-b) or DFS with backtracking
      * LM self-evaluation (value function) in [0,1]
      * Prune when value < vth
      * Choose final branch by highest value; get answer conditioned on that branch

    Returns:
      pred: Optional[int] parsed prediction
      all_raw: list of generation artifacts (dicts) for logging
      all_flagged: list of flagged prompts for logging
    """
    # --- bookkeeping
    all_raw: List[dict] = []
    all_flagged: List[dict] = []

    # --- sampling for "thought expansion" (one-shot; you can tune as needed)
    thought_sampling = SamplingParams(
        max_tokens=160,
        temperature=0.7,
        top_p=1.0,
        n=beam_width  # branch factor: produce up to b thoughts per expansion
    )

    # --- helper: prompt constructors
    def thought_prompt(prefix: str, state: str) -> str:
        # Encourage short, atomic progress steps (like ToT paper)
        return (
            f"{prefix}{state}\n\n"
            "Think step by step. Propose the NEXT short, concrete reasoning step that helps solve the task.\n"
            "Return ONLY the next step as a single sentence."
        )

    def answer_prompt(prefix: str, best_branch: str) -> str:
        if task == "binary":
            # require 0/1 so robust_parse_binary can parse it
            return (
                f"{prefix}{best_branch}\n\n"
                "Now give the FINAL ANSWER for the task above.\n"
                "Return ONLY one line in this exact format:\n"
                "Answer: 1  (if the emotion is present)  OR  Answer: 0 (if not)."
            )
        else:  # intensity
            return (
                f"{prefix}{best_branch}\n\n"
                "Now give the FINAL ANSWER for the task above.\n"
                "Return ONLY one line in this exact format:\n"
                "Answer: <0|1|2|3>"
            )

    # --- initial state (empty chain)
    initial_state = ""
    # Track states as dicts: {text, step, value, value_bin}
    from collections import deque

    def make_state(text: str, step: int, value: Optional[float]) -> dict:
        return {
            "text": text,
            "step": step,
            "value": value,
            "value_bin": _bin_from_score(value),
        }

    # Evaluate the initial state (optional but keeps code uniform)
    init_val, init_raw = evaluate_state(
        llm=llm,
        prompt_prefix=prompt_base,
        state_text=initial_state,
        value_trials=value_trials,
        max_retries=max_retries,
        flagged_list=all_flagged
    )
    all_raw.append({"stage": "value_init", "prompt": "(initial)", "outputs": [init_raw], "score": init_val})

    if search == "bfs":
        frontier = [make_state(initial_state, 0, init_val)]
    else:
        frontier = [make_state(initial_state, 0, init_val)]  # will use as a stack for DFS

    best_states: List[dict] = []

    # --- main search
    if search == "bfs":
        # Level-by-level expansion, keep top-b each level by value
        for step in range(1, max_steps + 1):
            print(f"[ToT][BFS] Step {step}/{max_steps} — Expanding {len(frontier)} states...")
            candidates: List[dict] = []

            # Expand each frontier state with up to 'beam_width' thoughts
            for st in frontier:
                # Skip hopeless states early
                if st["value"] is not None and st["value"] < vth:
                    continue

                # Generate next-step thoughts
                p = thought_prompt(prompt_base, st["text"])
                # Generate next-step thoughts directly (don’t parse these as final answers)
                try:
                    result = llm.generate([p], thought_sampling)[0]
                    raw_texts = result.texts if hasattr(result, "texts") else []
                except Exception:
                    raw_texts = []
                all_raw.append({"stage": f"thought_s{step}", "prompt": p, "outputs": raw_texts})

                # Create child states
                for t in raw_texts:
                    child_text = (st["text"] + ("\n" if st["text"] else "") + t).strip()
                    score, raw_eval = evaluate_state(
                        llm=llm,
                        prompt_prefix=prompt_base,
                        state_text=child_text,
                        value_trials=value_trials,
                        max_retries=max_retries,
                        flagged_list=all_flagged
                    )
                    all_raw.append({"stage": f"value_s{step}", "prompt": child_text, "outputs": [raw_eval], "score": score})
                    if score is None or score >= vth:
                        candidates.append(make_state(child_text, step, score))
                print(f"[ToT][BFS] Step {step} complete — kept {len(frontier)} states for next step.")
            if not candidates:
                break

            # Keep only top-b by value (fallback: None treated as 0)
            candidates.sort(key=lambda d: (d["value"] if d["value"] is not None else 0.0), reverse=True)
            frontier = candidates[:beam_width]
            best_states = frontier[:]  # track last level’s kept states

    else:
        # DFS with backtracking: expand the most promising branch first; backtrack on low value
        stack: List[dict] = [make_state(initial_state, 0, init_val)]
        visited = 0

        while stack and visited < 10000:  # safety
            visited += 1
            st = stack.pop()
            print(f"[ToT][DFS] Visited {visited} states so far, stack size={len(stack)}")

            # If reached depth
            if st["step"] >= max_steps:
                # treat as a completed candidate
                best_states.append(st)
                continue

            # Prune weak states
            if st["value"] is not None and st["value"] < vth:
                continue

            # Expand this state: get up to b thoughts
            p = thought_prompt(prompt_base, st["text"])
            try:
                result = llm.generate([p], thought_sampling)[0]
                raw_texts = result.texts if hasattr(result, "texts") else []
            except Exception:
                raw_texts = []
            all_raw.append({"stage": f"thought_s{st['step']+1}", "prompt": p, "outputs": raw_texts})

            children: List[dict] = []
            for t in raw_texts:
                child_text = (st["text"] + ("\n" if st["text"] else "") + t).strip()
                score, raw_eval = evaluate_state(
                    llm=llm,
                    prompt_prefix=prompt_base,
                    state_text=child_text,
                    value_trials=value_trials,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                all_raw.append({"stage": f"value_s{st['step']+1}", "prompt": child_text, "outputs": [raw_eval], "score": score})
                if score is None or score >= vth:
                    children.append(make_state(child_text, st["step"] + 1, score))

            # Push children onto stack in descending score so we explore best first
            children.sort(key=lambda d: (d["value"] if d["value"] is not None else 0.0), reverse=True)
            stack.extend(children)

        # If DFS never added candidates at depth == max_steps, fall back to whatever we have
        if not best_states and stack:
            best_states = stack[:beam_width]

    # --- pick best branch (highest value; tie-break by length)
    if not best_states:
        # final attempt: use initial state
        best_branch = initial_state
    else:
        best_states.sort(
            key=lambda d: (
                d["value"] if d["value"] is not None else 0.0,
                len(d["text"])
            ),
            reverse=True
        )
        best_branch = best_states[0]["text"]

    # --- ask for final answer, conditioned on best branch
    final_p = answer_prompt(prompt_base, best_branch)
    final_sampling = SamplingParams(max_tokens=160, temperature=0.0, top_p=1.0, n=1)

    def _gen_final(pp):
        return llm.generate([pp], final_sampling)[0]

    parsed, raw_obj = try_generate_with_retries(
        prompt=final_p,
        generator_fn=_gen_final,
        task=task,
        max_retries=max_retries,
        flagged_list=all_flagged
    )
    final_text = (raw_obj.texts[0] if getattr(raw_obj, "texts", None) else "")
    all_raw.append({"stage": "final_answer", "prompt": final_p, "outputs": [final_text]})

    pred = parsed
    # ---------- ToT confidence (search-based + model-declared) ----------
    # Collect leaf scores from the last kept states (best_states).
    leaf_scores = []
    if 'best_states' in locals() and best_states:
        leaf_scores = [(s.get("value") if s.get("value") is not None else 0.0) for s in best_states]
    # If we somehow have none, fallback to any recorded scores from value stages at the deepest step
    if not leaf_scores and isinstance(all_raw, list):
        try:
            # take scores from the latest 'value_s*' entries
            last_value_entries = [r for r in all_raw if isinstance(r, dict) and str(r.get("stage","")).startswith("value_s")]
            if last_value_entries:
                max_step = max(int(e["stage"].split("_s")[1]) for e in last_value_entries if "_s" in e["stage"])
                leaf_scores = [(e.get("score") or 0.0) for e in last_value_entries if e["stage"] == f"value_s{max_step}"]
        except Exception:
            pass

    # Compute search-based confidence
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except Exception:
            return 0.5
    def _safe_std(vals):
        try:
            return float(np.std(vals)) if len(vals) > 1 else 0.0
        except Exception:
            return 0.0

    if leaf_scores:
        # softmax mass for top leaf
        tau = 8.0
        exps = [math.exp(tau*s) for s in leaf_scores]
        Z = sum(exps) or 1.0
        softmax = [x / Z for x in exps]
        # chosen is index 0 if we sorted best_states earlier by value desc;
        # but to be safe, recompute the argmax:
        best_idx = int(max(range(len(leaf_scores)), key=lambda i: leaf_scores[i]))
        p_star = softmax[best_idx]
        sorted_scores = sorted(leaf_scores, reverse=True)
        delta = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else sorted_scores[0]
        sigma = _safe_std(leaf_scores)
        # pass-rate above vth
        try:
            pass_rate = (sum(1 for s in leaf_scores if s is not None and s >= vth) / max(1, len(leaf_scores)))
        except Exception:
            pass_rate = 0.0
        # weights tuned lightly; you can revisit after calibration
        conf_search = _sigmoid(2.0*p_star + 1.0*delta - 1.0*sigma - 0.5*pass_rate)
    else:
        conf_search = None

    # Model-declared confidence on the final answer
    conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
    conf_model = None
    try:
        # Only ask if we produced any final text that parses
        if parse_output(final_text, task) is not None:
            _bin, _score = query_confidence_bin(llm, final_text, conf_sampling)
            conf_model = _score
    except Exception:
        conf_model = None

    # Geometric blend (search has more weight initially). Clamp to [0,1].
    def _geo_blend(vals, weights):
        eps = 1e-6
        vs = [max(eps, v) for v in vals]
        wsum = sum(weights)
        prod = 1.0
        for v, w in zip(vs, weights):
            prod *= v**w
        return prod ** (1.0 / max(1e-6, wsum))

    parts = []
    ws    = []
    if conf_search is not None:
        parts.append(conf_search); ws.append(2.0)
    if conf_model is not None:
        parts.append(conf_model);  ws.append(1.0)
    conf_tot = None
    if parts:
        try:
            conf_tot = max(0.0, min(1.0, _geo_blend(parts, ws)))
        except Exception:
            conf_tot = None

    # Log a single structured record so the caller can read confidence cleanly
    all_raw.append({
        "stage": "tot_confidence",
        "leaf_scores": leaf_scores,
        "conf_search": conf_search,
        "conf_model": conf_model,
        "conf_tot": conf_tot
    })

    return pred, all_raw, all_flagged
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
    """
    Calls the generator function until we get a parseable output or hit max_retries.
    Returns:
        pred (int or None): Final parsed prediction
        texts (list[str]): Raw model outputs from the last attempt
    """
    attempt = 0
    last_texts: List[str] = []

    while attempt < max_retries:
        try:
            result = generator_fn(prompt)
            texts = result.texts if hasattr(result, "texts") else result

            # Parse outputs
            parsed = [parse_output(t, task) for t in texts]
            valid = [p for p in parsed if p is not None]

            if valid:
                pred = (
                    max(set(valid), key=valid.count)  # majority vote for binary
                    if task == "binary"
                    else Counter(valid).most_common(1)[0][0]  # majority vote for intensity
                )
                return pred, texts

            # No valid parse — record and retry
            last_texts = texts
            attempt += 1
            continue

        except OpenAIError as e:
            action = handle_openai_error(e, flagged_list, prompt, stage="try_generate")
            if action == "fatal":
                sys.exit(1)
            if action == "retry":
                time.sleep(2 ** attempt)  # exponential backoff
                continue  # don't increment attempt for recoverables
            if action == "flagged":
                return None, []
            attempt += 1

        except Exception as e:
            flagged_list.append({
                "prompt": prompt,
                "reason": f"Unexpected error: {e}"
            })
            attempt += 1

    # Out of retries — flag final failure
    flagged_list.append({
        "prompt": prompt,
        "outputs": last_texts if isinstance(last_texts, list) else [str(last_texts)],
        "reason": f"Unparseable after {max_retries} attempts"
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
    print("[DEBUG-ENTER] query_confidence_bin() called")
    confidence_prompt = (
        f"Based on your reasoning so far:\n\n"
        f"{step_text.strip()}\n\n"
        "How confident are you that your answer is correct?\n"
        "Please choose one of the following options exactly:\n"
        "A. 0-10%\nB. 10-20%\nC. 20-30%\nD. 30-40%\nE. 40-50%\n"
        "F. 50-60%\nG. 60-70%\nH. 70-80%\nI. 80-90%\nJ. 90-100%\n\n"
        "**Only respond with a single letter (A-J).**\n"
        "Confidence:"
    )

    max_tries = 3
    attempts = 0
    flagged = []  # collect any flagged prompts, if desired
    if isinstance(llm, MockLLM):
        return llm.query_confidence_bin(step_text, sampling)

    while attempts < max_tries:
        try:
            result = llm.generate([confidence_prompt], sampling)[0]

            # ── 2) unwrap the actual text string ──
            if hasattr(result, "texts"):
                text = result.texts[0].strip()
            elif isinstance(result, str):
                text = result.strip()
            else:
                text = str(result)

            # ── DEBUG: see exactly what the model said ──
            print(f"[DEBUG] Confidence response for step_text={step_text!r}:\n{text!r}", flush=True)

            # ── 3) try to parse A–J ──
            match = re.search(r"\b([A-J])\b", text.upper())
            if not match:
                # fallback: pick up any A–J anywhere
                match = re.search(r"([A-J])", text.upper())
            if match:
                letter = match.group(1)
                return letter, confidence_map.get(letter)

            # no valid letter → count as a failed attempt and retry
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

def cbp_score_complexity(prompt_base: str, llm, sampling, flagged_list) -> Tuple[str, float, str]:
    """
    Ask the LLM to rate reasoning complexity in {simple, medium, complex} with a 0..1 score.
    Returns: (level, score, raw_text)
    Fallbacks to 'medium', 0.5 if unparseable.
    """
    probe = (
        f"{prompt_base}\n\n"
        "Before answering, assess the inherent reasoning complexity required.\n"
        "Reply with exactly two lines:\n"
        "LEVEL: simple|medium|complex\n"
        "SCORE: <float between 0 and 1>\n"
        "LEVEL: "
    )
    try:
        out = llm.generate([probe], SamplingParams(160, 0.0, 1.0, 1))[0].texts[0]
        import re
        m_level = re.search(r"LEVEL:\s*(simple|medium|complex)", out, re.IGNORECASE)
        m_score = re.search(r"SCORE:\s*([01](?:\.\d+)?)", out, re.IGNORECASE)
        level = (m_level.group(1).lower() if m_level else "medium")
        try:
            score = float(m_score.group(1)) if m_score else 0.5
        except Exception:
            score = 0.5
        score = max(0.0, min(1.0, score))
        return level, score, out
    except OpenAIError as e:
        action = handle_openai_error(e, flagged_list, probe, stage="cbp_score_complexity")
        if action in ("fatal", "flagged"):
            return "medium", 0.5, ""
        # retry-ish fallback not needed; keep it tiny
        return "medium", 0.5, ""
    except Exception as e:
        flagged_list.append({"prompt": probe, "reason": str(e), "stage": "cbp_score_complexity"})
        return "medium", 0.5, ""

def rankcot_score_cot(query_text: str, doc_text: str, cot_text: str, llm, sampling, flagged_list, alpha: float = 0.5) -> float:
    """
    Score a chain-of-thought on two axes in [0,1]:
      - RELEVANCE to the query
      - FAITHFULNESS to the provided doc
    Returns combined score = alpha*REL + (1-alpha)*FAITH.
    Falls back gracefully to a single 1..10 score if needed.
    """
    prompt = (
        "You are ranking a chain-of-thought for answering a query using a retrieved document.\n\n"
        f"QUERY:\n{query_text}\n\n"
        f"DOCUMENT:\n{doc_text}\n\n"
        f"CHAIN-OF-THOUGHT:\n{cot_text}\n\n"
        "Output two lines ONLY:\n"
        "RELEVANCE: <float 0..1>\n"
        "FAITHFULNESS: <float 0..1>\n"
        "RELEVANCE: "
    )
    try:
        resp = llm.generate([prompt], sampling)[0].texts[0]
        import re
        m_r = re.search(r"RELEVANCE:\s*([01](?:\.\d+)?)", resp, re.IGNORECASE)
        m_f = re.search(r"FAITHFULNESS:\s*([01](?:\.\d+)?)", resp, re.IGNORECASE)
        if m_r and m_f:
            r = float(m_r.group(1)); f = float(m_f.group(1))
            r = max(0.0, min(1.0, r)); f = max(0.0, min(1.0, f))
            return float(alpha*r + (1.0 - alpha)*f)

        # fallback: single 1..10 somewhere in the text
        m10 = re.search(r"\b(10|[1-9])\b", resp.strip())
        if m10:
            return int(m10.group(1)) / 10.0
    except OpenAIError as e:
        action = handle_openai_error(e, flagged_list, prompt, stage="rankcot_score")
        if action == "retry":
            try:
                resp = llm.generate([prompt], sampling)[0].texts[0]
                m10 = re.search(r"\b(10|[1-9])\b", resp.strip())
                if m10:
                    return int(m10.group(1)) / 10.0
            except Exception:
                pass
        if action in ("fatal", "flagged"):
            return 0.0
    except Exception as e:
        flagged_list.append({"prompt": prompt, "reason": str(e), "stage": "rankcot_score"})
    return 0.0

###########################################################
# EVALUATION
###########################################################
def evaluate_model_on_test_set(
    model_name: str,
    llm: Union["AzureEngineWrapper", "MockLLM", "VLLMEngineWrapper"],
    test_data: List[dict],
    prompt_template: str,
    task: str,
    top_k: int,
    n_shot: int,
    reasoning_mode: str,
    max_steps: int,
    beam_width: int,
    out_json: str,
    # ---- NEW OPTIONAL ABLATION HOOKS ----
    rasc_min_conf: float = None,
    rasc_alpha: float = None,
    self_refine_max_refinements: int = None,
    cbp_force_level: str = None,          # "simple" | "medium" | "complex"
    tot_search: str = None,               # "bfs" | "dfs"
    tot_vth: float = None,
    tot_value_trials: int = None,
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
    prompts = []
    for sample in test_data:
        # grab the few-shot examples for this sample’s emotion
        fs = few_shot_examples_by_emotion[sample["emotion"]]
        # filter out any example whose input text equals the test text
        fs_filtered = [ex for ex in fs if ex["input"] != sample["text"]]

        # build the prompt using the filtered few-shot list
        p = construct_prompt(
            prompt_template,
            fs_filtered,
            sample["text"],
            sample["emotion"],
            task
        )
        prompts.append(p)
    all_raw = []
    all_preds = []
    all_flagged = []
    max_retries = 5
    auroc = float("nan")
    auprc = float("nan")

    # Only OpenAI GPT models support these advanced methods
    if model_name.startswith("openai"):
        if reasoning_mode == "default":
            default_confidence = []
            conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
            y_conf_pairs = []

            sampling = SamplingParams(max_tokens=200, temperature=0.0, top_p=0.95, n=1)
            def gen_fn(prompt_text: str):
                return llm.generate([prompt_text], sampling)[0]

            for sample, prompt in zip(test_data, prompts):
                pred, raw_texts = try_generate_with_retries(
                    prompt=prompt,
                    generator_fn=gen_fn,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)

                conf_bin, conf_score = None, None
                if raw_texts:
                    for response in reversed(raw_texts):
                        if response and parse_output(response, task) is not None:
                            conf_bin, conf_score = query_confidence_bin(llm, response, conf_sampling)
                            break

                default_confidence.append({
                    "raw_response": (raw_texts[-1] if raw_texts else ""),
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            confidence_default_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, default_confidence)):
                confidence_default_table.add_data(
                    idx, prompts[idx], sample["label"], pred,
                    conf_data["raw_response"], conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    sample["emotion"]
                )
            wandb.log({"confidence_table": confidence_default_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})

        elif reasoning_mode == "self_consistency":
            sc_confidence = []
            conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
            y_conf_pairs = []

            sampling_sc = SamplingParams(max_tokens=200, temperature=0.7, top_p=0.95, n=top_k)
            def gen_fn_sc(prompt_text: str):
                return llm.generate([prompt_text], sampling_sc)[0]

            for sample, prompt in zip(test_data, prompts):
                pred, raw_texts = try_generate_with_retries(
                    prompt=prompt,
                    generator_fn=gen_fn_sc,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged,
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)

                conf_bin, conf_score = None, None
                if raw_texts:
                    for response in reversed(raw_texts):
                        if response and parse_output(response, task) is not None:
                            conf_bin, conf_score = query_confidence_bin(llm, response, conf_sampling)
                            break

                sc_confidence.append({
                    "raw_response": (raw_texts[-1] if raw_texts else ""),
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            confidence_sc_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, sc_confidence)):
                confidence_sc_table.add_data(
                    idx, prompts[idx], sample["label"], pred,
                    conf_data["raw_response"], conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    sample["emotion"]
                )
            wandb.log({"confidence_table": confidence_sc_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})

        elif reasoning_mode == "tree_of_thoughts":
            tot_confidence = []
            y_conf_pairs = []
            beam_bw = max(1, int(top_k)) if top_k is not None else 3

            for sample, prompt in zip(test_data, prompts):
                pred, raw, flagged = run_tree_of_thoughts(
                    prompt_base=prompt,
                    emotion=sample["emotion"],
                    task=task,
                    input_text=sample["text"],
                    llm=llm,
                    max_steps=max_steps,
                    beam_width=beam_bw,
                    max_retries=max_retries,
                    search=tot_search or "bfs",
                    vth=tot_vth if tot_vth is not None else 0.5,
                    value_trials=tot_value_trials if tot_value_trials is not None else 1,
                )

                all_preds.append(pred)
                all_raw.append(raw)                 # <- was [last_text]
                all_flagged.extend(flagged or [])

                 # Prefer the structured 'tot_confidence' record from run_tree_of_thoughts
                conf_score = None
                conf_bin   = None
                raw_for_conf = ""
                if isinstance(raw, list):
                    # 1) read the structured record if present
                    recs = [e for e in raw if isinstance(e, dict) and e.get("stage") == "tot_confidence"]
                    if recs:
                        rec = recs[-1]
                        conf_score = rec.get("conf_tot")
                        # keep a raw explanation handle — optional
                        raw_for_conf = "tot_confidence"
                        if conf_score is not None:
                            conf_bin = _bin_from_score(conf_score)
                    # 2) fallback to your previous proxy: max of any recorded scores
                    if conf_score is None:
                        best_score = None
                        for entry in raw:
                            s = entry.get("score")
                            if s is not None and (best_score is None or s > best_score):
                                best_score = s
                                raw_for_conf = entry.get("prompt", raw_for_conf)
                        conf_score = best_score
                        conf_bin = _bin_from_score(conf_score) if conf_score is not None else None

                tot_confidence.append({
                    "raw_response": raw_for_conf,
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            confidence_tot_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, tot_confidence)):
                confidence_tot_table.add_data(
                    idx, prompts[idx], sample["label"], pred,
                    conf_data["raw_response"], conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    sample["emotion"]
                )
            wandb.log({"confidence_table": confidence_tot_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})
        elif reasoning_mode == "self_refine":
            sr_confidence = []
            y_conf_pairs = []
            # run_self_refine already queries pre/post confidence; reuse post if available
            for sample, prompt in zip(test_data, prompts):
                final_text, meta, new_flags = run_self_refine(
                    prompt=prompt,
                    task=task,
                    llm=llm,
                    flagged_prompts=all_flagged,
                    max_refinements=self_refine_max_refinements if self_refine_max_refinements is not None else 3,
                    max_tries=max_retries
                )
                all_flagged.extend(new_flags)
                pred = parse_output(final_text or "", task)
                all_preds.append(pred)
                all_raw.append([final_text or ""])

                conf_bin = meta.get("post_conf_bin")
                conf_score = meta.get("post_conf_score")

                sr_confidence.append({
                    "raw_response": final_text or "",
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            confidence_sr_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, sr_confidence)):
                confidence_sr_table.add_data(
                    idx, prompts[idx], sample["label"], pred,
                    conf_data["raw_response"], conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    sample["emotion"]
                )
            wandb.log({"confidence_table": confidence_sr_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})

        elif reasoning_mode == "complexity_based":
            cb_confidence = []
            conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
            y_conf_pairs = []
            cb_rows = []

            # routing table: level -> (prompt_variant_key, SamplingParams)
            routing = {
                "simple":  ("cbp_simple",  SamplingParams(180, 0.0, 0.95, 1)),
                "medium":  ("cbp_medium",  SamplingParams(200, 0.5, 0.95, 2)),
                "complex": ("cbp_complex", SamplingParams(220, 0.7, 0.95, 4)),
            }

            for idx, (sample, prompt_base) in enumerate(zip(test_data, prompts)):
                # 1) complexity probe
                if cbp_force_level in ("simple", "medium", "complex"):
                    level = cbp_force_level
                    level_score, raw_probe = None, ""   # no probe when forced
                else:
                    level, level_score, raw_probe = cbp_score_complexity(
                        prompt_base, llm, conf_sampling, all_flagged
                    )
                variant_key, gen_sampling = routing.get(level, routing["medium"])

                # 2) build a per-sample prompt using the selected CBP template
                #    (reuse your emotion-specific few-shot pool)
                fs = few_shot_examples_by_emotion[sample["emotion"]]
                fs_filtered = [ex for ex in fs if ex["input"] != sample["text"]]
                cbp_template = TASK_CONFIGS[task]["prompt_variants"][variant_key]
                dyn_prompt = construct_prompt(
                    cbp_template,
                    fs_filtered,
                    sample["text"],
                    sample["emotion"],
                    task
                )

                # 3) generate with scaled params (n varies by complexity)
                def gen_fn_cb(prompt_text: str):
                    return llm.generate([prompt_text], gen_sampling)[0]

                pred, raw_texts = try_generate_with_retries(
                    prompt=dyn_prompt,
                    generator_fn=gen_fn_cb,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)

                # confidence from the latest parseable response
                conf_bin, conf_score = None, None
                if raw_texts:
                    for response in reversed(raw_texts):
                        if response and parse_output(response, task) is not None:
                            conf_bin, conf_score = query_confidence_bin(llm, response, conf_sampling)
                            break

                cb_confidence.append({
                    "raw_response": (raw_texts[-1] if raw_texts else ""),
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                # log row for CBP analysis
                cb_rows.append({
                    "index": idx,
                    "prompt_variant": variant_key,
                    "complexity_level": level,
                    "complexity_score": level_score,
                    "emotion": sample["emotion"],
                    "gold": sample["label"],
                    "pred": pred,
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            # W&B table for CBP runs
            cbp_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred",
                "raw_response", "conf_bin", "conf_score",
                "emotion", "complexity_level", "complexity_score", "prompt_variant"
            ])
            for (row, prompt_text, conf_data) in zip(cb_rows, prompts, cb_confidence):
                cbp_table.add_data(
                    row["index"], prompt_text, row["gold"], row["pred"],
                    conf_data["raw_response"],
                    conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    row["emotion"], row["complexity_level"], row["complexity_score"], row["prompt_variant"]
                )
            wandb.log({"cbp_table": cbp_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})

        elif reasoning_mode == "plan_and_solve":
            pas_confidence = []
            conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
            y_conf_pairs = []

            sampling_pas = SamplingParams(max_tokens=220, temperature=0.5, top_p=0.95, n=1)
            def gen_fn_pas(prompt_text: str):
                # Pass 1 — PLAN (slightly higher temp for diversity)
                plan_trigger = "Let's first understand the problem and devise a plan to solve it. Then, let's carry out the plan step by step."
                plan_prompt = f"{prompt_text}\n\n{plan_trigger}\n\nPlan:"
                plan_sampling = SamplingParams(max_tokens=200, temperature=0.7, top_p=0.95, n=1)
                plan_result = llm.generate([plan_prompt], plan_sampling)[0]
                plan_text = (plan_result.texts[0] if getattr(plan_result, 'texts', None) else "").strip()

                # Pass 2 — SOLVE (condition on the plan, low temp for accuracy)
                solve_prompt = (
                    f"{prompt_text}\n\nPlan:\n{plan_text}\n\n"
                    "Now follow the plan carefully and produce the final answer.\n"
                    "End with exactly 'Answer: 1' or 'Answer: 0' for binary, "
                    "or 'Answer: 0/1/2/3' for intensity."
                )
                solve_sampling = SamplingParams(max_tokens=220, temperature=0.0, top_p=0.95, n=1)
                return llm.generate([solve_prompt], solve_sampling)[0]

            for sample, prompt in zip(test_data, prompts):
                pred, raw_texts = try_generate_with_retries(
                    prompt=prompt,
                    generator_fn=gen_fn_pas,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)

                conf_bin, conf_score = None, None
                if raw_texts:
                    for response in reversed(raw_texts):
                        if response and parse_output(response, task) is not None:
                            conf_bin, conf_score = query_confidence_bin(llm, response, conf_sampling)
                            break

                pas_confidence.append({
                    "raw_response": (raw_texts[-1] if raw_texts else ""),
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            confidence_pas_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, pas_confidence)):
                confidence_pas_table.add_data(
                    idx, prompts[idx], sample["label"], pred,
                    conf_data["raw_response"], conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    sample["emotion"]
                )
            wandb.log({"confidence_table": confidence_pas_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})

        elif reasoning_mode == "rasc":
            rasc_confidence = []
            conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
            y_conf_pairs = []

            for sample, prompt in zip(test_data, prompts):
                pred, meta, new_flags = run_rasc(
                    prompt=prompt,
                    task=task,
                    llm=llm,
                    max_samples=top_k,  # you already use top_k as samples
                    min_conf=rasc_min_conf if rasc_min_conf is not None else 0.7,
                    alpha=rasc_alpha if rasc_alpha is not None else 0.5,
                    max_tries=max_retries
                )
                # For consistency with other modes, we also want the raw last response; run_rasc returns meta only.
                # So we just log meta conf and leave raw_response blank (or you can include the best reasoning text if you return it).
                all_preds.append(pred)
                all_raw.append([""])  # placeholder
                all_flagged.extend(new_flags)

                conf_bin  = meta.get("conf_bin")
                conf_score= meta.get("conf_score")

                rasc_confidence.append({
                    "raw_response": meta.get("winning_rationale", ""),
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })

                if conf_score is not None and pred is not None:
                    y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_score))

            confidence_rasc_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred", "raw_response", "conf_bin", "conf_score", "emotion"
            ])
            for idx, (sample, pred, conf_data) in enumerate(zip(test_data, all_preds, rasc_confidence)):
                confidence_rasc_table.add_data(
                    idx, prompts[idx], sample["label"], pred,
                    conf_data["raw_response"], conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    sample["emotion"]
                )
            wandb.log({"confidence_table": confidence_rasc_table})

            if y_conf_pairs:
                y_correct, confs = zip(*y_conf_pairs)
                auroc = roc_auc_score(y_correct, confs)
                auprc = average_precision_score(y_correct, confs)
            else:
                auroc = float("nan"); auprc = float("nan")
            wandb.log({"auroc": auroc, "auprc": auprc})

        elif reasoning_mode == "rankcot":
            rk_confidence = []
            y_conf_pairs = []
            conf_sampling = SamplingParams(160, 0.0, 1.0, 1)
            cot_sampling  = SamplingParams(220, 0.7, 0.95, 1)   # for generating CoTs
            score_sampling = SamplingParams(160, 0.0, 1.0, 1)    # for scoring CoTs
            answer_sampling = SamplingParams(160, 0.0, 0.95, 1) # for final answer

            TOP_K_DOCS = 5
            FUSE_TOP2 = False  # set True to concatenate top-2 CoTs before answering

            rankcot_rows = []

            for idx, (sample, prompt_base) in enumerate(zip(test_data, prompts)):
                query_text = sample["text"]
                docs = retrieve_docs(query_text, k=TOP_K_DOCS)

                cots = []
                scores = []
                flagged_local = []

                # 1) Generate one CoT per doc
                for d in docs:
                    cot_prompt = (
                        f"{prompt_base}\n\n"
                        f"Use ONLY the following retrieved information when reasoning:\n"
                        f"=== RETRIEVED SNIPPET ===\n{d}\n=== END SNIPPET ===\n\n"
                        "Think step by step using the snippet. End with an explicit 'Answer:' line."
                    )
                    try:
                        cot_text = llm.generate([cot_prompt], cot_sampling)[0].texts[0].strip()
                    except OpenAIError as e:
                        action = handle_openai_error(e, flagged_local, cot_prompt, stage="rankcot_cot")
                        if action in ("fatal", "flagged"):
                            cot_text = ""
                        else:
                            cot_text = ""
                    cots.append((d, cot_text))

                # 2) Score each CoT for relevance+faithfulness
                for (doc_text, cot_text) in cots:
                    if not cot_text:
                        scores.append(0.0)
                        continue
                    s = rankcot_score_cot(query_text, doc_text, cot_text, llm, score_sampling, flagged_local, alpha=0.5)
                    scores.append(s)

                # 3) Pick best (or fuse top-2)
                best_idx = int(np.argmax(scores)) if scores else -1
                if best_idx < 0 or not cots:
                    # emergency fallback: plain prompt
                    def gen_fn_rank_fallback(p: str):
                        return llm.generate([p], answer_sampling)[0]
                    pred, raw_texts = try_generate_with_retries(
                        prompt=prompt_base,
                        generator_fn=gen_fn_rank_fallback,
                        task=task,
                        max_retries=max_retries,
                        flagged_list=all_flagged
                    )
                    all_preds.append(pred); all_raw.append(raw_texts); all_flagged.extend(flagged_local)
                    # confidence
                    conf_bin, conf_score = None, None
                    if raw_texts:
                        for response in reversed(raw_texts):
                            if response and parse_output(response, task) is not None:
                                conf_bin, conf_score = query_confidence_bin(llm, response, conf_sampling)
                                break
                    rk_confidence.append({"raw_response": (raw_texts[-1] if raw_texts else ""), "conf_bin": conf_bin, "conf_score": conf_score})
                    # row
                    rankcot_rows.append({
                        "index": idx, "emotion": sample["emotion"], "gold": sample["label"],
                        "pred": pred, "winning_cot": "", "winning_score": None
                    })
                    continue

                # best or fused CoT
                if FUSE_TOP2 and len(cots) >= 2:
                    order = list(reversed(np.argsort(scores)))
                    i1, i2 = order[0], order[1]
                    winning_cot = (cots[i1][1] + "\n\n" + cots[i2][1]).strip()
                    winning_score = float((scores[i1] + scores[i2]) / 2.0)
                else:
                    winning_cot = cots[best_idx][1]
                    winning_score = float(scores[best_idx])

                # 4) Ask for final answer conditioned on the winning CoT
                final_prompt = (
                    f"{prompt_base}\n\n"
                    f"=== SELECTED CHAIN-OF-THOUGHT ===\n{winning_cot}\n=== END COT ===\n\n"
                    "Now, produce ONLY the final decision. "
                    "Binary: end with exactly 'Answer: 1' or 'Answer: 0'. "
                    "Intensity: end with exactly 'Answer: 0/1/2/3'."
                )
                def gen_fn_rank_final(p: str):
                    return llm.generate([p], answer_sampling)[0]

                pred, raw_texts = try_generate_with_retries(
                    prompt=final_prompt,
                    generator_fn=gen_fn_rank_final,
                    task=task,
                    max_retries=max_retries,
                    flagged_list=all_flagged
                )
                all_preds.append(pred)
                all_raw.append(raw_texts)
                all_flagged.extend(flagged_local)

                # confidence from parseable response
                conf_bin, conf_score = None, None
                if raw_texts:
                    for response in reversed(raw_texts):
                        if response and parse_output(response, task) is not None:
                            conf_bin, conf_score = query_confidence_bin(llm, response, conf_sampling)
                            break

                rk_confidence.append({
                    "raw_response": winning_cot,
                    "conf_bin": conf_bin,
                    "conf_score": conf_score
                })
                rankcot_rows.append({
                    "index": idx,
                    "emotion": sample["emotion"],
                    "gold": sample["label"],
                    "pred": pred,
                    "winning_cot": winning_cot,
                    "winning_score": winning_score
                })

            # Log RankCoT table (mirrors other modes + extras)
            rankcot_table = wandb.Table(columns=[
                "index", "prompt", "gold", "pred",
                "raw_response", "conf_bin", "conf_score",
                "emotion", "winning_cot_score"
            ])
            for (row, prompt_text, conf_data) in zip(rankcot_rows, prompts, rk_confidence):
                rankcot_table.add_data(
                    row["index"], prompt_text, row["gold"], row["pred"],
                    conf_data["raw_response"],
                    conf_data["conf_bin"] or "error",
                    conf_data["conf_score"] if conf_data["conf_score"] is not None else None,
                    row["emotion"], row["winning_score"]
                )
            wandb.log({"rankcot_table": rankcot_table})

            # AUROC/AUPRC like other modes
            if rk_confidence:
                y_conf_pairs = []
                for (sample, row, conf_data) in zip(test_data, rankcot_rows, rk_confidence):
                    pred = row["pred"]
                    if pred is not None and (conf_data["conf_score"] is not None):
                        y_conf_pairs.append((1 if pred == sample["label"] else 0, conf_data["conf_score"]))
                if y_conf_pairs:
                    y_correct, confs = zip(*y_conf_pairs)
                    auroc = roc_auc_score(y_correct, confs)
                    auprc = average_precision_score(y_correct, confs)
                else:
                    auroc = float("nan"); auprc = float("nan")
                wandb.log({"auroc": auroc, "auprc": auprc})
        else:
            raise ValueError(f"Unsupported reasoning_mode: {reasoning_mode}")

    # ----------------------------------------
    # 6) Compute metrics
    # ----------------------------------------
    
    if all_flagged:
        wandb.log({"flagged_prompts_count": len(all_flagged)})
        flagged_table = wandb.Table(columns=["prompt", "reason", "stage", "outputs"])
        for item in all_flagged:
            flagged_table.add_data(
            item.get("prompt",  ""),
            item.get("reason",   ""),
            item.get("stage",    ""),
            # raw texts from the failed generation
            str(item.get("outputs", []))
        )
        wandb.log({"flagged_prompts": flagged_table})


    preds_table = wandb.Table(columns=["index", "prompt", "raw_outputs", "parsed", "gold", "emotion"])
    for idx, (sample, prompt) in enumerate(zip(test_data, prompts)):
        raw = all_raw[idx] if idx < len(all_raw) else []
        pred = all_preds[idx] if idx < len(all_preds) else None
        preds_table.add_data(idx, prompt, str(raw), pred, sample["label"], sample["emotion"])
    wandb.log({"predictions_table": preds_table})

    emotion2refs = defaultdict(list)
    emotion2preds = defaultdict(list)
 
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
    f1_per_emotion = {}
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
        tnrs, tprs = [], []
        for emo in emotion2refs:
            y_true = [l for l,e in zip(all_labels_flat, all_emotions_flat) if e==emo]
            y_pred = [p for p,e in zip(all_preds_flat,  all_emotions_flat) if e==emo]
            if len(set(y_true)) < 2:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            tnrs.append(tn/(tn+fp) if (tn+fp)>0 else 0)
            tprs.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        avg_tnr = np.mean(tnrs) if tnrs else float("nan")
        avg_tpr = np.mean(tprs) if tprs else float("nan")

        # Log to WandB
        wandb.log({
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "accuracy": accuracy,
            "true_negative_rate_avg": avg_tnr,
            "true_positive_rate_avg": avg_tpr,
        })

        return {
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "accuracy": accuracy,
            "true_negative_rate_avg": avg_tnr,
            "true_positive_rate_avg": avg_tpr,
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
        pearson_per_emotion = {}
        for emo in emotion2refs:
            refs = emotion2refs[emo]
            preds = emotion2preds[emo]
            if len(refs) > 1 and len(set(refs)) > 1 and len(set(preds)) > 1:
                try:
                    pearson_per_emotion[emo] = pearsonr(refs, preds)[0]
                except Exception:
                    pearson_per_emotion[emo] = float("nan")
            else:
                pearson_per_emotion[emo] = float("nan")
        avg_pearson = np.nanmean(list(pearson_per_emotion.values()))

        # Spearman
        if len(set(all_labels_flat)) > 1 and len(set(all_preds_flat)) > 1:
            try:
                spearman_corr = spearmanr(all_labels_flat, all_preds_flat).correlation
            except Exception:
                spearman_corr = float("nan")
        else:
            spearman_corr = float("nan")

        # MAE + RMSE
        mse = mean_squared_error(all_labels_flat, all_preds_flat)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(all_labels_flat, all_preds_flat)

        # QWK
        try:
            qwk = cohen_kappa_score(all_labels_flat, all_preds_flat, weights="quadratic")
        except Exception:
            qwk = float("nan")

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
    prefix = ""
    if isinstance(llm_engine, ErrorMockLLM):
        prefix = "error_"
    elif isinstance(llm_engine, MockLLM):
        prefix = "mock_"
    if not args.skip_ablations:
        results = {}
        #Note when running evaulate model on test set here, we don't actaully give a directory that code can save all the results

        # 1. Prompt variants
        print("=== Ablation: Prompt Variants ===")
        variant_results = {}
        for variant, tmpl in prompt_variants.items():
            # Only run CBP variants when in complexity_based; only run ToT when in tree_of_thoughts
            if variant.startswith("cbp_") and reasoning_mode != "complexity_based":
                continue
            if variant == "tree_of_thoughts" and reasoning_mode != "tree_of_thoughts":
                continue

            wandb_run_name = (
                f"{prefix}ablation_main_{args.model_name.replace('/', '-')}_{args.task}_{args.language or 'all'}_{args.reasoning_mode}_nshot{args.n_shot}_top_k{args.top_k}_samplesize{args.sample_size}_balancingstrategy{args.balancing_strategy}_promptvariant{args.prompt_variant}"
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
                f"{prefix}ablation_main_{args.model_name.replace('/', '-')}_{args.task}_{args.language or 'all'}_{args.reasoning_mode}_nshot{args.n_shot}_top_k{args.top_k}_samplesize{args.sample_size}_balancingstrategy{args.balancing_strategy}_promptvariant{args.prompt_variant}"
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

        if reasoning_mode == "tree_of_thoughts":
            print("=== Ablation: ToT Beam Widths ===")
            tot_bw_results = {}
            for bw in [1, 2, 3, 4]:
                wandb_run_name = (
                    f"{prefix}ablation_main_{args.model_name.replace('/', '_')}"
                    f"_task{task}_lang{language}_mode{reasoning_mode}"
                    f"_bal{str(balanced).lower()}_strategy{args.balancing_strategy}"
                    f"_promptvariant{args.prompt_variant}_totbw{bw}"
                )
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=wandb_run_name,
                    config={
                        "ablation_type": "tot_beam_width",
                        "beam_width": bw,
                        "model": model_name,
                        "task": task,
                        "language": language,
                        "reasoning_mode": reasoning_mode,
                        "top_k": main_top_k,
                        "n_shot": main_n_shot,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data,
                    prompt_template=main_prompt,
                    task=task,
                    top_k=main_top_k,
                    n_shot=main_n_shot,
                    model_name=model_name,
                    out_json="...",
                    llm=llm,
                    reasoning_mode=reasoning_mode,
                    max_steps=max_steps,
                    beam_width=bw
                )
                tot_bw_results[bw] = scores
                wandb.log(scores)
                wandb.finish()
                score_val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  beam_width = {bw}: {score_val:.4f}")

            print("=== Ablation: ToT Max Steps ===")
            tot_steps_results = {}
            for steps in [2, 3, 4]:
                wandb_run_name = (
                    f"{prefix}ablation_main_{args.model_name.replace('/', '_')}"
                    f"_task{task}_lang{language}_mode{reasoning_mode}"
                    f"_bal{str(balanced).lower()}_strategy{args.balancing_strategy}"
                    f"_promptvariant{args.prompt_variant}_totsteps{steps}"
                )
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=wandb_run_name,
                    config={
                        "ablation_type": "tot_max_steps",
                        "max_steps": steps,
                        "model": model_name,
                        "task": task,
                        "language": language,
                        "reasoning_mode": reasoning_mode,
                        "top_k": main_top_k,
                        "n_shot": main_n_shot,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data,
                    prompt_template=main_prompt,
                    task=task,
                    top_k=main_top_k,
                    n_shot=main_n_shot,
                    model_name=model_name,
                    out_json="...",
                    llm=llm,
                    reasoning_mode=reasoning_mode,
                    max_steps=steps,
                    beam_width=tot_beam_width
                )
                tot_steps_results[steps] = scores
                wandb.log(scores)
                wandb.finish()
                score_val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  max_steps = {steps}: {score_val:.4f}")

            # keep results collected
            results["tot_beam_width"] = tot_bw_results
            results["tot_max_steps"] = tot_steps_results

        if reasoning_mode == "self_refine":
            print("=== Ablation: Self-Refine max_refinements ===")
            sr_results = {}
            for refinements in [1, 2, 3, 4]:
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=f"{prefix}ablation_sr_refinements_{refinements}",
                    config={
                        "ablation_type": "self_refine_max_refinements",
                        "max_refinements": refinements,
                        "model": model_name, "task": task, "language": language,
                        "reasoning_mode": reasoning_mode, "top_k": main_top_k, "n_shot": main_n_shot,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data, prompt_template=main_prompt, task=task,
                    top_k=main_top_k, n_shot=main_n_shot, model_name=model_name,
                    out_json="...", llm=llm, reasoning_mode=reasoning_mode,
                    max_steps=max_steps, beam_width=tot_beam_width,
                    self_refine_max_refinements=refinements
                )
                sr_results[refinements] = scores
                wandb.log(scores); wandb.finish()
                val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  max_refinements = {refinements}: {val:.4f}")
            results["self_refine_max_refinements"] = sr_results

        if reasoning_mode == "rasc":
            print("=== Ablation: RASC min_conf ===")
            rasc_minconf = {}
            for mc in [0.6, 0.7, 0.8]:
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=f"{prefix}ablation_rasc_minconf_{mc}",
                    config={
                        "ablation_type": "rasc_min_conf",
                        "min_conf": mc,
                        "model": model_name, "task": task, "language": language,
                        "reasoning_mode": reasoning_mode, "n_shot": main_n_shot,
                        "top_k": main_top_k,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data, prompt_template=main_prompt, task=task,
                    top_k=main_top_k, n_shot=main_n_shot, model_name=model_name,
                    out_json="...", llm=llm, reasoning_mode=reasoning_mode,
                    max_steps=max_steps, beam_width=tot_beam_width,
                    rasc_min_conf=mc
                )
                rasc_minconf[mc] = scores
                wandb.log(scores); wandb.finish()
                val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  min_conf = {mc}: {val:.4f}")
            results["rasc_min_conf"] = rasc_minconf

            print("=== Ablation: RASC alpha (QUALITY vs CONSISTENCY weight) ===")
            rasc_alpha_res = {}
            for a in [0.25, 0.5, 0.75]:
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=f"{prefix}ablation_rasc_alpha_{a}",
                    config={
                        "ablation_type": "rasc_alpha",
                        "alpha": a,
                        "model": model_name, "task": task, "language": language,
                        "reasoning_mode": reasoning_mode, "n_shot": main_n_shot,
                        "top_k": main_top_k,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data, prompt_template=main_prompt, task=task,
                    top_k=main_top_k, n_shot=main_n_shot, model_name=model_name,
                    out_json="...", llm=llm, reasoning_mode=reasoning_mode,
                    max_steps=max_steps, beam_width=tot_beam_width,
                    rasc_alpha=a
                )
                rasc_alpha_res[a] = scores
                wandb.log(scores); wandb.finish()
                val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  alpha = {a}: {val:.4f}")
            results["rasc_alpha"] = rasc_alpha_res
        if reasoning_mode == "complexity_based":
            print("=== Ablation: CBP forced level (skip probe) ===")
            cbp_forced = {}
            for level in ["simple", "medium", "complex"]:
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=f"{prefix}ablation_cbp_forced_{level}",
                    config={
                        "ablation_type": "cbp_force_level",
                        "level": level,
                        "model": model_name, "task": task, "language": language,
                        "reasoning_mode": reasoning_mode, "n_shot": main_n_shot,
                        "top_k": main_top_k,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data, prompt_template=main_prompt, task=task,
                    top_k=main_top_k, n_shot=main_n_shot, model_name=model_name,
                    out_json="...", llm=llm, reasoning_mode=reasoning_mode,
                    max_steps=max_steps, beam_width=tot_beam_width,
                    cbp_force_level=level
                )
                cbp_forced[level] = scores
                wandb.log(scores); wandb.finish()
                val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  force_level = {level}: {val:.4f}")
            results["cbp_force_level"] = cbp_forced

        if reasoning_mode == "tree_of_thoughts":
            print("=== Ablation: ToT search strategy ===")
            tot_search_res = {}
            for s in ["bfs", "dfs"]:
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=f"{prefix}ablation_tot_search_{s}",
                    config={
                        "ablation_type": "tot_search",
                        "search": s,
                        "model": model_name, "task": task, "language": language,
                        "reasoning_mode": reasoning_mode, "n_shot": main_n_shot, "top_k": main_top_k,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data, prompt_template=main_prompt, task=task,
                    top_k=main_top_k, n_shot=main_n_shot, model_name=model_name,
                    out_json="...", llm=llm, reasoning_mode=reasoning_mode,
                    max_steps=max_steps, beam_width=tot_beam_width,
                    tot_search=s
                )
                tot_search_res[s] = scores
                wandb.log(scores); wandb.finish()
                val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  search = {s}: {val:.4f}")
            results["tot_search"] = tot_search_res

            print("=== Ablation: ToT value threshold vth ===")
            tot_vth_res = {}
            for v in [0.3, 0.5, 0.7]:
                wandb.init(
                    entity="CongAndSiy",
                    project="emotion-eval",
                    name=f"{prefix}ablation_tot_vth_{v}",
                    config={
                        "ablation_type": "tot_vth",
                        "vth": v,
                        "model": model_name, "task": task, "language": language,
                        "reasoning_mode": reasoning_mode, "n_shot": main_n_shot, "top_k": main_top_k,
                    },
                    reinit=True
                )
                scores = evaluate_model_on_test_set(
                    test_data=test_data, prompt_template=main_prompt, task=task,
                    top_k=main_top_k, n_shot=main_n_shot, model_name=model_name,
                    out_json="...", llm=llm, reasoning_mode=reasoning_mode,
                    max_steps=max_steps, beam_width=tot_beam_width,
                    tot_vth=v
                )
                tot_vth_res[v] = scores
                wandb.log(scores); wandb.finish()
                val = scores["f1_macro"] if task == "binary" else scores["avg_pearson"]
                print(f"  vth = {v}: {val:.4f}")
            results["tot_vth"] = tot_vth_res

        # 3. top_k
        if reasoning_mode in ("self_consistency", "rasc","tree_of_thoughts"):
            print("=== Ablation: Top_k Values (self_consistency only) ===")
            topk_results = {}
            for k in topk_list:
                wandb_run_name = (
                    f"{prefix}ablation_main_{args.model_name.replace('/', '-')}_{args.task}_{args.language or 'all'}_{args.reasoning_mode}_nshot{args.n_shot}_top_k{args.top_k}_samplesize{args.sample_size}_balancingstrategy{args.balancing_strategy}_promptvariant{args.prompt_variant}"
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
                f"{prefix}ablation_main_{args.model_name.replace('/', '-')}_{args.task}_{args.language or 'all'}_{args.reasoning_mode}_nshot{args.n_shot}_top_k{args.top_k}_samplesize{args.sample_size}_balancingstrategy{args.balancing_strategy}_promptvariant{args.prompt_variant}"
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
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["half", "float16", "auto", "bfloat16"],
        help="Model compute dtype for vLLM. Use 'half' (fp16) on Turing/Volta GPUs and for GPTQ."
    )
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
        "--skip_ablations",
        action="store_true",
        help="If set, only run the main evaluation and skip all ablation loops.",
    )
    parser.add_argument(
        "--error_test",
        action="store_true",
        help="If set, use ErrorMockLLM to randomly simulate API errors.",
    )
    parser.add_argument(
        "--tot_search",
        choices=["bfs", "dfs"],
        default="bfs",
        help="Search policy for Tree of Thoughts (paper-style)"
    )
    parser.add_argument(
            "--tot_vth",
            type=float,
            default=0.5,
            help="Value threshold in [0,1] to prune weak branches"
    )
    parser.add_argument(
            "--tot_value_trials",
            type=int,
            default=1,
            help="Number of value (lookahead) trials per state before scoring"
    )
    args = parser.parse_args()
    
    if args.reasoning_mode not in ["self_consistency", "rasc", "rankcot", "tree_of_thoughts"]:
        print(f"[DEBUG] For reasoning_mode={args.reasoning_mode}, forcing top_k=1 (was {args.top_k})")
        args.top_k = 1

    if USE_MOCK_LLM:
        engine_choice = "mock"
    elif args.model_name.startswith("openai/") or args.model_name.startswith("google/gemini"):
        engine_choice = "openai"
    else:
        engine_choice = "vllm"

    if engine_choice == "openai":
        llm_engine = AzureEngineWrapper(model_name=args.model_name)
    elif engine_choice == "mock":
        llm_engine = MockLLM()
    elif engine_choice == "vllm":
        llm_engine = VLLMEngineWrapper(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype, 
        )
    else:
        raise ValueError(f"Unknown engine choice: {engine_choice}")
    prefix = ""
    if isinstance(llm_engine, ErrorMockLLM):
        prefix = "error_"
    elif isinstance(llm_engine, MockLLM):
        prefix = "mock_"

    wandb.init(
        entity="CongAndSiy",
        project="emotion-eval",  # Change if needed
        name=f"{prefix}_main_{args.model_name.replace('/', '-')}_{args.task}_{args.language or 'all'}_{args.reasoning_mode}_nshot{args.n_shot}_top_k{args.top_k}_samplesize{args.sample_size}_balancingstrategy{args.balancing_strategy}_promptvariant{args.prompt_variant}",
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
    data = []
    for row in sampled_df.itertuples(index=False):
        for emo in emotions_to_use:
            val = getattr(row, emo, 0)
            if args.task == "binary":
                label = 1 if int(val) > 0 else 0
            else:
                label = max(0, min(3, int(val)))
            data.append({
                "text": getattr(row, "text"),
                "emotion": emo,
                "label": label
            })
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
                emotions_to_use = [
                    emo for emo in EMOTIONS
                    if emo in sampled_df.columns and sampled_df[emo].sum() > 0
                ]
            else:  # intensity
                sampled_df = sample_dataset_intensity(
                    csv_path,
                    args.sample_size,
                    balanced=True,
                    balancing_strategy=args.balancing_strategy  # <-- ADD THIS
                )
                emotions_to_use = [
                    emo for emo in EMOTIONS
                    if emo in sampled_df.columns and sampled_df[emo].max() > 0
                ]

            data = []
            for row in sampled_df.itertuples(index=False):
                for emo in emotions_to_use:
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
                emotions_to_use = [emo for emo in EMOTIONS if emo in df.columns and df[emo].sum() > 0]
            elif args.task == "intensity":
                csv_path = os.path.join(TEST_DIRS["intensity"], f"{args.language}.csv")
                df = sample_dataset_intensity(
                    csv_path=csv_path,
                    sample_size=args.sample_size,
                    balanced=args.balanced,
                    balancing_strategy=args.balancing_strategy
                )
                emotions_to_use = [emo for emo in EMOTIONS if emo in df.columns and df[emo].max() > 0]
            data = []
            for row in df.itertuples(index=False):
                for emo in emotions_to_use:
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
            beam_width=max(1, int(args.top_k)) if args.top_k is not None else 3
        )
        #wandb.log(main_res)
        # 1) Debug-print the raw values
        if args.task == "binary":
            f1 = float(main_res.get("f1_macro", 0.0))
            acc = float(main_res.get("accuracy", 0.0))
            auroc = float(main_res.get("auroc", 0.0))
            auprc = float(main_res.get("auprc", 0.0))
            tnr = float(main_res.get("true_negative_rate_avg", 0.0))
            tpr = float(main_res.get("true_positive_rate_avg", 0.0))

            summary_table = wandb.Table(columns=[
                "model", "task", "reasoning_mode", "prompt_variant", "language",
                "f1_macro", "accuracy", "auroc", "auprc", "tnr","tpr"
            ])
            summary_table.add_data(
                args.model_name, args.task, args.reasoning_mode, args.prompt_variant, args.language,
                f1, acc, auroc, auprc, tnr, tpr
            )
            wandb.log({"summary_metrics": summary_table})

        elif args.task == "intensity":
            pearson = float(main_res.get("avg_pearson", 0.0))
            spearman = float(main_res.get("spearman", 0.0))
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
                    max_steps=max_steps,
                    tot_beam_width=beam_width,
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
