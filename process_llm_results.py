import os
import glob
import json
import pandas as pd
import argparse

###############################################################################
# 1. Model Name Mapping & Desired Order
###############################################################################
model_name_map = {
    "final_bothTasks_Qwen_Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "final_bothTasks_databricks_dolly-v2-12b": "Dolly-v2-12B",
    "final_bothTasks_meta-llama_Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "final_bothTasks_mistralai_Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "final_bothTasks_deepseek-ai_DeepSeek-R1-Distill-Llama-70B": "DeepSeek-R1-70B",
    "final_bothTasks_openai_gpt-4": "ChatGPT-GPT4",
    "final_bothTasks_google_gemini-2F":"Google Gemini2Flash"

}

desired_model_order = [
    "Qwen2.5-72B",
    "Dolly-v2-12B",
    "Llama-3.3-70B",
    "Mixtral-8x7B",
    "DeepSeek-R1-70B",
    "ChatGPT-GPT4",
    "Google Gemini2Flash"
]

def get_short_model_name_from_data(data):
    """
    Convert the "model_name" string from the JSON file into a short model name.
    We replace "/" with "_" and add the prefix "final_bothTasks_" so that the key
    matches one in our model_name_map. If not found, we return the original string.
    """
    model_full = data.get("model_name", "")
    key = "final_bothTasks_" + model_full.replace("/", "_")
    return model_name_map.get(key, model_full)

###############################################################################
# 2. Utility: Load data
###############################################################################
def load_data(json_path):
    """Return the entire JSON as a dict (or None if load fails)."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[DEBUG] load_data failed on {json_path}: {e!r}")
        return None

    if not isinstance(data, dict):
        print(f"[DEBUG] load_data skipping {json_path}: top‑level is a {type(data).__name__}, not dict")
        return None

    return data

###############################################################################
# 3. Ablation Tables for the Binary Task 
###############################################################################
def collect_prompt_variant_eng(all_files):
    """
    Table 1:
    Rows = [Prompt v1, Prompt v2, Prompt v3]
    Columns = [models in desired order]
    For files where task is 'binary' and language is 'eng', 
    extract the ablation->prompt_variant->v1,v2,v3 (macro_f1) values.
    Multiply by 100 and round.
    """
    rows = ["Prompt v1", "Prompt v2", "Prompt v3"]
    table_data = {}

    for path in all_files:
        data = load_data(path)
        if not isinstance(data, dict) or data.get("task") != "binary":
            continue
        # Only consider English files for this table
        if data.get("language", "").lower() != "eng":
            continue

        model_name = get_short_model_name_from_data(data)
        ablation = data.get("ablation")
        if not ablation:
            continue
        
        prompt_var = ablation.get("prompt_variant", {})
        v1 = prompt_var.get("v1", {}).get("macro_f1", float('nan'))
        v2 = prompt_var.get("v2", {}).get("macro_f1", float('nan'))
        v3 = prompt_var.get("v3", {}).get("macro_f1", float('nan'))
        
        table_data[model_name] = [v1, v2, v3]
    
    df = pd.DataFrame(table_data, index=rows)
    # Reorder columns
    existing_cols = [m for m in desired_model_order if m in df.columns]
    df = df[existing_cols]
    df = df * 100
    df = df.round(2)
    return df

def collect_few_shot_eng(all_files):
    """
    Table 2:
    Rows = [0-shot, 1-shot, 2-shot, 4-shot]
    Columns = [models]
    For files where task is 'binary' and language is 'eng', 
    extract the ablation->few_shot->{0,1,2,4} (macro_f1) values.
    Multiply by 100 and round.
    """
    rows = ["0-shot", "1-shot", "2-shot", "4-shot", "8-shot", "16-shot"]
    table_data = {}

    for path in all_files:
        data = load_data(path)
        if data is None or data.get("task") != "binary":
            print(f"[DEBUG] SKIPPED {path} → load_data returned None")
            continue
        else:
            print(f"[DEBUG] LOADED  {path} → OK (type={type(data).__name__})")
        if data.get("language", "").lower() != "eng":
            continue

        model_name = get_short_model_name_from_data(data)
        ablation = data.get("ablation")
        if not ablation:
            continue
        
        fs_dict = ablation.get("few_shot", {})
        val_0 = fs_dict.get("0", {}).get("macro_f1", float('nan'))
        val_1 = fs_dict.get("1", {}).get("macro_f1", float('nan'))
        val_2 = fs_dict.get("2", {}).get("macro_f1", float('nan'))
        val_4 = fs_dict.get("4", {}).get("macro_f1", float('nan'))
        val_8 = fs_dict.get("8", {}).get("macro_f1", float('nan'))
        val_16 = fs_dict.get("16", {}).get("macro_f1", float('nan'))
        
        table_data[model_name] = [val_0, val_1, val_2, val_4, val_8, val_16]
    
    df = pd.DataFrame(table_data, index=rows)
    existing_cols = [m for m in desired_model_order if m in df.columns]
    df = df[existing_cols]
    df = df * 100
    df = df.round(2)
    return df

def collect_top_k_eng(all_files):
    """
    Table 3:
    Rows = [@1, @2, @4]
    Columns = [models]
    For files where task is 'binary' and language is 'eng', 
    extract the ablation->top_k->{1,2,4} (macro_f1) values.
    Multiply by 100 and round.
    """
    rows = ["@1", "@2", "@4", "@8"]
    table_data = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != "binary":
            continue
        if data.get("language", "").lower() != "eng":
            continue

        model_name = get_short_model_name_from_data(data)
        ablation = data.get("ablation")
        if not ablation:
            continue
        
        topk_dict = ablation.get("top_k", {})
        val_1 = topk_dict.get("1", {}).get("macro_f1", float('nan'))
        val_2 = topk_dict.get("2", {}).get("macro_f1", float('nan'))
        val_4 = topk_dict.get("4", {}).get("macro_f1", float('nan'))
        val_8 = topk_dict.get("8", {}).get("macro_f1", float('nan'))
        
        table_data[model_name] = [val_1, val_2, val_4, val_8]
    
    df = pd.DataFrame(table_data, index=rows)
    existing_cols = [m for m in desired_model_order if m in df.columns]
    df = df[existing_cols]
    df = df * 100
    df = df.round(2)
    return df

def collect_english_vs_native(all_files):
    """
    Table 4:
    Rows = multi-index of (Language, 'Prompt in eng') and (Language, 'Prompt in {lang}')
    Columns = [models]
    For each file (with task 'binary') that has non-null ablation->english_v1_vs_native_v1,
    extract:
      - f1_english_v1: the result when the prompt is in English,
      - f1_native_v1: the result when the prompt is in the target (native) language.
    (Both values are taken from the 'macro_f1' field.)
    """
    all_rows = set()  # Will hold tuples of (language, variant label)
    data_by_model = {}

    for path in all_files:
        data = load_data(path)
        if not data or data.get("task") != "binary":
            continue

        model_name = get_short_model_name_from_data(data)
        language = data.get("language", "")
        ablation = data.get("ablation")
        if not ablation:
            continue

        eng_vs_nat = ablation.get("english_v1_vs_native_v1")
        if not eng_vs_nat:
            continue

        f1_eng = eng_vs_nat.get("f1_english_v1", {}).get("macro_f1", float('nan'))
        f1_nat = eng_vs_nat.get("f1_native_v1", {}).get("macro_f1", float('nan'))

        if model_name not in data_by_model:
            data_by_model[model_name] = {}

        # The file's language is the target language.
        # We record both the result when using an English prompt and the result
        # when using a prompt in the target language.
        data_by_model[model_name][(language, "Prompt in eng")] = f1_eng
        data_by_model[model_name][(language, f"Prompt in {language}")] = f1_nat

        all_rows.add((language, "Prompt in eng"))
        all_rows.add((language, f"Prompt in {language}"))

    # Sort the row keys (first by language)
    sorted_rows = sorted(all_rows, key=lambda x: x[0])
    model_names = [m for m in desired_model_order if m in data_by_model]

    # Build table matrix
    table_values = []
    for row in sorted_rows:
        row_vals = []
        for m in model_names:
            val = data_by_model[m].get(row, float('nan'))
            row_vals.append(val)
        table_values.append(row_vals)

    idx = pd.MultiIndex.from_tuples(sorted_rows, names=["Language", "Ablation"])
    df = pd.DataFrame(table_values, index=idx, columns=model_names)
    df = df * 100
    df = df.round(2)
    return df

###############################################################################
# 4. Main Results Tables for 'binary' & 'intensity'
###############################################################################
def collect_main_results(all_files, task="binary", n_shot=None, prompt_variant=None, top_k=None):
    """
    Extract main_result (macro_f1 or avg_pearson) filtered by optional metadata:
      - n_shot (int)
      - prompt_variant (str)
      - top_k (int)

    Builds table with rows=languages, columns=models.
    """
    all_langs = set()
    data_by_model = {}

    for path in all_files:
        data = load_data(path)
        if data is None:
            print(f"[DEBUG main] SKIPPING {path}: load_data → None")
            continue
        print(f"[DEBUG main] LOADED   {path} → task={data.get('task')!r}, lang={data.get('language')!r}, n_shot={data.get('n_shot')!r}, variant={data.get('prompt_variant')!r}, top_k={data.get('top_k')!r}")
        if not data or data.get("task") != task:
            print(f"[DEBUG main] SKIPPING {path}: wrong task {data.get('task')!r} (expected {task!r})")
            continue
        # Check metadata filters if specified
        if n_shot is not None and data.get("n_shot") != n_shot:
            print(f"[DEBUG main] SKIPPING {path}: n_shot={data.get('n_shot')!r} (expected {n_shot})")
            continue
        if prompt_variant is not None and data.get("prompt_variant") != prompt_variant:
            print(f"[DEBUG main] SKIPPING {path}: prompt_variant={data.get('prompt_variant')!r} (expected {prompt_variant!r})")
            continue
        if top_k is not None and data.get("top_k") != top_k:
            print(f"[DEBUG main] SKIPPING {path}: top_k={data.get('top_k')!r} (expected {top_k})")
            continue
        print(f"[DEBUG main] INCLUDING {path}")
        model_name = get_short_model_name_from_data(data)
        language = data.get("language", "")
        main_result = data.get("main_result", {})
        if task == "intensity":
            mr = main_result.get("avg_pearson", float('nan'))
        else:
            mr = main_result.get("macro_f1", float('nan'))

        if model_name not in data_by_model:
            data_by_model[model_name] = {}
        data_by_model[model_name][language] = mr
        all_langs.add(language)

    sorted_langs = sorted(all_langs)
    sorted_models = [m for m in desired_model_order if m in data_by_model]

    # Build a table: rows = languages, columns = models
    matrix = []
    for lang in sorted_langs:
        row = []
        for model in sorted_models:
            val = data_by_model[model].get(lang, float('nan'))
            row.append(val)
        matrix.append(row)

    df = pd.DataFrame(matrix, index=sorted_langs, columns=sorted_models)
    df = df * 100
    df = df.round(2)
    return df

###############################################################################
# 5. main()
###############################################################################
def main(task, n_shot, prompt_variant, top_k):
    os.makedirs("llm_track_ab_results", exist_ok=True)
    # 1. Gather all JSON files from the results folder.
    all_files = [
    p for p in glob.glob("llm_track_ab_results/*.json")
    if os.path.basename(p).startswith("results_")
    ]
    print("\n[DEBUG] Found result files:")
    for p in all_files:
        print("  ", p)
    print(f"[DEBUG] Total files found: {len(all_files)}\n")

    # 2. Construct ablation tables for the binary task.
    df_prompt_variant = collect_prompt_variant_eng(all_files)
    df_few_shot = collect_few_shot_eng(all_files)
    df_top_k = collect_top_k_eng(all_files)
    df_eng_vs_native = collect_english_vs_native(all_files)

    # 3. Construct main results tables.
    df_main = collect_main_results(
        all_files,
        task=task,
        n_shot=n_shot,
        prompt_variant=prompt_variant,
        top_k=top_k
    )

    # 4. Output ablation tables (CSV + LaTeX)
    os.makedirs("llm_track_ab_results", exist_ok=True)
    df_prompt_variant.to_csv("llm_track_ab_results/table_prompt_variant.csv", float_format="%.2f")
    df_few_shot.to_csv("llm_track_ab_results/table_few_shot.csv", float_format="%.2f")
    df_top_k.to_csv("llm_track_ab_results/table_top_k.csv", float_format="%.2f")
    df_eng_vs_native.to_csv("llm_track_ab_results/table_english_vs_native.csv", float_format="%.2f")

    with open("llm_track_ab_results/table_prompt_variant.tex", "w") as f:
        f.write(df_prompt_variant.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_few_shot.tex", "w") as f:
        f.write(df_few_shot.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_top_k.tex", "w") as f:
        f.write(df_top_k.to_latex(float_format="%.2f"))
    with open("llm_track_ab_results/table_english_vs_native.tex", "w") as f:
        f.write(df_eng_vs_native.to_latex(float_format="%.2f", multirow=True))

    # 5. Output main results tables (CSV + LaTeX)
    csv_name = f"table_main_{task}_{n_shot}shot_{prompt_variant}_topk{top_k}.csv"
    tex_name = csv_name.replace(".csv", ".tex")
    df_main.to_csv(f"llm_track_ab_results/{csv_name}", float_format="%.2f")
    with open(f"llm_track_ab_results/{tex_name}", "w") as f:
        f.write(df_main.to_latex(float_format="%.2f"))

    print("All tables have been saved to CSV and LaTeX files in llm_track_ab_results/.")
    # 6. Collect flagged prompts if any
    flagged_files = glob.glob("llm_track_ab_results/*_flagged.json")
    all_flagged = []
    for path in flagged_files:
        with open(path, "r", encoding="utf-8") as f:
            flagged = json.load(f)
            for entry in flagged:
                entry["source_file"] = os.path.basename(path)
            all_flagged.extend(flagged)

    # Save as CSV for review
    if all_flagged:
        flagged_df = pd.DataFrame(all_flagged)
        flagged_df.to_csv("llm_track_ab_results/flagged_prompts.csv", index=False)
        print(f"{len(all_flagged)} flagged prompts saved to flagged_prompts.csv.")
    else:
        print("No flagged prompts found.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task",           choices=["binary","intensity"], default="binary")
    p.add_argument("--n_shot",         type=int,          default=4)
    p.add_argument("--prompt_variant", type=str,          default="v2")
    p.add_argument("--top_k",          type=int,          default=1)
    args = p.parse_args()

    main(
        task=args.task,
        n_shot=args.n_shot,
        prompt_variant=args.prompt_variant,
        top_k=args.top_k
    )