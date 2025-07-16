import os
from llms import evaluate_model_on_test_set, load_test_data_multicolumn, TASK_CONFIGS

# Use your actual test CSV path
test_csv = "path/to/test_data.csv"
task = "binary"  # or "intensity"
model_name = "openai/gpt-4"
n_shot = 0
top_k = 1

# Set dummy environment variables if using Azure
os.environ["AZURE_OPENAI_KEY"] = "18W17o5SWFUi6HTmovJNqtPn7xbMw4wvqwOZ6dX16u189b4Xg4qNJQQJ99BGACYeBjFXJ3w3AAABACOGDWaH"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt4t.openai.azure.com/"

# Load data
data = load_test_data_multicolumn(test_csv, task=task)

# Pick a prompt template
prompt_template = TASK_CONFIGS[task]["prompt_variants"]["v1"]

# Set dummy output file path
out_json = "llm_track_ab_results/test_eval.json"

# Call function with minimal setup (engine=None for GPT)
result = evaluate_model_on_test_set(
    engine=None,
    test_data=data,
    prompt_template=prompt_template,
    task=task,
    top_k=top_k,
    n_shot=n_shot,
    model_name=model_name,
    out_json=out_json
)

print("EVALUATION RESULT:")
print(result)
