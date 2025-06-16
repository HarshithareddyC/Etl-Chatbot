import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize your Ollama model
llm = ChatOllama(model="llama3")  # or "mistral", "flan", etc.

# List of allowed operation names from your cleaning.py and etl_engine.py
VALID_OPS = [
    "remove_duplicates", "impute_missing", "fill_missing", "rename_columns",
    "drop_columns", "normalize_case", "split_column", "combine_columns",
    "scale_columns", "parse_dates", "convert_dtypes", "filter_rows",
    "sort_by", "set_index", "reset_index", "group_and_aggregate",
    "regex_transform", "derive_date_parts", "create_ratio"
]

# LangChain prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert ETL assistant. Convert the user's natural language instruction into a JSON list of ETL steps. "
        "Use only the following operations: " + ", ".join(VALID_OPS) + ". "
        "Each step must be a dictionary with an 'operation' and any necessary parameters. "
        "DO NOT include explanation, notes, or 'description' fields. Return ONLY valid JSON starting with [ and ending with ]."
    )),
    ("user", "{input}")
])

# LangChain chain
chain = prompt | llm

# Step cleaner for operation key + extras
def clean_step(step):
    # Fix common wrong LLM outputs
    op_map = {
        "drop_duplicates": "remove_duplicates",
        "fill_missing_values": "impute_missing",
        "impute": "impute_missing",
    }

    if 'step' in step:
        step['operation'] = step.pop('step')

    if 'operation' in step:
        step['operation'] = op_map.get(step['operation'], step['operation'])

    # Remove non-functional / extra keys
    for key in ['description', 'parameters', 'params']:
        step.pop(key, None)

    # Strip out irrelevant keys based on known function signatures
    op = step.get("operation")
    if op == "remove_duplicates":
        # This function takes only (df), so remove everything else
        step = {"operation": "remove_duplicates"}

    return step



# Main function called from app.py
def parse_etl_instruction(text):
    response = chain.invoke({"input": f"Return a JSON list of ETL steps for: {text}"})
    raw = response.content if hasattr(response, "content") else str(response)

    # Extract JSON using regex
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        raise ValueError(f"Invalid JSON output from LLM:\n{raw}")

    json_str = match.group(0)

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return [clean_step(s) for s in parsed]
        return clean_step(parsed)
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode cleaned JSON:\n{json_str}")
