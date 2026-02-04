import os
import re
import json

def extract_code_from_llm_response(llm_code: str) -> str | None:
    if not llm_code:
        return None
    try:
        if llm_code.strip().startswith("{") and llm_code.strip().endswith("}"):
            data = json.loads(llm_code)
            if isinstance(data, dict):
                for key in ["python", "code", "solution", "content"]:
                    if key in data and isinstance(data[key], str) and "def solver" in data[key]:
                        return extract_code_from_llm_response(data[key])
    except json.JSONDecodeError:
        pass
    code = llm_code
    code_search_area = None
    if "### FINAL SOLUTION ###" in llm_code:
        parts = llm_code.split("### FINAL SOLUTION ###")
        code_search_area = parts[-1]
    pattern = r"```python(.*?)```"
    if not code_search_area:
        blocks = re.findall(pattern, llm_code, re.DOTALL)
        for block in reversed(blocks):
            if "def solver" in block:
                code = block.strip()
                code_search_area = "FOUND_IN_BLOCK"
                break
    if code_search_area and code_search_area != "FOUND_IN_BLOCK":
        match = re.search(pattern, code_search_area, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            if "def solver" in code_search_area:
                lines = code_search_area.splitlines()
                start_idx = -1
                for i, line in enumerate(lines):
                    if "def solver" in line:
                        start_idx = i
                        break
                if start_idx != -1:
                    code = "\n".join(lines[start_idx:])
    if not code_search_area:
        if "def solver" in llm_code:
            lines = llm_code.splitlines()
            start_idx = -1
            for i, line in enumerate(lines):
                if "def solver" in line:
                    start_idx = i
                    break
            if start_idx != -1:
                code = "\n".join(lines[start_idx:])
    return code if "def solver" in code else None

def find_calls(data):
    calls = []
    if isinstance(data, dict):
        if "Full raw LLM response" in data:
            name = data.get("name", "unknown")
            calls.append((name, data))
        else:
            for k, v in data.items():
                if isinstance(v, dict) and "Full raw LLM response" in v:
                    calls.append((k, v))
                else:
                    calls.extend(find_calls(v))
    elif isinstance(data, list):
        for item in data:
            calls.extend(find_calls(item))
    return calls

def generate():
    directory = "logs_safe_15"
    target_task = "dfadab01"
    target_test = "1"
    
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    pattern = re.compile(rf'.*{target_task}_{target_test}_step_([a-zA-Z0-9]+)\.json$')
    
    solutions = []
    original_prompt = None
    total_runs = 0
    total_pass = 0
    
    for filename in sorted(files):
        if not pattern.match(filename):
            continue
        
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        calls = find_calls(data)
        for name, call_val in calls:
            if "codegen" not in name.lower():
                continue
                
            total_runs += 1
            
            if original_prompt is None:
                original_prompt = call_val.get("Full raw LLM call", "")
            
            v_details = call_val.get("verification_details")
            if v_details and isinstance(v_details, dict):
                status = v_details.get("status", "")
                if status == "PASS":
                    total_pass += 1
                    llm_resp = call_val.get("Full raw LLM response", "")
                    code = extract_code_from_llm_response(llm_resp)
                    if code and code not in solutions:
                        solutions.append(code)

    if not original_prompt:
        print("Error: Could not find original prompt.")
        return

    with open("prompt.txt", "w") as f:
        f.write(f"Below is a prompt that was run {total_runs} times:\n\n")
        f.write(f"<PROMPT START>\n{original_prompt}\n<PROMPT STOP>\n\n\n")
        f.write(f"For {total_pass} of these prompts, the code generated solved all the training examples, but the code is different and generates different results on the test input data. Below are the {len(solutions)} different solutions:\n\n")
        
        for i, sol in enumerate(solutions, 1):
            f.write(f"<SOLUTION {i} START>\n{sol}\n<SOLUTION {i} STOP>\n\n")
            
        f.write(f"Your task is to understand these {len(solutions)} solutions, and assess how well they've understood the problem, and how likely their solutions are to provide the correct solution to the test input. Often, new mechanics are introduced in the test example for which the solutions do not generalize well.\n\n")
        f.write("Please output what you think is the right mechanic for solving the problem, and express it both in natural language and in a complete solver() function implementation.\n")
    
    print(f"Generated prompt.txt with {total_runs} runs, {total_pass} PASSes, and {len(solutions)} unique solutions.")

if __name__ == "__main__":
    generate()
