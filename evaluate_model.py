import json
import torch
from tqdm import tqdm # 用於顯示進度條 (如果沒安裝，請先 pip install tqdm)
from unsloth import FastLanguageModel

# ==========================================
# 1. 設定區
# ==========================================
MODEL_PATH = "lora_model"          # 你的微調模型路徑
TEST_DATA_PATH = "test.json" # 先前產生的 50 筆測試資料檔案
ERROR_REPORT_PATH = "error_report.json" # 驗證錯誤的詳細報告輸出位置

SYSTEM_PROMPT = """You control a network system. You are a classification task agent. Respond in the specified JSON format. RULES: 1. If a command is potentially disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute), your default action is to enter the \"discussion\" state to ask for confirmation. 2. If the user's prompt includes a confirmation flag like --confirm or --force, skip the discussion and proceed directly to \"answer\". 3. Keep the \"explanation\" field extremely concise (under 15 words). DO NOT think out loud or explain your internal logic."""

# ==========================================
# 2. 載入模型 (沿用你的設定)
# ==========================================
print(f"Loading model from: {MODEL_PATH} ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# ==========================================
# 3. 推論與解析輔助函數
# ==========================================
def ask_agent(user_input):
    messages = [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human", "value": user_input}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")

    input_length = inputs.shape[1]

    outputs = model.generate(
        inputs,
        max_new_tokens = 512,
        use_cache = True,
        temperature = 0.1,
        pad_token_id = tokenizer.eos_token_id
    )
    
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response

def extract_json(response_str):
    """安全地從字串中萃取並解析 JSON"""
    try:
        # 清理可能殘留的 Markdown 標記
        if response_str.startswith("```json"): response_str = response_str[7:]
        if response_str.startswith("```"): response_str = response_str[3:]
        if response_str.endswith("```"): response_str = response_str[:-3]
        
        start_idx = response_str.find('{')
        end_idx = response_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_str[start_idx:end_idx]
            return json.loads(json_str)
        return None
    except Exception:
        return None

# ==========================================
# 4. 核心比對邏輯
# ==========================================
def compare_outputs(expected, actual):
    """比較預期輸出與實際輸出，返回 (是否正確, 錯誤原因)"""
    if actual is None:
        return False, "無法解析為有效的 JSON 格式"
        
    # 檢查 1: State 是否一致 (最重要)
    if expected.get("state") != actual.get("state"):
        return False, f"State 不符 (預期: {expected.get('state')}, 實際: {actual.get('state')})"
        
    # 檢查 2: 若為 answer，檢查 tasks 是否正確
    if expected.get("state") == "answer":
        # 檢查 valid
        if expected.get("valid") != actual.get("valid"):
            return False, f"Valid 標記不符 (預期: {expected.get('valid')}, 實際: {actual.get('valid')})"
            
        exp_tasks = expected.get("tasks", [])
        act_tasks = actual.get("tasks", [])
        
        if len(exp_tasks) != len(act_tasks):
            return False, f"Tasks 數量不符 (預期: {len(exp_tasks)}, 實際: {len(act_tasks)})"
            
        # 逐一檢查 Task 類型與關鍵參數
        for et, at in zip(exp_tasks, act_tasks):
            if et.get("type") != at.get("type"):
                return False, f"Task 類型不符 (預期: {et.get('type')}, 實際: {at.get('type')})"
            
            # 檢查設備名稱是否抓取正確 (如果預期結果有這個參數的話)
            et_params = et.get("parameters", {})
            at_params = at.get("parameters", {})
            if "device_name" in et_params and et_params.get("device_name") != at_params.get("device_name"):
                return False, f"Device Name 解析錯誤 (預期: {et_params.get('device_name')}, 實際: {at_params.get('device_name')})"

    return True, "Success"

# ==========================================
# 5. 執行批量測試
# ==========================================
def run_evaluation():
    print(f"Loading test cases from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    total_cases = len(test_cases)
    parsed_count = 0
    correct_count = 0
    failed_reports = []

    print(f"Starting evaluation of {total_cases} test cases...\n")
    
    # 使用 tqdm 顯示進度條
    for idx, item in enumerate(tqdm(test_cases, desc="Evaluating")):
        conversations = item.get("conversations", [])
        human_input = next(c["value"] for c in conversations if c["from"] == "human")
        expected_str = next(c["value"] for c in conversations if c["from"] == "gpt")
        expected_json = json.loads(expected_str)
        
        # 呼叫模型推論
        actual_str = ask_agent(human_input)
        actual_json = extract_json(actual_str)
        
        if actual_json is not None:
            parsed_count += 1
            
        is_correct, reason = compare_outputs(expected_json, actual_json)
        
        if is_correct:
            correct_count += 1
        else:
            failed_reports.append({
                "id": idx + 1,
                "input": human_input,
                "reason": reason,
                "expected": expected_json,
                "actual_raw": actual_str,
                "actual_parsed": actual_json
            })

    # ==========================================
    # 6. 輸出測試報告
    # ==========================================
    print("\n" + "="*40)
    print("📊 Evaluation Report")
    print("="*40)
    print(f"Total Test Cases : {total_cases}")
    print(f"JSON Parse Rate  : {parsed_count}/{total_cases} ({parsed_count/total_cases*100:.1f}%)")
    print(f"Accuracy (Exact) : {correct_count}/{total_cases} ({correct_count/total_cases*100:.1f}%)")
    print("="*40)
    
    if failed_reports:
        print(f"⚠️ Found {len(failed_reports)} mismatches. Saving details to {ERROR_REPORT_PATH}...")
        with open(ERROR_REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(failed_reports, f, indent=2, ensure_ascii=False)
        print("Please check the error report to see where the model hallucinated or failed the rules.")
    else:
        print("🎉 Perfect Score! The model perfectly learned all tasks and edge cases.")

if __name__ == "__main__":
    run_evaluation()