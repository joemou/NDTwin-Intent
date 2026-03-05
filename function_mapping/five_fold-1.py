import json
import torch
import gc
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# 0. Global Configurations
# ==========================================
ALL_DATA_PATH = "all_data.json" # 包含擴充資料的訓練/交叉驗證集
TEST_DATA_PATH = "test.json"    # ★ 你的獨立 50 題乾淨測試集
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096

# ★ 已替換為強化版 System Prompt，杜絕 API 幻覺與過度謹慎
SYSTEM_PROMPT = """You control a network system. You are a classification task agent. Respond in the specified JSON format. RULES: 1. If a command is potentially disruptive (e.g., Disable, PowerOff, Block, Delete, Drop, Modify, Reroute), your default action is to enter the \"discussion\" state to ask for confirmation. 2. If the user's prompt includes a confirmation flag like --confirm or --force, skip the discussion and proceed directly to \"answer\". 3. Keep the \"explanation\" field extremely concise (under 15 words). DO NOT think out loud or explain your internal logic."""

# ==========================================
# 1. Helper Functions (JSON Extraction & Comparison)
# ==========================================
def extract_json(response_str):
    try:
        if response_str.startswith("```json"): response_str = response_str[7:]
        if response_str.startswith("```"): response_str = response_str[3:]
        if response_str.endswith("```"): response_str = response_str[:-3]
        
        start_idx = response_str.find('{')
        end_idx = response_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            return json.loads(response_str[start_idx:end_idx])
        return None
    except Exception:
        return None

def compare_outputs(expected, actual):
    if actual is None: return False, "Failed to parse into a valid JSON format"
    if expected.get("state") != actual.get("state"):
        return False, f"State mismatch (Expected: {expected.get('state')}, Actual: {actual.get('state')})"
        
    if expected.get("state") == "answer":
        if expected.get("valid") != actual.get("valid"):
            return False, "Valid flag mismatch"
        exp_tasks = expected.get("tasks", [])
        act_tasks = actual.get("tasks", [])
        if len(exp_tasks) != len(act_tasks):
            return False, "Task count mismatch"
            
        for et, at in zip(exp_tasks, act_tasks):
            if et.get("type") != at.get("type"):
                return False, "Task type mismatch"
            
            et_params, at_params = et.get("parameters", {}), at.get("parameters", {})
            if "device_name" in et_params and et_params.get("device_name") != at_params.get("device_name"):
                return False, "Device Name parsing error"
    return True, "Success"

# ==========================================
# 2. Training Function (Single Fold)
# ==========================================
def train_fold(train_data_list, fold_idx):
    print(f"\n[Fold {fold_idx}] Loading model and setting up LoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    hf_dataset = Dataset.from_list(train_data_list)
    hf_dataset = hf_dataset.map(formatting_prompts_func, batched = True)

    print(f"[Fold {fold_idx}] Starting training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = hf_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2, 
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            num_train_epochs = 4, 
            learning_rate = 3e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"outputs_fold_{fold_idx}",
        ),
    )
    
    trainer.train()
    
    # 這裡可以選擇性註解掉，如果你不想每一折都存檔佔用硬碟空間
    # model.save_pretrained(f"lora_model_fold_{fold_idx}")
    # tokenizer.save_pretrained(f"lora_model_fold_{fold_idx}")
    
    return model, tokenizer, trainer

# ==========================================
# 3. Inference and Evaluation Function
# ==========================================
# ★ 新增 dataset_name 參數，用來區分報表名稱
def evaluate_fold(model, tokenizer, test_data_list, fold_idx, dataset_name):
    print(f"\n[Fold {fold_idx}] Starting inference and validation on [{dataset_name}] ({len(test_data_list)} samples)...")
    FastLanguageModel.for_inference(model)
    
    correct_count = 0
    failed_reports = []

    for idx, item in enumerate(tqdm(test_data_list, desc=f"Evaluating Fold {fold_idx} ({dataset_name})")):
        conversations = item.get("conversations", [])
        human_input = next(c["value"] for c in conversations if c["from"] == "human")
        expected_str = next(c["value"] for c in conversations if c["from"] == "gpt")
        expected_json = json.loads(expected_str)
        
        messages = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": human_input}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")

        input_length = inputs.shape[1]
        outputs = model.generate(
            inputs, max_new_tokens=512, use_cache=True, temperature=0.1, pad_token_id=tokenizer.eos_token_id
        )
        
        response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        actual_json = extract_json(response_str)
        
        is_correct, reason = compare_outputs(expected_json, actual_json)
        if is_correct:
            correct_count += 1
        else:
            failed_reports.append({
                "input": human_input,
                "reason": reason,
                "expected": expected_json,
                "actual_parsed": actual_json
            })
            
    accuracy = correct_count / len(test_data_list)
    print(f"[Fold {fold_idx} - {dataset_name}] Accuracy: {accuracy*100:.1f}% ({correct_count}/{len(test_data_list)})")
    
    if failed_reports:
        # ★ 錯誤報告檔名加上 dataset_name 避免覆蓋
        with open(f"error_report_fold_{fold_idx}_{dataset_name}.json", 'w', encoding='utf-8') as f:
            json.dump(failed_reports, f, indent=2, ensure_ascii=False)
            
    return accuracy

# ==========================================
# 4. Main Program: 5-Fold Cross-Validation Pipeline
# ==========================================
def main():
    print(f"Loading ALL_DATA from: {ALL_DATA_PATH}")
    with open(ALL_DATA_PATH, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        
    print(f"Loading TEST_DATA from: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        external_test_data = json.load(f)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_accuracies = []       # 紀錄每一折自己的 Test 準確率
    external_accuracies = [] # 紀錄每一折對外部 test.json 的準確率

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data), start=1):
        print("\n" + "="*50)
        print(f"🚀 Starting Fold {fold}/5")
        print("="*50)
        
        train_data = [all_data[i] for i in train_idx]
        cv_test_data = [all_data[i] for i in test_idx]
        
        print(f"Train size: {len(train_data)}, CV Test size: {len(cv_test_data)}, External Test size: {len(external_test_data)}")
        
        # 1. 訓練
        model, tokenizer, trainer = train_fold(train_data, fold)
        
        # 2. 評估: Fold 內部的測試集
        cv_acc = evaluate_fold(model, tokenizer, cv_test_data, fold, dataset_name="CV_Test")
        cv_accuracies.append(cv_acc)
        
        # 3. 評估: 外部獨立的 test.json
        ext_acc = evaluate_fold(model, tokenizer, external_test_data, fold, dataset_name="External_Test")
        external_accuracies.append(ext_acc)
        
        # 4. 釋放記憶體
        print(f"[Fold {fold}] Clearing VRAM memory...")
        del model
        del tokenizer
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 輸出最終雙料報表
    print("\n" + "="*50)
    print("📊 5-Fold Cross-Validation & External Test Final Report")
    print("="*50)
    
    print("【Cross-Validation Test Results】 (模型泛化能力測試)")
    for i, acc in enumerate(cv_accuracies, start=1):
        print(f"  Fold {i} Accuracy : {acc*100:.2f}%")
    print(f"  🏆 CV Average Accuracy: {sum(cv_accuracies) / len(cv_accuracies)*100:.2f}%\n")
    
    print("【External 'test.json' Results】 (零資料洩漏極限測試)")
    for i, acc in enumerate(external_accuracies, start=1):
        print(f"  Fold {i} Accuracy : {acc*100:.2f}%")
    print(f"  🏆 External Average Accuracy: {sum(external_accuracies) / len(external_accuracies)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()